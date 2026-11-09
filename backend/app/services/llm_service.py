"""LLM access behind a small provider interface.

The RAG pipeline only calls ``LLMProvider.generate(system_prompt, turns)``,
so supporting another API or a local model means adding one class here and
one entry in ``_PROVIDERS``; nothing else changes.

Errors from the provider are translated into the application's own error
types with messages a user can act on. The API key is never included in an
error message or a log line.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from app.config import Settings
from app.exceptions import (
    RateLimitedError,
    ResearchMateError,
    ServiceUnavailableError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    # The model hit its output limit, so the text may end mid-sentence.
    truncated: bool = False


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        *,
        system_prompt: str,
        turns: Sequence[ChatTurn],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


# --- Gemini --------------------------------------------------------------------

# Rate limits and server errors are often momentary, so the SDK retries them
# with exponential backoff before an error reaches the user.
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
RETRY_ATTEMPTS = 3


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: int,
        client=None,  # injectable for tests
    ) -> None:
        from google import genai
        from google.genai import types

        self.model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,  # milliseconds
                retry_options=types.HttpRetryOptions(
                    attempts=RETRY_ATTEMPTS,
                    initial_delay=1.0,
                    max_delay=8.0,
                    http_status_codes=RETRY_STATUS_CODES,
                ),
            ),
        )

    def generate(
        self,
        *,
        system_prompt: str,
        turns: Sequence[ChatTurn],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        from google.genai import errors, types

        contents = [
            types.Content(
                role="user" if turn.role == "user" else "model",
                parts=[types.Part(text=turn.content)],
            )
            for turn in turns
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature if temperature is None else temperature,
            max_output_tokens=max_output_tokens or self._max_output_tokens,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            raise self._translate_api_error(exc) from exc
        except Exception as exc:
            raise self._translate_transport_error(exc) from exc
        return self._parse(response)

    def _parse(self, response) -> LLMResponse:
        from google.genai import types

        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            raise UpstreamServiceError(
                "Gemini declined to process this request (safety filter).", code="llm_blocked"
            )
        candidate = response.candidates[0] if response.candidates else None
        finish = candidate.finish_reason if candidate else None
        text = (response.text or "").strip()
        if not text:
            if finish == types.FinishReason.MAX_TOKENS:
                raise UpstreamServiceError(
                    "The model used up its output limit before answering. "
                    "Increase LLM_MAX_OUTPUT_TOKENS in backend/.env.",
                    code="llm_truncated",
                )
            if finish == types.FinishReason.SAFETY:
                raise UpstreamServiceError(
                    "Gemini withheld its answer (safety filter).", code="llm_blocked"
                )
            raise UpstreamServiceError("Gemini returned an empty answer.", code="llm_empty")

        usage = response.usage_metadata
        return LLMResponse(
            text=text,
            model=response.model_version or self.model,
            prompt_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            truncated=finish == types.FinishReason.MAX_TOKENS,
        )

    def _translate_api_error(self, exc) -> ResearchMateError:
        # Only the status is logged: provider messages can echo request data.
        logger.warning("Gemini API error %s %s", exc.code, exc.status)
        if exc.code in (401, 403) or "API_KEY_INVALID" in str(exc.details):
            return ServiceUnavailableError(
                "Gemini rejected the API key. Check GEMINI_API_KEY in backend/.env "
                "and restart the backend.",
                code="llm_auth_failed",
            )
        if exc.code == 429:
            return RateLimitedError(
                "Gemini's rate limit or quota was reached. Wait a minute and try again, "
                "or check the limits of your API plan.",
                code="llm_quota_exceeded",
            )
        if exc.code == 404:
            return ServiceUnavailableError(
                f"Gemini model '{self.model}' was not found. Check GEMINI_MODEL in "
                "backend/.env against https://ai.google.dev/gemini-api/docs/models.",
                code="llm_model_not_found",
            )
        if exc.code is not None and exc.code >= 500:
            return UpstreamServiceError(
                "Gemini is temporarily unavailable. Please try again shortly.",
                code="llm_unavailable",
            )
        return UpstreamServiceError(
            f"Gemini could not process the request ({exc.status or exc.code}).",
            code="llm_request_failed",
        )

    def _translate_transport_error(self, exc: Exception) -> ResearchMateError:
        if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
            return UpstreamTimeoutError(
                "Gemini did not respond in time. Try again, or increase LLM_TIMEOUT_SECONDS.",
                code="llm_timeout",
            )
        logger.warning("Could not reach Gemini: %s", type(exc).__name__)
        return UpstreamServiceError(
            "Could not reach Gemini. Check the internet connection.", code="llm_unreachable"
        )


# --- Provider selection ------------------------------------------------------

_PROVIDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "gemini": lambda settings: GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    ),
}


def create_llm_provider(settings: Settings) -> LLMProvider:
    if not settings.llm_configured:
        raise ServiceUnavailableError(
            "No language model is configured. Add GEMINI_API_KEY to backend/.env and restart "
            "the backend (a free key is available at https://aistudio.google.com/apikey).",
            code="llm_not_configured",
        )
    return _PROVIDERS[settings.llm_provider](settings)


def cached_provider(factory: Callable[[], LLMProvider]) -> Callable[[], LLMProvider]:
    """Create the provider on first use and reuse it; failures are retried next time."""
    lock = threading.Lock()
    instance: list[LLMProvider] = []

    def get() -> LLMProvider:
        with lock:
            if not instance:
                instance.append(factory())
            return instance[0]

    return get
