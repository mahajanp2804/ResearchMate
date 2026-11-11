"""The Gemini provider against a stand-in client: request mapping, parsing, errors."""

from pathlib import Path

import httpx
import pytest
from google.genai import errors, types

from app.exceptions import ResearchMateError
from app.services.llm_service import ChatTurn, GeminiProvider, cached_provider, create_llm_provider
from tests.conftest import make_settings

API_KEY = "secret-key-for-tests"


class FakeModels:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.requests: list[dict] = []

    def generate_content(self, **request):
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome) -> None:
        self.models = FakeModels(outcome)


def response(text: str | None, finish=types.FinishReason.STOP) -> types.GenerateContentResponse:
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=120, candidates_token_count=15
        ),
        model_version="gemini-3.6-flash",
    )


def provider(outcome) -> tuple[GeminiProvider, FakeClient]:
    client = FakeClient(outcome)
    gemini = GeminiProvider(
        api_key=API_KEY,
        model="gemini-3.6-flash",
        temperature=0.1,
        max_output_tokens=8192,
        timeout_seconds=60,
        client=client,
    )
    return gemini, client


def generate(gemini: GeminiProvider):
    return gemini.generate(
        system_prompt="Answer from the sources.",
        turns=[
            ChatTurn("user", "What is FedProx?"),
            ChatTurn("assistant", "A method."),
            ChatTurn("user", "Why?"),
        ],
    )


def test_request_maps_turns_system_prompt_and_settings() -> None:
    gemini, client = provider(response("FedProx adds a proximal term [S1]."))
    generate(gemini)

    request = client.models.requests[0]
    assert request["model"] == "gemini-3.6-flash"
    assert [content.role for content in request["contents"]] == ["user", "model", "user"]
    assert request["contents"][0].parts[0].text == "What is FedProx?"
    assert request["config"].system_instruction == "Answer from the sources."
    assert request["config"].temperature == 0.1
    assert request["config"].max_output_tokens == 8192


def test_per_call_overrides() -> None:
    gemini, client = provider(response("Standalone question?"))
    gemini.generate(
        system_prompt="Rewrite.",
        turns=[ChatTurn("user", "q")],
        temperature=0.0,
        max_output_tokens=200,
    )
    config = client.models.requests[0]["config"]
    assert (config.temperature, config.max_output_tokens) == (0.0, 200)


def test_json_schema_requests_structured_output() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    gemini, client = provider(response('{"answer": "a proximal term"}'))
    result = gemini.generate(
        system_prompt="Extract.", turns=[ChatTurn("user", "q")], json_schema=schema
    )
    config = client.models.requests[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert result.text == '{"answer": "a proximal term"}'


def test_plain_text_by_default() -> None:
    gemini, client = provider(response("text"))
    generate(gemini)
    config = client.models.requests[0]["config"]
    assert config.response_mime_type is None and config.response_json_schema is None


def test_response_is_parsed_with_usage() -> None:
    gemini, _ = provider(response("  FedProx adds a proximal term [S1].  "))
    result = generate(gemini)
    assert result.text == "FedProx adds a proximal term [S1]."
    assert result.model == "gemini-3.6-flash"
    assert (result.prompt_tokens, result.output_tokens) == (120, 15)
    assert result.truncated is False


def test_text_cut_off_by_the_output_limit_is_flagged() -> None:
    gemini, _ = provider(response("FedProx adds a", finish=types.FinishReason.MAX_TOKENS))
    assert generate(gemini).truncated is True


def client_error(code: int, status: str, message: str, reason: str | None = None):
    body = {"error": {"code": code, "status": status, "message": message}}
    if reason:
        body["error"]["details"] = [{"reason": reason}]
    error_type = errors.ServerError if code >= 500 else errors.ClientError
    return error_type(code, body)


@pytest.mark.parametrize(
    ("outcome", "status_code", "code"),
    [
        (
            client_error(400, "INVALID_ARGUMENT", "API key not valid.", "API_KEY_INVALID"),
            503,
            "llm_auth_failed",
        ),
        (client_error(403, "PERMISSION_DENIED", "Permission denied."), 503, "llm_auth_failed"),
        (client_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded."), 429, "llm_quota_exceeded"),
        (client_error(404, "NOT_FOUND", "models/x is not found."), 503, "llm_model_not_found"),
        (client_error(503, "UNAVAILABLE", "Overloaded."), 502, "llm_unavailable"),
        (client_error(400, "INVALID_ARGUMENT", "Bad request."), 502, "llm_request_failed"),
        (httpx.ReadTimeout("timed out"), 504, "llm_timeout"),
        (httpx.ConnectError("no route"), 502, "llm_unreachable"),
        (response(None, finish=types.FinishReason.MAX_TOKENS), 502, "llm_truncated"),
        (response(None, finish=types.FinishReason.SAFETY), 502, "llm_blocked"),
        (
            types.GenerateContentResponse(
                prompt_feedback=types.GenerateContentResponsePromptFeedback(
                    block_reason=types.BlockedReason.SAFETY
                )
            ),
            502,
            "llm_blocked",
        ),
        (response(None), 502, "llm_empty"),
    ],
)
def test_failures_become_clear_errors(outcome, status_code: int, code: str) -> None:
    gemini, _ = provider(outcome)
    with pytest.raises(ResearchMateError) as error:
        generate(gemini)
    assert (error.value.status_code, error.value.code) == (status_code, code)
    assert API_KEY not in error.value.message


def test_missing_key_is_reported_before_any_call(tmp_path: Path) -> None:
    with pytest.raises(ResearchMateError) as error:
        create_llm_provider(make_settings(tmp_path))
    assert error.value.code == "llm_not_configured"
    assert "GEMINI_API_KEY" in error.value.message


def test_configured_key_creates_a_gemini_provider(tmp_path: Path) -> None:
    created = create_llm_provider(make_settings(tmp_path, gemini_api_key=API_KEY))
    assert created.name == "gemini"
    assert created.model == "gemini-3.6-flash"


def test_cached_provider_creates_once_and_retries_failures() -> None:
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first attempt fails")
        return "provider"

    get = cached_provider(factory)
    with pytest.raises(RuntimeError):
        get()
    assert get() == "provider"
    assert get() == "provider"
    assert len(calls) == 2
