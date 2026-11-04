"""Domain errors raised by services.

Services raise these instead of ``HTTPException`` so they stay independent of
the web layer; ``main.py`` turns them into JSON error responses in one place.
"""

from typing import Any


class ResearchMateError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        # A specific code (e.g. "encrypted_pdf") lets the UI react to the exact
        # reason while the class still decides the HTTP status.
        if code is not None:
            self.code = code
        self.details = details or {}


class InvalidInputError(ResearchMateError):
    status_code = 422
    code = "invalid_input"


class NotFoundError(ResearchMateError):
    status_code = 404
    code = "not_found"


class ConflictError(ResearchMateError):
    status_code = 409
    code = "conflict"


class PayloadTooLargeError(ResearchMateError):
    status_code = 413
    code = "payload_too_large"


class ServiceUnavailableError(ResearchMateError):
    """A required dependency is not configured or not reachable (e.g. no LLM key)."""

    status_code = 503
    code = "service_unavailable"


class UpstreamServiceError(ResearchMateError):
    """An external service (e.g. the LLM API) returned an error or timed out."""

    status_code = 502
    code = "upstream_error"
