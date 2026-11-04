"""FastAPI application factory.

``create_app`` wires settings, the database, middleware, error handlers and
routers together. Tests call it with their own ``Settings``; uvicorn uses the
module-level ``app``.
"""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import ResearchMateError
from app.routers import system
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _error_response(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ResearchMateError)
    async def handle_domain_error(_request: Request, exc: ResearchMateError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {"location": list(error["loc"]), "message": error["msg"]} for error in exc.errors()
        ]
        return _error_response(
            422, "invalid_input", "Request validation failed", {"errors": errors}
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Full traceback goes to the log; the client only gets a generic message
        # so internal paths and library details are not leaked.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error_response(500, "internal_error", "An unexpected error occurred")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    database = Database(settings.database_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings.ensure_directories()
        database.create_tables()
        if not settings.llm_configured:
            logger.warning("GEMINI_API_KEY is not set; question answering will be unavailable")
        logger.info("%s %s started", settings.app_name, settings.app_version)
        yield
        database.dispose()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered research paper analysis with hybrid RAG and citations.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.1f ms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    _register_exception_handlers(app)
    app.include_router(system.router)
    return app


app = create_app()
