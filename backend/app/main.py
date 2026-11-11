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
from app.routers import chat, documents, research, retrieval, system
from app.services.bm25_service import BM25Service
from app.services.embedding_service import get_sentence_transformer_embedder
from app.services.ingestion_service import EmbedderProvider, IngestionService, TokenizerProvider
from app.services.llm_service import cached_provider, create_llm_provider
from app.services.paper_profile_service import PaperProfileService
from app.services.rag_pipeline import LLMProviderFactory, RagPipeline
from app.services.reranker import get_cross_encoder_reranker
from app.services.retrieval_service import RerankerProvider, RetrievalService
from app.services.tokenization import get_huggingface_tokenizer
from app.services.vector_store import VectorStore
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Allowance for multipart boundaries and headers on top of the file bytes.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


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


def create_app(
    settings: Settings | None = None,
    *,
    tokenizer_provider: TokenizerProvider | None = None,
    embedder_provider: EmbedderProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
    llm_provider: LLMProviderFactory | None = None,
) -> FastAPI:
    """Build the application.

    The providers let tests swap in deterministic fakes. By default the
    embedding model, its tokenizer, the reranker and the LLM client are created
    on first use, so the API starts quickly and only pays the loading cost when
    it is needed.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    database = Database(settings.database_url)
    if tokenizer_provider is None:

        def tokenizer_provider():
            return get_huggingface_tokenizer(settings.embedding_model_name)

    if embedder_provider is None:

        def embedder_provider():
            return get_sentence_transformer_embedder(
                settings.embedding_model_name,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                query_prefix=settings.embedding_query_prefix,
            )

    if reranker_provider is None:

        def reranker_provider():
            return get_cross_encoder_reranker(
                settings.reranker_model_name, device=settings.embedding_device
            )

    if llm_provider is None:
        # Created on first use; a missing key is reported then, not at startup.
        llm_provider = cached_provider(lambda: create_llm_provider(settings))

    ingestion = IngestionService(
        settings=settings,
        database=database,
        vector_store=VectorStore(settings.index_dir),
        tokenizer_provider=tokenizer_provider,
        embedder_provider=embedder_provider,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings.ensure_directories()
        database.create_tables()
        ingestion.reconcile()
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
    app.state.ingestion = ingestion
    app.state.retrieval = RetrievalService(
        settings=settings,
        ingestion=ingestion,
        bm25=BM25Service(k1=settings.bm25_k1, b=settings.bm25_b),
        reranker_provider=reranker_provider,
    )
    app.state.rag = RagPipeline(
        settings=settings, retrieval=app.state.retrieval, llm_provider=llm_provider
    )
    app.state.profiles = PaperProfileService(
        settings=settings, retrieval=app.state.retrieval, llm_provider=llm_provider
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    max_upload_request_bytes = (
        settings.max_upload_size_bytes * settings.max_files_per_upload + MULTIPART_OVERHEAD_BYTES
    )

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        # Starlette buffers the whole multipart body before the endpoint runs,
        # so an obviously oversized upload is refused from its Content-Length
        # header first. Each file is still size-checked while it is read.
        if request.method == "POST" and request.url.path == documents.UPLOAD_PATH:
            content_length = request.headers.get("content-length", "")
            if content_length.isdigit() and int(content_length) > max_upload_request_bytes:
                return _error_response(
                    413,
                    "payload_too_large",
                    f"Upload is too large. Each file may be at most "
                    f"{settings.max_upload_size_mb} MB, "
                    f"with up to {settings.max_files_per_upload} files per upload.",
                )
        return await call_next(request)

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
    app.include_router(documents.router)
    app.include_router(retrieval.router)
    app.include_router(chat.router)
    app.include_router(research.router)
    return app


app = create_app()
