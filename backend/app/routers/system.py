from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies import DbSession, IngestionDep, SettingsDep
from app.schemas.system import (
    AnalysisInfo,
    ChunkingInfo,
    DashboardStats,
    HealthResponse,
    LLMInfo,
    RetrievalInfo,
    SystemInfo,
    VectorIndexInfo,
)
from app.services import stats_service
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(db: DbSession, settings: SettingsDep) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        database_status = "ok"
    except SQLAlchemyError:
        logger.exception("Database health check failed")
        database_status = "error"

    return HealthResponse(
        status="ok" if database_status == "ok" else "degraded",
        version=settings.app_version,
        database=database_status,
        llm_configured=settings.llm_configured,
    )


@router.get("/system/info", response_model=SystemInfo)
def system_info(settings: SettingsDep, ingestion: IngestionDep) -> SystemInfo:
    # Built field by field on purpose: dumping Settings wholesale would risk
    # sending secrets or local filesystem paths to the browser.
    store = ingestion.vector_store
    return SystemInfo(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        embedding_model_name=settings.embedding_model_name,
        embedding_device=settings.embedding_device,
        vector_index=VectorIndexInfo(
            vector_count=store.vector_count,
            model_name=store.model_name,
            dimension=store.dimension,
        ),
        max_upload_size_mb=settings.max_upload_size_mb,
        max_files_per_upload=settings.max_files_per_upload,
        max_pages_per_document=settings.max_pages_per_document,
        chunking=ChunkingInfo(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        ),
        retrieval=RetrievalInfo(
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            rrf_k=settings.rrf_k,
            reranker_enabled=settings.reranker_enabled,
            reranker_model_name=settings.reranker_model_name,
            rerank_candidates=settings.rerank_candidates,
            min_rerank_score=settings.min_rerank_score,
            final_context_chunks=settings.final_context_chunks,
            max_context_tokens=settings.max_context_tokens,
            min_passages_per_paper=settings.min_passages_per_paper,
        ),
        analysis=AnalysisInfo(
            profile_passages_per_aspect=settings.profile_passages_per_aspect,
            profile_max_context_tokens=settings.profile_max_context_tokens,
            max_compare_papers=settings.max_compare_papers,
            gap_passages_per_question=settings.gap_passages_per_question,
            gap_max_tokens_per_paper=settings.gap_max_tokens_per_paper,
            max_gap_papers=settings.max_gap_papers,
        ),
        llm=LLMInfo(
            provider=settings.llm_provider,
            model=settings.gemini_model,
            configured=settings.llm_configured,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
    )


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(db: DbSession) -> DashboardStats:
    return stats_service.get_dashboard_stats(db)
