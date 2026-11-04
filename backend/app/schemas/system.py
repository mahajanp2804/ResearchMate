from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.models import ActivityType, IndexingStatus, ProcessingStatus
from app.schemas.common import ORMModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: Literal["ok", "error"]
    llm_configured: bool


class ChunkingInfo(BaseModel):
    chunk_size_tokens: int
    chunk_overlap_tokens: int


class RetrievalInfo(BaseModel):
    dense_top_k: int
    bm25_top_k: int
    rrf_k: int
    reranker_enabled: bool
    reranker_model_name: str
    rerank_candidates: int
    final_context_chunks: int
    max_context_tokens: int


class LLMInfo(BaseModel):
    provider: str
    model: str
    configured: bool
    temperature: float
    max_output_tokens: int


class SystemInfo(BaseModel):
    """Non-secret runtime configuration shown on the Settings page."""

    app_name: str
    version: str
    environment: str
    embedding_model_name: str
    max_upload_size_mb: int
    max_files_per_upload: int
    max_pages_per_document: int
    chunking: ChunkingInfo
    retrieval: RetrievalInfo
    llm: LLMInfo


class RecentDocument(ORMModel):
    id: str
    filename: str
    title: str | None
    page_count: int | None
    processing_status: ProcessingStatus
    indexing_status: IndexingStatus
    created_at: datetime


class ActivityItem(ORMModel):
    id: int
    event_type: ActivityType
    description: str
    document_id: str | None
    created_at: datetime


class DashboardStats(BaseModel):
    total_papers: int
    indexed_papers: int
    total_chunks: int
    total_questions: int
    total_conversations: int
    recent_documents: list[RecentDocument]
    recent_activity: list[ActivityItem]
