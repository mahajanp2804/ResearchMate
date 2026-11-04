from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from app.models import IndexingStatus, ProcessingStatus
from app.schemas.common import ErrorDetail, ORMModel


class DocumentSummary(ORMModel):
    id: str
    filename: str
    title: str | None
    page_count: int | None
    file_size_bytes: int
    processing_status: ProcessingStatus
    indexing_status: IndexingStatus
    # Explanation of a failure, or a warning such as pages without text.
    status_message: str | None
    chunk_count: int
    extracted_metadata: dict[str, Any]
    upload_date: datetime
    updated_at: datetime


class DocumentDetail(DocumentSummary):
    file_hash: str
    embedding_model: str | None
    indexed_at: datetime | None
    pages_without_text: list[int]


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int


class UploadResult(BaseModel):
    filename: str
    status: Literal["created", "duplicate", "rejected"]
    # For duplicates this is the copy already in the library.
    document: DocumentSummary | None = None
    error: ErrorDetail | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]
    created: int
    duplicates: int
    rejected: int
