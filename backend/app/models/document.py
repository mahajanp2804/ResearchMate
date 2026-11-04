import enum
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, BigInteger, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin, UTCDateTime, new_id, string_enum

if TYPE_CHECKING:
    from app.models.chunk import Chunk


class ProcessingStatus(enum.StrEnum):
    """Text extraction and chunking progress."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexingStatus(enum.StrEnum):
    """Embedding and vector-index progress."""

    NOT_INDEXED = "not_indexed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Path of the stored file relative to the upload directory. Files are saved
    # under a generated name, never the user-supplied one.
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # SHA-256 of the file bytes; the unique constraint enforces duplicate detection.
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    title: Mapped[str | None] = mapped_column(String(512))
    page_count: Mapped[int | None] = mapped_column(Integer)
    # Author, subject, keywords, creation date, etc. as reported by the PDF.
    extracted_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    processing_status: Mapped[ProcessingStatus] = mapped_column(
        string_enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False
    )
    indexing_status: Mapped[IndexingStatus] = mapped_column(
        string_enum(IndexingStatus), default=IndexingStatus.NOT_INDEXED, nullable=False
    )
    # User-facing explanation of a failure or warning (e.g. "no extractable text").
    status_message: Mapped[str | None] = mapped_column(Text)

    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # What the current vectors were built with. If the file hash, model and
    # chunking settings all match on re-index, re-embedding is skipped.
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    chunking_signature: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )

    @property
    def upload_date(self) -> datetime:
        return self.created_at

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r}>"
