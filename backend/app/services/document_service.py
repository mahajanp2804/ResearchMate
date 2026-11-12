"""Uploading, listing and deleting research papers."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import InvalidInputError, NotFoundError, ResearchMateError
from app.models import ActivityType, Chunk, Document, DocumentPage, ProcessingStatus
from app.models.base import new_id
from app.services import ingestion_service, pdf_service
from app.services.activity_service import record_activity
from app.services.ingestion_service import IngestionService
from app.services.research_gap_service import forget_document
from app.utils.file_validation import (
    has_pdf_extension,
    looks_like_pdf,
    sanitize_filename,
    stream_to_file,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

UploadStatus = Literal["created", "duplicate", "rejected"]


@dataclass
class UploadOutcome:
    filename: str
    status: UploadStatus
    document: Document | None = None
    error: ResearchMateError | None = None
    # True when the caller should schedule background indexing.
    needs_indexing: bool = False


def ingest_upload(
    db: Session,
    settings: Settings,
    ingestion: IngestionService,
    raw_filename: str | None,
    stream: BinaryIO,
) -> UploadOutcome:
    """Validate, de-duplicate, extract, store and chunk one uploaded file.

    Every failure before storage becomes a ``rejected`` outcome with a
    user-facing reason, so one bad file in a batch does not fail the others.
    Once stored, a chunking failure leaves the paper in the library with
    status ``failed`` so it can be retried.
    """
    filename = sanitize_filename(raw_filename)
    try:
        outcome = _store(db, settings, filename, stream)
    except ResearchMateError as error:
        logger.info("Rejected upload %r: %s", filename, error.code)
        return UploadOutcome(filename=filename, status="rejected", error=error)

    if outcome.status == "created" and outcome.document is not None:
        outcome.needs_indexing = ingestion.process(db, outcome.document)
    return outcome


def _store(db: Session, settings: Settings, filename: str, stream: BinaryIO) -> UploadOutcome:
    if not has_pdf_extension(filename):
        raise InvalidInputError("Only PDF files are supported.", code="unsupported_file_type")

    document_id = new_id()
    # Written under a temporary name first, so a half-processed upload never
    # looks like a stored document.
    temp_path = settings.upload_dir / f".{document_id}.part"
    stored = stream_to_file(stream, temp_path, settings.max_upload_size_bytes)
    try:
        if stored.size_bytes == 0:
            raise InvalidInputError("The file is empty.", code="empty_file")
        if not looks_like_pdf(stored.header):
            raise InvalidInputError(
                "This file is not a PDF, even though its name ends in .pdf.",
                code="unsupported_file_type",
            )

        existing = _find_by_hash(db, stored.sha256)
        if existing is not None:
            return UploadOutcome(filename=filename, status="duplicate", document=existing)

        extracted = pdf_service.extract_pdf(
            temp_path,
            original_filename=filename,
            max_pages=settings.max_pages_per_document,
        )

        final_path = settings.upload_dir / f"{document_id}.pdf"
        document = Document(
            id=document_id,
            filename=filename,
            stored_path=final_path.name,
            file_hash=stored.sha256,
            file_size_bytes=stored.size_bytes,
            title=extracted.title,
            page_count=extracted.page_count,
            extracted_metadata={**extracted.metadata, "title_source": extracted.title_source},
            # Text is extracted; chunking runs next and moves this to completed.
            processing_status=ProcessingStatus.PROCESSING,
            status_message=ingestion_service.missing_text_warning(
                extracted.pages_without_text, extracted.page_count
            ),
            pages=[
                DocumentPage(
                    page_number=page.page_number,
                    text=page.text,
                    char_count=page.char_count,
                    has_text=page.has_text,
                )
                for page in extracted.pages
            ],
        )
        db.add(document)
        # Flush first: ActivityEvent has no ORM relationship to Document, so
        # SQLAlchemy would not otherwise insert the document row before it.
        db.flush()
        record_activity(
            db, ActivityType.DOCUMENT_UPLOADED, f"Uploaded “{extracted.title}”", document_id
        )

        temp_path.replace(final_path)
        try:
            db.commit()
        except IntegrityError:
            # Another request stored the same file between our check and commit.
            db.rollback()
            final_path.unlink(missing_ok=True)
            existing = _find_by_hash(db, stored.sha256)
            if existing is None:
                raise
            return UploadOutcome(filename=filename, status="duplicate", document=existing)

        logger.info("Stored document %s (%d pages)", document_id, extracted.page_count)
        return UploadOutcome(filename=filename, status="created", document=document)
    finally:
        temp_path.unlink(missing_ok=True)


def _find_by_hash(db: Session, file_hash: str) -> Document | None:
    return db.scalar(select(Document).where(Document.file_hash == file_hash))


def list_chunks(
    db: Session, document_id: str, *, limit: int, offset: int
) -> tuple[list[Chunk], int]:
    get_document(db, document_id)  # 404 for unknown documents
    total = (
        db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id))
        or 0
    )
    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    ).all()
    return list(chunks), total


def list_documents(db: Session, *, limit: int, offset: int) -> tuple[list[Document], int]:
    total = db.scalar(select(func.count()).select_from(Document)) or 0
    documents = db.scalars(
        select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return list(documents), total


def get_document(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found.", details={"document_id": document_id})
    return document


def get_document_file(db: Session, settings: Settings, document_id: str) -> tuple[Document, Path]:
    document = get_document(db, document_id)
    upload_dir = settings.upload_dir.resolve()
    path = (upload_dir / document.stored_path).resolve()
    # stored_path is generated by us, but never serve anything outside the
    # upload directory even if the database were tampered with.
    if not path.is_relative_to(upload_dir) or not path.is_file():
        raise NotFoundError("The file for this document is missing.")
    return document, path


def delete_document(
    db: Session, settings: Settings, ingestion: IngestionService, document_id: str
) -> None:
    """Delete a document, its pages, chunks and profile (via cascade), the gap
    analyses that include it, its vectors and its file."""
    document = get_document(db, document_id)
    path = settings.upload_dir / document.stored_path
    title = document.title or document.filename
    chunk_ids = ingestion.chunk_ids(db, document_id)

    db.delete(document)
    forget_document(db, document_id)
    record_activity(db, ActivityType.DOCUMENT_DELETED, f"Deleted “{title}”")
    db.commit()

    # Vectors and the file are removed after the commit. If either step
    # failed, the leftovers would be harmless: vectors without a chunk row are
    # never returned and are cleaned up at the next startup.
    ingestion.remove_vectors(chunk_ids)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove file for deleted document %s", document_id)
    logger.info("Deleted document %s", document_id)
