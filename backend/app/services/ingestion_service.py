"""Turns a stored document's page text into chunks (and, from Phase 4, vectors)."""

from collections.abc import Callable

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import ResearchMateError
from app.models import ActivityType, Chunk, Document, IndexingStatus, ProcessingStatus
from app.services import chunking_service
from app.services.activity_service import record_activity
from app.services.tokenization import Tokenizer
from app.utils.logging import get_logger

logger = get_logger(__name__)

TokenizerProvider = Callable[[], Tokenizer]

# Listing every page number in a warning stops being useful past this point.
MAX_PAGES_LISTED_IN_WARNING = 10


def missing_text_warning(pages_without_text: list[int], page_count: int) -> str | None:
    if not pages_without_text:
        return None
    count = len(pages_without_text)
    if count <= MAX_PAGES_LISTED_IN_WARNING:
        label = "Page" if count == 1 else "Pages"
        pages = ", ".join(str(number) for number in pages_without_text)
        subject = f"{label} {pages} {'has' if count == 1 else 'have'}"
    else:
        subject = f"{count} of {page_count} pages have"
    return (
        f"{subject} no extractable text (likely images or scans). "
        "That content won't be searchable because OCR is not supported yet."
    )


def chunk_document(
    db: Session, settings: Settings, document: Document, tokenizer: Tokenizer
) -> int:
    """Replace the document's chunks with fresh ones built from its stored pages."""
    pages = [(page.page_number, page.text) for page in document.pages if page.has_text]
    drafts = chunking_service.chunk_pages(
        pages,
        tokenizer,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    # Delete explicitly before inserting: the ORM would otherwise insert the
    # new rows first and collide with the (document_id, chunk_index) constraint.
    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    db.expire(document, ["chunks"])
    db.add_all(
        Chunk(
            document_id=document.id,
            chunk_index=draft.chunk_index,
            page_start=draft.page_start,
            page_end=draft.page_end,
            text=draft.text,
            token_count=draft.token_count,
            char_count=draft.char_count,
            section=draft.section,
        )
        for draft in drafts
    )

    document.chunk_count = len(drafts)
    document.chunking_signature = chunking_service.chunking_signature(
        tokenizer.name, settings.chunk_size_tokens, settings.chunk_overlap_tokens
    )
    # Any vectors were built from the old chunks, so they no longer apply.
    document.indexing_status = IndexingStatus.NOT_INDEXED
    if drafts:
        document.processing_status = ProcessingStatus.COMPLETED
        document.status_message = missing_text_warning(
            document.pages_without_text, document.page_count or 0
        )
    else:
        document.processing_status = ProcessingStatus.FAILED
        document.status_message = "No searchable passages could be created from this PDF."
    return len(drafts)


def process_document(
    db: Session, settings: Settings, document: Document, tokenizer_provider: TokenizerProvider
) -> None:
    """Chunk a stored document and commit.

    Failures are recorded on the document (status ``failed`` plus a message
    the UI shows) rather than raised, so the paper stays in the library and
    can be retried with re-index.
    """
    document_id = document.id
    try:
        count = chunk_document(db, settings, document, tokenizer_provider())
        db.commit()
        logger.info("Chunked document %s into %d chunks", document_id, count)
    except Exception as error:
        db.rollback()
        if isinstance(error, ResearchMateError):
            message = error.message
        else:
            logger.exception("Unexpected error while chunking document %s", document_id)
            message = "An unexpected error occurred while splitting the paper into passages."
        document.processing_status = ProcessingStatus.FAILED
        document.status_message = message
        db.commit()


def reindex_document(
    db: Session, settings: Settings, document: Document, tokenizer_provider: TokenizerProvider
) -> Document:
    process_document(db, settings, document, tokenizer_provider)
    title = document.title or document.filename
    record_activity(db, ActivityType.DOCUMENT_REINDEXED, f"Re-indexed “{title}”", document.id)
    db.commit()
    return document
