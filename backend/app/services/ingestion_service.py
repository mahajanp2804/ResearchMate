"""Turns stored documents into searchable chunks and vectors.

Two stages with different costs:
- Chunking (fast, milliseconds) runs inside the upload/re-index request.
- Embedding and indexing (seconds; the model may need loading) runs as a
  background task. The document's ``indexing_status`` goes
  ``indexing`` -> ``indexed`` (or ``failed``), and the UI polls it.

Failures are recorded on the document (status ``failed`` plus a message the
UI shows) instead of being raised, so a paper is never lost and can be retried.
"""

from collections.abc import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import Database
from app.exceptions import ResearchMateError
from app.models import ActivityType, Chunk, Document, IndexingStatus, ProcessingStatus
from app.models.base import utcnow
from app.services import chunking_service
from app.services.activity_service import record_activity
from app.services.embedding_service import Embedder, warn_if_chunks_exceed_model
from app.services.tokenization import Tokenizer
from app.services.vector_store import VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

TokenizerProvider = Callable[[], Tokenizer]
EmbedderProvider = Callable[[], Embedder]

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


def _failure_message(error: Exception, stage: str) -> str:
    if isinstance(error, ResearchMateError):
        return error.message
    return f"An unexpected error occurred while {stage}."


def _chunk_ids(db: Session, document_id: str) -> list[int]:
    return list(db.scalars(select(Chunk.id).where(Chunk.document_id == document_id)))


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        vector_store: VectorStore,
        tokenizer_provider: TokenizerProvider,
        embedder_provider: EmbedderProvider,
    ) -> None:
        self.settings = settings
        self.database = database
        self.vector_store = vector_store
        self._tokenizer_provider = tokenizer_provider
        self._embedder_provider = embedder_provider
        self._checked_chunk_size = False

    # --- Chunking (synchronous) --------------------------------------------

    def _chunk(self, db: Session, document: Document, tokenizer: Tokenizer) -> int:
        """Replace the document's chunks with fresh ones built from its stored pages."""
        pages = [(page.page_number, page.text) for page in document.pages if page.has_text]
        drafts = chunking_service.chunk_pages(
            pages,
            tokenizer,
            chunk_size=self.settings.chunk_size_tokens,
            chunk_overlap=self.settings.chunk_overlap_tokens,
        )

        # Old vectors belong to chunk ids that are about to disappear.
        self.vector_store.remove(_chunk_ids(db, document.id))
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
            tokenizer.name, self.settings.chunk_size_tokens, self.settings.chunk_overlap_tokens
        )
        document.indexing_status = IndexingStatus.NOT_INDEXED
        document.embedding_model = None
        document.indexed_at = None
        if drafts:
            document.processing_status = ProcessingStatus.COMPLETED
            document.status_message = missing_text_warning(
                document.pages_without_text, document.page_count or 0
            )
        else:
            document.processing_status = ProcessingStatus.FAILED
            document.status_message = "No searchable passages could be created from this PDF."
        return len(drafts)

    def process(self, db: Session, document: Document) -> bool:
        """Chunk a stored document and mark it ready for background indexing.

        Returns True if the caller should schedule ``index_document``.
        """
        document_id = document.id
        try:
            count = self._chunk(db, document, self._tokenizer_provider())
            if count:
                document.indexing_status = IndexingStatus.INDEXING
            db.commit()
            logger.info("Chunked document %s into %d chunks", document_id, count)
            return count > 0
        except Exception as error:
            db.rollback()
            if not isinstance(error, ResearchMateError):
                logger.exception("Unexpected error while chunking document %s", document_id)
            document.processing_status = ProcessingStatus.FAILED
            document.status_message = _failure_message(error, "splitting the paper into passages")
            db.commit()
            return False

    def reindex(self, db: Session, document: Document) -> bool:
        """Rebuild a document's chunks; returns True if indexing should be scheduled."""
        queued = self.process(db, document)
        title = document.title or document.filename
        record_activity(db, ActivityType.DOCUMENT_REINDEXED, f"Re-indexed “{title}”", document.id)
        db.commit()
        return queued

    # --- Embedding and indexing (background) -------------------------------

    def _load_embedder(self) -> Embedder:
        embedder = self._embedder_provider()
        if not self._checked_chunk_size:
            warn_if_chunks_exceed_model(embedder, self.settings.chunk_size_tokens)
            self._checked_chunk_size = True
        self.vector_store.ensure_model(embedder.name, embedder.dimension)
        return embedder

    def index_document(self, document_id: str) -> None:
        """Embed a document's chunks and add them to the vector index.

        Runs after the response is sent, with its own database session.
        """
        with self.database.session() as db:
            document = db.get(Document, document_id)
            if document is None or document.processing_status != ProcessingStatus.COMPLETED:
                return  # deleted or re-chunked unsuccessfully in the meantime
            document.indexing_status = IndexingStatus.INDEXING
            db.commit()
            try:
                embedder = self._load_embedder()
                chunks = list(
                    db.scalars(
                        select(Chunk)
                        .where(Chunk.document_id == document_id)
                        .order_by(Chunk.chunk_index)
                    )
                )
                vectors = embedder.embed_documents([chunk.text for chunk in chunks])
                chunk_ids = [chunk.id for chunk in chunks]
                self.vector_store.add(chunk_ids, vectors)

                # The paper may have been deleted or re-chunked while embedding.
                db.expire_all()
                if set(_chunk_ids(db, document_id)) != set(chunk_ids):
                    self.vector_store.remove(chunk_ids)
                    logger.info("Document %s changed during indexing; vectors dropped", document_id)
                    return

                document.indexing_status = IndexingStatus.INDEXED
                document.embedding_model = embedder.name
                document.indexed_at = utcnow()
                db.commit()
                logger.info("Indexed %d chunks of document %s", len(chunk_ids), document_id)
            except Exception as error:
                db.rollback()
                if not isinstance(error, ResearchMateError):
                    logger.exception("Unexpected error while indexing document %s", document_id)
                document = db.get(Document, document_id)
                if document is not None:
                    document.indexing_status = IndexingStatus.FAILED
                    document.status_message = _failure_message(error, "indexing the paper")
                    db.commit()

    # --- Deletion ----------------------------------------------------------

    def remove_vectors(self, chunk_ids: list[int]) -> None:
        removed = self.vector_store.remove(chunk_ids)
        logger.info("Removed %d vectors", removed)

    def chunk_ids(self, db: Session, document_id: str) -> list[int]:
        return _chunk_ids(db, document_id)
