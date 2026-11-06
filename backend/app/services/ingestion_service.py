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
from typing import Literal

from sqlalchemy import delete, select, update
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
ReindexAction = Literal["unchanged", "reembedding", "rechunked", "failed"]

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

    def _expected_signature(self) -> str | None:
        try:
            tokenizer_name = self._tokenizer_provider().name
        except Exception:
            return None  # re-chunking will fail too and record the reason
        return chunking_service.chunking_signature(
            tokenizer_name, self.settings.chunk_size_tokens, self.settings.chunk_overlap_tokens
        )

    def _is_fully_indexed(self, db: Session, document: Document) -> bool:
        return (
            document.indexing_status == IndexingStatus.INDEXED
            and document.embedding_model == self.settings.embedding_model_name
            and set(_chunk_ids(db, document.id)) <= self.vector_store.ids()
        )

    def reindex(self, db: Session, document: Document, *, force: bool = False) -> ReindexAction:
        """Bring a document up to date with the current settings, doing as little as possible.

        - ``rechunked``: chunking settings changed, processing had failed, or
          ``force`` was set. Chunks are rebuilt, then re-embedded.
        - ``reembedding``: chunks are current but vectors are missing, failed,
          or came from another embedding model. Only embedding is redone.
        - ``unchanged``: nothing to do; no model is loaded and nothing is embedded.
        - ``failed``: re-chunking failed; the reason is on the document.

        The caller schedules ``index_document`` for ``rechunked`` and ``reembedding``.
        """
        needs_rechunk = (
            force
            or document.processing_status != ProcessingStatus.COMPLETED
            or document.chunking_signature != self._expected_signature()
        )
        if needs_rechunk:
            action: ReindexAction = "rechunked" if self.process(db, document) else "failed"
        elif self._is_fully_indexed(db, document):
            return "unchanged"
        else:
            document.indexing_status = IndexingStatus.INDEXING
            action = "reembedding"

        title = document.title or document.filename
        record_activity(db, ActivityType.DOCUMENT_REINDEXED, f"Re-indexed “{title}”", document.id)
        db.commit()
        return action

    # --- Embedding and indexing (background) -------------------------------

    def _load_embedder(self, db: Session) -> Embedder:
        embedder = self._embedder_provider()
        if not self._checked_chunk_size:
            warn_if_chunks_exceed_model(embedder, self.settings.chunk_size_tokens)
            self._checked_chunk_size = True
        if self.vector_store.ensure_model(embedder.name, embedder.dimension):
            # Vectors from the previous model were discarded, so no paper is
            # searchable any more until it is re-embedded.
            db.execute(
                update(Document)
                .where(Document.indexing_status == IndexingStatus.INDEXED)
                .values(indexing_status=IndexingStatus.NOT_INDEXED)
            )
            db.commit()
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
                embedder = self._load_embedder(db)
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
                # Clears any earlier indexing error, keeping content warnings.
                document.status_message = missing_text_warning(
                    document.pages_without_text, document.page_count or 0
                )
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

    # --- Startup consistency check -----------------------------------------

    def reconcile(self) -> None:
        """Make document statuses match the vector index after a restart.

        Handles work interrupted by a shutdown, a missing or unreadable index
        file, an embedding model changed in the configuration, and vectors
        left behind by a deletion that did not finish.
        """
        with self.database.session() as db:
            interrupted_processing = db.execute(
                update(Document)
                .where(Document.processing_status == ProcessingStatus.PROCESSING)
                .values(
                    processing_status=ProcessingStatus.FAILED,
                    status_message="Processing was interrupted. Retry to finish it.",
                )
            ).rowcount
            interrupted_indexing = db.execute(
                update(Document)
                .where(Document.indexing_status == IndexingStatus.INDEXING)
                .values(indexing_status=IndexingStatus.NOT_INDEXED)
            ).rowcount

            stored_ids = self.vector_store.ids()
            # Vectors from another model are not comparable with new queries.
            store_matches_model = self.vector_store.model_name == self.settings.embedding_model_name
            chunk_ids_by_document: dict[str, set[int]] = {}
            for document_id, chunk_id in db.execute(select(Chunk.document_id, Chunk.id)):
                chunk_ids_by_document.setdefault(document_id, set()).add(chunk_id)

            unindexed = 0
            indexed = db.scalars(
                select(Document).where(Document.indexing_status == IndexingStatus.INDEXED)
            )
            for document in indexed:
                if not (
                    store_matches_model
                    and document.embedding_model == self.settings.embedding_model_name
                    and chunk_ids_by_document.get(document.id, set()) <= stored_ids
                ):
                    document.indexing_status = IndexingStatus.NOT_INDEXED
                    unindexed += 1
            db.commit()

        orphans: set[int] = set()
        if store_matches_model:
            all_chunk_ids = set().union(*chunk_ids_by_document.values())
            orphans = stored_ids - all_chunk_ids
            self.vector_store.remove(orphans)

        if interrupted_processing or interrupted_indexing or unindexed or orphans:
            logger.warning(
                "Startup check: %d interrupted uploads, %d interrupted indexing runs, "
                "%d papers need re-indexing, %d orphaned vectors removed",
                interrupted_processing,
                interrupted_indexing,
                unindexed,
                len(orphans),
            )

    # --- Deletion ----------------------------------------------------------

    def remove_vectors(self, chunk_ids: list[int]) -> None:
        removed = self.vector_store.remove(chunk_ids)
        logger.info("Removed %d vectors", removed)

    def chunk_ids(self, db: Session, document_id: str) -> list[int]:
        return _chunk_ids(db, document_id)
