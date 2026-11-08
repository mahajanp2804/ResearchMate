"""Search entry point: query processing, paper scoping and result hydration.

Phase 5 provides the two first-stage retrievers side by side:
- dense: the query embedding against FAISS (finds paraphrases and meaning)
- keyword: BM25 over the same chunks (finds exact names, acronyms, numbers)

Both search only papers that are fully indexed, and the paper filter is
applied inside each retriever, so out-of-scope chunks are never candidates.
"""

import unicodedata
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.exceptions import InvalidInputError
from app.models import Chunk, Document, IndexingStatus
from app.services.bm25_service import BM25Service
from app.services.ingestion_service import IngestionService

MAX_QUERY_CHARS = 1000


def normalize_query(query: str) -> str:
    """Unicode-normalise and collapse whitespace; reject empty or oversized queries."""
    query = " ".join(unicodedata.normalize("NFKC", query).split())
    if not query:
        raise InvalidInputError("Enter a question or search terms.", code="empty_query")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidInputError(
            f"Queries can be at most {MAX_QUERY_CHARS} characters.", code="query_too_long"
        )
    return query


@dataclass(frozen=True)
class SearchScope:
    """Which papers a search covers."""

    # Papers to search; None means every searchable paper.
    document_ids: frozenset[str] | None
    searchable_document_ids: frozenset[str]
    # Requested papers that are not searchable yet (still indexing, failed, ...).
    skipped_document_ids: tuple[str, ...] = ()

    @property
    def effective_document_ids(self) -> frozenset[str]:
        if self.document_ids is None:
            return self.searchable_document_ids
        return self.document_ids


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int  # 1-based position in this result list
    score: float  # cosine similarity (dense) or BM25 score (keyword)
    chunk_id: int
    document_id: str
    document_title: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str | None
    text: str
    token_count: int


class RetrievalService:
    def __init__(self, ingestion: IngestionService, bm25: BM25Service) -> None:
        self._ingestion = ingestion
        self._bm25 = bm25

    def resolve_scope(self, db: Session, document_ids: list[str] | None) -> SearchScope:
        searchable = frozenset(
            db.scalars(
                select(Document.id).where(Document.indexing_status == IndexingStatus.INDEXED)
            )
        )
        if document_ids is None:
            return SearchScope(document_ids=None, searchable_document_ids=searchable)

        requested = list(dict.fromkeys(document_ids))  # de-duplicate, keep order
        if not requested:
            raise InvalidInputError(
                "Select at least one paper, or search all papers.", code="empty_document_filter"
            )
        known = set(db.scalars(select(Document.id).where(Document.id.in_(requested))))
        unknown = [document_id for document_id in requested if document_id not in known]
        if unknown:
            raise InvalidInputError(
                "Some selected papers do not exist.",
                code="unknown_documents",
                details={"document_ids": unknown},
            )
        return SearchScope(
            document_ids=frozenset(requested) & searchable,
            searchable_document_ids=searchable,
            skipped_document_ids=tuple(d for d in requested if d not in searchable),
        )

    def dense_search(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[RetrievedChunk]:
        documents = scope.effective_document_ids
        if not documents:
            return []  # nothing to search, so don't load the model
        allowed_chunk_ids = set(
            db.scalars(select(Chunk.id).where(Chunk.document_id.in_(documents)))
        )
        embedder = self._ingestion.load_embedder(db)
        hits = self._ingestion.vector_store.search(
            embedder.embed_query(query), top_k, allowed_ids=allowed_chunk_ids
        )
        return self._hydrate(db, [(hit.chunk_id, hit.score) for hit in hits])

    def keyword_search(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[RetrievedChunk]:
        if not scope.effective_document_ids:
            return []
        hits = self._bm25.search(db, query, top_k, document_ids=scope.document_ids)
        return self._hydrate(db, [(hit.chunk_id, hit.score) for hit in hits])

    def _hydrate(self, db: Session, scored: list[tuple[int, float]]) -> list[RetrievedChunk]:
        """Attach chunk text and paper details, keeping the retriever's order."""
        if not scored:
            return []
        rows = db.execute(
            select(Chunk, Document.title, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_([chunk_id for chunk_id, _ in scored]))
        ).all()
        by_id = {chunk.id: (chunk, title or filename) for chunk, title, filename in rows}

        results: list[RetrievedChunk] = []
        for chunk_id, score in scored:
            if chunk_id not in by_id:
                continue  # deleted between search and lookup
            chunk, title = by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    rank=len(results) + 1,
                    score=score,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=title,
                    chunk_index=chunk.chunk_index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    text=chunk.text,
                    token_count=chunk.token_count,
                )
            )
        return results
