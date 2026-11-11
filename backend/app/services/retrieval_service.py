"""Search entry point: query processing, paper scoping, retrieval and context.

The retrieval pipeline, in order:
1. dense: the query embedding against FAISS (finds paraphrases and meaning)
2. bm25: keyword matching over the same chunks (finds names, acronyms, numbers)
3. hybrid: Reciprocal Rank Fusion of the two ranked lists
4. hybrid_rerank: a cross-encoder rescoring the top fused candidates
5. context selection: what is finally sent to the LLM

Every stage is exposed on its own so the playground and the evaluation can
compare them. All stages search only fully indexed papers, and the paper
filter is applied inside each retriever, so out-of-scope chunks are never
candidates at any later stage.
"""

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import InvalidInputError
from app.models import Chunk, Document, DocumentPage, IndexingStatus
from app.services.bm25_service import BM25Service
from app.services.context_selection import ContextSelection, select_context
from app.services.hybrid_retriever import reciprocal_rank_fusion
from app.services.ingestion_service import IngestionService
from app.services.reranker import Reranker

MAX_QUERY_CHARS = 1000

RerankerProvider = Callable[[], Reranker]


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
    # Cosine similarity (dense), BM25 score, RRF score (hybrid) or
    # cross-encoder logit (hybrid_rerank); only comparable within one list.
    score: float
    chunk_id: int
    document_id: str
    document_title: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str | None
    text: str
    token_count: int
    # Where the chunk ranked in each stage that saw it, e.g. {"dense": 3, "bm25": 1}.
    source_ranks: dict[str, int] = field(default_factory=dict)


def page_texts_for(db: Session, chunks: Sequence[RetrievedChunk]) -> dict[str, dict[int, str]]:
    """Raw page texts of the passages that span pages, by document and page number.

    Citation checks use them to pin an excerpt to the exact page it is on.
    """
    texts: dict[str, dict[int, str]] = {}
    for chunk in chunks:
        if chunk.page_start == chunk.page_end:
            continue
        rows = db.execute(
            select(DocumentPage.page_number, DocumentPage.text).where(
                DocumentPage.document_id == chunk.document_id,
                DocumentPage.page_number.between(chunk.page_start, chunk.page_end),
            )
        )
        texts.setdefault(chunk.document_id, {}).update(dict(rows.all()))
    return texts


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        ingestion: IngestionService,
        bm25: BM25Service,
        reranker_provider: RerankerProvider,
    ) -> None:
        self._settings = settings
        self._ingestion = ingestion
        self._bm25 = bm25
        self._reranker_provider = reranker_provider

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

    # --- First-stage retrievers --------------------------------------------

    def _dense_hits(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[tuple[int, float]]:
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
        return [(hit.chunk_id, hit.score) for hit in hits]

    def _keyword_hits(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[tuple[int, float]]:
        if not scope.effective_document_ids:
            return []
        hits = self._bm25.search(db, query, top_k, document_ids=scope.document_ids)
        return [(hit.chunk_id, hit.score) for hit in hits]

    def dense_search(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[RetrievedChunk]:
        hits = self._dense_hits(db, query, top_k, scope)
        return self._hydrate(db, hits, [{"dense": rank} for rank in range(1, len(hits) + 1)])

    def keyword_search(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[RetrievedChunk]:
        hits = self._keyword_hits(db, query, top_k, scope)
        return self._hydrate(db, hits, [{"bm25": rank} for rank in range(1, len(hits) + 1)])

    # --- Fusion and reranking ----------------------------------------------

    def hybrid_search(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[RetrievedChunk]:
        """Dense and BM25 candidates fused with Reciprocal Rank Fusion."""
        dense = self._dense_hits(db, query, self._settings.dense_top_k, scope)
        keyword = self._keyword_hits(db, query, self._settings.bm25_top_k, scope)
        fused = reciprocal_rank_fusion(
            {
                "dense": [chunk_id for chunk_id, _ in dense],
                "bm25": [chunk_id for chunk_id, _ in keyword],
            },
            k=self._settings.rrf_k,
        )[:top_k]
        return self._hydrate(
            db,
            [(candidate.chunk_id, candidate.score) for candidate in fused],
            [candidate.ranks for candidate in fused],
        )

    def reranked_search(
        self, db: Session, query: str, top_k: int, scope: SearchScope
    ) -> list[RetrievedChunk]:
        """Hybrid candidates rescored by the cross-encoder, best first."""
        candidates = self.hybrid_search(db, query, self._settings.rerank_candidates, scope)
        if not candidates:
            return []  # nothing to rerank, so don't load the model
        scores = self._reranker_provider().score(query, [chunk.text for chunk in candidates])
        # Python's sort is stable, so equal scores keep their fused order.
        ordered = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return [
            replace(
                chunk,
                rank=rank,
                score=score,
                source_ranks={**chunk.source_ranks, "hybrid": chunk.rank},
            )
            for rank, (chunk, score) in enumerate(ordered[:top_k], start=1)
        ]

    # --- Context for the LLM -----------------------------------------------

    def retrieve_context(self, db: Session, query: str, scope: SearchScope) -> ContextSelection:
        """The passages the answer pipeline sends to the LLM, best first.

        With reranking on, the optional ``min_rerank_score`` gate applies to
        cross-encoder scores. RRF scores are not relevance estimates, so no
        score gate is applied without the reranker.
        """
        settings = self._settings
        if settings.reranker_enabled:
            candidates = self.reranked_search(db, query, settings.rerank_candidates, scope)
            min_score = settings.min_rerank_score
        else:
            candidates = self.hybrid_search(db, query, settings.rerank_candidates, scope)
            min_score = None
        return select_context(
            candidates,
            max_chunks=settings.final_context_chunks,
            max_tokens=settings.max_context_tokens,
            min_score=min_score,
        )

    # --- Hydration ---------------------------------------------------------

    def _hydrate(
        self,
        db: Session,
        scored: list[tuple[int, float]],
        source_ranks: list[dict[str, int]],
    ) -> list[RetrievedChunk]:
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
        for (chunk_id, score), ranks in zip(scored, source_ranks, strict=True):
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
                    source_ranks=ranks,
                )
            )
        return results
