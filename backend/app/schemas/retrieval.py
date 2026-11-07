from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel

# Phase 6 adds "hybrid" (RRF fusion) and "hybrid_rerank" (fusion + cross-encoder).
SearchMethod = Literal["dense", "bm25"]


class SearchRequest(BaseModel):
    query: str = Field(max_length=4000, description="Question or search terms")
    methods: list[SearchMethod] = Field(
        default=["dense", "bm25"],
        min_length=1,
        description="Retrievers to run; each returns its own ranked list",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    document_ids: list[str] | None = Field(
        default=None,
        max_length=500,
        description="Restrict the search to these papers; omit to search all papers",
    )


class SearchHit(ORMModel):
    rank: int
    # Cosine similarity for dense, BM25 score for bm25. Scores are only
    # comparable within one method's list, never across methods.
    score: float
    chunk_id: int
    document_id: str
    document_title: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str | None
    text: str


class MethodResults(BaseModel):
    method: SearchMethod
    hits: list[SearchHit]
    latency_ms: float


class SearchResponse(BaseModel):
    query: str  # after normalisation
    results: list[MethodResults]
    searched_document_count: int
    # Requested papers that were not searched because they are not indexed yet.
    skipped_document_ids: list[str]
