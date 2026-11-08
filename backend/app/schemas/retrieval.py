from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel

# dense and bm25 are the first-stage retrievers, hybrid fuses them with RRF,
# and hybrid_rerank rescores the fused candidates with a cross-encoder.
SearchMethod = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class SearchRequest(BaseModel):
    query: str = Field(max_length=4000, description="Question or search terms")
    methods: list[SearchMethod] = Field(
        default=["hybrid_rerank"],
        min_length=1,
        description="Pipeline stages to run; each returns its own ranked list",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    document_ids: list[str] | None = Field(
        default=None,
        max_length=500,
        description="Restrict the search to these papers; omit to search all papers",
    )


class SearchHit(ORMModel):
    rank: int
    # Cosine similarity (dense), BM25 score, RRF score (hybrid) or cross-encoder
    # logit (hybrid_rerank). Only comparable within one method's list.
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
    # Rank in each stage that saw this passage, e.g. {"dense": 3, "bm25": 1, "hybrid": 2}.
    source_ranks: dict[str, int]


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


class ContextRequest(BaseModel):
    query: str = Field(max_length=4000)
    document_ids: list[str] | None = Field(default=None, max_length=500)


class DroppedCounts(BaseModel):
    low_score: int
    duplicates: int
    over_budget: int


class ContextResponse(BaseModel):
    """The passages the answer pipeline would send to the LLM, as sources S1..Sn."""

    query: str
    reranked: bool
    has_evidence: bool
    passages: list[SearchHit]
    total_tokens: int
    max_tokens: int
    max_passages: int
    # Evidence threshold applied to cross-encoder scores; null when disabled.
    min_score: float | None
    dropped: DroppedCounts
    searched_document_count: int
    skipped_document_ids: list[str]
    latency_ms: float
