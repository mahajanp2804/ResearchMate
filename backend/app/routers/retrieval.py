import time

from fastapi import APIRouter

from app.dependencies import DbSession, RetrievalDep, SettingsDep
from app.schemas.retrieval import (
    ContextRequest,
    ContextResponse,
    DroppedCounts,
    MethodResults,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.retrieval_service import normalize_query

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, db: DbSession, retrieval: RetrievalDep) -> SearchResponse:
    """Run each requested pipeline stage and return their ranked passages side by side.

    Used by the retrieval playground and the evaluation module to inspect what
    each stage finds before anything reaches the LLM.
    """
    query = normalize_query(request.query)
    scope = retrieval.resolve_scope(db, request.document_ids)
    stages = {
        "dense": retrieval.dense_search,
        "bm25": retrieval.keyword_search,
        "hybrid": retrieval.hybrid_search,
        "hybrid_rerank": retrieval.reranked_search,
    }

    results = []
    for method in dict.fromkeys(request.methods):  # de-duplicate, keep order
        started = time.perf_counter()
        hits = stages[method](db, query, request.top_k, scope)
        results.append(
            MethodResults(
                method=method,
                hits=[SearchHit.model_validate(hit) for hit in hits],
                latency_ms=_elapsed_ms(started),
            )
        )
    return SearchResponse(
        query=query,
        results=results,
        searched_document_count=len(scope.effective_document_ids),
        skipped_document_ids=list(scope.skipped_document_ids),
    )


@router.post("/context", response_model=ContextResponse)
def context(
    request: ContextRequest, db: DbSession, retrieval: RetrievalDep, settings: SettingsDep
) -> ContextResponse:
    """Show exactly which passages the answer pipeline would give the LLM, and why
    others were left out (evidence threshold, near-duplicate, token budget)."""
    started = time.perf_counter()
    query = normalize_query(request.query)
    scope = retrieval.resolve_scope(db, request.document_ids)
    selection = retrieval.retrieve_context(db, query, scope)
    return ContextResponse(
        query=query,
        reranked=settings.reranker_enabled,
        has_evidence=selection.has_evidence,
        passages=[SearchHit.model_validate(chunk) for chunk in selection.chunks],
        total_tokens=selection.total_tokens,
        max_tokens=settings.max_context_tokens,
        max_passages=settings.final_context_chunks,
        min_score=settings.min_rerank_score if settings.reranker_enabled else None,
        dropped=DroppedCounts(
            low_score=selection.dropped_low_score,
            duplicates=selection.dropped_duplicates,
            over_budget=selection.dropped_over_budget,
        ),
        searched_document_count=len(scope.effective_document_ids),
        skipped_document_ids=list(scope.skipped_document_ids),
        latency_ms=_elapsed_ms(started),
    )
