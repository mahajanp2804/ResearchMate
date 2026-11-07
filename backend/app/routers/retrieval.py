import time

from fastapi import APIRouter

from app.dependencies import DbSession, RetrievalDep
from app.schemas.retrieval import MethodResults, SearchHit, SearchRequest, SearchResponse
from app.services.retrieval_service import normalize_query

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, db: DbSession, retrieval: RetrievalDep) -> SearchResponse:
    """Run each requested retriever and return their ranked passages side by side.

    Used by the retrieval playground and the evaluation module to inspect what
    each stage of the pipeline finds before anything reaches the LLM.
    """
    query = normalize_query(request.query)
    scope = retrieval.resolve_scope(db, request.document_ids)
    retrievers = {"dense": retrieval.dense_search, "bm25": retrieval.keyword_search}

    results = []
    for method in dict.fromkeys(request.methods):  # de-duplicate, keep order
        started = time.perf_counter()
        hits = retrievers[method](db, query, request.top_k, scope)
        results.append(
            MethodResults(
                method=method,
                hits=[SearchHit.model_validate(hit) for hit in hits],
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        )
    return SearchResponse(
        query=query,
        results=results,
        searched_document_count=len(scope.effective_document_ids),
        skipped_document_ids=list(scope.skipped_document_ids),
    )
