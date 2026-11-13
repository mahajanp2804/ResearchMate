from collections import Counter
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query, Response, status

from app.dependencies import DbSession, EvaluationDep, IngestionDep
from app.schemas.evaluation import (
    DatasetListResponse,
    DatasetOut,
    EvaluationPaperOut,
    RunDetailOut,
    RunListResponse,
    RunRequest,
    RunSummaryOut,
    SamplePaperResult,
    SamplePapersResponse,
)
from app.services.evaluation_data import list_dataset_specs, load_dataset_spec

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(db: DbSession, evaluation: EvaluationDep) -> DatasetListResponse:
    """The curated datasets and whether their papers are in the library and searchable."""
    items = []
    for spec in list_dataset_specs():
        papers = evaluation.paper_statuses(db, spec)
        items.append(
            DatasetOut(
                slug=spec.slug,
                name=spec.name,
                description=spec.description,
                sample_collection=spec.sample_collection,
                question_count=len(spec.questions),
                categories=dict(Counter(question.category for question in spec.questions)),
                papers=[
                    EvaluationPaperOut(
                        key=paper.key,
                        title=paper.title,
                        document_id=paper.document.id if paper.document else None,
                        indexing_status=paper.document.indexing_status if paper.document else None,
                    )
                    for paper in papers
                ],
                ready=all(paper.ready for paper in papers),
            )
        )
    return DatasetListResponse(items=items)


@router.post("/sample-papers", response_model=SamplePapersResponse)
def install_sample_papers(
    db: DbSession,
    evaluation: EvaluationDep,
    ingestion: IngestionDep,
    background_tasks: BackgroundTasks,
    response: Response,
) -> SamplePapersResponse:
    """Add the bundled synthetic sample papers to the library (once).

    They are generated as PDFs and go through the normal upload pipeline;
    indexing runs in the background. Returns 201 if any paper was added.
    """
    results, to_index = evaluation.install_sample_papers(db)
    for document_id in to_index:
        background_tasks.add_task(ingestion.index_document, document_id)
    if any(result_status == "created" for _, _, result_status in results):
        response.status_code = status.HTTP_201_CREATED
    return SamplePapersResponse(
        papers=[
            SamplePaperResult(title=title, document_id=document_id, status=result_status)
            for title, document_id, result_status in results
        ]
    )


@router.post("/run", response_model=RunSummaryOut, status_code=status.HTTP_202_ACCEPTED)
def run_evaluation(
    request: RunRequest,
    db: DbSession,
    evaluation: EvaluationDep,
    background_tasks: BackgroundTasks,
) -> RunSummaryOut:
    """Start an evaluation run; poll ``GET /api/evaluation/results/{id}`` for progress.

    Retrieval metrics (Precision@K, Recall@K, MRR, per method) need no LLM.
    With ``answer_quality`` the pipeline also answers every question and the
    LLM judges faithfulness, answer relevance, context relevance and
    correctness; these are reported separately.
    """
    spec = load_dataset_spec(request.dataset)
    run = evaluation.start_run(
        db, spec, methods=request.methods, answer_quality=request.answer_quality
    )
    background_tasks.add_task(evaluation.execute, run.id)
    return RunSummaryOut.from_run(run)


@router.get("/results", response_model=RunListResponse)
def list_results(
    db: DbSession,
    evaluation: EvaluationDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> RunListResponse:
    """Evaluation runs, newest first, with their aggregate metrics."""
    runs, total = evaluation.list_runs(db, limit=limit)
    return RunListResponse(items=[RunSummaryOut.from_run(run) for run in runs], total=total)


@router.get("/results/{run_id}", response_model=RunDetailOut)
def get_result(run_id: str, db: DbSession, evaluation: EvaluationDep) -> RunDetailOut:
    """One run with its per-question results."""
    return RunDetailOut.from_run(evaluation.get_run(db, run_id))


@router.delete("/results/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_result(run_id: str, db: DbSession, evaluation: EvaluationDep) -> Response:
    evaluation.delete_run(db, run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
