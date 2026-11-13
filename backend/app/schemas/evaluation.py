from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models import EvaluationRun, EvaluationRunStatus, IndexingStatus
from app.schemas.retrieval import SearchMethod
from app.services.evaluation_service import METHODS


class EvaluationPaperOut(BaseModel):
    key: str
    title: str
    # The uploaded paper with this exact title, if any.
    document_id: str | None
    indexing_status: IndexingStatus | None


class DatasetOut(BaseModel):
    slug: str
    name: str
    description: str
    # Its papers can be installed from the bundled sample collection.
    sample_collection: bool
    question_count: int
    # Questions per category (factual, paraphrase, keyword, multi_paper, unanswerable).
    categories: dict[str, int]
    papers: list[EvaluationPaperOut]
    # Every paper is uploaded and searchable, so a run can start.
    ready: bool


class DatasetListResponse(BaseModel):
    items: list[DatasetOut]


class SamplePaperResult(BaseModel):
    title: str
    document_id: str
    status: Literal["created", "existing", "duplicate"]


class SamplePapersResponse(BaseModel):
    papers: list[SamplePaperResult]


class RunRequest(BaseModel):
    dataset: str = Field(default="sample_papers", max_length=100)
    methods: list[SearchMethod] = Field(default_factory=lambda: list(METHODS), min_length=1)
    # Also generate and judge answers with the LLM (about two calls per question).
    answer_quality: bool = False


class RunSummaryOut(BaseModel):
    """A run with its aggregate results.

    ``metrics`` holds ``retrieval`` (per method: precision@k, recall@k,
    hit@k, mrr, mean_latency_ms, questions), ``retrieval_by_category`` (the
    same per method and question category), ``answers`` (``summary`` of the
    LLM-judged scores, if requested) and ``progress``.
    """

    id: str
    dataset: str
    dataset_name: str
    status: EvaluationRunStatus
    config: dict[str, Any]
    metrics: dict[str, Any]
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_run(cls, run: EvaluationRun) -> "RunSummaryOut":
        return cls(
            id=run.id,
            dataset=run.config.get("dataset", ""),
            dataset_name=run.config.get("dataset_name", ""),
            status=run.status,
            config=run.config,
            metrics=run.metrics,
            error_message=run.error_message,
            created_at=run.created_at,
            completed_at=run.completed_at,
        )


class RunDetailOut(RunSummaryOut):
    # Per question: its category, evidence and reference answer, the ranks at
    # which each method retrieved the evidence, and (if requested) the answer
    # with its judgement.
    per_question_results: list[dict[str, Any]]

    @classmethod
    def from_run(cls, run: EvaluationRun) -> "RunDetailOut":
        summary = RunSummaryOut.from_run(run)
        return cls(**summary.model_dump(), per_question_results=run.per_question_results)


class RunListResponse(BaseModel):
    items: list[RunSummaryOut]
    total: int
