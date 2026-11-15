"""Evaluation experiments over a curated dataset.

A run has two parts, reported separately:

1. Retrieval (no LLM): every answerable question is searched with each
   retrieval method (dense, BM25, hybrid, hybrid + rerank) within the
   dataset's papers, and the ranked passages are judged against the
   question's evidence text (see ``retrieval_metrics``).
2. Answer quality (optional, needs the LLM): the full question-answering
   pipeline answers each question, and the LLM judges the answer's
   faithfulness, relevance and correctness and the relevance of the passages
   it was given (see ``answer_evaluation``). Whether the pipeline refused is
   checked in code against the question's category.

Runs execute in the background and record their progress, so the UI can
poll them; a run interrupted by a restart is marked failed at startup.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from statistics import fmean
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import Database
from app.exceptions import ConflictError, InvalidInputError, NotFoundError
from app.models import (
    ActivityType,
    Chunk,
    Document,
    DocumentPage,
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationRun,
    EvaluationRunStatus,
    IndexingStatus,
)
from app.models.base import utcnow
from app.services import document_service
from app.services.activity_service import record_activity
from app.services.citation_service import letters_and_digits
from app.services.evaluation_data import (
    DatasetSpec,
    build_sample_pdf,
    load_sample_collection,
)
from app.services.ingestion_service import IngestionService
from app.services.retrieval_metrics import Passage, RankJudgement, judge_ranking, summarise
from app.services.retrieval_service import RetrievalService, RetrievedChunk, SearchScope
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.services.answer_evaluation import AnswerEvaluator

logger = get_logger(__name__)

METHODS = ("dense", "bm25", "hybrid", "hybrid_rerank")
# Passages retrieved per question and method, and the cut-offs reported.
RETRIEVE_TOP_K = 10
CUTOFFS = (1, 3, 5, 10)

Search = Callable[[Session, str, int, SearchScope], list[RetrievedChunk]]


@dataclass(frozen=True)
class PaperStatus:
    key: str
    title: str
    document: Document | None

    @property
    def ready(self) -> bool:
        return self.document is not None and self.document.indexing_status == IndexingStatus.INDEXED


def _summary_by_category(
    judged: list[tuple[str, RankJudgement]],
) -> dict[str, dict[str, float | int]]:
    categories = sorted({category for category, _ in judged})
    return {
        category: summarise([j for c, j in judged if c == category], CUTOFFS)
        for category in categories
    }


class EvaluationService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        ingestion: IngestionService,
        retrieval: RetrievalService,
        answer_evaluator: "AnswerEvaluator | None" = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._answers = answer_evaluator

    # --- Papers and datasets --------------------------------------------------

    @staticmethod
    def paper_statuses(db: Session, spec: DatasetSpec) -> list[PaperStatus]:
        """Each paper of the dataset and the uploaded document with that exact title."""
        titles = list(spec.papers.values())
        documents = {
            document.title: document
            for document in db.scalars(select(Document).where(Document.title.in_(titles)))
        }
        return [PaperStatus(key, title, documents.get(title)) for key, title in spec.papers.items()]

    def install_sample_papers(self, db: Session) -> tuple[list[tuple[str, str, str]], list[str]]:
        """Upload the bundled sample papers that are not in the library yet.

        Returns (title, document id, "created" | "existing") per paper and the
        ids that still need background indexing.
        """
        collection = load_sample_collection()
        existing = {
            document.title: document
            for document in db.scalars(
                select(Document).where(Document.title.in_([p.title for p in collection.papers]))
            )
        }
        results: list[tuple[str, str, str]] = []
        to_index: list[str] = []
        for paper in collection.papers:
            if paper.title in existing:
                results.append((paper.title, existing[paper.title].id, "existing"))
                continue
            outcome = document_service.ingest_upload(
                db,
                self._settings,
                self._ingestion,
                f"{paper.key}.pdf",
                BytesIO(build_sample_pdf(paper)),
            )
            if outcome.document is None:
                message = outcome.error.message if outcome.error else "unknown error"
                raise ConflictError(
                    f"The sample paper “{paper.title}” could not be added: {message}",
                    code="sample_paper_failed",
                )
            results.append((paper.title, outcome.document.id, outcome.status))
            if outcome.needs_indexing:
                to_index.append(outcome.document.id)
        return results, to_index

    def import_dataset(self, db: Session, spec: DatasetSpec) -> EvaluationDataset:
        """Store the dataset with its papers resolved to the current documents.

        The evidence is located in the papers' pages (for display) and chunks
        (``expected_chunk_ids``, valid until the next re-chunking). Re-importing
        replaces the questions; past runs keep their own copy of the results.
        """
        statuses = self.paper_statuses(db, spec)
        missing = [status.title for status in statuses if not status.ready]
        if missing:
            raise ConflictError(
                "Some papers of this dataset are not in the library or not searchable yet.",
                code="dataset_papers_missing",
                details={"titles": missing},
            )
        documents = {status.key: status.document for status in statuses}
        dataset = db.scalar(select(EvaluationDataset).where(EvaluationDataset.name == spec.name))
        if dataset is None:
            dataset = EvaluationDataset(name=spec.name)
            db.add(dataset)
        dataset.description = spec.description
        dataset.questions = []
        db.flush()

        for question in spec.questions:
            evidence = []
            chunk_ids: list[int] = []
            for item in question.evidence:
                document = documents[item.paper]
                needle = letters_and_digits(item.text)
                page = next(
                    (
                        number
                        for number, text in db.execute(
                            select(DocumentPage.page_number, DocumentPage.text)
                            .where(DocumentPage.document_id == document.id)
                            .order_by(DocumentPage.page_number)
                        )
                        if needle in letters_and_digits(text)
                    ),
                    None,
                )
                chunk_ids.extend(
                    chunk_id
                    for chunk_id, text in db.execute(
                        select(Chunk.id, Chunk.text).where(Chunk.document_id == document.id)
                    )
                    if needle in letters_and_digits(text)
                )
                evidence.append({"document_id": document.id, "page": page, "text": item.text})
            dataset.questions.append(
                EvaluationQuestion(
                    key=question.key,
                    category=question.category,
                    question=question.question,
                    expected_document_ids=list(dict.fromkeys(e["document_id"] for e in evidence)),
                    expected_chunk_ids=list(dict.fromkeys(chunk_ids)),
                    expected_evidence=evidence,
                    reference_answer=question.reference_answer,
                )
            )
        db.flush()
        return dataset

    # --- Runs -------------------------------------------------------------------

    def start_run(
        self,
        db: Session,
        spec: DatasetSpec,
        *,
        methods: Sequence[str] = METHODS,
        answer_quality: bool = False,
    ) -> EvaluationRun:
        """Create a pending run; ``execute`` does the work (in the background)."""
        methods = list(dict.fromkeys(methods))
        unknown = [method for method in methods if method not in METHODS]
        if not methods or unknown:
            raise InvalidInputError(
                f"Choose retrieval methods from {', '.join(METHODS)}.",
                code="invalid_methods",
                details={"methods": unknown},
            )
        active = db.scalar(
            select(EvaluationRun.id).where(
                EvaluationRun.status.in_([EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING])
            )
        )
        if active:
            raise ConflictError(
                "An evaluation is already running. Wait for it to finish.",
                code="evaluation_running",
                details={"run_id": active},
            )
        if answer_quality:
            if self._answers is None:
                raise InvalidInputError(
                    "Answer quality evaluation is not available.", code="answers_unavailable"
                )
            self._answers.check_ready()

        dataset = self.import_dataset(db, spec)
        document_ids = [status.document.id for status in self.paper_statuses(db, spec)]
        settings = self._settings
        run = EvaluationRun(
            dataset=dataset,
            status=EvaluationRunStatus.PENDING,
            config={
                "dataset": spec.slug,
                "dataset_name": spec.name,
                "methods": methods,
                "answer_quality": answer_quality,
                "top_k": RETRIEVE_TOP_K,
                "cutoffs": list(CUTOFFS),
                "document_ids": document_ids,
                # What the numbers depend on, so runs can be compared fairly.
                "settings": {
                    "embedding_model": settings.embedding_model_name,
                    "reranker_model": settings.reranker_model_name,
                    "chunk_size_tokens": settings.chunk_size_tokens,
                    "chunk_overlap_tokens": settings.chunk_overlap_tokens,
                    "dense_top_k": settings.dense_top_k,
                    "bm25_top_k": settings.bm25_top_k,
                    "rrf_k": settings.rrf_k,
                    "rerank_candidates": settings.rerank_candidates,
                    "llm_model": settings.gemini_model if answer_quality else None,
                },
            },
            metrics={"progress": {"phase": "pending", "done": 0, "total": 0}},
        )
        db.add(run)
        db.commit()
        return run

    def execute(self, run_id: str) -> None:
        """Run the experiment and store the results. Never raises: failures are recorded."""
        with self._database.session() as db:
            run = db.get(EvaluationRun, run_id)
            if run is None or run.status != EvaluationRunStatus.PENDING:
                return
            run.status = EvaluationRunStatus.RUNNING
            db.commit()
            try:
                self._run_retrieval(db, run)
                if run.config.get("answer_quality") and self._answers is not None:
                    self._answers.run(db, run, self._scope(db, run))
                run.status = EvaluationRunStatus.COMPLETED
                run.metrics = {**run.metrics, "progress": {"phase": "done", "done": 1, "total": 1}}
                answers = run.config.get("answer_quality")
                measured = "retrieval and answers" if answers else "retrieval"
                record_activity(
                    db,
                    ActivityType.EVALUATION_RUN,
                    f"Evaluated {measured} on “{run.config['dataset_name']}”",
                )
            except Exception as error:  # recorded on the run, shown in the UI
                logger.exception("Evaluation run %s failed", run_id)
                db.rollback()
                run = db.get(EvaluationRun, run_id)
                run.status = EvaluationRunStatus.FAILED
                run.error_message = getattr(error, "message", None) or "The evaluation failed."
            run.completed_at = utcnow()
            db.commit()

    def _scope(self, db: Session, run: EvaluationRun) -> SearchScope:
        return self._retrieval.resolve_scope(db, run.config["document_ids"])

    def _searches(self) -> dict[str, Search]:
        retrieval = self._retrieval
        return {
            "dense": retrieval.dense_search,
            "bm25": retrieval.keyword_search,
            "hybrid": retrieval.hybrid_search,
            "hybrid_rerank": retrieval.reranked_search,
        }

    def _run_retrieval(self, db: Session, run: EvaluationRun) -> None:
        scope = self._scope(db, run)
        if scope.skipped_document_ids:
            raise ConflictError(
                "Some papers of this dataset are no longer searchable.",
                code="dataset_papers_missing",
            )
        searches = self._searches()
        methods: list[str] = run.config["methods"]
        questions = list(run.dataset.questions)
        judged: dict[str, list[tuple[str, RankJudgement]]] = {method: [] for method in methods}
        latencies: dict[str, list[float]] = {method: [] for method in methods}
        results: list[dict[str, Any]] = []

        # One untimed search per method first, so loading (or downloading) the
        # embedding model and cross-encoder doesn't count as query latency.
        if questions:
            for method in methods:
                searches[method](db, questions[0].question, RETRIEVE_TOP_K, scope)

        for index, question in enumerate(questions):
            entry: dict[str, Any] = {
                "key": question.key,
                "question": question.question,
                "category": question.category,
                "evidence": question.expected_evidence,
                "reference_answer": question.reference_answer,
                "retrieval": {},
            }
            if question.expected_evidence:
                evidence = [
                    Passage(item["document_id"], item["text"])
                    for item in question.expected_evidence
                ]
                for method in methods:
                    started = time.perf_counter()
                    ranked = searches[method](db, question.question, RETRIEVE_TOP_K, scope)
                    latencies[method].append((time.perf_counter() - started) * 1000)
                    judgement = judge_ranking(
                        [Passage(chunk.document_id, chunk.text) for chunk in ranked], evidence
                    )
                    judged[method].append((question.category or "uncategorised", judgement))
                    entry["retrieval"][method] = {
                        "first_relevant_rank": judgement.first_relevant_rank,
                        "evidence_ranks": judgement.evidence_ranks,
                        "reciprocal_rank": round(judgement.reciprocal_rank, 4),
                        **{f"recall@{k}": round(judgement.recall_at(k), 4) for k in CUTOFFS},
                        **{f"precision@{k}": round(judgement.precision_at(k), 4) for k in CUTOFFS},
                        "top": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "page_start": chunk.page_start,
                                "page_end": chunk.page_end,
                                "relevant": relevant,
                            }
                            for chunk, relevant in zip(ranked, judgement.relevant, strict=True)
                        ],
                    }
            results.append(entry)
            run.metrics = {
                **run.metrics,
                "progress": {"phase": "retrieval", "done": index + 1, "total": len(questions)},
            }
            db.commit()

        run.per_question_results = results
        run.metrics = {
            **run.metrics,
            "retrieval": {
                method: {
                    **summarise([j for _, j in judged[method]], CUTOFFS),
                    "mean_latency_ms": round(fmean(latencies[method]), 1)
                    if latencies[method]
                    else None,
                }
                for method in methods
            },
            "retrieval_by_category": {
                method: _summary_by_category(judged[method]) for method in methods
            },
        }
        db.commit()

    def reconcile(self) -> int:
        """Mark runs interrupted by a restart as failed. Returns how many."""
        with self._database.session() as db:
            runs = db.scalars(
                select(EvaluationRun).where(
                    EvaluationRun.status.in_(
                        [EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING]
                    )
                )
            ).all()
            for run in runs:
                run.status = EvaluationRunStatus.FAILED
                run.error_message = "The run was interrupted when the backend stopped."
                run.completed_at = utcnow()
            db.commit()
            return len(runs)

    @staticmethod
    def get_run(db: Session, run_id: str) -> EvaluationRun:
        run = db.get(EvaluationRun, run_id)
        if run is None:
            raise NotFoundError("Evaluation run not found.", details={"run_id": run_id})
        return run

    @staticmethod
    def list_runs(db: Session, *, limit: int) -> tuple[list[EvaluationRun], int]:
        """Runs, newest first."""
        total = db.scalar(select(func.count()).select_from(EvaluationRun)) or 0
        runs = db.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        ).all()
        return list(runs), total

    def delete_run(self, db: Session, run_id: str) -> None:
        run = self.get_run(db, run_id)
        if run.status in (EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING):
            raise ConflictError("This run is still in progress.", code="evaluation_running")
        db.delete(run)
        db.commit()
