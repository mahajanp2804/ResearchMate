"""Evaluation runs over the bundled sample dataset, with offline stand-in models."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.exceptions import ConflictError, InvalidInputError
from app.models import (
    ActivityEvent,
    ActivityType,
    Document,
    EvaluationRun,
    EvaluationRunStatus,
)
from app.services.evaluation_data import load_dataset_spec
from app.services.evaluation_service import CUTOFFS, METHODS
from tests.conftest import make_app, make_settings

SPEC = load_dataset_spec("sample_papers")
ANSWERABLE = [q for q in SPEC.questions if q.category != "unanswerable"]


def install(app: FastAPI) -> None:
    """Add the sample papers and index them now (the API does it in the background)."""
    service = app.state.evaluation
    with app.state.database.session() as db:
        _, to_index = service.install_sample_papers(db)
    for document_id in to_index:
        app.state.ingestion.index_document(document_id)


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FastAPI]:
    app = make_app(make_settings(tmp_path_factory.mktemp("eval") / "data"))
    with TestClient(app):
        install(app)
        yield app


def run_evaluation(app: FastAPI, **options) -> EvaluationRun:
    service = app.state.evaluation
    with app.state.database.session() as db:
        run_id = service.start_run(db, SPEC, **options).id
    service.execute(run_id)
    with app.state.database.session() as db:
        run = db.get(EvaluationRun, run_id)
        db.expunge(run)
        return run


def test_sample_papers_are_installed_once(library: FastAPI) -> None:
    with library.state.database.session() as db:
        results, to_index = library.state.evaluation.install_sample_papers(db)
        titles = set(db.scalars(select(Document.title)))
    assert {status for _, _, status in results} == {"existing"} and to_index == []
    assert titles == set(SPEC.papers.values())


def test_every_evidence_passage_lands_in_a_chunk(library: FastAPI) -> None:
    # The premise of text-based relevance: chunking never splits an evidence sentence.
    with library.state.database.session() as db:
        dataset = library.state.evaluation.import_dataset(db, SPEC)
        db.commit()
        for question in dataset.questions:
            if question.category == "unanswerable":
                assert question.expected_evidence == [] and question.expected_chunk_ids == []
                continue
            assert question.expected_chunk_ids, question.key
            assert all(item["page"] for item in question.expected_evidence), question.key


def test_retrieval_run_measures_every_method(library: FastAPI) -> None:
    run = run_evaluation(library)

    assert run.status is EvaluationRunStatus.COMPLETED, run.error_message
    assert run.config["methods"] == list(METHODS)
    assert len(run.config["document_ids"]) == 5
    retrieval = run.metrics["retrieval"]
    for method in METHODS:
        summary = retrieval[method]
        assert summary["questions"] == len(ANSWERABLE)
        for k in CUTOFFS:
            assert 0 <= summary[f"precision@{k}"] <= 1 and 0 <= summary[f"recall@{k}"] <= 1
        # Recall can only grow with K.
        recalls = [summary[f"recall@{k}"] for k in CUTOFFS]
        assert recalls == sorted(recalls)
        assert summary["mean_latency_ms"] >= 0
    # Even the offline stand-ins find most evidence among 10 passages from 5 short papers.
    assert retrieval["bm25"]["hit@10"] > 0.8
    by_category = run.metrics["retrieval_by_category"]["bm25"]
    assert set(by_category) == {"factual", "paraphrase", "keyword", "multi_paper"}
    assert run.metrics["progress"]["phase"] == "done"


def test_per_question_results_show_where_the_evidence_ranked(library: FastAPI) -> None:
    run = run_evaluation(library, methods=["bm25"])
    results = {item["key"]: item for item in run.per_question_results}
    assert len(results) == len(SPEC.questions)

    unanswerable = results["q32"]
    assert unanswerable["category"] == "unanswerable" and unanswerable["retrieval"] == {}

    multi = results["q29"]["retrieval"]["bm25"]
    assert len(multi["evidence_ranks"]) == 2
    top = multi["top"]
    assert len(top) == 10 and all({"chunk_id", "relevant", "page_start"} <= set(t) for t in top)
    first = multi["first_relevant_rank"]
    assert first is None or top[first - 1]["relevant"]


def test_model_loading_is_not_timed_as_query_latency(library: FastAPI) -> None:
    reranker = library.state.test_reranker
    reranker.calls.clear()
    run_evaluation(library, methods=["hybrid_rerank"])
    # One warm-up search, then one per answerable question.
    assert len(reranker.calls) == 1 + len(ANSWERABLE)


def test_runs_are_logged_and_not_started_twice(library: FastAPI) -> None:
    service = library.state.evaluation
    with library.state.database.session() as db:
        pending = service.start_run(db, SPEC)
        with pytest.raises(ConflictError) as error:
            service.start_run(db, SPEC)
        assert error.value.code == "evaluation_running"
    service.execute(pending.id)
    with library.state.database.session() as db:
        events = db.scalars(
            select(ActivityEvent).where(ActivityEvent.event_type == ActivityType.EVALUATION_RUN)
        ).all()
    assert events


def test_unknown_methods_are_rejected(library: FastAPI) -> None:
    with library.state.database.session() as db, pytest.raises(InvalidInputError):
        library.state.evaluation.start_run(db, SPEC, methods=["dense", "magic"])


def test_dataset_needs_its_papers(tmp_path) -> None:
    app = make_app(make_settings(tmp_path / "data"))
    with TestClient(app), app.state.database.session() as db:
        with pytest.raises(ConflictError) as error:
            app.state.evaluation.start_run(db, SPEC)
        assert error.value.code == "dataset_papers_missing"
        assert len(error.value.details["titles"]) == 5


def test_interrupted_runs_are_marked_failed_at_startup(tmp_path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings)
    with TestClient(app):
        install(app)
        with app.state.database.session() as db:
            run_id = app.state.evaluation.start_run(db, SPEC).id
    # A restart: the new app reconciles during startup.
    restarted = make_app(settings)
    with TestClient(restarted), restarted.state.database.session() as db:
        run = db.get(EvaluationRun, run_id)
        assert run.status is EvaluationRunStatus.FAILED
        assert "interrupted" in run.error_message
