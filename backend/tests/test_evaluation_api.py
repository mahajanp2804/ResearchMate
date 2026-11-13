"""The evaluation API: datasets, sample papers, runs and results."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.llm_service import create_llm_provider
from tests.conftest import make_app, make_settings


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(make_app(make_settings(tmp_path / "data"))) as client:
        yield client


def dataset(client: TestClient) -> dict:
    (item,) = client.get("/api/evaluation/datasets").json()["items"]
    return item


def test_dataset_reports_whether_its_papers_are_ready(client: TestClient) -> None:
    before = dataset(client)
    assert before["slug"] == "sample_papers" and before["question_count"] == 35
    assert before["categories"]["unanswerable"] == 4
    assert before["ready"] is False
    assert all(paper["document_id"] is None for paper in before["papers"])

    # Installing runs the normal upload pipeline; the test client runs the
    # background indexing before returning.
    response = client.post("/api/evaluation/sample-papers")
    assert response.status_code == 201
    assert {paper["status"] for paper in response.json()["papers"]} == {"created"}

    after = dataset(client)
    assert after["ready"] is True
    assert {paper["indexing_status"] for paper in after["papers"]} == {"indexed"}

    again = client.post("/api/evaluation/sample-papers")
    assert again.status_code == 200
    assert {paper["status"] for paper in again.json()["papers"]} == {"existing"}


def test_a_run_is_started_and_its_results_are_stored(client: TestClient) -> None:
    client.post("/api/evaluation/sample-papers")
    response = client.post("/api/evaluation/run", json={"methods": ["bm25", "dense"]})
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    assert response.json()["status"] == "pending"

    # The background task has finished by the time the client returns.
    listed = client.get("/api/evaluation/results").json()
    assert listed["total"] == 1
    summary = listed["items"][0]
    assert summary["id"] == run_id and summary["status"] == "completed"
    assert summary["dataset"] == "sample_papers" and summary["dataset_name"] == "Sample papers"
    assert set(summary["metrics"]["retrieval"]) == {"bm25", "dense"}
    assert "per_question_results" not in summary

    detail = client.get(f"/api/evaluation/results/{run_id}").json()
    assert len(detail["per_question_results"]) == 35
    assert detail["config"]["settings"]["chunk_size_tokens"] == 250

    assert client.delete(f"/api/evaluation/results/{run_id}").status_code == 204
    assert client.get(f"/api/evaluation/results/{run_id}").status_code == 404


def test_a_run_needs_the_datasets_papers(client: TestClient) -> None:
    response = client.post("/api/evaluation/run", json={})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "dataset_papers_missing"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"dataset": "missing"}, 404),
        ({"dataset": "../sample_papers"}, 404),
        ({"methods": ["magic"]}, 422),
        ({"methods": []}, 422),
    ],
)
def test_invalid_run_requests(client: TestClient, body: dict, status: int) -> None:
    assert client.post("/api/evaluation/run", json=body).status_code == status


def test_answer_quality_without_a_key_is_explained(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app) as client:
        client.post("/api/evaluation/sample-papers")
        response = client.post("/api/evaluation/run", json={"answer_quality": True})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"


def test_unknown_runs_are_not_found(client: TestClient) -> None:
    assert client.get("/api/evaluation/results/nope").status_code == 404
    assert client.delete("/api/evaluation/results/nope").status_code == 404
