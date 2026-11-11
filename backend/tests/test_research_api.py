"""The research API: paper summaries and comparisons."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.exceptions import UpstreamServiceError
from app.models import ActivityEvent, ActivityType, Chunk, Document, IndexingStatus
from app.prompts.profile_prompt import ASPECTS_BY_KEY, COMPARISON_ASPECTS, SUMMARY_SECTIONS
from app.services.llm_service import create_llm_provider
from tests.conftest import make_app, make_settings
from tests.fakes import ScriptedLLM
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload
from tests.test_paper_profile_service import FEDPROX, SCAFFOLD


@contextmanager
def library(
    tmp_path: Path, llm: ScriptedLLM | None = None, **overrides
) -> Iterator[tuple[FastAPI, TestClient, dict[str, str]]]:
    settings = make_settings(
        tmp_path / "data", chunk_size_tokens=32, chunk_overlap_tokens=0, **overrides
    )
    app = make_app(settings, llm=llm)
    with TestClient(app) as client:
        papers = {
            "fedprox": upload(client, make_pdf(FEDPROX, title="Federated Optimization"))["id"],
            "scaffold": upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD"), "b.pdf")["id"],
        }
        yield app, client, papers


def activity(app: FastAPI, event_type: ActivityType) -> list[str]:
    with app.state.database.session() as db:
        events = db.scalars(select(ActivityEvent).where(ActivityEvent.event_type == event_type))
        return [event.description for event in events]


# --- Summaries ---------------------------------------------------------------


def test_summary_has_nine_cited_sections(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        response = client.post(f"/api/research/summarize/{papers['fedprox']}")
        assert response.status_code == 200, response.text
        summary = response.json()

        assert summary["document_title"] == "Federated Optimization"
        assert summary["cached"] is False and summary["model"] == "scripted-model"
        assert [section["key"] for section in summary["aspects"]] == list(SUMMARY_SECTIONS)
        assert [section["label"] for section in summary["aspects"]] == [
            ASPECTS_BY_KEY[key].label for key in SUMMARY_SECTIONS
        ]
        # The scripted model cites S1 for every section.
        first = summary["aspects"][0]
        assert first["status"] == "found" and first["citation_status"] == "cited"
        assert first["citations"][0]["source_id"] == "S1"
        assert summary["sources"][0]["source_id"] == "S1"
        assert activity(app, ActivityType.SUMMARY_GENERATED) == [
            "Summarised “Federated Optimization”"
        ]


def test_summary_is_saved_and_reused(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        url = f"/api/research/summaries/{papers['fedprox']}"
        missing = client.get(url)
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "summary_not_found"

        created = client.post(f"/api/research/summarize/{papers['fedprox']}").json()
        again = client.post(f"/api/research/summarize/{papers['fedprox']}").json()
        saved = client.get(url).json()

        assert again["cached"] is True and saved["cached"] is True
        assert again["aspects"] == created["aspects"] == saved["aspects"]
        assert again["generated_at"] == created["generated_at"]
        assert len(app.state.test_llm.calls_of("profile")) == 1
        assert len(activity(app, ActivityType.SUMMARY_GENERATED)) == 1

        refreshed = client.post(f"/api/research/summarize/{papers['fedprox']}?refresh=true")
        assert refreshed.json()["cached"] is False
        assert len(app.state.test_llm.calls_of("profile")) == 2


def test_sections_the_paper_does_not_cover_are_not_found(tmp_path: Path) -> None:
    llm = ScriptedLLM(profile={"datasets": {"found": False, "text": ""}})
    with library(tmp_path, llm) as (_, client, papers):
        summary = client.post(f"/api/research/summarize/{papers['fedprox']}").json()
    sections = {section["key"]: section for section in summary["aspects"]}
    assert sections["datasets"]["status"] == "not_found"
    assert sections["datasets"]["text"] is None
    # Aspects missing from the reply are not found too, never invented.
    assert {section["status"] for section in summary["aspects"]} == {"not_found"}


def test_summary_needs_a_searchable_paper(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        with app.state.database.session() as db:
            db.get(Document, papers["fedprox"]).indexing_status = IndexingStatus.FAILED
            db.commit()
        response = client.post(f"/api/research/summarize/{papers['fedprox']}")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "paper_not_indexed"
        assert client.post("/api/research/summarize/unknown").status_code == 404
        assert client.get("/api/research/summaries/unknown").status_code == 404


def test_summary_without_an_api_key_is_explained(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app) as client:
        paper = upload(client, make_pdf(FEDPROX, title="Federated Optimization"))["id"]
        response = client.post(f"/api/research/summarize/{paper}")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"


def test_reindexing_with_new_chunks_makes_the_summary_stale(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        client.post(f"/api/research/summarize/{papers['fedprox']}")
        with app.state.database.session() as db:
            db.get(Document, papers["fedprox"]).chunking_signature = "v9:rechunked"
            db.commit()
        assert client.get(f"/api/research/summaries/{papers['fedprox']}").status_code == 404
        rebuilt = client.post(f"/api/research/summarize/{papers['fedprox']}").json()
        assert rebuilt["cached"] is False


# --- Comparisons ---------------------------------------------------------------


def compare(client: TestClient, document_ids: list[str], **body):
    return client.post("/api/research/compare", json={"document_ids": document_ids, **body})


def test_comparison_has_one_cited_column_per_paper(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        response = compare(client, [papers["scaffold"], papers["fedprox"]])
        assert response.status_code == 200, response.text
        comparison = response.json()
        with app.state.database.session() as db:
            owner = dict(db.query(Chunk.id, Chunk.document_id).all())

    assert [row["key"] for row in comparison["aspects"]] == list(COMPARISON_ASPECTS)
    assert comparison["aspects"][5] == {"key": "evaluation_metrics", "label": "Evaluation metrics"}
    columns = comparison["papers"]
    assert [column["document_id"] for column in columns] == [papers["scaffold"], papers["fedprox"]]
    for column in columns:
        assert [cell["key"] for cell in column["aspects"]] == list(COMPARISON_ASPECTS)
        # Every passage behind a column comes from that column's paper.
        assert {owner[source["chunk_id"]] for source in column["sources"]} == {
            column["document_id"]
        }
    # Each LLM call saw one paper only.
    prompts = [call["turns"][0].content for call in app.state.test_llm.calls_of("profile")]
    assert sorted(prompt.split("\n", 1)[0] for prompt in prompts) == [
        'Paper: "Federated Optimization"',
        'Paper: "SCAFFOLD"',
    ]
    assert activity(app, ActivityType.COMPARISON_RUN) == [
        "Compared “SCAFFOLD” and “Federated Optimization”"
    ]


def test_profiles_are_built_in_parallel(tmp_path: Path) -> None:
    # Each call waits until both are in flight: sequential calls would time out.
    both_running = threading.Barrier(2, timeout=5)

    def profile(_prompt: str) -> None:
        both_running.wait()
        return None  # the default scripted profile

    with library(tmp_path, ScriptedLLM(profile=profile)) as (_, client, papers):
        response = compare(client, [papers["fedprox"], papers["scaffold"]])
    assert response.status_code == 200, response.text


def test_saved_profiles_are_reused(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        client.post(f"/api/research/summarize/{papers['fedprox']}")
        columns = compare(client, [papers["fedprox"], papers["scaffold"]]).json()["papers"]
        assert [column["cached"] for column in columns] == [True, False]
        assert len(app.state.test_llm.calls_of("profile")) == 2

        columns = compare(client, [papers["fedprox"], papers["scaffold"]]).json()["papers"]
        assert [column["cached"] for column in columns] == [True, True]
        assert len(app.state.test_llm.calls_of("profile")) == 2

        refreshed = compare(client, [papers["fedprox"], papers["scaffold"]], refresh=True)
        assert [column["cached"] for column in refreshed.json()["papers"]] == [False, False]
        assert len(app.state.test_llm.calls_of("profile")) == 4


def test_a_failed_paper_keeps_the_finished_ones(tmp_path: Path) -> None:
    failing = {"on": True}

    def profile(prompt: str) -> None:
        if failing["on"] and prompt.startswith('Paper: "SCAFFOLD"'):
            raise UpstreamServiceError("Gemini is unavailable.", code="llm_unavailable")
        return None

    with library(tmp_path, ScriptedLLM(profile=profile)) as (app, client, papers):
        response = compare(client, [papers["fedprox"], papers["scaffold"]])
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "llm_unavailable"
        assert client.get(f"/api/research/summaries/{papers['fedprox']}").status_code == 200
        assert activity(app, ActivityType.COMPARISON_RUN) == []

        failing["on"] = False
        columns = compare(client, [papers["fedprox"], papers["scaffold"]]).json()["papers"]
        assert [column["cached"] for column in columns] == [True, False]
        assert len(app.state.test_llm.calls_of("profile")) == 3


@pytest.mark.parametrize(
    ("ids", "code"),
    [
        (["fedprox"], "too_few_papers"),
        (["fedprox", "fedprox"], "too_few_papers"),
        (["fedprox", "scaffold", "unknown"], "too_many_papers"),
        (["fedprox", "unknown"], "unknown_documents"),
    ],
)
def test_invalid_selections_are_rejected(tmp_path: Path, ids: list[str], code: str) -> None:
    with library(tmp_path, max_compare_papers=2) as (app, client, papers):
        response = compare(client, [papers.get(name, name) for name in ids])
        assert response.status_code == 422
        assert response.json()["error"]["code"] == code
        assert app.state.test_llm.calls == []


def test_comparison_needs_searchable_papers(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        with app.state.database.session() as db:
            db.get(Document, papers["scaffold"]).indexing_status = IndexingStatus.INDEXING
            db.commit()
        response = compare(client, [papers["fedprox"], papers["scaffold"]])
    assert response.status_code == 409
    body = response.json()["error"]
    assert body["code"] == "papers_not_indexed"
    assert body["details"]["document_ids"] == [papers["scaffold"]]
