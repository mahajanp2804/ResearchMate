"""The research API: paper summaries and comparisons."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.models import ActivityEvent, ActivityType, Document, IndexingStatus
from app.prompts.profile_prompt import ASPECTS_BY_KEY, SUMMARY_SECTIONS
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
