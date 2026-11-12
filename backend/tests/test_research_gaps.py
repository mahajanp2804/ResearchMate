"""Research gap analysis: evidence per paper, checks on every finding, caching."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.models import ActivityEvent, ActivityType, Document, GapAnalysis
from app.prompts.research_gap_prompt import GAP_QUESTIONS
from app.services.llm_service import create_llm_provider
from tests.conftest import make_app, make_settings
from tests.fakes import ScriptedLLM
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

FEDPROX = [
    "We propose FedProx for federated optimization in heterogeneous networks. "
    "Statistical heterogeneity across devices causes client drift.",
    "Limitations: our analysis assumes bounded dissimilarity between devices, which may "
    "not hold in practice.",
    "We evaluate only on small benchmarks such as MNIST with at most one thousand devices.",
]
SCAFFOLD = [
    "SCAFFOLD corrects client drift caused by heterogeneous data using control variates.",
    "A limitation of SCAFFOLD is that it doubles the communication per round.",
    "Experiments use EMNIST with 400 clients. Future work should test larger deployments.",
]


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


def analyse(client: TestClient, document_ids: list[str], **body):
    return client.post("/api/research/gaps", json={"document_ids": document_ids, **body})


def source_with(prompt: str, text: str) -> str:
    """The id of the source block in ``prompt`` whose passage contains ``text``."""
    for source_id, body in re.findall(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S):
        if text in " ".join(body.split()):
            return source_id
    raise AssertionError(f"no source contains {text!r}")


# --- Evidence ----------------------------------------------------------------


def test_every_paper_is_searched_separately(tmp_path: Path) -> None:
    with library(tmp_path, gap_max_tokens_per_paper=40) as (app, client, papers):
        response = analyse(client, [papers["fedprox"], papers["scaffold"]])
        assert response.status_code == 200, response.text
        body = response.json()
        (call,) = app.state.test_llm.calls_of("gaps")

    assert [paper["label"] for paper in body["papers"]] == ["P1", "P2"]
    assert [paper["title"] for paper in body["papers"]] == ["Federated Optimization", "SCAFFOLD"]
    # Both papers' passages were sent, each paper labelled, sources numbered across them.
    prompt = call["turns"][0].content
    assert 'paper="P1"' in prompt and 'paper="P2"' in prompt
    by_paper = {paper: [] for paper in papers.values()}
    for source in body["sources"]:
        by_paper[source["document_id"]].append(source)
    assert all(by_paper.values())
    # One search per gap question and paper, and the per-paper budget holds
    # (the test tokenizer counts words).
    assert len(app.state.test_reranker.calls) == 2 * len(GAP_QUESTIONS)
    for sources in by_paper.values():
        assert sum(len(source["text"].split()) for source in sources) <= 40
    assert "2 selected papers" in body["scope_note"]
    assert "not the research field" in body["scope_note"]


# --- Checks on every finding ---------------------------------------------------


def scripted_gaps(prompt: str) -> dict:
    bounded = source_with(prompt, "assumes bounded dissimilarity")
    small = source_with(prompt, "small benchmarks such as MNIST")
    doubles = source_with(prompt, "doubles the communication")
    drift_a = source_with(prompt, "causes client drift")
    drift_b = source_with(prompt, "SCAFFOLD corrects client drift")
    return {
        "stated_limitations": [
            {"paper": "P1", "text": f"The analysis assumes bounded dissimilarity [{bounded}]."},
            # Cites one passage of its own paper and one of the other paper.
            {"paper": "P2", "text": f"SCAFFOLD doubles communication [{doubles}][{bounded}]."},
            # Cites only the other paper.
            {"paper": "P1", "text": f"FedProx doubles communication [{doubles}]."},
        ],
        "recurring_challenges": [
            {"title": "Client drift", "text": f"Both address client drift [{drift_a}][{drift_b}]."},
            {"title": "Drift again", "text": f"Client drift is a recurring issue [{drift_a}]."},
        ],
        "differences": [
            {"title": "Invented", "text": "They differ in everything [S99]."},
        ],
        "limited_discussion": [
            {"title": "Scale", "text": f"Only one paper discusses evaluation scale [{small}]."},
        ],
        "opportunities": [
            {
                "title": "Larger evaluations",
                "evidence": (
                    f"Evaluations stay small [{small}]. Larger deployments remain to be tested."
                ),
                "suggestion": f"Evaluate on larger, unexplored deployments [{small}].",
            },
            {"title": "No evidence", "evidence": "Nobody has tried it.", "suggestion": "Try it."},
            "not an object",
        ],
    }


def test_findings_are_checked_before_they_are_shown(tmp_path: Path) -> None:
    with library(tmp_path, ScriptedLLM(gaps=scripted_gaps)) as (_, client, papers):
        body = analyse(client, [papers["fedprox"], papers["scaffold"]]).json()
    fedprox, scaffold = papers["fedprox"], papers["scaffold"]

    first, second = body["stated_limitations"]
    assert first["document_ids"] == [fedprox] and first["citation_status"] == "cited"
    assert first["citations"][0]["excerpt"].startswith("Limitations: our analysis assumes")
    # The citation to the other paper's passage was removed from SCAFFOLD's limitation.
    assert second["document_ids"] == [scaffold]
    assert len(second["misattributed_citation_ids"]) == 1
    assert second["misattributed_citation_ids"][0] not in second["text"]

    (challenge,) = body["recurring_challenges"]
    assert challenge["title"] == "Client drift"
    assert challenge["document_ids"] == [fedprox, scaffold]
    assert body["differences"] == []
    assert body["limited_discussion"][0]["document_ids"] == [fedprox]

    (opportunity,) = body["opportunities"]
    assert opportunity["suggestion"] == "Evaluate on larger, unexplored deployments."
    assert opportunity["beyond_scope"] is True  # "unexplored"
    assert opportunity["citation_status"] == "partially_cited"
    assert opportunity["uncited_statements"] == ["Larger deployments remain to be tested."]

    left_out = {(item["kind"], item["title"], item["reason"]) for item in body["left_out"]}
    assert left_out == {
        ("stated_limitations", "FedProx doubles communication.", "cites_other_paper"),
        ("recurring_challenges", "Drift again", "cites_one_paper"),
        ("differences", "Invented", "no_valid_citation"),
        ("opportunities", "No evidence", "no_valid_citation"),
    }
    assert all(item["explanation"] for item in body["left_out"])
    assert body["malformed_items"] == 1


@pytest.mark.parametrize("reply", ["The gaps are:", "[]"])
def test_unreadable_reply_is_a_clear_error(tmp_path: Path, reply: str) -> None:
    with library(tmp_path, ScriptedLLM(gaps=reply)) as (app, client, papers):
        response = analyse(client, [papers["fedprox"], papers["scaffold"]])
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "llm_bad_output"
        with app.state.database.session() as db:
            assert db.scalars(select(GapAnalysis)).all() == []


# --- Saving and reuse ----------------------------------------------------------


def activity(app: FastAPI) -> list[str]:
    with app.state.database.session() as db:
        events = db.scalars(
            select(ActivityEvent).where(ActivityEvent.event_type == ActivityType.GAP_ANALYSIS_RUN)
        )
        return [event.description for event in events]


def test_analysis_is_saved_and_reused_for_the_same_papers(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        first = analyse(client, [papers["fedprox"], papers["scaffold"]]).json()
        # The same papers in another order reuse the saved analysis and its labels.
        again = analyse(client, [papers["scaffold"], papers["fedprox"]]).json()
        assert (first["cached"], again["cached"]) == (False, True)
        assert again["papers"] == first["papers"]
        assert again["stated_limitations"] == first["stated_limitations"]
        assert len(app.state.test_llm.calls_of("gaps")) == 1
        assert activity(app) == ["Analysed research gaps in “Federated Optimization”, “SCAFFOLD”"]

        refreshed = analyse(client, [papers["fedprox"], papers["scaffold"]], refresh=True)
        assert refreshed.json()["cached"] is False
        assert len(app.state.test_llm.calls_of("gaps")) == 2

        # Re-processing a paper makes the saved analysis stale.
        with app.state.database.session() as db:
            db.get(Document, papers["scaffold"]).chunking_signature = "v9:rechunked"
            db.commit()
        assert analyse(client, [papers["fedprox"], papers["scaffold"]]).json()["cached"] is False
        assert len(app.state.test_llm.calls_of("gaps")) == 3


def test_deleting_a_paper_deletes_analyses_that_include_it(tmp_path: Path) -> None:
    with library(tmp_path) as (app, client, papers):
        analyse(client, [papers["fedprox"], papers["scaffold"]])
        assert client.delete(f"/api/documents/{papers['scaffold']}").status_code == 204
        with app.state.database.session() as db:
            assert db.scalars(select(GapAnalysis)).all() == []


# --- Requests ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("names", "code"),
    [
        (["fedprox"], "too_few_papers"),
        (["fedprox", "scaffold", "other"], "too_many_papers"),
        (["fedprox", "missing"], "unknown_documents"),
    ],
)
def test_invalid_selections_are_rejected(tmp_path: Path, names: list[str], code: str) -> None:
    with library(tmp_path, max_gap_papers=2) as (app, client, papers):
        response = analyse(client, [papers.get(name, name) for name in names])
        assert response.status_code == 422
        assert response.json()["error"]["code"] == code
        assert "analyse" in response.json()["error"]["message"] or code == "unknown_documents"
        assert app.state.test_llm.calls == []


def test_analysis_without_an_api_key_is_explained(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app) as client:
        ids = [
            upload(client, make_pdf(FEDPROX, title="Federated Optimization"))["id"],
            upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD"), "b.pdf")["id"],
        ]
        response = analyse(client, ids)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"
