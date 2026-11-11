"""Paper profiles: evidence per aspect, extraction, validation and caching."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions import ConflictError, NotFoundError, UpstreamServiceError
from app.models import Document, IndexingStatus
from app.prompts.answer_prompt import REFUSAL
from app.prompts.profile_prompt import ASPECTS
from tests.conftest import make_app, make_settings
from tests.fakes import ScriptedLLM
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

FEDPROX = [
    "We propose FedProx, a framework for federated optimization in heterogeneous networks. "
    "Existing methods such as FedAvg diverge when devices hold non-IID data.",
    "FedProx adds a proximal term to the local objective of each device. "
    "Devices may perform a variable number of local updates.",
    "We evaluate on MNIST, FEMNIST, Shakespeare and Sent140. "
    "Training uses a learning rate of 0.01 and 10 local epochs.",
    "FedProx improves absolute test accuracy by 22% on average in highly heterogeneous "
    "settings. Our analysis assumes bounded dissimilarity, which may not always hold.",
]
SCAFFOLD = [
    "SCAFFOLD uses control variates to correct client drift in local updates.",
    "Experiments on EMNIST show SCAFFOLD needs fewer communication rounds.",
]


@contextmanager
def library(
    tmp_path: Path, llm: ScriptedLLM | None = None, **overrides
) -> Iterator[tuple[FastAPI, dict[str, str]]]:
    settings = make_settings(
        tmp_path / "data", chunk_size_tokens=32, chunk_overlap_tokens=0, **overrides
    )
    app = make_app(settings, llm=llm)
    with TestClient(app) as client:
        papers = {
            "fedprox": upload(client, make_pdf(FEDPROX, title="Federated Optimization"))["id"],
            "scaffold": upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD"), "b.pdf")["id"],
        }
        yield app, papers


def build(app: FastAPI, document_id: str, refresh: bool = False):
    with app.state.database.session() as db:
        document = app.state.profiles.require_indexed(db, document_id)
        profile, cached = app.state.profiles.get_or_build(db, document, refresh=refresh)
        db.commit()
        return profile, cached


def source_with(prompt: str, text: str) -> str:
    """The id of the source block in ``prompt`` that contains ``text``."""
    for source_id, body in re.findall(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S):
        if text in " ".join(body.split()):
            return source_id
    raise AssertionError(f"no source contains {text!r}")


# --- Evidence ----------------------------------------------------------------


def test_evidence_comes_only_from_the_paper_in_reading_order(tmp_path: Path) -> None:
    with library(tmp_path) as (app, papers):
        with app.state.database.session() as db:
            document = db.get(Document, papers["fedprox"])
            evidence = app.state.profiles.gather_evidence(db, document)

        assert {chunk.document_id for chunk in evidence.chunks} == {papers["fedprox"]}
        indexes = [chunk.chunk_index for chunk in evidence.chunks]
        assert indexes == sorted(indexes) and len(set(indexes)) == len(indexes)
        # One reranked search per aspect.
        assert len(app.state.test_reranker.calls) == len(ASPECTS)


def test_evidence_stays_within_the_token_budget(tmp_path: Path) -> None:
    with (
        library(tmp_path, profile_max_context_tokens=40) as (app, papers),
        app.state.database.session() as db,
    ):
        evidence = app.state.profiles.gather_evidence(db, db.get(Document, papers["fedprox"]))
    assert evidence.chunks
    assert sum(chunk.token_count for chunk in evidence.chunks) <= 40


# --- Extraction and validation -------------------------------------------------


def scripted_profile(prompt: str) -> dict:
    datasets = source_with(prompt, "We evaluate on MNIST")
    results = source_with(prompt, "improves absolute test accuracy")
    return {
        "research_objective": {"found": True, "text": REFUSAL},
        "datasets": {
            "found": True,
            "text": f"The paper evaluates on MNIST, FEMNIST, Shakespeare and Sent140 [{datasets}].",
        },
        "key_results": {
            "found": True,
            "text": f"FedProx improves test accuracy by 22% on average [{results}][S99].",
        },
        "strengths": {
            "found": True,
            "text": "FedProx is the most widely adopted framework in industry today.",
        },
        "limitations": {"found": True, "text": "The paper does not discuss its limitations."},
        "future_work": {"found": False, "text": "Probably more experiments."},
        # methodology and the other aspects are left out of the reply.
    }


def test_each_aspect_is_validated_on_its_own(tmp_path: Path) -> None:
    with library(tmp_path, ScriptedLLM(profile=scripted_profile)) as (app, papers):
        profile, cached = build(app, papers["fedprox"])

    assert cached is False and profile.model == "scripted-model"
    aspects = {aspect["key"]: aspect for aspect in profile.content["aspects"]}
    assert list(aspects) == [aspect.key for aspect in ASPECTS]

    datasets = aspects["datasets"]
    assert datasets["status"] == "found" and datasets["citation_status"] == "cited"
    (citation,) = datasets["citations"]
    assert citation["excerpt"].startswith("We evaluate on MNIST")
    assert citation["page"] == 3

    results = aspects["key_results"]
    assert results["invalid_citation_ids"] == ["S99"]
    assert "[S99]" not in results["text"]

    assert aspects["strengths"]["status"] == "found"
    assert aspects["strengths"]["citation_status"] == "uncited"
    assert aspects["strengths"]["uncited_statements"] == [
        "FedProx is the most widely adopted framework in industry today."
    ]

    # A refusal, a statement that information is missing, found: false, and a
    # field missing from the reply are all reported as not found, with no text.
    for key in ("research_objective", "limitations", "future_work", "methodology"):
        assert aspects[key]["status"] == "not_found", key
        assert aspects[key]["text"] is None and aspects[key]["citations"] == []


def test_sources_keep_their_text_and_pages(tmp_path: Path) -> None:
    with library(tmp_path) as (app, papers):
        profile, _ = build(app, papers["fedprox"])
        (call,) = app.state.test_llm.calls_of("profile")

    sources = profile.content["sources"]
    assert [source["source_id"] for source in sources] == [
        f"S{n}" for n in range(1, len(sources) + 1)
    ]
    assert all(f'<source id="{s["source_id"]}"' in call["turns"][0].content for s in sources)
    assert call["json_schema"]["required"] == [aspect.key for aspect in ASPECTS]
    assert all(source["text"] and source["page_start"] >= 1 for source in sources)


@pytest.mark.parametrize("reply", ["Sure! Here is the profile.", '["datasets"]'])
def test_unreadable_reply_is_a_clear_error(tmp_path: Path, reply: str) -> None:
    with (
        library(tmp_path, ScriptedLLM(profile=reply)) as (app, papers),
        pytest.raises(UpstreamServiceError) as error,
    ):
        build(app, papers["fedprox"])
    assert error.value.code == "llm_bad_output"


# --- Caching -------------------------------------------------------------------


def test_profile_is_reused_until_the_paper_changes(tmp_path: Path) -> None:
    with library(tmp_path) as (app, papers):
        first, cached_first = build(app, papers["fedprox"])
        again, cached_again = build(app, papers["fedprox"])
        assert (cached_first, cached_again) == (False, True)
        assert again.content == first.content
        assert len(app.state.test_llm.calls_of("profile")) == 1

        # Re-processing changes the chunking signature: the old profile is stale.
        with app.state.database.session() as db:
            db.get(Document, papers["fedprox"]).chunking_signature = "v9:changed"
            db.commit()
        _, cached = build(app, papers["fedprox"])
        assert cached is False
        assert len(app.state.test_llm.calls_of("profile")) == 2

        _, cached = build(app, papers["fedprox"], refresh=True)
        assert cached is False
        assert len(app.state.test_llm.calls_of("profile")) == 3


def test_only_searchable_papers_can_be_profiled(tmp_path: Path) -> None:
    with library(tmp_path) as (app, papers):
        with app.state.database.session() as db:
            db.get(Document, papers["fedprox"]).indexing_status = IndexingStatus.INDEXING
            db.commit()
        with pytest.raises(ConflictError) as error:
            build(app, papers["fedprox"])
        assert error.value.code == "paper_not_indexed"
        with pytest.raises(NotFoundError):
            build(app, "missing")
        assert app.state.test_llm.calls_of("profile") == []
