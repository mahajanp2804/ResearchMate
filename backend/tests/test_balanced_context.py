"""Context for questions about several selected papers: every paper gets a share."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import make_app, make_settings
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

# One paper reports results on every page; the other mentions them once.
RESULTS_HEAVY = [
    "Experimental results on CIFAR-10 show that the proposed method reaches 81% accuracy "
    "after two hundred rounds of training.",
    "On FEMNIST the experimental results improve accuracy to 84%, which outperforms FedAvg by "
    "three points in the same budget.",
    "Shakespeare next-character prediction experimental results: accuracy rises to 88% once "
    "the proximal term is enabled for every device.",
    "Across all experimental settings the average accuracy results exceed the strongest "
    "baseline by six percent, with lower variance.",
]
RESULTS_LIGHT = [
    "SCAFFOLD uses control variates to correct client drift. Its results are reported in "
    "communication rounds rather than final model quality."
]
QUESTION = "Compare the experimental results and accuracy of the papers"


@contextmanager
def library(tmp_path: Path, **overrides) -> Iterator[tuple[FastAPI, TestClient, dict[str, str]]]:
    settings = make_settings(
        tmp_path / "data",
        chunk_size_tokens=32,
        chunk_overlap_tokens=0,
        final_context_chunks=3,
        **overrides,
    )
    app = make_app(settings)
    with TestClient(app) as client:
        papers = {
            "heavy": upload(client, make_pdf(RESULTS_HEAVY, title="Results Paper"))["id"],
            "light": upload(client, make_pdf(RESULTS_LIGHT, title="SCAFFOLD"), "b.pdf")["id"],
        }
        yield app, client, papers


def context_for(app: FastAPI, document_ids: list[str] | None):
    with app.state.database.session() as db:
        scope = app.state.retrieval.resolve_scope(db, document_ids)
        return app.state.retrieval.retrieve_context(db, QUESTION, scope)


def test_without_balancing_one_paper_can_take_every_slot(tmp_path: Path) -> None:
    with library(tmp_path, min_passages_per_paper=0) as (app, _, papers):
        context = context_for(app, [papers["heavy"], papers["light"]])
    assert {chunk.document_id for chunk in context.chunks} == {papers["heavy"]}


def test_each_selected_paper_gets_its_best_passages(tmp_path: Path) -> None:
    with library(tmp_path) as (app, _, papers):
        context = context_for(app, [papers["heavy"], papers["light"]])

    # 3 slots for 2 papers: one guaranteed each, the third by score.
    assert len(context.chunks) == 3
    assert {chunk.document_id for chunk in context.chunks} == {papers["heavy"], papers["light"]}
    # Sources stay best first, numbered from 1.
    scores = [chunk.score for chunk in context.chunks]
    assert scores == sorted(scores, reverse=True)
    assert [chunk.rank for chunk in context.chunks] == [1, 2, 3]


def test_all_papers_or_one_paper_are_ranked_as_before(tmp_path: Path) -> None:
    with library(tmp_path) as (app, _, papers):
        # No selection: the question isn't about particular papers.
        everything = context_for(app, None)
        assert {chunk.document_id for chunk in everything.chunks} == {papers["heavy"]}
        only_light = context_for(app, [papers["light"]])
        assert {chunk.document_id for chunk in only_light.chunks} == {papers["light"]}


def test_the_evidence_gate_still_applies_to_guaranteed_passages(tmp_path: Path) -> None:
    # The stand-in reranker scores by shared words; the light paper's best
    # passage falls below this threshold, so it is not forced in.
    with library(tmp_path, min_rerank_score=0.0) as (app, _, papers):
        context = context_for(app, [papers["heavy"], papers["light"]])
    assert context.chunks
    assert {chunk.document_id for chunk in context.chunks} == {papers["heavy"]}
    assert context.dropped_low_score > 0


def test_follow_up_questions_on_a_comparison_see_both_papers(tmp_path: Path) -> None:
    # A comparison's follow-up is a chat question limited to the compared papers.
    with library(tmp_path) as (_, client, papers):
        response = client.post(
            "/api/chat/query",
            json={"question": QUESTION, "document_ids": [papers["heavy"], papers["light"]]},
        )
    sources = response.json()["assistant_message"]["sources"]
    assert {source["document_title"] for source in sources} == {"Results Paper", "SCAFFOLD"}
