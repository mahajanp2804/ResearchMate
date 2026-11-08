"""Hybrid fusion, reranking and context selection through the retrieval service."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.exceptions import ServiceUnavailableError
from app.models import Document, IndexingStatus
from tests.conftest import make_app, make_settings
from tests.fakes import unavailable_reranker
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

FEDPROX = [
    "FedProx adds a proximal term to the local objective of each device.",
    "Devices with limited computation may perform fewer local updates in FedProx.",
    "We evaluate FedProx on Sent140 and Shakespeare with non-IID partitions.",
    "Our analysis assumes bounded dissimilarity, which may not always hold.",
]
SCAFFOLD = [
    "SCAFFOLD uses control variates to correct client drift in local updates.",
    "Client drift slows FedAvg when data is heterogeneous across clients.",
    "Experiments on EMNIST show SCAFFOLD needs fewer communication rounds.",
]


@contextmanager
def library_app(tmp_path: Path, **overrides) -> Iterator[tuple[FastAPI, dict[str, str]]]:
    # Small chunks give each sentence its own passage, so there is something to rank.
    settings = make_settings(
        tmp_path / "data", chunk_size_tokens=32, chunk_overlap_tokens=0, **overrides
    )
    app = make_app(settings)
    with TestClient(app) as client:
        papers = {
            "fedprox": upload(client, make_pdf(FEDPROX, title="Federated Optimization"), "a.pdf")[
                "id"
            ],
            "scaffold": upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"), "b.pdf")[
                "id"
            ],
        }
        yield app, papers


@pytest.fixture
def library(tmp_path: Path) -> Iterator[tuple[FastAPI, dict[str, str]]]:
    with library_app(tmp_path) as built:
        yield built


@pytest.fixture
def db(library) -> Iterator[Session]:
    app, _ = library
    with app.state.database.session() as session:
        yield session


def run(library, db, method: str, query: str, top_k: int = 10, document_ids=None):
    app, _ = library
    retrieval = app.state.retrieval
    scope = retrieval.resolve_scope(db, document_ids)
    return getattr(retrieval, method)(db, query, top_k, scope)


def test_hybrid_fuses_dense_and_bm25_ranks(library, db) -> None:
    results = run(library, db, "hybrid_search", "client drift control variates")

    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert "control variates" in results[0].text
    # Found by both retrievers, so it carries both ranks.
    assert set(results[0].source_ranks) == {"dense", "bm25"}
    assert all(set(result.source_ranks) <= {"dense", "bm25"} for result in results)


def test_reranking_reorders_by_cross_encoder_score(library, db) -> None:
    app, _ = library
    results = run(library, db, "reranked_search", "devices with limited computation", top_k=5)

    assert "limited computation" in results[0].text
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert "hybrid" in results[0].source_ranks  # its position before reranking
    query, scored = app.state.test_reranker.calls[-1]
    assert query == "devices with limited computation"
    assert scored <= app.state.settings.rerank_candidates


@pytest.mark.parametrize("method", ["hybrid_search", "reranked_search"])
def test_paper_filter_holds_through_every_stage(library, db, method) -> None:
    _, papers = library
    results = run(
        library, db, method, "client drift FedProx devices", document_ids=[papers["fedprox"]]
    )
    assert results
    assert {result.document_id for result in results} == {papers["fedprox"]}


def test_reranker_is_not_called_when_nothing_is_searchable(library, db) -> None:
    app, papers = library
    for document_id in papers.values():
        db.get(Document, document_id).indexing_status = IndexingStatus.NOT_INDEXED
    db.commit()
    calls = len(app.state.test_reranker.calls)

    assert run(library, db, "reranked_search", "client drift") == []
    assert len(app.state.test_reranker.calls) == calls


def test_reranker_outage_is_reported(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, reranker_provider=unavailable_reranker)
    with TestClient(app) as client:
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"))
        with app.state.database.session() as session:
            retrieval = app.state.retrieval
            scope = retrieval.resolve_scope(session, None)
            with pytest.raises(ServiceUnavailableError) as error:
                retrieval.reranked_search(session, "client drift", 5, scope)
    assert error.value.code == "reranker_unavailable"


def test_context_respects_chunk_and_token_limits(library, db) -> None:
    app, _ = library
    retrieval = app.state.retrieval
    context = retrieval.retrieve_context(
        db, "client drift and devices", retrieval.resolve_scope(db, None)
    )
    settings = app.state.settings
    assert context.has_evidence
    assert len(context.chunks) <= settings.final_context_chunks
    assert context.total_tokens == sum(chunk.token_count for chunk in context.chunks)
    assert context.total_tokens <= settings.max_context_tokens
    assert "drift" in context.chunks[0].text  # reranked best first


def test_context_without_reranking_uses_fused_order(tmp_path: Path) -> None:
    with library_app(tmp_path, reranker_enabled=False) as (app, _):
        with app.state.database.session() as session:
            retrieval = app.state.retrieval
            context = retrieval.retrieve_context(
                session, "client drift", retrieval.resolve_scope(session, None)
            )
        assert context.has_evidence
        assert app.state.test_reranker.calls == []
        assert all("hybrid" not in chunk.source_ranks for chunk in context.chunks)


def test_evidence_threshold_can_refuse_weak_matches(tmp_path: Path) -> None:
    # The fake reranker scores a passage sharing no query word at -11.
    with (
        library_app(tmp_path, min_rerank_score=-10.0) as (app, _),
        app.state.database.session() as session,
    ):
        retrieval = app.state.retrieval
        scope = retrieval.resolve_scope(session, None)
        weak = retrieval.retrieve_context(session, "sourdough baking recipe", scope)
        assert not weak.has_evidence
        assert retrieval.retrieve_context(session, "client drift", scope).has_evidence
