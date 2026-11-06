from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.exceptions import InvalidInputError
from app.models import Document, IndexingStatus
from app.services.retrieval_service import RetrievalService, normalize_query
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

FEDPROX = [
    "FedProx adds a proximal term to the local objective so that devices with "
    "heterogeneous data stay close to the global model.",
    "We evaluate FedProx on the Sent140 and Shakespeare datasets with non-IID partitions.",
]
SCAFFOLD = [
    "SCAFFOLD uses control variates to correct client drift in federated optimisation.",
    "Experiments on EMNIST show SCAFFOLD needs fewer communication rounds than FedAvg.",
]


@pytest.fixture
def papers(client: TestClient) -> dict[str, dict]:
    return {
        "fedprox": upload(client, make_pdf(FEDPROX, title="Federated Optimization"), "a.pdf"),
        "scaffold": upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"), "b.pdf"),
    }


@pytest.fixture
def retrieval(app: FastAPI) -> RetrievalService:
    return app.state.retrieval


@pytest.fixture
def db(app: FastAPI, client: TestClient) -> Iterator[Session]:
    with app.state.database.session() as session:
        yield session


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query("  What   is\n FedProx? ") == "What is FedProx?"
    assert normalize_query("ＦｅｄＰｒｏｘ") == "FedProx"


@pytest.mark.parametrize(
    ("query", "code"), [("   ", "empty_query"), ("x" * 1001, "query_too_long")]
)
def test_normalize_query_rejects_bad_input(query: str, code: str) -> None:
    with pytest.raises(InvalidInputError) as error:
        normalize_query(query)
    assert error.value.code == code


def test_scope_defaults_to_all_searchable_papers(db, papers, retrieval) -> None:
    scope = retrieval.resolve_scope(db, None)
    assert scope.document_ids is None
    assert scope.effective_document_ids == {paper["id"] for paper in papers.values()}


def test_scope_rejects_unknown_and_empty_filters(db, papers, retrieval) -> None:
    with pytest.raises(InvalidInputError) as error:
        retrieval.resolve_scope(db, [papers["fedprox"]["id"], "0" * 32])
    assert error.value.code == "unknown_documents"
    assert error.value.details["document_ids"] == ["0" * 32]
    with pytest.raises(InvalidInputError) as error:
        retrieval.resolve_scope(db, [])
    assert error.value.code == "empty_document_filter"


def test_scope_reports_papers_that_are_not_searchable(db, papers, retrieval) -> None:
    document = db.get(Document, papers["scaffold"]["id"])
    document.indexing_status = IndexingStatus.FAILED
    db.commit()

    ids = [papers["fedprox"]["id"], papers["scaffold"]["id"], papers["fedprox"]["id"]]
    scope = retrieval.resolve_scope(db, ids)
    assert scope.document_ids == {papers["fedprox"]["id"]}
    assert scope.skipped_document_ids == (papers["scaffold"]["id"],)


@pytest.mark.parametrize("method", ["dense_search", "keyword_search"])
def test_results_carry_paper_details_and_ranks(db, papers, retrieval, method) -> None:
    scope = retrieval.resolve_scope(db, None)
    results = getattr(retrieval, method)(db, "control variates client drift", 3, scope)

    assert results[0].document_id == papers["scaffold"]["id"]
    assert results[0].document_title == "SCAFFOLD Paper Title"
    assert "control variates" in results[0].text
    # The two-page test paper is a single passage, so it spans both pages.
    assert (results[0].page_start, results[0].page_end) == (1, 2)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))


@pytest.mark.parametrize("method", ["dense_search", "keyword_search"])
def test_filter_keeps_other_papers_out(db, papers, retrieval, method) -> None:
    search = getattr(retrieval, method)
    query = "FedAvg rounds and datasets"  # "FedAvg" is in SCAFFOLD, "datasets" in FedProx
    unfiltered = search(db, query, 10, retrieval.resolve_scope(db, None))
    assert {result.document_id for result in unfiltered} == {p["id"] for p in papers.values()}

    filtered = search(db, query, 10, retrieval.resolve_scope(db, [papers["fedprox"]["id"]]))
    assert {result.document_id for result in filtered} == {papers["fedprox"]["id"]}


@pytest.mark.parametrize("method", ["dense_search", "keyword_search"])
def test_nothing_searchable_returns_nothing(db, papers, retrieval, method, app) -> None:
    for paper in papers.values():
        db.get(Document, paper["id"]).indexing_status = IndexingStatus.NOT_INDEXED
    db.commit()
    embedded = app.state.test_embedder.texts_embedded

    scope = retrieval.resolve_scope(db, None)
    assert getattr(retrieval, method)(db, "FedProx", 5, scope) == []
    assert app.state.test_embedder.texts_embedded == embedded  # model not even used
