from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_app, make_settings
from tests.fakes import HashingEmbedder, unavailable_embedder
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload
from tests.test_retrieval_service import FEDPROX, SCAFFOLD

SEARCH_URL = "/api/retrieval/search"


@pytest.fixture
def papers(client: TestClient) -> dict[str, dict]:
    return {
        "fedprox": upload(client, make_pdf(FEDPROX, title="Federated Optimization"), "a.pdf"),
        "scaffold": upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"), "b.pdf"),
    }


def search(client: TestClient, **body):
    return client.post(SEARCH_URL, json=body)


def test_runs_dense_and_bm25_by_default(client: TestClient, papers) -> None:
    response = search(client, query="  control variates   for client drift ")

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "control variates for client drift"
    assert [result["method"] for result in body["results"]] == ["dense", "bm25"]
    assert body["searched_document_count"] == 2
    assert body["skipped_document_ids"] == []
    for result in body["results"]:
        assert result["latency_ms"] >= 0
        top = result["hits"][0]
        assert top["document_title"] == "SCAFFOLD Paper Title"
        assert top["rank"] == 1
        assert {"chunk_id", "page_start", "page_end", "section", "text", "score"} <= top.keys()


def test_single_method_and_top_k(client: TestClient, papers) -> None:
    body = search(client, query="FedAvg datasets rounds", methods=["bm25"], top_k=1).json()
    assert [result["method"] for result in body["results"]] == ["bm25"]
    assert len(body["results"][0]["hits"]) == 1


def test_document_filter_and_skipped_papers(client: TestClient, papers) -> None:
    body = search(
        client, query="FedAvg datasets rounds", document_ids=[papers["fedprox"]["id"]]
    ).json()
    for result in body["results"]:
        assert {hit["document_id"] for hit in result["hits"]} == {papers["fedprox"]["id"]}
    assert body["searched_document_count"] == 1


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"query": "   "}, "empty_query"),
        ({"query": "x" * 1500}, "query_too_long"),
        ({"query": "drift", "document_ids": []}, "empty_document_filter"),
        ({"query": "drift", "document_ids": ["0" * 32]}, "unknown_documents"),
        ({"query": "drift", "methods": ["magic"]}, "invalid_input"),
        ({"query": "drift", "methods": []}, "invalid_input"),
        ({"query": "drift", "top_k": 0}, "invalid_input"),
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, papers, body, code) -> None:
    response = search(client, **body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


def test_empty_library_returns_empty_results(client: TestClient) -> None:
    body = search(client, query="anything").json()
    assert body["searched_document_count"] == 0
    assert all(result["hits"] == [] for result in body["results"])


def test_embedding_outage_fails_dense_but_not_bm25(tmp_path: Path) -> None:
    class SwitchableEmbedder:
        available = True

        def __init__(self, name: str) -> None:
            self.embedder = HashingEmbedder(name=name)

        def __call__(self):
            if not self.available:
                unavailable_embedder()
            return self.embedder

    settings = make_settings(tmp_path / "data")
    provider = SwitchableEmbedder(settings.embedding_model_name)
    with TestClient(make_app(settings, embedder_provider=provider)) as client:
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"))
        provider.available = False

        dense = search(client, query="client drift", methods=["dense"])
        assert dense.status_code == 503
        assert dense.json()["error"]["code"] == "embedding_model_unavailable"

        keyword = search(client, query="client drift", methods=["bm25"])
        assert keyword.status_code == 200
        assert keyword.json()["results"][0]["hits"]
