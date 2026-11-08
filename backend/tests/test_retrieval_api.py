from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_app, make_settings
from tests.fakes import HashingEmbedder, unavailable_embedder, unavailable_reranker
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


def test_runs_each_requested_method(client: TestClient, papers) -> None:
    response = search(
        client, query="  control variates   for client drift ", methods=["dense", "bm25"]
    )

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


# --- Hybrid, reranking and context (Phase 6) ----------------------------------


def test_hybrid_rerank_is_the_default_method(client: TestClient, papers) -> None:
    body = search(client, query="control variates client drift").json()
    assert [result["method"] for result in body["results"]] == ["hybrid_rerank"]
    top = body["results"][0]["hits"][0]
    assert top["document_title"] == "SCAFFOLD Paper Title"
    assert {"dense", "bm25", "hybrid"} & top["source_ranks"].keys()
    assert top["token_count"] > 0


def test_all_four_stages_can_be_compared(client: TestClient, papers) -> None:
    methods = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    body = search(client, query="FedAvg rounds and datasets", methods=methods).json()
    assert [result["method"] for result in body["results"]] == methods
    for result in body["results"]:
        assert result["hits"]


def test_filter_holds_for_fused_and_reranked_results(client: TestClient, papers) -> None:
    body = search(
        client,
        query="FedAvg rounds and datasets",
        methods=["hybrid", "hybrid_rerank"],
        document_ids=[papers["fedprox"]["id"]],
    ).json()
    for result in body["results"]:
        assert {hit["document_id"] for hit in result["hits"]} == {papers["fedprox"]["id"]}


def test_context_endpoint_shows_what_the_llm_would_receive(client: TestClient, papers) -> None:
    response = client.post("/api/retrieval/context", json={"query": "control variates drift"})

    assert response.status_code == 200
    body = response.json()
    assert body["reranked"] is True
    assert body["has_evidence"] is True
    assert body["passages"][0]["document_title"] == "SCAFFOLD Paper Title"
    assert body["total_tokens"] == sum(passage["token_count"] for passage in body["passages"])
    assert body["max_passages"] == 8
    assert body["min_score"] is None  # evidence gate off by default
    assert body["dropped"] == {"low_score": 0, "duplicates": 0, "over_budget": 0}
    assert body["searched_document_count"] == 2


def test_context_with_no_papers_has_no_evidence(client: TestClient) -> None:
    body = client.post("/api/retrieval/context", json={"query": "anything"}).json()
    assert body["has_evidence"] is False
    assert body["passages"] == []


def test_context_validates_input(client: TestClient, papers) -> None:
    response = client.post("/api/retrieval/context", json={"query": " ", "document_ids": None})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_query"


def test_reranker_outage_fails_reranked_methods_only(tmp_path: Path) -> None:

    settings = make_settings(tmp_path / "data")
    with TestClient(make_app(settings, reranker_provider=unavailable_reranker)) as client:
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"))

        reranked = search(client, query="client drift")
        assert reranked.status_code == 503
        assert reranked.json()["error"]["code"] == "reranker_unavailable"

        hybrid = search(client, query="client drift", methods=["hybrid"])
        assert hybrid.status_code == 200
        assert hybrid.json()["results"][0]["hits"]
