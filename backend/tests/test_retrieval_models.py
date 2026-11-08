"""Why both retrievers exist, shown with the real embedding model.

Dense retrieval finds a passage from a paraphrase that shares no keywords
with it; BM25 finds a rare identifier exactly. Phase 6 fuses the two.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.embedding_service import get_sentence_transformer_embedder
from tests.conftest import make_app, make_settings
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

pytestmark = pytest.mark.models

PAPERS = {
    "partial_work": (
        "Tolerating Partial Work",
        "FedProx tolerates partial work: devices with limited computation may perform "
        "fewer local updates, and a proximal term keeps them close to the global model.",
    ),
    "benchmark": (
        "A New Handwriting Benchmark",
        "We benchmark on the XQ-17 corpus, a collection of handwritten characters "
        "contributed by 3,500 writers.",
    ),
    "privacy": (
        "Private Aggregation",
        "Differential privacy adds calibrated noise to gradients before aggregation "
        "to bound information leakage about any single participant.",
    ),
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    settings = make_settings(Path(tmp_path_factory.mktemp("models")) / "data")

    def real_embedder():
        return get_sentence_transformer_embedder(
            settings.embedding_model_name, device="cpu", batch_size=32
        )

    with TestClient(make_app(settings, embedder_provider=real_embedder)) as test_client:
        for key, (title, text) in PAPERS.items():
            upload(test_client, make_pdf([text], title=title), f"{key}.pdf")
        yield test_client


def top_titles(client: TestClient, query: str) -> dict[str, list[str]]:
    body = client.post(
        "/api/retrieval/search", json={"query": query, "top_k": 3, "methods": ["dense", "bm25"]}
    ).json()
    return {
        result["method"]: [hit["document_title"] for hit in result["hits"]]
        for result in body["results"]
    }


def test_dense_finds_a_paraphrase_with_no_shared_keywords(client: TestClient) -> None:
    titles = top_titles(
        client, "How are phones with weak processors that cannot finish their training handled?"
    )
    assert titles["dense"][0] == "Tolerating Partial Work"
    # None of the query's content words appear in that passage, so BM25 cannot find it.
    assert "Tolerating Partial Work" not in titles["bm25"]


def test_bm25_finds_a_rare_identifier_exactly(client: TestClient) -> None:
    titles = top_titles(client, "XQ-17")
    assert titles["bm25"] == ["A New Handwriting Benchmark"]


def test_both_agree_when_wording_and_meaning_match(client: TestClient) -> None:
    titles = top_titles(client, "differential privacy noise added to gradients")
    assert titles["dense"][0] == titles["bm25"][0] == "Private Aggregation"
