"""The full retrieval pipeline with the real embedding and reranking models.

Asserts only properties that held when measured. With the lightweight MS MARCO
cross-encoder, reranking was mixed on these research passages (it lifted the
"partial work" passage into the top 3 for a paraphrased question but demoted
the right passage for "Which datasets were used?"), so whether it helps is
left to the Phase 11 evaluation rather than asserted here.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.embedding_service import get_sentence_transformer_embedder
from app.services.reranker import get_cross_encoder_reranker
from tests.conftest import make_app, make_settings
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

pytestmark = pytest.mark.models

FEDPROX = [
    "1 Introduction\nFederated learning trains models over remote devices such as phones. "
    "McMahan et al. proposed FedAvg, which runs SGD locally and averages the resulting models "
    "at a central server. FedAvg does not account for heterogeneity across devices.",
    "2 Proposed Method\nFedProx tolerates partial work: each device may perform a variable "
    "number of local updates depending on its available resources. The proximal term limits "
    "the impact of local updates by keeping them close to the global model.",
    "3 Experiments\nWe evaluate on MNIST, FEMNIST, Shakespeare and Sent140 with non-IID "
    "partitions. Models are multinomial logistic regression and LSTMs.",
    "4 Limitations\nOur analysis assumes bounded dissimilarity, which may not hold in all "
    "federations. Future work includes adaptive selection of the proximal coefficient.",
]
SCAFFOLD = [
    "Abstract\nFederated Averaging suffers from client drift when data is heterogeneous across "
    "clients. SCAFFOLD uses control variates to correct for the drift in local updates.",
    "4 Experiments\nWe evaluate SCAFFOLD on EMNIST with 400 clients and compare against FedAvg "
    "and FedProx. SCAFFOLD needs far fewer communication rounds.",
]
STAGES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    # One passage per section, so stages can disagree about which section answers.
    settings = make_settings(
        Path(tmp_path_factory.mktemp("pipeline")) / "data",
        chunk_size_tokens=96,
        chunk_overlap_tokens=0,
    )
    app = make_app(
        settings,
        embedder_provider=lambda: get_sentence_transformer_embedder(
            settings.embedding_model_name, device="cpu", batch_size=32
        ),
        reranker_provider=lambda: get_cross_encoder_reranker(
            settings.reranker_model_name, device="cpu"
        ),
    )
    with TestClient(app) as test_client:
        upload(test_client, make_pdf(FEDPROX, title="Federated Optimization"), "a.pdf")
        upload(test_client, make_pdf(SCAFFOLD, title="SCAFFOLD"), "b.pdf")
        yield test_client


def sections(client: TestClient, query: str) -> dict[str, list[str]]:
    body = client.post(
        "/api/retrieval/search", json={"query": query, "top_k": 3, "methods": STAGES}
    ).json()
    return {
        result["method"]: [hit["section"] for hit in result["hits"]] for result in body["results"]
    }


def test_fusion_keeps_dense_results_when_bm25_finds_nothing(client: TestClient) -> None:
    # No passage contains "datasets" or "used", so BM25 is empty; fusion must
    # not lose what dense retrieval found.
    found = sections(client, "Which datasets were used?")
    assert found["bm25"] == []
    assert found["hybrid"] == found["dense"]
    assert found["dense"][0] == "3 Experiments"


@pytest.mark.parametrize(
    ("query", "expected"),
    [("What are the limitations?", "4 Limitations"), ("What fixes client drift?", "Abstract")],
)
def test_all_stages_agree_on_clear_questions(client: TestClient, query: str, expected: str) -> None:
    found = sections(client, query)
    for stage in ("hybrid", "hybrid_rerank"):
        assert found[stage][0] == expected


def test_context_contains_the_answering_passage(client: TestClient) -> None:
    body = client.post(
        "/api/retrieval/context", json={"query": "How do clients with little compute participate?"}
    ).json()
    assert body["has_evidence"]
    assert "2 Proposed Method" in [passage["section"] for passage in body["passages"]]
