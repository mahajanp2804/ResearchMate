"""Embedding and vector indexing through the API, using offline fakes."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import make_app, make_settings
from tests.fakes import HashingEmbedder, unavailable_embedder
from tests.pdf_factory import make_pdf, paper_text, sample_paper

UPLOAD_URL = "/api/documents/upload"


def upload(client: TestClient, data: bytes, name: str = "paper.pdf") -> dict:
    response = client.post(UPLOAD_URL, files=[("files", (name, data, "application/pdf"))])
    return response.json()["results"][0]["document"]


def chunk_ids(client: TestClient, document_id: str) -> set[int]:
    items = client.get(f"/api/documents/{document_id}/chunks?limit=200").json()["items"]
    return {chunk["id"] for chunk in items}


def vector_ids(app: FastAPI) -> set[int]:
    return app.state.ingestion.vector_store.ids()


def test_upload_embeds_and_indexes_every_chunk(client: TestClient, app: FastAPI) -> None:
    document = upload(client, sample_paper(pages=4))

    detail = client.get(f"/api/documents/{document['id']}").json()
    assert detail["indexing_status"] == "indexed"
    assert detail["embedding_model"] == app.state.settings.embedding_model_name
    assert detail["indexed_at"] is not None
    # Vector ids are the chunk ids, one vector per chunk.
    assert vector_ids(app) == chunk_ids(client, document["id"])


def test_indexed_chunks_are_searchable(client: TestClient, app: FastAPI) -> None:
    federated = upload(client, sample_paper("Federated Optimization Methods"), "fl.pdf")
    other = make_pdf(["Ovens bake bread with flour, yeast, butter and sugar."], title="Baking")
    baking = upload(client, other, "baking.pdf")

    embedder = app.state.test_embedder
    store = app.state.ingestion.vector_store
    top = store.search(embedder.embed_query("bread flour yeast butter"), 1)[0]
    assert top.chunk_id in chunk_ids(client, baking["id"])
    top = store.search(embedder.embed_query("federated clients non-IID dataset"), 1)[0]
    assert top.chunk_id in chunk_ids(client, federated["id"])


def test_index_persists_across_restarts(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    with TestClient(make_app(settings)) as client:
        document = upload(client, sample_paper())
        ids = chunk_ids(client, document["id"])
    restarted = make_app(settings)
    with TestClient(restarted):
        assert vector_ids(restarted) == ids


def test_delete_removes_only_that_papers_vectors(client: TestClient, app: FastAPI) -> None:
    first = upload(client, sample_paper("First Paper on Federated Learning"), "a.pdf")
    second = upload(client, sample_paper("Second Paper on Client Drift"), "b.pdf")
    second_ids = chunk_ids(client, second["id"])

    client.delete(f"/api/documents/{first['id']}")

    assert vector_ids(app) == second_ids


def test_reindex_replaces_vectors(client: TestClient, app: FastAPI) -> None:
    document = upload(client, sample_paper())
    before = vector_ids(app)

    client.post(f"/api/documents/{document['id']}/reindex")

    after = vector_ids(app)
    assert after == chunk_ids(client, document["id"])
    assert after.isdisjoint(before)  # old chunk ids and their vectors are gone


def test_embedding_outage_is_recorded_and_recoverable(tmp_path: Path) -> None:
    class SwitchableEmbedder:
        available = False

        def __init__(self, name: str) -> None:
            self.embedder = HashingEmbedder(name=name)

        def __call__(self):
            if not self.available:
                unavailable_embedder()
            return self.embedder

    settings = make_settings(tmp_path / "data")
    provider = SwitchableEmbedder(settings.embedding_model_name)
    app = make_app(settings, embedder_provider=provider)
    with TestClient(app) as client:
        document = upload(client, make_pdf([paper_text("x", 1)], title="An Unlucky Paper"))
        detail = client.get(f"/api/documents/{document['id']}").json()
        # Text processing succeeded; only indexing failed, with a reason.
        assert detail["processing_status"] == "completed"
        assert detail["indexing_status"] == "failed"
        assert "embedding model" in detail["status_message"]
        assert vector_ids(app) == set()

        provider.available = True
        client.post(f"/api/documents/{document['id']}/reindex")
        detail = client.get(f"/api/documents/{document['id']}").json()
        assert detail["indexing_status"] == "indexed"
        assert detail["status_message"] is None
        assert vector_ids(app) == chunk_ids(client, document["id"])


def test_dashboard_counts_indexed_papers(client: TestClient) -> None:
    upload(client, sample_paper())
    assert client.get("/api/dashboard/stats").json()["indexed_papers"] == 1
