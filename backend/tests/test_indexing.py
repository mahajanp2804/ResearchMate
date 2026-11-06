"""Embedding and vector indexing through the API, using offline fakes."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models import Document
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


def reindex(client: TestClient, document_id: str, force: bool = False) -> str:
    query = "?force=true" if force else ""
    return client.post(f"/api/documents/{document_id}/reindex{query}").json()["action"]


def test_forced_reindex_replaces_vectors(client: TestClient, app: FastAPI) -> None:
    document = upload(client, sample_paper())
    before = vector_ids(app)

    assert reindex(client, document["id"], force=True) == "rechunked"

    after = vector_ids(app)
    assert after == chunk_ids(client, document["id"])
    assert after.isdisjoint(before)  # old chunk ids and their vectors are gone


def test_reindex_skips_an_up_to_date_paper(client: TestClient, app: FastAPI) -> None:
    document = upload(client, sample_paper())
    ids = chunk_ids(client, document["id"])
    embedded = app.state.test_embedder.texts_embedded

    assert reindex(client, document["id"]) == "unchanged"

    assert chunk_ids(client, document["id"]) == ids
    assert app.state.test_embedder.texts_embedded == embedded  # nothing re-embedded


def test_missing_vectors_are_re_embedded_without_rechunking(
    client: TestClient, app: FastAPI
) -> None:
    document = upload(client, sample_paper())
    ids = chunk_ids(client, document["id"])
    app.state.ingestion.vector_store.remove(ids)

    assert reindex(client, document["id"]) == "reembedding"

    assert chunk_ids(client, document["id"]) == ids  # same chunks
    assert vector_ids(app) == ids  # vectors restored


LONG_PAGES = [
    " ".join(f"Sentence {page}.{index} about topic {page * 10 + index}." for index in range(30))
    for page in range(1, 4)
]


def test_changed_chunk_settings_trigger_rechunking(tmp_path: Path) -> None:
    with TestClient(make_app(make_settings(tmp_path / "data"))) as client:
        document = upload(client, make_pdf(LONG_PAGES, title="A Paper With Long Pages"))

    smaller_chunks = make_settings(tmp_path / "data", chunk_size_tokens=40, chunk_overlap_tokens=8)
    app = make_app(smaller_chunks)
    with TestClient(app) as client:
        assert reindex(client, document["id"]) == "rechunked"
        detail = client.get(f"/api/documents/{document['id']}").json()
        assert detail["chunking_signature"].endswith(":40/8")
        assert detail["chunk_count"] > document["chunk_count"]
        assert vector_ids(app) == chunk_ids(client, document["id"])


def test_changed_embedding_model_re_embeds_every_paper(tmp_path: Path) -> None:
    with TestClient(make_app(make_settings(tmp_path / "data"))) as client:
        first = upload(client, sample_paper("First Paper on Federated Learning"), "a.pdf")
        second = upload(client, sample_paper("Second Paper on Client Drift"), "b.pdf")

    app = make_app(make_settings(tmp_path / "data", embedding_model_name="another/model"))
    with TestClient(app) as client:
        ids = chunk_ids(client, first["id"])
        assert reindex(client, first["id"]) == "reembedding"
        assert chunk_ids(client, first["id"]) == ids  # chunks kept, only vectors rebuilt

        store = app.state.ingestion.vector_store
        assert store.model_name == "another/model"
        assert store.ids() == ids  # vectors from the old model were all discarded
        # The other paper's old vectors are gone too, and it says so.
        assert client.get(f"/api/documents/{second['id']}").json()["indexing_status"] == (
            "not_indexed"
        )
        assert reindex(client, second["id"]) == "reembedding"
        assert client.get(f"/api/documents/{second['id']}").json()["indexing_status"] == "indexed"


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


# --- Startup consistency check -----------------------------------------------


def set_statuses(app: FastAPI, document_id: str, **values) -> None:
    with app.state.database.session() as db:
        document = db.get(Document, document_id)
        for field, value in values.items():
            setattr(document, field, value)
        db.commit()


def test_restart_resets_interrupted_work(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings)
    with TestClient(app) as client:
        first = upload(client, sample_paper("First Paper on Federated Learning"), "a.pdf")
        second = upload(client, sample_paper("Second Paper on Client Drift"), "b.pdf")
        # Simulate a shutdown in the middle of uploading and indexing.
        set_statuses(app, first["id"], processing_status="processing")
        set_statuses(app, second["id"], indexing_status="indexing")

    with TestClient(make_app(settings)) as client:
        first = client.get(f"/api/documents/{first['id']}").json()
        assert first["processing_status"] == "failed"
        assert "interrupted" in first["status_message"]
        second = client.get(f"/api/documents/{second['id']}").json()
        assert second["indexing_status"] == "not_indexed"


def test_restart_without_index_file_marks_papers_not_indexed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    with TestClient(make_app(settings)) as client:
        document = upload(client, sample_paper())
    for path in settings.index_dir.iterdir():
        path.unlink()

    with TestClient(make_app(settings)) as client:
        status = client.get(f"/api/documents/{document['id']}").json()["indexing_status"]
        assert status == "not_indexed"
        assert reindex(client, document["id"]) == "reembedding"
        status = client.get(f"/api/documents/{document['id']}").json()["indexing_status"]
        assert status == "indexed"


def test_restart_with_a_different_model_marks_papers_not_indexed(tmp_path: Path) -> None:
    with TestClient(make_app(make_settings(tmp_path / "data"))) as client:
        document = upload(client, sample_paper())

    settings = make_settings(tmp_path / "data", embedding_model_name="another/model")
    with TestClient(make_app(settings)) as client:
        status = client.get(f"/api/documents/{document['id']}").json()["indexing_status"]
        assert status == "not_indexed"


def test_restart_removes_orphaned_vectors(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings)
    with TestClient(app) as client:
        document = upload(client, sample_paper())
        ids = chunk_ids(client, document["id"])
        store = app.state.ingestion.vector_store
        store.add([999_999], app.state.test_embedder.embed_documents(["left behind"]))

    restarted = make_app(settings)
    with TestClient(restarted):
        assert vector_ids(restarted) == ids
