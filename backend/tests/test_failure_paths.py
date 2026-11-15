"""Races and failures that must leave the library consistent."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.dependencies import get_db
from app.models import Document, IndexingStatus
from app.services import document_service
from tests.conftest import make_app, make_settings
from tests.fakes import HashingEmbedder
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

PAPER = ["FedProx adds a proximal term to the local objective.", "We evaluate on MNIST."]


class InterruptedEmbedder(HashingEmbedder):
    """Runs ``during_embedding`` once, while the first batch is being embedded."""

    def __init__(self, name: str, during_embedding) -> None:
        super().__init__(name=name)
        self.during_embedding = during_embedding

    def embed_documents(self, texts):
        if self.during_embedding is not None:
            hook, self.during_embedding = self.during_embedding, None
            hook()
        return super().embed_documents(texts)


def test_vectors_are_dropped_if_the_paper_changes_while_it_is_embedded(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    holder: dict = {}

    def rechunk_meanwhile() -> None:
        # Another request replaces the paper's chunks mid-embedding.
        with holder["app"].state.database.session() as db:
            db.execute(text("DELETE FROM chunks"))
            db.commit()

    embedder = InterruptedEmbedder(settings.embedding_model_name, rechunk_meanwhile)
    app = make_app(settings, embedder_provider=lambda: embedder)
    holder["app"] = app
    with TestClient(app) as client:
        document = upload(client, make_pdf(PAPER, title="Federated Optimization"))
        with app.state.database.session() as db:
            status = db.get(Document, document["id"]).indexing_status

    # The vectors for the old chunks were removed rather than left orphaned,
    # and the paper was not marked searchable.
    assert app.state.ingestion.vector_store.ids() == set()
    assert status is not IndexingStatus.INDEXED


def test_two_uploads_of_one_file_at_once_give_a_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path / "data")
    with TestClient(make_app(settings)) as client:
        pdf = make_pdf(PAPER, title="Federated Optimization")
        first = upload(client, pdf)

        # Simulate the race: the second request checks for the file before the
        # first one has committed, so its own check finds nothing.
        real_find = document_service._find_by_hash
        calls = {"count": 0}

        def find_after_race(db, file_hash):
            calls["count"] += 1
            return None if calls["count"] == 1 else real_find(db, file_hash)

        monkeypatch.setattr(document_service, "_find_by_hash", find_after_race)
        response = client.post(
            "/api/documents/upload", files=[("files", ("again.pdf", pdf, "application/pdf"))]
        )

    result = response.json()["results"][0]
    assert result["status"] == "duplicate"
    assert result["document"]["id"] == first["id"]
    # The second copy's file was removed again.
    assert len(list(settings.upload_dir.iterdir())) == 1


@pytest.fixture
def broken_database_client(tmp_path: Path) -> Iterator[TestClient]:
    app = make_app(make_settings(tmp_path / "data"))

    class UnreachableSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("database is unreachable"))

    app.dependency_overrides[get_db] = lambda: UnreachableSession()
    with TestClient(app) as client:
        yield client


def test_health_reports_a_database_failure(broken_database_client: TestClient) -> None:
    response = broken_database_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "error"
