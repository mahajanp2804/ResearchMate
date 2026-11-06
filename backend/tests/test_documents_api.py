from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ActivityEvent, Chunk, DocumentPage
from tests.conftest import make_app, make_settings
from tests.fakes import WhitespaceTokenizer, unavailable_tokenizer
from tests.pdf_factory import make_pdf, paper_text, sample_paper

UPLOAD_URL = "/api/documents/upload"


def upload(client: TestClient, *files: tuple[str, bytes]):
    return client.post(
        UPLOAD_URL,
        files=[("files", (name, data, "application/pdf")) for name, data in files],
    )


def test_upload_creates_document_with_pages(client: TestClient, app: FastAPI) -> None:
    response = upload(client, ("fedavg.pdf", sample_paper("Deep Networks from Decentralized Data")))

    assert response.status_code == 201
    body = response.json()
    assert (body["created"], body["duplicates"], body["rejected"]) == (1, 0, 0)
    result = body["results"][0]
    assert result["status"] == "created"
    document = result["document"]
    assert document["title"] == "Deep Networks from Decentralized Data"
    assert document["filename"] == "fedavg.pdf"
    assert document["page_count"] == 3
    assert document["processing_status"] == "completed"
    # Embedding runs in the background after the response is sent.
    assert document["indexing_status"] == "indexing"
    assert document["status_message"] is None

    stored = app.state.settings.upload_dir / f"{document['id']}.pdf"
    assert stored.is_file()
    # No temporary files are left behind.
    assert [path.name for path in app.state.settings.upload_dir.iterdir()] == [stored.name]


def test_list_and_get_document(client: TestClient) -> None:
    created = upload(client, ("a.pdf", sample_paper("First Paper About Federated Learning")))
    document_id = created.json()["results"][0]["document"]["id"]

    listing = client.get("/api/documents").json()
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == document_id

    detail = client.get(f"/api/documents/{document_id}").json()
    assert detail["pages_without_text"] == []
    assert len(detail["file_hash"]) == 64
    assert detail["extracted_metadata"]["title_source"] == "first_page"


def test_file_endpoint_serves_original_pdf(client: TestClient) -> None:
    data = sample_paper()
    document_id = upload(client, ("paper.pdf", data)).json()["results"][0]["document"]["id"]

    response = client.get(f"/api/documents/{document_id}/file")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline")
    assert response.content == data


def test_duplicate_upload_returns_existing_document(client: TestClient) -> None:
    data = sample_paper()
    first = upload(client, ("paper.pdf", data)).json()["results"][0]
    second = upload(client, ("renamed-copy.pdf", data))

    assert second.status_code == 200  # nothing new was created
    result = second.json()["results"][0]
    assert result["status"] == "duplicate"
    assert result["document"]["id"] == first["document"]["id"]
    assert client.get("/api/documents").json()["total"] == 1


@pytest.mark.parametrize(
    ("filename", "data", "code"),
    [
        ("notes.txt", b"plain text", "unsupported_file_type"),
        ("fake.pdf", b"MZ\x90\x00 an executable", "unsupported_file_type"),
        ("empty.pdf", b"", "empty_file"),
        ("broken.pdf", b"%PDF-1.7\nnot really a pdf", "invalid_pdf"),
        ("scan.pdf", make_pdf([None, None]), "no_extractable_text"),
        ("locked.pdf", make_pdf([paper_text("x", 1)], user_password="pw"), "encrypted_pdf"),
    ],
)
def test_invalid_files_are_rejected_with_reason(
    client: TestClient, app: FastAPI, filename: str, data: bytes, code: str
) -> None:
    response = upload(client, (filename, data))

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert result["error"]["code"] == code
    assert result["error"]["message"]
    assert result["document"] is None
    assert list(app.state.settings.upload_dir.iterdir()) == []


def test_mixed_batch_reports_each_file(client: TestClient) -> None:
    response = upload(
        client,
        ("good.pdf", sample_paper("A Good Paper on Federated Averaging")),
        ("scan.pdf", make_pdf([None])),
    )
    assert response.status_code == 201
    statuses = [result["status"] for result in response.json()["results"]]
    assert statuses == ["created", "rejected"]


def test_partially_scanned_pdf_is_accepted_with_warning(client: TestClient) -> None:
    data = make_pdf([paper_text("x", 1), None, paper_text("x", 3)], title="Partly Scanned Paper")
    document = upload(client, ("partial.pdf", data)).json()["results"][0]["document"]

    assert "Page 2 has no extractable text" in document["status_message"]
    detail = client.get(f"/api/documents/{document['id']}").json()
    assert detail["pages_without_text"] == [2]


def test_path_traversal_filename_is_neutralised(client: TestClient, app: FastAPI) -> None:
    document = upload(client, ("../../outside.pdf", sample_paper())).json()["results"][0][
        "document"
    ]
    assert document["filename"] == "outside.pdf"
    assert not (app.state.settings.data_dir.parent / "outside.pdf").exists()


def test_too_many_files_is_rejected(tmp_path: Path) -> None:
    app = make_app(make_settings(tmp_path / "data", max_files_per_upload=1))
    with TestClient(app) as client:
        response = upload(client, ("a.pdf", sample_paper()), ("b.pdf", sample_paper("Other")))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "too_many_files"


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    app = make_app(make_settings(tmp_path / "data", max_upload_size_mb=1))
    oversized = b"%PDF-1.7\n" + b"0" * (1024 * 1024)
    with TestClient(app) as client:
        result = upload(client, ("big.pdf", oversized)).json()["results"][0]
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "file_too_large"


def test_oversized_request_is_refused_before_parsing(tmp_path: Path) -> None:
    app = make_app(make_settings(tmp_path / "data", max_upload_size_mb=1, max_files_per_upload=1))
    with TestClient(app) as client:
        response = upload(client, ("huge.pdf", b"%PDF-1.7\n" + b"0" * (3 * 1024 * 1024)))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_upload_without_files_is_invalid(client: TestClient) -> None:
    response = client.post(UPLOAD_URL)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_delete_removes_document_pages_and_file(
    client: TestClient, app: FastAPI, db_session: Session
) -> None:
    document = upload(client, ("paper.pdf", sample_paper())).json()["results"][0]["document"]
    stored = app.state.settings.upload_dir / f"{document['id']}.pdf"

    response = client.delete(f"/api/documents/{document['id']}")

    assert response.status_code == 204
    assert not stored.exists()
    assert client.get(f"/api/documents/{document['id']}").status_code == 404
    assert db_session.scalars(select(DocumentPage)).all() == []
    events = db_session.scalars(select(ActivityEvent).order_by(ActivityEvent.id)).all()
    assert [event.event_type.value for event in events] == ["document_uploaded", "document_deleted"]
    assert all(event.document_id is None for event in events)


def test_deleted_document_can_be_uploaded_again(client: TestClient) -> None:
    data = sample_paper()
    first_id = upload(client, ("paper.pdf", data)).json()["results"][0]["document"]["id"]
    client.delete(f"/api/documents/{first_id}")

    result = upload(client, ("paper.pdf", data)).json()["results"][0]
    assert result["status"] == "created"
    assert result["document"]["id"] != first_id


@pytest.mark.parametrize("method", ["get", "delete"])
def test_unknown_document_returns_404(client: TestClient, method: str) -> None:
    response = getattr(client, method)("/api/documents/0123456789abcdef0123456789abcdef")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_dashboard_reflects_uploads(client: TestClient) -> None:
    upload(client, ("paper.pdf", sample_paper("Dashboard Visible Paper Title")))
    stats = client.get("/api/dashboard/stats").json()
    assert stats["total_papers"] == 1
    assert stats["recent_documents"][0]["title"] == "Dashboard Visible Paper Title"
    assert stats["recent_activity"][0]["description"] == "Uploaded “Dashboard Visible Paper Title”"


# --- Chunking (Phase 3) ------------------------------------------------------


def uploaded_document(client: TestClient, data: bytes | None = None, name: str = "paper.pdf"):
    return upload(client, (name, data or sample_paper(pages=4))).json()["results"][0]["document"]


def test_upload_splits_paper_into_chunks(client: TestClient) -> None:
    document = uploaded_document(client)
    assert document["processing_status"] == "completed"
    assert document["chunk_count"] > 0

    detail = client.get(f"/api/documents/{document['id']}").json()
    assert detail["chunking_signature"] == "v2:test-whitespace:250/50"


def test_chunks_endpoint_lists_passages_in_order_with_pages(client: TestClient) -> None:
    document = uploaded_document(client)
    body = client.get(f"/api/documents/{document['id']}/chunks").json()

    assert body["total"] == document["chunk_count"]
    indexes = [chunk["chunk_index"] for chunk in body["items"]]
    assert indexes == sorted(indexes)
    for chunk in body["items"]:
        assert 1 <= chunk["page_start"] <= chunk["page_end"] <= 4
        assert chunk["token_count"] <= 250
        # Each test page mentions its own number, which must fall in the range.
        for page in map(int, __import__("re").findall(r"This page (\d+)", chunk["text"])):
            assert chunk["page_start"] <= page <= chunk["page_end"]


def test_chunks_endpoint_paginates(client: TestClient) -> None:
    pages = [paper_text("x", number) * 8 for number in range(1, 7)]
    document = uploaded_document(client, make_pdf(pages, title="A Long Paper For Paging"))
    first = client.get(f"/api/documents/{document['id']}/chunks?limit=2").json()
    second = client.get(f"/api/documents/{document['id']}/chunks?limit=2&offset=2").json()
    assert len(first["items"]) == 2
    assert first["items"][-1]["chunk_index"] < second["items"][0]["chunk_index"]


def test_chunks_for_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/" + "0" * 32 + "/chunks").status_code == 404


def test_forced_reindex_rebuilds_chunks_and_records_activity(
    client: TestClient, db_session: Session
) -> None:
    document = uploaded_document(client)
    before = client.get(f"/api/documents/{document['id']}/chunks").json()["items"]

    response = client.post(f"/api/documents/{document['id']}/reindex?force=true")

    assert response.status_code == 200
    assert response.json()["action"] == "rechunked"
    assert response.json()["document"]["processing_status"] == "completed"
    after = client.get(f"/api/documents/{document['id']}/chunks").json()["items"]
    assert [chunk["text"] for chunk in after] == [chunk["text"] for chunk in before]
    # New rows: chunk ids are never reused (they double as vector ids).
    assert min(chunk["id"] for chunk in after) > max(chunk["id"] for chunk in before)
    events = db_session.scalars(select(ActivityEvent.event_type)).all()
    assert events[-1].value == "document_reindexed"


def test_reindex_unknown_document_is_404(client: TestClient) -> None:
    assert client.post("/api/documents/" + "0" * 32 + "/reindex").status_code == 404


def test_tokenizer_outage_keeps_paper_and_reindex_recovers(tmp_path: Path) -> None:
    class SwitchableTokenizer:
        available = False

        def __call__(self):
            if not self.available:
                unavailable_tokenizer()
            return WhitespaceTokenizer()

    provider = SwitchableTokenizer()
    app = make_app(make_settings(tmp_path / "data"), tokenizer_provider=provider)
    with TestClient(app) as client:
        result = upload(client, ("paper.pdf", sample_paper())).json()["results"][0]
        # The paper is stored; only the chunking step failed.
        assert result["status"] == "created"
        document = result["document"]
        assert document["processing_status"] == "failed"
        assert "tokenizer" in document["status_message"]
        assert document["chunk_count"] == 0

        provider.available = True
        response = client.post(f"/api/documents/{document['id']}/reindex").json()
        assert response["action"] == "rechunked"
        recovered = response["document"]
        assert recovered["processing_status"] == "completed"
        assert recovered["status_message"] is None
        assert recovered["chunk_count"] > 0


def test_partial_scan_warning_survives_reindex(client: TestClient) -> None:
    data = make_pdf([paper_text("x", 1), None, paper_text("x", 3)], title="Partly Scanned Paper")
    document = uploaded_document(client, data)
    reindexed = client.post(f"/api/documents/{document['id']}/reindex?force=true").json()
    assert "Page 2 has no extractable text" in reindexed["document"]["status_message"]


def test_delete_removes_chunks(client: TestClient, db_session: Session) -> None:
    document = uploaded_document(client)
    client.delete(f"/api/documents/{document['id']}")
    assert db_session.scalars(select(Chunk)).all() == []
