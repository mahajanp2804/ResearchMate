from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.main import create_app
from app.models import ActivityEvent, DocumentPage
from tests.conftest import make_settings
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
    assert document["indexing_status"] == "not_indexed"
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
    app = create_app(make_settings(tmp_path / "data", max_files_per_upload=1))
    with TestClient(app) as client:
        response = upload(client, ("a.pdf", sample_paper()), ("b.pdf", sample_paper("Other")))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "too_many_files"


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "data", max_upload_size_mb=1))
    oversized = b"%PDF-1.7\n" + b"0" * (1024 * 1024)
    with TestClient(app) as client:
        result = upload(client, ("big.pdf", oversized)).json()["results"][0]
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "file_too_large"


def test_oversized_request_is_refused_before_parsing(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "data", max_upload_size_mb=1, max_files_per_upload=1))
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
