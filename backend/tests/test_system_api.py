from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import (
    ActivityEvent,
    ActivityType,
    Conversation,
    IndexingStatus,
    Message,
    MessageRole,
)
from tests.conftest import make_app, make_settings
from tests.test_models import make_chunk, make_document


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["llm_configured"] is False


def test_lifespan_creates_data_directories(client: TestClient, app: FastAPI) -> None:
    settings = app.state.settings
    assert settings.upload_dir.is_dir()
    assert settings.index_dir.is_dir()


def test_system_info_never_exposes_secrets(tmp_path) -> None:
    settings = make_settings(tmp_path, gemini_api_key="top-secret-value")
    with TestClient(make_app(settings)) as client:
        response = client.get("/api/system/info")

    assert response.status_code == 200
    assert "top-secret-value" not in response.text
    assert str(tmp_path) not in response.text
    body = response.json()
    assert body["llm"]["configured"] is True
    assert body["chunking"] == {"chunk_size_tokens": 250, "chunk_overlap_tokens": 50}
    assert body["embedding_model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert body["retrieval"]["min_rerank_score"] is None  # evidence gate off by default


def test_dashboard_stats_empty(client: TestClient) -> None:
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    assert response.json() == {
        "total_papers": 0,
        "indexed_papers": 0,
        "total_chunks": 0,
        "total_questions": 0,
        "total_conversations": 0,
        "recent_documents": [],
        "recent_activity": [],
    }


def test_dashboard_stats_counts_real_rows(client: TestClient, db_session: Session) -> None:
    now = datetime.now(UTC)
    older = make_document("a" * 64, title="Older", created_at=now - timedelta(days=1))
    newer = make_document(
        "b" * 64, title="Newer", created_at=now, indexing_status=IndexingStatus.INDEXED
    )
    newer.chunks = [make_chunk(0), make_chunk(1)]
    conversation = Conversation(title="Questions")
    conversation.messages = [
        Message(role=MessageRole.USER, content="Q1"),
        Message(role=MessageRole.ASSISTANT, content="A1"),
        Message(role=MessageRole.USER, content="Q2"),
    ]
    db_session.add_all([older, newer, conversation])
    db_session.flush()
    db_session.add(
        ActivityEvent(
            event_type=ActivityType.DOCUMENT_UPLOADED,
            description="Uploaded Newer",
            document_id=newer.id,
        )
    )
    db_session.commit()

    body = client.get("/api/dashboard/stats").json()
    assert body["total_papers"] == 2
    assert body["indexed_papers"] == 1
    assert body["total_chunks"] == 2
    assert body["total_questions"] == 2
    assert body["total_conversations"] == 1
    assert [doc["title"] for doc in body["recent_documents"]] == ["Newer", "Older"]
    assert body["recent_activity"][0]["event_type"] == "document_uploaded"
    # Timestamps carry an explicit UTC offset so browsers don't misread them.
    assert body["recent_documents"][0]["created_at"].endswith("Z")


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"


def test_request_id_header_is_returned(client: TestClient) -> None:
    response = client.get("/api/health", headers={"X-Request-ID": "trace-123"})
    assert response.headers["X-Request-ID"] == "trace-123"
