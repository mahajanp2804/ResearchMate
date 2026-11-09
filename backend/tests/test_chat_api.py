from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.prompts.answer_prompt import REFUSAL
from app.services.llm_service import create_llm_provider
from tests.conftest import make_app, make_settings
from tests.pdf_factory import make_pdf
from tests.test_hybrid_search import FEDPROX, SCAFFOLD
from tests.test_indexing import upload

QUERY_URL = "/api/chat/query"


@pytest.fixture
def papers(client: TestClient) -> dict[str, str]:
    return {
        "fedprox": upload(client, make_pdf(FEDPROX, title="Federated Optimization"), "a.pdf")["id"],
        "scaffold": upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"), "b.pdf")["id"],
    }


def ask(client: TestClient, question: str, **body):
    return client.post(QUERY_URL, json={"question": question, **body})


def test_first_question_starts_a_conversation(client: TestClient, papers) -> None:
    response = ask(client, "How does SCAFFOLD correct client drift?")

    assert response.status_code == 200
    body = response.json()
    conversation = body["conversation"]
    assert conversation["title"] == "How does SCAFFOLD correct client drift?"
    assert conversation["message_count"] == 2
    assert body["user_message"]["role"] == "user"

    answer = body["assistant_message"]
    assert answer["role"] == "assistant"
    assert "[S1]" in answer["content"]
    assert answer["details"]["outcome"] == "answered"
    assert answer["details"]["model"] == "scripted-model"
    assert answer["details"]["searched_document_count"] == 2
    sources = answer["sources"]
    assert [source["source_id"] for source in sources] == [
        f"S{i}" for i in range(1, len(sources) + 1)
    ]
    assert sources[0]["document_title"] == "SCAFFOLD Paper Title"
    assert sources[0]["text"] and sources[0]["page_start"] >= 1


def test_follow_up_uses_the_conversation(client: TestClient, app, papers) -> None:
    first = ask(client, "What is SCAFFOLD?").json()
    conversation_id = first["conversation"]["id"]

    second = ask(client, "How does it handle drift?", conversation_id=conversation_id).json()

    assert second["conversation"]["id"] == conversation_id
    assert second["conversation"]["message_count"] == 4
    (rewrite,) = app.state.test_llm.calls_of("rewrite")
    assert "User: What is SCAFFOLD?" in rewrite["turns"][0].content
    assert (
        second["assistant_message"]["details"]["standalone_question"] == "How does it handle drift?"
    )


def test_refusal_needs_no_api_key(tmp_path: Path) -> None:
    # No papers and no GEMINI_API_KEY: the refusal is produced without the LLM.
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app) as client:
        response = ask(client, "What is FedProx?")

    assert response.status_code == 200
    answer = response.json()["assistant_message"]
    assert answer["content"] == REFUSAL
    assert answer["details"]["outcome"] == "no_evidence"
    assert answer["sources"] == []


def test_missing_api_key_is_a_clear_error_and_saves_nothing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app) as client:
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"))
        response = ask(client, "What is SCAFFOLD?")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "llm_not_configured"
        assert "GEMINI_API_KEY" in response.json()["error"]["message"]
        assert client.get("/api/chat/conversations").json()["total"] == 0


def test_paper_selection_limits_sources_and_is_remembered(client: TestClient, papers) -> None:
    body = ask(client, "client drift control variates", document_ids=[papers["fedprox"]]).json()
    assert {source["document_id"] for source in body["assistant_message"]["sources"]} == {
        papers["fedprox"]
    }
    assert body["conversation"]["document_ids"] == [papers["fedprox"]]


@pytest.mark.parametrize(
    ("body", "status_code", "code"),
    [
        ({"question": "  "}, 422, "empty_query"),
        ({"question": "drift", "conversation_id": "0" * 32}, 404, "not_found"),
        ({"question": "drift", "document_ids": ["0" * 32]}, 422, "unknown_documents"),
        ({"question": "drift", "document_ids": []}, 422, "empty_document_filter"),
    ],
)
def test_invalid_requests(client: TestClient, papers, body, status_code, code) -> None:
    response = client.post(QUERY_URL, json=body)
    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code


def test_long_questions_get_a_shortened_title(client: TestClient, papers) -> None:
    question = "How does SCAFFOLD " + "really " * 30 + "correct drift?"
    title = ask(client, question).json()["conversation"]["title"]
    assert len(title) <= 81 and title.endswith("…")


def test_conversations_are_listed_newest_first_and_can_be_read(client: TestClient, papers) -> None:
    older = ask(client, "What is FedProx?").json()["conversation"]["id"]
    newer = ask(client, "What is SCAFFOLD?").json()["conversation"]["id"]
    ask(client, "And its limitations?", conversation_id=older)  # older becomes most recent

    listing = client.get("/api/chat/conversations").json()
    assert listing["total"] == 2
    assert [item["id"] for item in listing["items"]] == [older, newer]
    assert listing["items"][0]["message_count"] == 4

    detail = client.get(f"/api/chat/conversations/{older}").json()
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"] * 2
    assert detail["messages"][1]["sources"]


def test_delete_conversation(client: TestClient, papers) -> None:
    conversation_id = ask(client, "What is FedProx?").json()["conversation"]["id"]
    assert client.delete(f"/api/chat/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/chat/conversations/{conversation_id}").status_code == 404
    assert client.delete(f"/api/chat/conversations/{conversation_id}").status_code == 404


def test_questions_appear_on_the_dashboard(client: TestClient, papers) -> None:
    ask(client, "What is FedProx?")
    stats = client.get("/api/dashboard/stats").json()
    assert stats["total_questions"] == 1
    assert stats["total_conversations"] == 1
    assert stats["recent_activity"][0]["description"] == "Asked “What is FedProx?”"


def test_sources_survive_deleting_the_paper(client: TestClient, papers) -> None:
    conversation_id = ask(client, "How does SCAFFOLD correct client drift?").json()["conversation"][
        "id"
    ]
    client.delete(f"/api/documents/{papers['scaffold']}")

    answer = client.get(f"/api/chat/conversations/{conversation_id}").json()["messages"][1]
    assert answer["sources"][0]["document_title"] == "SCAFFOLD Paper Title"
    assert "control variates" in answer["sources"][0]["text"]
