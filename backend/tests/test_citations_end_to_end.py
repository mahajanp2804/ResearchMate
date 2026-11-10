"""Citation validation through the answer pipeline and the chat API."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.prompts.answer_prompt import REFUSAL
from tests.conftest import make_app, make_settings
from tests.fakes import ScriptedLLM
from tests.pdf_factory import make_pdf
from tests.test_indexing import upload

SCAFFOLD = [
    "SCAFFOLD uses control variates to correct client drift in local updates.",
    "Experiments on EMNIST show SCAFFOLD needs fewer communication rounds than FedAvg.",
]
# One passage spanning two pages: the second sentence starts on page 1 and
# ends on page 2, and the third is entirely on page 2.
CROSS_PAGE = [
    "Clients train locally on private data. The server then aggregates the",
    "updates from every client in each round. Momentum correction reduces the variance "
    "of the aggregated update.",
]


@contextmanager
def library(tmp_path: Path, answer: str) -> Iterator[tuple[FastAPI, TestClient]]:
    llm = ScriptedLLM(answer=answer)
    app = make_app(make_settings(tmp_path / "data"), llm=llm)
    with TestClient(app) as client:
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"), "a.pdf")
        upload(client, make_pdf(CROSS_PAGE, title="Variance Paper"), "b.pdf")
        yield app, client


def ask(client: TestClient, question: str) -> dict:
    response = client.post("/api/chat/query", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()["assistant_message"]


def cited(message: dict) -> dict[str, dict]:
    return {source["source_id"]: source for source in message["sources"] if source["cited"]}


def source_of(message: dict, title: str) -> str:
    return next(s["source_id"] for s in message["sources"] if s["document_title"] == title)


def test_valid_citation_gets_excerpt_page_and_support(tmp_path: Path) -> None:
    # S1 is the best-ranked passage; the scripted answer cites it by default.
    with library(tmp_path, answer=None) as (_, client):
        message = ask(client, "How does SCAFFOLD correct client drift with control variates?")

    sources = cited(message)
    assert list(sources) == ["S1"]
    s1 = sources["S1"]
    assert "control variates" in s1["excerpt"]
    assert s1["excerpt_page"] == s1["page_start"]
    assert s1["support"] > 0.5
    assert message["details"]["citation_status"] == "cited"
    assert message["details"]["invalid_citation_ids"] == []
    assert all(not s["cited"] and s["excerpt"] is None for s in message["sources"][1:])


def test_invented_citation_is_removed_from_the_saved_answer(tmp_path: Path) -> None:
    answer = "SCAFFOLD corrects client drift using control variates [S1][S9]."
    with library(tmp_path, answer) as (_, client):
        message = ask(client, "How does SCAFFOLD correct client drift?")
        conversation = client.get("/api/chat/conversations").json()["items"][0]["id"]
        saved = client.get(f"/api/chat/conversations/{conversation}").json()["messages"][1]

    assert message["content"] == "SCAFFOLD corrects client drift using control variates [S1]."
    assert message["details"]["invalid_citation_ids"] == ["S9"]
    assert saved["content"] == message["content"]
    assert "[S9]" not in saved["content"]


def test_answer_without_citations_is_flagged(tmp_path: Path) -> None:
    answer = "SCAFFOLD is the most widely deployed algorithm in production federated systems."
    with library(tmp_path, answer) as (_, client):
        message = ask(client, "How does SCAFFOLD correct client drift?")

    assert message["details"]["citation_status"] == "uncited"
    assert message["details"]["uncited_statements"] == [answer]
    assert cited(message) == {}


def test_partially_cited_answer_lists_unsupported_claims(tmp_path: Path) -> None:
    answer = (
        "SCAFFOLD corrects client drift using control variates [S1]. "
        "It has replaced FedAvg in every commercial deployment."
    )
    with library(tmp_path, answer) as (_, client):
        message = ask(client, "How does SCAFFOLD correct client drift?")
    assert message["details"]["citation_status"] == "partially_cited"
    assert message["details"]["uncited_statements"] == [
        "It has replaced FedAvg in every commercial deployment."
    ]


def test_citation_into_a_two_page_passage_reports_the_exact_page(tmp_path: Path) -> None:
    question = "momentum correction variance aggregated update"
    with library(tmp_path, answer=None) as (app, client):
        variance = source_of(ask(client, question), "Variance Paper")
        claims = {
            2: "Momentum correction reduces the variance of the aggregated update",
            1: "Clients train locally on private data",
        }
        for page, claim in claims.items():
            app.state.test_llm.answer = f"{claim} [{variance}]."
            source = cited(ask(client, question))[variance]

            assert (source["page_start"], source["page_end"]) == (1, 2)
            assert source["excerpt"].startswith(claim)  # title "Variance Paper" stripped
            assert source["excerpt_page"] == page


def test_refusal_with_a_stray_citation_is_still_a_refusal(tmp_path: Path) -> None:
    with library(tmp_path, answer=f"{REFUSAL} [S1]") as (_, client):
        message = ask(client, "What GPU was used?")
    assert message["content"] == REFUSAL
    assert message["details"]["outcome"] == "model_declined"
    assert message["details"]["citation_status"] is None
    assert cited(message) == {}


def test_messages_saved_before_citation_checks_still_load(tmp_path: Path) -> None:
    # A message stored without the Phase 8 fields reads back with defaults.
    from app.models import Conversation, Message, MessageRole

    with library(tmp_path, answer=None) as (app, client):
        with app.state.database.session() as db:
            conversation = Conversation(title="Older conversation")
            conversation.messages = [
                Message(role=MessageRole.USER, content="Old question?"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Old answer [S1].",
                    citations=[
                        {
                            "source_id": "S1",
                            "chunk_id": 1,
                            "document_id": "d",
                            "document_title": "Paper",
                            "page_start": 1,
                            "page_end": 1,
                            "section": None,
                            "text": "Old passage.",
                            "score": 1.0,
                        }
                    ],
                    retrieval_info={
                        "outcome": "answered",
                        "standalone_question": "Old question?",
                        "model": "m",
                        "context_tokens": 3,
                        "truncated": False,
                        "timings_ms": {},
                        "searched_document_count": 1,
                        "skipped_document_ids": [],
                    },
                ),
            ]
            db.add(conversation)
            db.commit()
            conversation_id = conversation.id

        old = client.get(f"/api/chat/conversations/{conversation_id}").json()["messages"][1]
    assert old["sources"][0]["cited"] is False
    assert old["details"]["citation_status"] is None
