"""The answer pipeline with a scripted LLM: refusals, generation and follow-ups."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions import ResearchMateError, UpstreamServiceError
from app.models import Document, IndexingStatus
from app.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT, REFUSAL
from app.services.llm_service import ChatTurn, create_llm_provider
from tests.conftest import make_app, make_settings
from tests.fakes import ScriptedLLM
from tests.pdf_factory import make_pdf
from tests.test_hybrid_search import FEDPROX, SCAFFOLD
from tests.test_indexing import upload


@contextmanager
def library(tmp_path: Path, llm: ScriptedLLM | None = None, **overrides) -> Iterator[FastAPI]:
    settings = make_settings(
        tmp_path / "data", chunk_size_tokens=32, chunk_overlap_tokens=0, **overrides
    )
    app = make_app(settings, llm=llm)
    with TestClient(app) as client:
        upload(client, make_pdf(FEDPROX, title="Federated Optimization"), "a.pdf")
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"), "b.pdf")
        yield app


def ask(app: FastAPI, question: str, history=(), document_ids=None):
    with app.state.database.session() as db:
        scope = app.state.retrieval.resolve_scope(db, document_ids)
        return app.state.rag.answer(db, question, scope, history)


# --- Refusing without the LLM (the Phase 7 acceptance test) --------------------


def test_no_searchable_papers_refuses_without_calling_the_llm(tmp_path: Path) -> None:
    with library(tmp_path) as app:
        with app.state.database.session() as db:
            for document in db.query(Document):
                document.indexing_status = IndexingStatus.NOT_INDEXED
            db.commit()
        history = [ChatTurn("user", "What is FedProx?"), ChatTurn("assistant", "A method [S1].")]

        result = ask(app, "What does it assume?", history)

        assert result.answer == REFUSAL
        assert result.outcome == "no_evidence"
        assert result.sources == []
        assert app.state.test_llm.calls == []  # not even the follow-up rewrite


def test_evidence_gate_refuses_without_calling_the_llm(tmp_path: Path) -> None:
    # The stand-in reranker scores passages sharing no query word at -11.
    with library(tmp_path, min_rerank_score=-10.0) as app:
        result = ask(app, "sourdough baking recipe")
        assert (result.answer, result.outcome) == (REFUSAL, "no_evidence")
        assert app.state.test_llm.calls_of("answer") == []


# --- Answering ---------------------------------------------------------------------


def test_answers_from_the_retrieved_sources(tmp_path: Path) -> None:
    with library(tmp_path) as app:
        result = ask(app, "How does SCAFFOLD correct client drift?")

        assert result.outcome == "answered"
        assert "control variates" in result.answer and "[S1]" in result.answer
        assert result.model == "scripted-model"
        assert result.standalone_question == result.question  # no history, no rewrite
        assert set(result.timings_ms) == {"retrieval", "generation", "total"}

        (call,) = app.state.test_llm.calls_of("answer")
        assert call["system"] == ANSWER_SYSTEM_PROMPT
        prompt = call["turns"][0].content
        # The sources in the prompt are exactly the returned sources, in order.
        for number, source in enumerate(result.sources, start=1):
            assert f'<source id="S{number}" paper="{source.document_title}"' in prompt
        assert result.context_tokens == sum(source.token_count for source in result.sources)


def test_answers_stay_within_the_selected_papers(tmp_path: Path) -> None:
    with library(tmp_path) as app:
        with app.state.database.session() as db:
            fedprox = db.query(Document).filter_by(title="Federated Optimization").one().id
        result = ask(app, "client drift control variates", document_ids=[fedprox])
        assert {source.document_id for source in result.sources} == {fedprox}


def test_model_refusal_is_normalised_to_the_exact_sentence(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        answer="  i couldn’t find sufficient supporting information in the "
        "uploaded research papers to answer this question  "
    )
    with library(tmp_path, llm) as app:
        result = ask(app, "What GPU was used?")
        assert result.answer == REFUSAL
        assert result.outcome == "model_declined"
        assert result.sources  # the passages it was shown are still reported


def test_missing_api_key_is_reported_when_an_answer_is_needed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app) as client:
        upload(client, make_pdf(SCAFFOLD, title="SCAFFOLD Paper Title"))
        with pytest.raises(ResearchMateError) as error:
            ask(app, "What is SCAFFOLD?")
    assert error.value.code == "llm_not_configured"


# --- Follow-up questions -----------------------------------------------------


def test_follow_up_is_rewritten_before_retrieval(tmp_path: Path) -> None:
    llm = ScriptedLLM(rewrite="How does SCAFFOLD correct client drift?")
    history = [
        ChatTurn("user", "What is SCAFFOLD?"),
        ChatTurn("assistant", "SCAFFOLD is a federated optimisation method [S1]."),
    ]
    with library(tmp_path, llm) as app:
        result = ask(app, "How does it correct drift?", history)

        assert result.standalone_question == "How does SCAFFOLD correct client drift?"
        assert "control variates" in result.sources[0].text
        assert "rewrite" in result.timings_ms
        (rewrite,) = llm.calls_of("rewrite")
        assert rewrite["temperature"] == 0.0
        assert "User: What is SCAFFOLD?" in rewrite["turns"][0].content
        # The answer call carries both phrasings but not the history itself.
        answer_prompt = llm.calls_of("answer")[0]["turns"][0].content
        assert "Question: How does it correct drift?" in answer_prompt
        assert "this means: How does SCAFFOLD correct client drift?" in answer_prompt
        assert "What is SCAFFOLD?" not in answer_prompt


def test_only_recent_history_is_used(tmp_path: Path) -> None:
    history = [ChatTurn("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(10)]
    with library(tmp_path, llm_history_messages=4) as app:
        ask(app, "And SCAFFOLD?", history)
        prompt = app.state.test_llm.calls_of("rewrite")[0]["turns"][0].content
        assert "message 5" not in prompt and "message 6" in prompt and "message 9" in prompt


def test_history_is_ignored_when_disabled(tmp_path: Path) -> None:
    history = [ChatTurn("user", "What is SCAFFOLD?"), ChatTurn("assistant", "A method.")]
    with library(tmp_path, llm_history_messages=0) as app:
        result = ask(app, "client drift", history)
        assert app.state.test_llm.calls_of("rewrite") == []
        assert result.standalone_question == "client drift"


def test_unusable_rewrite_falls_back_to_the_question(tmp_path: Path) -> None:
    def truncated(_prompt: str) -> str:
        raise UpstreamServiceError("cut off", code="llm_truncated")

    history = [ChatTurn("user", "What is SCAFFOLD?"), ChatTurn("assistant", "A method.")]
    with library(tmp_path, ScriptedLLM(rewrite=truncated)) as app:
        result = ask(app, "client drift control variates", history)
        assert result.standalone_question == "client drift control variates"
        assert result.outcome == "answered"


def test_serious_rewrite_errors_are_not_hidden(tmp_path: Path) -> None:
    def quota(_prompt: str) -> str:
        raise UpstreamServiceError("quota", code="llm_quota_exceeded")

    history = [ChatTurn("user", "What is SCAFFOLD?"), ChatTurn("assistant", "A method.")]
    with (
        library(tmp_path, ScriptedLLM(rewrite=quota)) as app,
        pytest.raises(UpstreamServiceError),
    ):
        ask(app, "client drift", history)
