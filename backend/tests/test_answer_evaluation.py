"""LLM-judged answer quality: the judge prompt and full runs with a scripted LLM."""

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions import RateLimitedError, ResearchMateError
from app.models import EvaluationRun, EvaluationRunStatus
from app.prompts.answer_prompt import REFUSAL
from app.prompts.judge_prompt import build_judge_turns, judge_json_schema, parse_judgement
from app.services.evaluation_data import load_dataset_spec
from app.services.llm_service import create_llm_provider
from tests.conftest import make_app, make_settings
from tests.test_evaluation_service import install

SPEC = load_dataset_spec("sample_papers")
UNANSWERABLE_WORDS = ("GPU", "BLEU", "trainable parameters", "differential privacy")


# --- The judge prompt ------------------------------------------------------------


def test_judge_prompt_numbers_statements_and_escapes_text() -> None:
    (turn,) = build_judge_turns(
        "What is <b>X</b>?",
        [("S1", "Passage with <tags> & text.")],
        ["First claim.", "Second claim."],
        None,
    )
    assert "Question: What is &lt;b&gt;X&lt;/b&gt;?" in turn.content
    assert "Reference answer: (none)" in turn.content
    assert '<passage id="S1">\nPassage with &lt;tags&gt; &amp; text.\n</passage>' in turn.content
    assert "Answer statements:\n1. First claim.\n2. Second claim." in turn.content


def test_correctness_is_only_asked_for_with_a_reference_answer() -> None:
    assert "correctness" in judge_json_schema(True)["required"]
    assert "correctness" not in judge_json_schema(False)["properties"]


def test_judgement_parsing_ignores_what_it_cannot_use() -> None:
    reply = json.dumps(
        {
            "statements": [
                {"index": 2, "supported": False},
                {"index": 9, "supported": True},  # no such statement
                {"index": 1, "supported": "yes"},  # not a boolean
            ],
            "passages": [{"id": "S1", "relevant": True}, {"id": "S2"}],
            "answer_relevance": 7,  # out of range
            "correctness": 4,
        }
    )
    judgement = parse_judgement(reply, statement_count=2)
    assert judgement.supported == [None, False]
    assert judgement.relevant == {"S1": True}
    assert (judgement.answer_relevance, judgement.correctness) == (None, 4)
    with pytest.raises(ValueError):
        parse_judgement("Looks good to me.", 1)


# --- Runs ----------------------------------------------------------------------------


def answer_or_refuse(prompt: str) -> str | None:
    question = prompt.rsplit("Question: ", 1)[1]
    if any(word in question for word in UNANSWERABLE_WORDS):
        return REFUSAL
    return None  # the scripted default: first sentence of S1, cited


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FastAPI]:
    app = make_app(make_settings(tmp_path_factory.mktemp("answers") / "data"))
    with TestClient(app):
        install(app)
        yield app


@pytest.fixture
def llm(library: FastAPI):
    llm = library.state.test_llm
    llm.answer, llm.judge = answer_or_refuse, None
    llm.calls.clear()
    return llm


def run_with_answers(app: FastAPI) -> EvaluationRun:
    service = app.state.evaluation
    with app.state.database.session() as db:
        run_id = service.start_run(db, SPEC, methods=["bm25"], answer_quality=True).id
    service.execute(run_id)
    with app.state.database.session() as db:
        run = db.get(EvaluationRun, run_id)
        db.expunge(run)
        return run


def test_answers_are_generated_checked_and_judged(library: FastAPI, llm) -> None:
    run = run_with_answers(library)
    assert run.status is EvaluationRunStatus.COMPLETED, run.error_message

    summary = run.metrics["answers"]["summary"]
    assert (summary["questions"], summary["answered"], summary["refused"]) == (35, 31, 4)
    assert summary["refusal_accuracy"] == 1.0
    assert summary["unanswerable_refused"] == 1.0 and summary["answerable_answered"] == 1.0
    for metric in ("faithfulness", "answer_relevance", "context_relevance", "correctness"):
        assert summary[metric] == 1.0, metric
    # Retrieval metrics are still there, separately.
    assert run.metrics["retrieval"]["bm25"]["questions"] == 31

    judge_calls = llm.calls_of("judge")
    assert len(judge_calls) == 31  # refusals are not judged
    assert all(call["temperature"] == 0.0 for call in judge_calls)
    results = {item["key"]: item for item in run.per_question_results}
    refused = results["q32"]["answer"]
    assert refused["refused"] and refused["refusal_correct"] and "faithfulness" not in refused
    answered = results["q01"]["answer"]
    assert answered["statements"] and answered["sources"][0]["source_id"] == "S1"


def test_scores_follow_the_judgement(library: FastAPI, llm) -> None:
    def harsh(prompt: str) -> dict:
        passages = prompt.count("<passage id=")
        return {
            "statements": [{"index": 1, "supported": False}],
            "passages": [{"id": f"S{n}", "relevant": n == 1} for n in range(1, passages + 1)],
            "answer_relevance": 3,
            "correctness": 1,
        }

    llm.judge = harsh
    run = run_with_answers(library)
    answer = {item["key"]: item for item in run.per_question_results}["q01"]["answer"]
    assert answer["faithfulness"] == 0.0
    assert answer["answer_relevance"] == 0.5
    assert answer["correctness"] == 0.0
    assert answer["context_relevance"] == pytest.approx(1 / len(answer["sources"]), abs=1e-4)


def test_a_wrong_refusal_is_counted(library: FastAPI, llm) -> None:
    llm.answer = None  # answers everything, including the unanswerable questions
    summary = run_with_answers(library).metrics["answers"]["summary"]
    assert summary["unanswerable_refused"] == 0.0
    assert summary["refusal_accuracy"] == pytest.approx(31 / 35, abs=1e-4)


def test_a_quota_error_keeps_the_answers_judged_so_far(library: FastAPI, llm) -> None:
    def limited(_prompt: str) -> None:
        if len(llm.calls_of("judge")) > 3:
            raise RateLimitedError("Gemini's rate limit was reached.", code="llm_quota_exceeded")

    llm.judge = limited
    run = run_with_answers(library)
    assert run.status is EvaluationRunStatus.FAILED
    assert run.error_message.startswith("Answer evaluation stopped after 3 of 35 questions")
    assert run.metrics["answers"]["stopped_after"] == 3
    assert run.metrics["answers"]["summary"]["questions"] == 3
    assert run.metrics["retrieval"]["bm25"]["questions"] == 31


def test_answer_quality_needs_an_api_key(tmp_path) -> None:
    settings = make_settings(tmp_path / "data")
    app = make_app(settings, llm_provider=lambda: create_llm_provider(settings))
    with TestClient(app):
        install(app)
        with app.state.database.session() as db, pytest.raises(ResearchMateError) as error:
            app.state.evaluation.start_run(db, SPEC, answer_quality=True)
    assert error.value.code == "llm_not_configured"
