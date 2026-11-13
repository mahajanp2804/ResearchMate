"""Answer quality: the full pipeline's answers, judged by the LLM.

For each question of a run:
1. The question-answering pipeline answers it from the dataset's papers,
   exactly as the Research Assistant would (nothing is saved to chat).
2. Refusals are checked in code: an unanswerable question should be
   refused and an answerable one answered.
3. Each answer that was given is judged (see ``judge_prompt``):
   - faithfulness: share of its statements the passages support;
   - answer relevance: how directly it addresses the question (1-5 -> 0-1);
   - context relevance: share of the passages it was given that help;
   - correctness: agreement with the reference answer (1-5 -> 0-1).

The judge is a language model (by default the same one that answered), so
these scores are indicative; they are stored and shown apart from the
retrieval metrics, which need no model.
"""

from statistics import fmean
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import ResearchMateError
from app.models import EvaluationRun
from app.prompts.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_turns,
    judge_json_schema,
    parse_judgement,
)
from app.services.citation_service import answer_statements
from app.services.rag_pipeline import LLMProviderFactory, RagAnswer, RagPipeline
from app.services.retrieval_service import SearchScope
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _scale(score: int | None) -> float | None:
    """A 1-5 rating as 0-1."""
    return None if score is None else round((score - 1) / 4, 4)


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(fmean(present), 4) if present else None


def models_used(entries: list[dict[str, Any]]) -> dict[str, str | None]:
    """The models that actually answered and judged (as the provider reported them)."""
    return {
        "answer": next((e["model"] for e in entries if e.get("model")), None),
        "judge": next((e["judge_model"] for e in entries if e.get("judge_model")), None),
    }


def summarise_answers(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean scores over the questions evaluated so far."""
    answerable = [e for e in entries if e["category"] != "unanswerable"]
    unanswerable = [e for e in entries if e["category"] == "unanswerable"]
    answered = [e for e in entries if not e["refused"]]
    return {
        "questions": len(entries),
        "answered": len(answered),
        "refused": len(entries) - len(answered),
        # Refused exactly the unanswerable questions.
        "refusal_accuracy": _mean([1.0 if e["refusal_correct"] else 0.0 for e in entries]),
        "answerable_answered": _mean([0.0 if e["refused"] else 1.0 for e in answerable]),
        "unanswerable_refused": _mean([1.0 if e["refused"] else 0.0 for e in unanswerable]),
        "faithfulness": _mean([e.get("faithfulness") for e in answered]),
        "answer_relevance": _mean([e.get("answer_relevance") for e in answered]),
        "context_relevance": _mean([e.get("context_relevance") for e in answered]),
        "correctness": _mean([e.get("correctness") for e in answered]),
    }


class AnswerEvaluator:
    def __init__(
        self, *, settings: Settings, rag: RagPipeline, llm_provider: LLMProviderFactory
    ) -> None:
        self._settings = settings
        self._rag = rag
        self._llm_provider = llm_provider

    def check_ready(self) -> None:
        """Raise the provider's own error (e.g. a missing API key) before a run starts."""
        self._llm_provider()

    def run(self, db: Session, run: EvaluationRun, scope: SearchScope) -> None:
        questions = list(run.dataset.questions)
        results = [dict(item) for item in run.per_question_results]
        entries: list[dict[str, Any]] = []
        try:
            for index, question in enumerate(questions):
                entry = self._evaluate(
                    db, question.question, question.category, question.reference_answer, scope
                )
                entries.append(entry)
                results[index] = {**results[index], "answer": entry}
                run.per_question_results = list(results)
                run.metrics = {
                    **run.metrics,
                    "answers": {
                        "summary": summarise_answers(entries),
                        "models": models_used(entries),
                    },
                    "progress": {"phase": "answers", "done": index + 1, "total": len(questions)},
                }
                db.commit()
        except ResearchMateError as error:
            # Keep what was judged; the run is marked failed with this message.
            run.metrics = {
                **run.metrics,
                "answers": {
                    "summary": summarise_answers(entries),
                    "models": models_used(entries),
                    "stopped_after": len(entries),
                },
            }
            db.commit()
            error.message = (
                f"Answer evaluation stopped after {len(entries)} of {len(questions)} "
                f"questions: {error.message}"
            )
            raise

    def _evaluate(
        self,
        db: Session,
        question: str,
        category: str | None,
        reference_answer: str | None,
        scope: SearchScope,
    ) -> dict[str, Any]:
        answer = self._rag.answer(db, question, scope)
        refused = answer.outcome != "answered"
        entry: dict[str, Any] = {
            "category": category,
            "outcome": answer.outcome,
            "refused": refused,
            "refusal_correct": refused == (category == "unanswerable"),
            "answer": answer.answer,
            "model": answer.model,
            "citation_status": answer.coverage.status if answer.coverage else None,
            "sources": [
                {
                    "source_id": f"S{number}",
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                }
                for number, chunk in enumerate(answer.sources, start=1)
            ],
        }
        if not refused:
            entry.update(self._judge(question, answer, reference_answer))
        return entry

    def _judge(
        self, question: str, answer: RagAnswer, reference_answer: str | None
    ) -> dict[str, Any]:
        statements = answer_statements(answer.answer)
        passages = [(f"S{n}", chunk.text) for n, chunk in enumerate(answer.sources, start=1)]
        response = self._llm_provider().generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            turns=build_judge_turns(question, passages, statements, reference_answer),
            temperature=0.0,
            json_schema=judge_json_schema(with_correctness=reference_answer is not None),
        )
        try:
            judgement = parse_judgement(response.text, len(statements))
        except ValueError as error:
            logger.warning("Unreadable judgement for %r: %s", question, error)
            return {"judge_error": "The judge's reply could not be read."}

        judged = [flag for flag in judgement.supported if flag is not None]
        relevant = [judgement.relevant[pid] for pid, _ in passages if pid in judgement.relevant]
        return {
            "judge_model": response.model,
            "statements": [
                {"text": text, "supported": flag}
                for text, flag in zip(statements, judgement.supported, strict=True)
            ],
            "faithfulness": round(sum(judged) / len(judged), 4) if judged else None,
            "context_relevance": round(sum(relevant) / len(relevant), 4) if relevant else None,
            "answer_relevance": _scale(judgement.answer_relevance),
            "correctness": _scale(judgement.correctness) if reference_answer else None,
        }
