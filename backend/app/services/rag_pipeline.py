"""Question answering: rewrite the question, retrieve evidence, generate an answer.

1. Query processing: normalise the question; for a follow-up, rewrite it into
   a standalone question using recent conversation turns.
2. Retrieval: hybrid search, reranking and context selection (Phase 6).
3. Evidence check: with no usable passages the fixed refusal is returned and
   the LLM is never called, so this path cannot hallucinate.
4. Generation: the LLM answers from the numbered sources only.
5. Citation validation: invented [S#] markers are removed, each cited source
   gets a supporting excerpt and exact page, and claims without citations
   are flagged.
6. The model may still decline; its refusal is normalised to the exact
   required sentence.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import UpstreamServiceError
from app.prompts.answer_prompt import (
    ANSWER_SYSTEM_PROMPT,
    REFUSAL,
    build_answer_turns,
    is_refusal,
)
from app.prompts.rewrite_prompt import REWRITE_SYSTEM_PROMPT, build_rewrite_turns, clean_rewrite
from app.services.citation_service import (
    Citation,
    Coverage,
    SourcePassage,
    build_citations,
    citation_coverage,
    validate_citations,
)
from app.services.llm_service import ChatTurn, LLMProvider
from app.services.retrieval_service import (
    RetrievalService,
    RetrievedChunk,
    SearchScope,
    normalize_query,
    page_texts_for,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

LLMProviderFactory = Callable[[], LLMProvider]

# no_evidence: refused by the pipeline without calling the LLM.
# model_declined: the LLM read the sources and judged them insufficient.
AnswerOutcome = Literal["answered", "no_evidence", "model_declined"]

# A failed rewrite is not worth failing the question for: these errors fall
# back to searching with the question as asked.
_REWRITE_FALLBACK_CODES = {"llm_truncated", "llm_empty", "llm_blocked"}


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    outcome: AnswerOutcome
    question: str
    standalone_question: str
    # The passages given to the LLM, in order, as sources S1..Sn.
    sources: list[RetrievedChunk]
    context_tokens: int
    model: str | None
    truncated: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    # One entry per source the answer cites, in order of first citation.
    citations: list[Citation] = field(default_factory=list)
    # Ids the model cited that were not among the sources (removed from the answer).
    invalid_citation_ids: list[str] = field(default_factory=list)
    # None for refusals.
    coverage: Coverage | None = None

    @property
    def refused(self) -> bool:
        return self.outcome != "answered"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


class RagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        retrieval: RetrievalService,
        llm_provider: LLMProviderFactory,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._llm_provider = llm_provider

    def answer(
        self,
        db: Session,
        question: str,
        scope: SearchScope,
        history: Sequence[ChatTurn] = (),
    ) -> RagAnswer:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        question = normalize_query(question)

        def refuse(standalone: str, outcome: AnswerOutcome) -> RagAnswer:
            timings["total"] = _elapsed_ms(started)
            return RagAnswer(
                answer=REFUSAL,
                outcome=outcome,
                question=question,
                standalone_question=standalone,
                sources=[],
                context_tokens=0,
                model=None,
                truncated=False,
                timings_ms=timings,
            )

        # Nothing to search: refuse before any model call, even for follow-ups.
        if not scope.effective_document_ids:
            return refuse(question, "no_evidence")

        recent = list(history)[-self._settings.llm_history_messages :] if history else []
        standalone = question
        if recent and self._settings.llm_history_messages:
            step = time.perf_counter()
            standalone = self._rewrite(question, recent)
            timings["rewrite"] = _elapsed_ms(step)

        step = time.perf_counter()
        context = self._retrieval.retrieve_context(db, standalone, scope)
        timings["retrieval"] = _elapsed_ms(step)
        if not context.has_evidence:
            return refuse(standalone, "no_evidence")

        step = time.perf_counter()
        response = self._llm_provider().generate(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            turns=build_answer_turns(question, standalone, context.chunks),
        )
        timings["generation"] = _elapsed_ms(step)

        sources = list(context.chunks)
        check = validate_citations(response.text, [f"S{n}" for n in range(1, len(sources) + 1)])
        declined = is_refusal(check.text)
        citations: list[Citation] = []
        coverage: Coverage | None = None
        if not declined:
            passages = [
                SourcePassage(
                    f"S{n}",
                    chunk.text,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.document_id,
                    chunk.document_title,
                )
                for n, chunk in enumerate(sources, start=1)
            ]
            citations = build_citations(check.text, passages, page_texts_for(db, sources))
            coverage = citation_coverage(check.text)
            if check.invalid_ids:
                logger.info("Removed citations to unknown sources: %s", check.invalid_ids)
        timings["total"] = _elapsed_ms(started)

        return RagAnswer(
            answer=REFUSAL if declined else check.text,
            outcome="model_declined" if declined else "answered",
            question=question,
            standalone_question=standalone,
            sources=sources,
            context_tokens=context.total_tokens,
            model=response.model,
            truncated=response.truncated,
            timings_ms=timings,
            citations=citations,
            invalid_citation_ids=check.invalid_ids,
            coverage=coverage,
        )

    def _rewrite(self, question: str, history: Sequence[ChatTurn]) -> str:
        try:
            response = self._llm_provider().generate(
                system_prompt=REWRITE_SYSTEM_PROMPT,
                turns=build_rewrite_turns(question, history),
                temperature=0.0,
            )
        except UpstreamServiceError as error:
            if error.code not in _REWRITE_FALLBACK_CODES:
                raise
            logger.info("Question rewrite failed (%s); searching with the original", error.code)
            return question
        return clean_rewrite(response.text, fallback=question)
