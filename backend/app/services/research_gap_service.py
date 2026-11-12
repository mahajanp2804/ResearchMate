"""Research gap analysis across a few selected papers.

1. Evidence: each paper is searched separately with gap-related questions
   (limitations, assumptions, future work, problem, evaluation scope,
   failure cases) within a per-paper token budget, so every paper is heard.
   The passages become sources S1..Sn, grouped by paper (P1, P2, ...).
2. One LLM call returns stated limitations, observed patterns and inferred
   opportunities as JSON (see ``research_gap_prompt``).
3. Every item is checked in code before it is shown:
   - citations to passages that were not sent are removed (as in Phase 8);
   - a stated limitation may only cite the paper it is attributed to;
   - a recurring challenge or a difference must cite at least two papers;
   - every item needs at least one valid citation (an opportunity, for its
     evidence);
   items that fail are left out and listed with the reason. Wording that
   claims a gap is new beyond the selected papers is flagged, and claims
   without a citation are flagged as in answers.
4. The analysis is saved and reused for the same papers until one of them
   is re-processed.
"""

import hashlib
import re
import time
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import UpstreamServiceError
from app.models import ActivityType, Document, GapAnalysis
from app.models.base import utcnow
from app.prompts.research_gap_prompt import (
    GAP_PROMPT_VERSION,
    GAP_QUESTIONS,
    GAP_SYSTEM_PROMPT,
    PATTERN_KINDS,
    build_gap_turns,
    claims_beyond_scope,
    gap_json_schema,
    paper_label,
    parse_gap_reply,
)
from app.services.activity_service import record_activity
from app.services.citation_service import (
    CITATION_MARKER,
    SourcePassage,
    build_citations,
    citation_coverage,
    validate_citations,
)
from app.services.rag_pipeline import LLMProviderFactory
from app.services.research_service import select_papers
from app.services.retrieval_service import RetrievalService, RetrievedChunk, page_texts_for
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Patterns that compare papers must rest on at least this many of them.
CROSS_PAPER_KINDS = {"recurring_challenges", "differences"}
LEFT_OUT_REASONS = {
    "no_valid_citation": "It cited no passage that was given to the model.",
    "cites_other_paper": "It was attributed to one paper but cited only other papers.",
    "cites_one_paper": "It was presented as a pattern across papers but cited only one paper.",
}
MAX_LEFT_OUT_TITLE = 120


def _title(document: Document) -> str:
    return document.title or document.filename


def analysis_key(documents: Sequence[Document]) -> str:
    """Identifies an analysis of these papers as they are now, in any order."""
    parts = sorted(
        f"{document.id}:{document.chunking_signature}:{document.embedding_model}"
        for document in documents
    )
    raw = f"gaps-v{GAP_PROMPT_VERSION}|" + "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def scope_note(paper_count: int, passage_count: int) -> str:
    return (
        f"Based only on the {paper_count} selected papers and the {passage_count} passages "
        "retrieved from them. These findings describe this collection, not the research "
        "field: a possible gap here may already be addressed in work you haven't uploaded."
    )


def forget_document(db: Session, document_id: str) -> int:
    """Delete saved analyses that include a paper being deleted. The caller commits."""
    analyses = [
        analysis
        for analysis in db.scalars(select(GapAnalysis))
        if document_id in analysis.document_ids
    ]
    for analysis in analyses:
        db.delete(analysis)
    return len(analyses)


class _Checker:
    """Validates the findings of one reply against the sources that were sent."""

    def __init__(
        self,
        sources: list[tuple[int, RetrievedChunk]],
        documents: list[Document],
        page_texts: dict[str, dict[int, str]],
    ) -> None:
        self.all_ids = [f"S{number}" for number in range(1, len(sources) + 1)]
        self.paper_of = {f"S{n}": paper for n, (paper, _) in enumerate(sources, start=1)}
        self.documents = documents
        self.page_texts = page_texts
        self.passages = [
            SourcePassage(
                f"S{number}",
                chunk.text,
                chunk.page_start,
                chunk.page_end,
                chunk.document_id,
                _title(documents[paper]),
            )
            for number, (paper, chunk) in enumerate(sources, start=1)
        ]
        self.left_out: list[dict[str, str]] = []

    def check(
        self, kind: str, title: str | None, text: str, *, paper: int | None = None
    ) -> dict[str, Any] | None:
        """The validated finding, or None if it was left out (and recorded)."""
        check = validate_citations(text, self.all_ids)
        invented = check.invalid_ids
        misattributed: list[str] = []
        if paper is not None:
            # A stated limitation may only cite the paper it is attributed to.
            own = [i for i in self.all_ids if self.paper_of[i] == paper]
            check = validate_citations(check.text, own)
            misattributed = check.invalid_ids

        papers = list(dict.fromkeys(self.paper_of[i] for i in check.cited_ids))
        reason = None
        if not papers:
            reason = "cites_other_paper" if misattributed else "no_valid_citation"
        elif kind in CROSS_PAPER_KINDS and len(papers) < 2:
            reason = "cites_one_paper"
        if reason:
            label = title or _strip_citations(text)
            self.left_out.append(
                {"kind": kind, "title": label[:MAX_LEFT_OUT_TITLE], "reason": reason}
            )
            return None

        coverage = citation_coverage(check.text)
        citations = build_citations(check.text, self.passages, self.page_texts)
        return {
            "title": title,
            "text": check.text,
            "document_ids": [self.documents[index].id for index in papers],
            "citations": [
                {
                    "source_id": citation.source_id,
                    "excerpt": citation.excerpt,
                    "page": citation.page,
                    "support": citation.support,
                }
                for citation in citations
            ],
            "citation_status": coverage.status,
            "uncited_statements": coverage.uncited_statements,
            "invalid_citation_ids": invented,
            "misattributed_citation_ids": misattributed,
            "beyond_scope": claims_beyond_scope(f"{title or ''} {check.text}"),
        }


def _strip_citations(text: str) -> str:
    """A suggestion is inference, so any markers in it are removed."""
    text = " ".join(CITATION_MARKER.sub("", text).split())
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


class ResearchGapService:
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

    def analyse(
        self, db: Session, document_ids: Sequence[str], *, refresh: bool = False
    ) -> tuple[GapAnalysis, bool]:
        """The analysis of these papers and whether it was already saved."""
        documents = select_papers(
            db, document_ids, max_papers=self._settings.max_gap_papers, verb="analyse"
        )
        key = analysis_key(documents)
        saved = db.scalar(select(GapAnalysis).where(GapAnalysis.key == key))
        if saved is not None and not refresh:
            return saved, True

        content, model = self._build(db, documents)
        analysis = saved or GapAnalysis(key=key)
        analysis.document_ids = [document.id for document in documents]
        analysis.model = model
        analysis.content = content
        analysis.created_at = utcnow()
        db.add(analysis)
        titles = ", ".join(f"“{_title(document)}”" for document in documents)
        record_activity(db, ActivityType.GAP_ANALYSIS_RUN, f"Analysed research gaps in {titles}")
        try:
            db.commit()
        except IntegrityError:
            # The same analysis was saved by a concurrent request; use that one.
            db.rollback()
            existing = db.scalar(select(GapAnalysis).where(GapAnalysis.key == key))
            if existing is None:
                raise
            return existing, True
        return analysis, False

    def _build(self, db: Session, documents: list[Document]) -> tuple[dict[str, Any], str]:
        settings = self._settings
        started = time.perf_counter()
        sources: list[tuple[int, RetrievedChunk]] = []
        for index, document in enumerate(documents):
            chunks = self._retrieval.passages_for_questions(
                db,
                document.id,
                GAP_QUESTIONS,
                per_question=settings.gap_passages_per_question,
                max_tokens=settings.gap_max_tokens_per_paper,
            )
            sources.extend((index, chunk) for chunk in chunks)
        page_texts = page_texts_for(db, [chunk for _, chunk in sources])
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)

        started = time.perf_counter()
        response = self._llm_provider().generate(
            system_prompt=GAP_SYSTEM_PROMPT,
            turns=build_gap_turns([_title(document) for document in documents], sources),
            json_schema=gap_json_schema(len(documents)),
        )
        try:
            reply = parse_gap_reply(response.text, len(documents))
        except ValueError as error:
            logger.warning("Unreadable gap analysis reply: %s", error)
            if response.truncated:
                raise UpstreamServiceError(
                    "The model ran out of output space during the analysis. Increase "
                    "LLM_MAX_OUTPUT_TOKENS in backend/.env, or analyse fewer papers.",
                    code="llm_truncated",
                ) from error
            raise UpstreamServiceError(
                "The model's reply was not in the expected format. Please try again.",
                code="llm_bad_output",
            ) from error

        checker = _Checker(sources, documents, page_texts)
        limitations = [
            finding
            for item in reply.stated_limitations
            if (finding := checker.check("stated_limitations", None, item.text, paper=item.paper))
        ]
        patterns = {
            kind: [
                finding
                for item in reply.patterns[kind]
                if (finding := checker.check(kind, item.title, item.text))
            ]
            for kind in PATTERN_KINDS
        }
        opportunities = []
        for item in reply.opportunities:
            finding = checker.check("opportunities", item.title, item.evidence)
            if finding is None:
                continue
            suggestion = _strip_citations(item.suggestion)
            finding["suggestion"] = suggestion
            finding["beyond_scope"] = finding["beyond_scope"] or claims_beyond_scope(suggestion)
            opportunities.append(finding)

        content = {
            "papers": [
                {"document_id": document.id, "title": _title(document), "label": paper_label(i)}
                for i, document in enumerate(documents)
            ],
            "sources": [
                {
                    "source_id": f"S{number}",
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "text": chunk.text,
                }
                for number, (_, chunk) in enumerate(sources, start=1)
            ],
            "stated_limitations": limitations,
            **patterns,
            "opportunities": opportunities,
            "left_out": checker.left_out,
            "malformed_items": reply.malformed,
            "truncated": response.truncated,
            "timings_ms": {
                "retrieval": retrieval_ms,
                "generation": round((time.perf_counter() - started) * 1000, 1),
            },
        }
        return content, response.model
