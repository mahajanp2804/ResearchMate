"""Paper profiles: the cited facts behind summaries and comparisons.

A profile answers a fixed set of aspects for one paper (objective, problem,
methodology, datasets, results, limitations, ...). Building one:

1. Evidence: each aspect has its own retrieval question, searched within
   that paper only (hybrid search + reranking). Every aspect's best passage
   is kept before any aspect's second best, within a token budget, and the
   kept passages are put back in reading order as sources S1..Sn.
2. Extraction: one LLM call fills in every aspect as JSON, citing sources,
   or marks it as not found.
3. Validation, per aspect (as in Phase 8): citations to passages that were
   not sent are removed, each cited passage gets its supporting excerpt and
   exact page, and claims without a citation are flagged. An aspect whose
   text only says the information is missing counts as not found.

The profile is saved and reused until the paper is re-processed, so a
summary or a comparison involving the paper costs no further LLM calls.
The steps are separate methods so a comparison can gather evidence for
several papers (which needs the database) and then run the LLM calls in
parallel (which does not).
"""

import time
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import ConflictError, NotFoundError, UpstreamServiceError
from app.models import Document, IndexingStatus, PaperProfile
from app.models.base import utcnow
from app.prompts.answer_prompt import is_refusal
from app.prompts.profile_prompt import (
    ASPECTS,
    PROFILE_SYSTEM_PROMPT,
    PROFILE_VERSION,
    ExtractedField,
    build_profile_turns,
    parse_profile_reply,
    profile_json_schema,
)
from app.services.citation_service import (
    SourcePassage,
    build_citations,
    citation_coverage,
    validate_citations,
)
from app.services.context_selection import select_context
from app.services.rag_pipeline import LLMProviderFactory
from app.services.retrieval_service import (
    RetrievalService,
    RetrievedChunk,
    SearchScope,
    page_texts_for,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

AspectStatus = Literal["found", "not_found"]


@dataclass(frozen=True)
class ProfileEvidence:
    """The passages a profile is built from: sources S1..Sn, in reading order."""

    document_id: str
    title: str
    chunks: list[RetrievedChunk]
    # Raw page texts of passages spanning pages, to pin citations to a page.
    page_texts: dict[str, dict[int, str]]
    retrieval_ms: float


@dataclass(frozen=True)
class ProfileDraft:
    """A freshly extracted profile, ready to be saved."""

    document_id: str
    model: str
    content: dict[str, Any]


def profile_signature(document: Document) -> str:
    """What a profile depends on; when it changes the profile is rebuilt."""
    return f"profile-v{PROFILE_VERSION}|{document.chunking_signature}|{document.embedding_model}"


def _not_found(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "status": "not_found",
        "text": None,
        "citations": [],
        "citation_status": None,
        "invalid_citation_ids": [],
        "uncited_statements": [],
    }


def check_aspect(
    key: str,
    field: ExtractedField | None,
    passages: list[SourcePassage],
    page_texts: dict[str, dict[int, str]],
) -> dict[str, Any]:
    """Validate one aspect's text and resolve its citations."""
    if field is None or not field.found or not field.text or is_refusal(field.text):
        return _not_found(key)

    check = validate_citations(field.text, [passage.source_id for passage in passages])
    coverage = citation_coverage(check.text)
    if not check.cited_ids and not coverage.uncited_statements:
        # Marked as found, but it only says the information is missing.
        return _not_found(key)

    citations = build_citations(check.text, passages, page_texts)
    return {
        "key": key,
        "status": "found",
        "text": check.text,
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
        "invalid_citation_ids": check.invalid_ids,
        "uncited_statements": coverage.uncited_statements,
    }


class PaperProfileService:
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

    # --- Lookup ----------------------------------------------------------------

    @staticmethod
    def require_indexed(db: Session, document_id: str) -> Document:
        document = db.get(Document, document_id)
        if document is None:
            raise NotFoundError("Paper not found.", details={"document_id": document_id})
        if document.indexing_status != IndexingStatus.INDEXED:
            raise ConflictError(
                f"“{document.title or document.filename}” isn't searchable yet. Wait for "
                "indexing to finish, or index it from My Research Papers.",
                code="paper_not_indexed",
                details={"document_id": document_id},
            )
        return document

    @staticmethod
    def current(db: Session, document: Document) -> PaperProfile | None:
        """The saved profile, unless the paper changed since it was built."""
        profile = db.get(PaperProfile, document.id)
        if profile is None or profile.signature != profile_signature(document):
            return None
        return profile

    # --- Building ----------------------------------------------------------------

    def gather_evidence(self, db: Session, document: Document) -> ProfileEvidence:
        started = time.perf_counter()
        settings = self._settings
        only_this_paper = frozenset({document.id})
        scope = SearchScope(document_ids=only_this_paper, searchable_document_ids=only_this_paper)
        search = (
            self._retrieval.reranked_search
            if settings.reranker_enabled
            else self._retrieval.hybrid_search
        )
        per_aspect = settings.profile_passages_per_aspect
        ranked = [search(db, aspect.query, per_aspect, scope) for aspect in ASPECTS]

        # Interleave so every aspect gets its best passage before any aspect
        # gets a second one; the budget then trims the least important.
        candidates: list[RetrievedChunk] = []
        seen: set[int] = set()
        for position in range(per_aspect):
            for hits in ranked:
                if position < len(hits) and hits[position].chunk_id not in seen:
                    seen.add(hits[position].chunk_id)
                    candidates.append(hits[position])
        if not candidates:
            raise ConflictError(
                "This paper has no passages to read. Re-index it from My Research Papers.",
                code="paper_has_no_passages",
                details={"document_id": document.id},
            )
        selection = select_context(
            candidates,
            max_chunks=len(candidates),
            max_tokens=settings.profile_max_context_tokens,
        )
        chunks = sorted(selection.chunks, key=lambda chunk: chunk.chunk_index)
        return ProfileEvidence(
            document_id=document.id,
            title=document.title or document.filename,
            chunks=chunks,
            page_texts=page_texts_for(db, chunks),
            retrieval_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def extract(self, evidence: ProfileEvidence) -> ProfileDraft:
        """Ask the LLM for every aspect and validate the reply. Uses no database."""
        started = time.perf_counter()
        response = self._llm_provider().generate(
            system_prompt=PROFILE_SYSTEM_PROMPT,
            turns=build_profile_turns(evidence.title, evidence.chunks),
            json_schema=profile_json_schema(),
        )
        try:
            fields = parse_profile_reply(response.text)
        except ValueError as error:
            logger.warning("Unreadable profile reply for %s: %s", evidence.document_id, error)
            if response.truncated:
                raise UpstreamServiceError(
                    "The model ran out of output space while profiling the paper. Increase "
                    "LLM_MAX_OUTPUT_TOKENS in backend/.env.",
                    code="llm_truncated",
                ) from error
            raise UpstreamServiceError(
                "The model's reply was not in the expected format. Please try again.",
                code="llm_bad_output",
            ) from error
        missing = [aspect.key for aspect in ASPECTS if aspect.key not in fields]
        if missing:
            logger.info("Profile reply left out %s; treated as not found", missing)

        passages = [
            SourcePassage(
                f"S{number}",
                chunk.text,
                chunk.page_start,
                chunk.page_end,
                chunk.document_id,
                evidence.title,
            )
            for number, chunk in enumerate(evidence.chunks, start=1)
        ]
        aspects = [
            check_aspect(aspect.key, fields.get(aspect.key), passages, evidence.page_texts)
            for aspect in ASPECTS
        ]
        sources = [
            {
                "source_id": f"S{number}",
                "chunk_id": chunk.chunk_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "text": chunk.text,
            }
            for number, chunk in enumerate(evidence.chunks, start=1)
        ]
        content = {
            "aspects": aspects,
            "sources": sources,
            "truncated": response.truncated,
            "timings_ms": {
                "retrieval": evidence.retrieval_ms,
                "generation": round((time.perf_counter() - started) * 1000, 1),
            },
        }
        return ProfileDraft(document_id=evidence.document_id, model=response.model, content=content)

    @staticmethod
    def save(db: Session, document: Document, draft: ProfileDraft) -> PaperProfile:
        """Store the profile, replacing an older one. The caller commits."""
        profile = db.get(PaperProfile, document.id)
        if profile is None:
            profile = PaperProfile(document_id=document.id)
            db.add(profile)
        profile.signature = profile_signature(document)
        profile.model = draft.model
        profile.content = draft.content
        profile.created_at = utcnow()
        return profile

    def get_or_build(
        self, db: Session, document: Document, *, refresh: bool = False
    ) -> tuple[PaperProfile, bool]:
        """The paper's profile and whether it came from the cache. The caller commits."""
        if not refresh and (existing := self.current(db, document)) is not None:
            return existing, True
        draft = self.extract(self.gather_evidence(db, document))
        return self.save(db, document, draft), False
