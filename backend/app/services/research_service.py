"""Summaries and comparisons, both built on cached paper profiles.

A summary is one paper's profile shown as nine sections; a comparison shows
the profiles of several papers side by side. Building a profile is the only
step that calls the LLM, so summarising a paper that was already compared
(or comparing papers that were already summarised) costs nothing.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.exceptions import ConflictError, InvalidInputError, NotFoundError, ResearchMateError
from app.models import ActivityType, Document, IndexingStatus, PaperProfile
from app.services.activity_service import record_activity
from app.services.paper_profile_service import (
    PaperProfileService,
    ProfileDraft,
    ProfileEvidence,
)

# Profiles built at the same time for one comparison (one LLM call each).
MAX_PARALLEL_PROFILES = 4


def _title(document: Document) -> str:
    return document.title or document.filename


def summarize(
    db: Session, profiles: PaperProfileService, document_id: str, *, refresh: bool = False
) -> tuple[Document, PaperProfile, bool]:
    """The paper's profile, built if needed, and whether it was already saved."""
    document = profiles.require_indexed(db, document_id)
    profile, cached = profiles.get_or_build(db, document, refresh=refresh)
    if not cached:
        record_activity(
            db,
            ActivityType.SUMMARY_GENERATED,
            f"Summarised “{_title(document)}”",
            document_id=document.id,
        )
    db.commit()
    return document, profile, cached


def saved_summary(
    db: Session, profiles: PaperProfileService, document_id: str
) -> tuple[Document, PaperProfile]:
    """The saved, current profile of a paper, without calling the LLM."""
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Paper not found.", details={"document_id": document_id})
    profile = profiles.current(db, document)
    if profile is None:
        raise NotFoundError(
            "This paper hasn't been summarised since it was last indexed.",
            code="summary_not_found",
            details={"document_id": document_id},
        )
    return document, profile


def select_papers(
    db: Session, document_ids: Sequence[str], *, max_papers: int, verb: str
) -> list[Document]:
    """Two to ``max_papers`` different, searchable papers, in the requested order.

    ``verb`` names the action in error messages, e.g. "compare".
    """
    requested = list(dict.fromkeys(document_ids))  # de-duplicate, keep order
    if len(requested) < 2:
        raise InvalidInputError(
            f"Select at least two different papers to {verb}.", code="too_few_papers"
        )
    if len(requested) > max_papers:
        raise InvalidInputError(
            f"Select at most {max_papers} papers to {verb} at a time.",
            code="too_many_papers",
            details={"max_papers": max_papers},
        )
    found = {
        document.id: document
        for document in db.scalars(select(Document).where(Document.id.in_(requested)))
    }
    unknown = [document_id for document_id in requested if document_id not in found]
    if unknown:
        raise InvalidInputError(
            "Some selected papers do not exist.",
            code="unknown_documents",
            details={"document_ids": unknown},
        )
    not_ready = [i for i in requested if found[i].indexing_status != IndexingStatus.INDEXED]
    if not_ready:
        raise ConflictError(
            "Some selected papers aren't searchable yet. Wait for indexing to finish, or "
            "index them from My Research Papers.",
            code="papers_not_indexed",
            details={"document_ids": not_ready},
        )
    return [found[document_id] for document_id in requested]


def _extract_all(
    profiles: PaperProfileService, evidence: list[ProfileEvidence]
) -> tuple[dict[str, ProfileDraft], list[ResearchMateError]]:
    """Run the LLM step for several papers at once; it needs no database."""
    drafts: dict[str, ProfileDraft] = {}
    failures: list[ResearchMateError] = []
    if not evidence:
        return drafts, failures
    workers = min(MAX_PARALLEL_PROFILES, len(evidence))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="profile") as pool:
        futures = {pool.submit(profiles.extract, item): item.document_id for item in evidence}
        for future in as_completed(futures):
            try:
                drafts[futures[future]] = future.result()
            except ResearchMateError as error:
                failures.append(error)
    return drafts, failures


def _comparison_description(documents: Sequence[Document]) -> str:
    titles = [f"“{_title(document)}”" for document in documents]
    return f"Compared {', '.join(titles[:-1])} and {titles[-1]}"


def compare(
    db: Session,
    profiles: PaperProfileService,
    settings: Settings,
    document_ids: Sequence[str],
    *,
    refresh: bool = False,
) -> list[tuple[Document, PaperProfile, bool]]:
    """Each paper's profile, in the requested order, and whether it was already saved.

    Missing profiles are built in parallel. If some fail, the ones that
    succeeded are still saved, so trying again only redoes the rest.
    """
    documents = select_papers(
        db, document_ids, max_papers=settings.max_compare_papers, verb="compare"
    )
    saved = (
        {} if refresh else {document.id: profiles.current(db, document) for document in documents}
    )
    to_build = [document for document in documents if saved.get(document.id) is None]

    # Retrieval uses the database, so it runs here; only the LLM calls run in parallel.
    evidence = [profiles.gather_evidence(db, document) for document in to_build]
    drafts, failures = _extract_all(profiles, evidence)
    built = {
        document.id: profiles.save(db, document, drafts[document.id])
        for document in to_build
        if document.id in drafts
    }
    if failures:
        db.commit()
        raise failures[0]

    # Logged when papers were read, like summaries; reopening a comparison
    # whose profiles are all saved (e.g. reloading the page) is not.
    if built:
        record_activity(db, ActivityType.COMPARISON_RUN, _comparison_description(documents))
    db.commit()
    return [
        (document, built.get(document.id) or saved[document.id], document.id not in built)
        for document in documents
    ]
