"""Summaries and comparisons, both built on cached paper profiles.

A summary is one paper's profile shown as nine sections; a comparison shows
the profiles of several papers side by side. Building a profile is the only
step that calls the LLM, so summarising a paper that was already compared
(or comparing papers that were already summarised) costs nothing.
"""

from sqlalchemy.orm import Session

from app.exceptions import NotFoundError
from app.models import ActivityType, Document, PaperProfile
from app.services.activity_service import record_activity
from app.services.paper_profile_service import PaperProfileService


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
