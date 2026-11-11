from fastapi import APIRouter

from app.dependencies import DbSession, ProfilesDep
from app.prompts.profile_prompt import SUMMARY_SECTIONS
from app.schemas.research import PaperProfileOut
from app.services import research_service

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("/summarize/{document_id}", response_model=PaperProfileOut)
def summarize(
    document_id: str, db: DbSession, profiles: ProfilesDep, refresh: bool = False
) -> PaperProfileOut:
    """Structured summary of one paper in nine sections, every statement cited.

    Each section's evidence is retrieved from the paper separately and one
    LLM call writes all sections. A section the passages don't cover is
    returned as ``not_found`` rather than guessed. The result is saved and
    returned again (``cached: true``) until the paper is re-indexed or
    ``refresh=true`` is passed.
    """
    document, profile, cached = research_service.summarize(
        db, profiles, document_id, refresh=refresh
    )
    return PaperProfileOut.from_profile(document, profile, SUMMARY_SECTIONS, cached=cached)


@router.get("/summaries/{document_id}", response_model=PaperProfileOut)
def get_summary(document_id: str, db: DbSession, profiles: ProfilesDep) -> PaperProfileOut:
    """The saved summary of a paper, without generating one (404 if there is none)."""
    document, profile = research_service.saved_summary(db, profiles, document_id)
    return PaperProfileOut.from_profile(document, profile, SUMMARY_SECTIONS, cached=True)
