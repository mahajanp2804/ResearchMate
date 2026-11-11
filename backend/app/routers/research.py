from fastapi import APIRouter

from app.dependencies import DbSession, ProfilesDep, SettingsDep
from app.prompts.profile_prompt import ASPECTS_BY_KEY, COMPARISON_ASPECTS, SUMMARY_SECTIONS
from app.schemas.research import (
    AspectLabel,
    CompareRequest,
    ComparisonResponse,
    PaperProfileOut,
)
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


@router.post("/compare", response_model=ComparisonResponse)
def compare(
    request: CompareRequest, db: DbSession, profiles: ProfilesDep, settings: SettingsDep
) -> ComparisonResponse:
    """Compare two or more papers aspect by aspect, every cell cited from its own paper.

    Each paper's column comes from its saved profile (the same one behind its
    summary), built in parallel for papers that don't have one yet. A cell
    the paper's passages don't cover is ``not_found`` ("not reported"), never
    a guess. Ask follow-up questions with ``POST /api/chat/query`` and the same
    ``document_ids``.
    """
    results = research_service.compare(
        db, profiles, settings, request.document_ids, refresh=request.refresh
    )
    return ComparisonResponse(
        aspects=[
            AspectLabel(key=key, label=ASPECTS_BY_KEY[key].label) for key in COMPARISON_ASPECTS
        ],
        papers=[
            PaperProfileOut.from_profile(document, profile, COMPARISON_ASPECTS, cached=cached)
            for document, profile, cached in results
        ],
    )
