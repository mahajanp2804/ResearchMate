from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models import Document, GapAnalysis, PaperProfile
from app.prompts.profile_prompt import ASPECTS_BY_KEY
from app.services.research_gap_service import LEFT_OUT_REASONS, scope_note


class ProfileCitationOut(BaseModel):
    """A source an aspect cites, with the sentence that supports it."""

    source_id: str
    excerpt: str
    # The page the excerpt is on (exact even when the passage spans pages).
    page: int
    # Share of the citing statements' key terms found in the source (0-1).
    support: float


class ProfileSourceOut(BaseModel):
    """A passage of the paper the profile was built from."""

    source_id: str
    chunk_id: int
    page_start: int
    page_end: int
    section: str | None
    text: str


class AspectOut(BaseModel):
    key: str
    label: str
    # not_found: the retrieved passages don't state this; no text is given.
    status: Literal["found", "not_found"]
    text: str | None
    citations: list[ProfileCitationOut]
    citation_status: Literal["cited", "partially_cited", "uncited"] | None
    # Ids the model cited that were not among the sources; removed from the text.
    invalid_citation_ids: list[str]
    # Claims in the text that cite no source.
    uncited_statements: list[str]


class PaperProfileOut(BaseModel):
    """One paper's aspects (summary sections or comparison cells), all cited."""

    document_id: str
    document_title: str
    aspects: list[AspectOut]
    sources: list[ProfileSourceOut]
    model: str
    generated_at: datetime
    # True when this was saved earlier and no LLM call was made.
    cached: bool
    truncated: bool

    @classmethod
    def from_profile(
        cls, document: Document, profile: PaperProfile, keys: tuple[str, ...], *, cached: bool
    ) -> "PaperProfileOut":
        content = profile.content
        by_key = {aspect["key"]: aspect for aspect in content["aspects"]}
        return cls(
            document_id=document.id,
            document_title=document.title or document.filename,
            aspects=[
                AspectOut(label=ASPECTS_BY_KEY[key].label, **by_key[key])
                for key in keys
                if key in by_key
            ],
            sources=[ProfileSourceOut(**source) for source in content["sources"]],
            model=profile.model,
            generated_at=profile.created_at,
            cached=cached,
            truncated=content.get("truncated", False),
        )


class CompareRequest(BaseModel):
    # The papers to compare, in column order: at least two different papers
    # and at most MAX_COMPARE_PAPERS (checked by the service, with clear errors).
    document_ids: list[str] = Field(max_length=50)
    # Rebuild the papers' profiles even if saved ones are current.
    refresh: bool = False


class AspectLabel(BaseModel):
    key: str
    label: str


class ComparisonResponse(BaseModel):
    # The rows of the comparison, in order.
    aspects: list[AspectLabel]
    # One column per paper, in the requested order; each holds its own sources.
    papers: list[PaperProfileOut]


# --- Research gaps -----------------------------------------------------------


class GapAnalysisRequest(BaseModel):
    # The papers to analyse: two to MAX_GAP_PAPERS different papers.
    document_ids: list[str] = Field(max_length=50)
    # Analyse again even if a saved analysis of these papers is current.
    refresh: bool = False


class GapPaperOut(BaseModel):
    document_id: str
    title: str
    # P1, P2, ...: how findings and passages refer to the paper.
    label: str


class GapSourceOut(BaseModel):
    """A passage the analysis was given, numbered across all papers."""

    source_id: str
    document_id: str
    chunk_id: int
    page_start: int
    page_end: int
    section: str | None
    text: str


class GapFindingOut(BaseModel):
    # Short label; null for stated limitations.
    title: str | None
    # The finding (for an opportunity: its evidence), with [S#] citations.
    text: str
    # The papers it cites, in label order of first citation.
    document_ids: list[str]
    citations: list[ProfileCitationOut]
    citation_status: Literal["cited", "partially_cited", "uncited"]
    uncited_statements: list[str]
    # Removed: ids that were not among the sources...
    invalid_citation_ids: list[str]
    # ...and, for a stated limitation, ids of another paper's passages.
    misattributed_citation_ids: list[str]
    # Its wording claims more than the selected papers can show (e.g. "unexplored").
    beyond_scope: bool


class GapOpportunityOut(GapFindingOut):
    # The inferred research direction; not a finding of the papers, so uncited.
    suggestion: str


class LeftOutFindingOut(BaseModel):
    """A finding the model proposed that failed the evidence checks."""

    kind: str
    title: str
    reason: Literal["no_valid_citation", "cites_other_paper", "cites_one_paper"]
    explanation: str


class GapAnalysisResponse(BaseModel):
    papers: list[GapPaperOut]
    # What the findings can and cannot claim, given how they were made.
    scope_note: str
    # Stated by the papers about their own work.
    stated_limitations: list[GapFindingOut]
    # Observed across the selected papers.
    recurring_challenges: list[GapFindingOut]
    differences: list[GapFindingOut]
    limited_discussion: list[GapFindingOut]
    # Inferred from the findings above.
    opportunities: list[GapOpportunityOut]
    left_out: list[LeftOutFindingOut]
    # Items not in the expected shape, which were ignored.
    malformed_items: int
    sources: list[GapSourceOut]
    model: str
    generated_at: datetime
    cached: bool
    truncated: bool

    @classmethod
    def from_analysis(cls, analysis: GapAnalysis, *, cached: bool) -> "GapAnalysisResponse":
        content = analysis.content
        return cls(
            papers=[GapPaperOut(**paper) for paper in content["papers"]],
            scope_note=scope_note(len(content["papers"]), len(content["sources"])),
            stated_limitations=content["stated_limitations"],
            recurring_challenges=content["recurring_challenges"],
            differences=content["differences"],
            limited_discussion=content["limited_discussion"],
            opportunities=content["opportunities"],
            left_out=[
                LeftOutFindingOut(**item, explanation=LEFT_OUT_REASONS[item["reason"]])
                for item in content["left_out"]
            ],
            malformed_items=content["malformed_items"],
            sources=content["sources"],
            model=analysis.model,
            generated_at=analysis.created_at,
            cached=cached,
            truncated=content["truncated"],
        )
