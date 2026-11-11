from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models import Document, PaperProfile
from app.prompts.profile_prompt import ASPECTS_BY_KEY


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
    # The papers to compare, in column order; the limit is checked against
    # MAX_COMPARE_PAPERS.
    document_ids: list[str] = Field(min_length=2, max_length=50)
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
