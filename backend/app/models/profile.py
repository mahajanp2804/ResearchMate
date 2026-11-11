from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import UTCDateTime, utcnow


class PaperProfile(Base):
    """A paper's cached profile: the cited facts behind its summary and comparisons.

    One row per paper. Building a profile costs an LLM call, so it is reused
    until the paper is re-processed (or the profile prompt changes), which
    ``signature`` detects.
    """

    __tablename__ = "paper_profiles"

    # Deleting the paper deletes its profile (ON DELETE CASCADE).
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    # Profile prompt version, chunking signature and embedding model it was
    # built from, e.g. "profile-v1|v2:...:250/50|sentence-transformers/...".
    signature: Mapped[str] = mapped_column(String(512), nullable=False)
    # The LLM that wrote it.
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    # Aspects with their validated text and citations, and the source passages
    # (with their text, so the profile stays traceable after re-chunking).
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
