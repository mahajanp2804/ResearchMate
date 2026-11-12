from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import UTCDateTime, utcnow


class GapAnalysis(Base):
    """A saved research gap analysis over a set of papers.

    It costs an LLM call over all the papers' passages, so it is reused for
    the same papers until one of them is re-processed (or the prompt
    changes), which a change of ``key`` detects. Deleting any of its papers
    deletes it (see ``research_gap_service.forget_document``).
    """

    __tablename__ = "gap_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # SHA-256 of the prompt version and each paper's id, chunking signature and
    # embedding model, independent of the order the papers were selected in.
    key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    # The analysed papers, in the order their labels (P1, P2, ...) refer to.
    document_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    # The LLM that wrote it.
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    # Validated findings, the source passages (with their text) and what was left out.
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
