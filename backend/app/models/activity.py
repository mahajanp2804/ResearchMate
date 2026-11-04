import enum
from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import UTCDateTime, string_enum, utcnow


class ActivityType(enum.StrEnum):
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_REINDEXED = "document_reindexed"
    QUESTION_ASKED = "question_asked"
    SUMMARY_GENERATED = "summary_generated"
    COMPARISON_RUN = "comparison_run"
    GAP_ANALYSIS_RUN = "gap_analysis_run"
    EVALUATION_RUN = "evaluation_run"


class ActivityEvent(Base):
    """Feed of research actions shown on the dashboard."""

    __tablename__ = "activity_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[ActivityType] = mapped_column(string_enum(ActivityType), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    # SET NULL keeps the history entry after its document is deleted.
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, nullable=False, index=True
    )
