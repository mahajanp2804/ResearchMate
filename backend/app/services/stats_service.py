"""Aggregate counts and recent items for the dashboard."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    ActivityEvent,
    Chunk,
    Conversation,
    Document,
    IndexingStatus,
    Message,
    MessageRole,
)
from app.schemas.system import ActivityItem, DashboardStats, RecentDocument

RECENT_DOCUMENTS_LIMIT = 5
RECENT_ACTIVITY_LIMIT = 10


def _count(db: Session, statement) -> int:
    return db.scalar(statement) or 0


def get_dashboard_stats(db: Session) -> DashboardStats:
    recent_documents = db.scalars(
        select(Document).order_by(Document.created_at.desc()).limit(RECENT_DOCUMENTS_LIMIT)
    ).all()
    recent_activity = db.scalars(
        select(ActivityEvent)
        .order_by(ActivityEvent.created_at.desc(), ActivityEvent.id.desc())
        .limit(RECENT_ACTIVITY_LIMIT)
    ).all()

    return DashboardStats(
        total_papers=_count(db, select(func.count()).select_from(Document)),
        indexed_papers=_count(
            db,
            select(func.count())
            .select_from(Document)
            .where(Document.indexing_status == IndexingStatus.INDEXED),
        ),
        total_chunks=_count(db, select(func.count()).select_from(Chunk)),
        total_questions=_count(
            db,
            select(func.count()).select_from(Message).where(Message.role == MessageRole.USER),
        ),
        total_conversations=_count(db, select(func.count()).select_from(Conversation)),
        recent_documents=[RecentDocument.model_validate(doc) for doc in recent_documents],
        recent_activity=[ActivityItem.model_validate(event) for event in recent_activity],
    )
