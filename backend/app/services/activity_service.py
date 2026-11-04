from sqlalchemy.orm import Session

from app.models import ActivityEvent, ActivityType

MAX_DESCRIPTION_LENGTH = 500


def record_activity(
    db: Session,
    event_type: ActivityType,
    description: str,
    document_id: str | None = None,
) -> ActivityEvent:
    """Add an event to the dashboard feed.

    The caller commits, so the event is saved in the same transaction as the
    action it describes and never appears for an action that failed.
    """
    event = ActivityEvent(
        event_type=event_type,
        description=description[:MAX_DESCRIPTION_LENGTH],
        document_id=document_id,
    )
    db.add(event)
    return event
