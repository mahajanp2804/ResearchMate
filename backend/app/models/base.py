"""Shared column types and helpers for the ORM models."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class UTCDateTime(TypeDecorator):
    """Stores timestamps as naive UTC and always returns timezone-aware UTC.

    SQLite has no timezone support, so without this the API would emit naive
    datetimes and browsers would interpret them in local time.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed; use timezone-aware UTC")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


def string_enum(enum_cls: type[enum.Enum]) -> Enum:
    """Persist an enum by its value as a plain string column.

    ``native_enum=False`` keeps the schema portable between SQLite and PostgreSQL.
    """
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
