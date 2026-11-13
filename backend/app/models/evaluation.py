import enum
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin, UTCDateTime, new_id, string_enum, utcnow


class EvaluationDataset(TimestampMixin, Base):
    __tablename__ = "evaluation_datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    questions: Mapped[list["EvaluationQuestion"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="EvaluationQuestion.id",
    )
    runs: Mapped[list["EvaluationRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )


class EvaluationQuestion(Base):
    """One curated question with its ground truth."""

    __tablename__ = "evaluation_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # Stable id from the dataset file (e.g. "q07"), so runs can be compared.
    key: Mapped[str | None] = mapped_column(String(64))
    # factual, paraphrase, keyword, multi_paper or unanswerable: lets results
    # show, for example, that dense search finds paraphrases BM25 misses.
    category: Mapped[str | None] = mapped_column(String(32))
    expected_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    expected_chunk_ids: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)
    # Ground-truth passages as {"document_id", "page", "text"}. Chunk ids change
    # whenever chunking settings change, so evidence text is the stable anchor.
    expected_evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    reference_answer: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped[EvaluationDataset] = relationship(back_populates="questions")


class EvaluationRunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[EvaluationRunStatus] = mapped_column(
        string_enum(EvaluationRunStatus), default=EvaluationRunStatus.PENDING, nullable=False
    )
    # Retrieval configurations compared in this run and their settings.
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # Aggregate metrics per configuration (retrieval and answer quality kept apart).
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    per_question_results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    dataset: Mapped[EvaluationDataset] = relationship(back_populates="runs")
