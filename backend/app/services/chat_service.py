"""Conversations: asking questions in context and keeping the history.

A question and its answer are saved together, only after the answer exists,
so a failed LLM call leaves nothing half-written and can simply be retried.

Each answer is stored with the passages it was given (sources S1..Sn) in
the message's ``citations`` column, including their text, so an answer stays
traceable even after its paper is deleted or re-chunked. Sources the answer
actually cites are marked, with their supporting excerpt and exact page.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.exceptions import NotFoundError
from app.models import ActivityType, Conversation, Message, MessageRole
from app.models.base import utcnow
from app.services.activity_service import record_activity
from app.services.llm_service import ChatTurn
from app.services.rag_pipeline import RagAnswer, RagPipeline
from app.services.retrieval_service import RetrievalService, normalize_query

MAX_TITLE_LENGTH = 80
MAX_ACTIVITY_QUESTION_LENGTH = 120


def _title_from(question: str) -> str:
    if len(question) <= MAX_TITLE_LENGTH:
        return question
    return question[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0] + "…"


def _source_records(answer: RagAnswer) -> list[dict[str, Any]]:
    citations = {citation.source_id: citation for citation in answer.citations}
    records = []
    for number, source in enumerate(answer.sources, start=1):
        source_id = f"S{number}"
        citation = citations.get(source_id)
        records.append(
            {
                "source_id": source_id,
                "chunk_id": source.chunk_id,
                "document_id": source.document_id,
                "document_title": source.document_title,
                "page_start": source.page_start,
                "page_end": source.page_end,
                "section": source.section,
                "text": source.text,
                "score": source.score,
                "cited": citation is not None,
                "excerpt": citation.excerpt if citation else None,
                "excerpt_page": citation.page if citation else None,
                "support": citation.support if citation else None,
            }
        )
    return records


def _answer_details(answer: RagAnswer, searched: int, skipped: list[str]) -> dict[str, Any]:
    coverage = answer.coverage
    return {
        "outcome": answer.outcome,
        "standalone_question": answer.standalone_question,
        "model": answer.model,
        "context_tokens": answer.context_tokens,
        "truncated": answer.truncated,
        "timings_ms": answer.timings_ms,
        "searched_document_count": searched,
        "skipped_document_ids": skipped,
        "citation_status": coverage.status if coverage else None,
        "invalid_citation_ids": answer.invalid_citation_ids,
        "uncited_statements": coverage.uncited_statements if coverage else [],
    }


def get_conversation(db: Session, conversation_id: str) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError("Conversation not found.", details={"conversation_id": conversation_id})
    return conversation


def ask(
    db: Session,
    rag: RagPipeline,
    retrieval: RetrievalService,
    *,
    question: str,
    conversation_id: str | None,
    document_ids: list[str] | None,
) -> tuple[Conversation, Message, Message]:
    """Answer a question, continuing ``conversation_id`` if given, and save both messages.

    ``document_ids`` limits the papers searched for this question (None = all).
    """
    question = normalize_query(question)
    conversation = get_conversation(db, conversation_id) if conversation_id else None
    scope = retrieval.resolve_scope(db, document_ids)
    history = (
        [ChatTurn(message.role.value, message.content) for message in conversation.messages]
        if conversation
        else []
    )

    answer = rag.answer(db, question, scope, history)

    if conversation is None:
        conversation = Conversation(title=_title_from(question))
        db.add(conversation)
    conversation.document_ids = document_ids
    conversation.updated_at = utcnow()
    user_message = Message(role=MessageRole.USER, content=question)
    assistant_message = Message(
        role=MessageRole.ASSISTANT,
        content=answer.answer,
        citations=_source_records(answer),
        retrieval_info=_answer_details(
            answer, len(scope.effective_document_ids), list(scope.skipped_document_ids)
        ),
    )
    conversation.messages.extend([user_message, assistant_message])
    question_preview = question[:MAX_ACTIVITY_QUESTION_LENGTH]
    record_activity(db, ActivityType.QUESTION_ASKED, f"Asked “{question_preview}”")
    db.commit()
    return conversation, user_message, assistant_message


def list_conversations(
    db: Session, *, limit: int, offset: int
) -> tuple[list[tuple[Conversation, int]], int]:
    """Conversations, most recently active first, with their message counts."""
    total = db.scalar(select(func.count()).select_from(Conversation)) or 0
    counts = (
        select(Message.conversation_id, func.count(Message.id).label("message_count"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    rows = db.execute(
        select(Conversation, func.coalesce(counts.c.message_count, 0))
        .outerjoin(counts, counts.c.conversation_id == Conversation.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [(conversation, count) for conversation, count in rows], total


def delete_conversation(db: Session, conversation_id: str) -> None:
    db.delete(get_conversation(db, conversation_id))
    db.commit()
