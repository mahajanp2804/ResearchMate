from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models import Conversation, Message, MessageRole


class ChatRequest(BaseModel):
    question: str = Field(max_length=4000)
    # Continue this conversation; omit to start a new one.
    conversation_id: str | None = None
    # Papers to search for this question; omit to search all papers.
    document_ids: list[str] | None = Field(default=None, max_length=500)


class SourceOut(BaseModel):
    """A passage given to the LLM, numbered as the answer cites it."""

    source_id: str  # "S1", "S2", ...
    chunk_id: int
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section: str | None
    text: str
    score: float


class AnswerDetails(BaseModel):
    # no_evidence: refused without calling the LLM; model_declined: the LLM
    # judged the sources insufficient.
    outcome: Literal["answered", "no_evidence", "model_declined"]
    # The follow-up rewritten as a standalone question (equal to the question otherwise).
    standalone_question: str
    model: str | None
    context_tokens: int
    truncated: bool
    timings_ms: dict[str, float]
    searched_document_count: int
    skipped_document_ids: list[str]


class MessageOut(BaseModel):
    id: int
    role: MessageRole
    content: str
    created_at: datetime
    sources: list[SourceOut] = []
    details: AnswerDetails | None = None

    @classmethod
    def from_message(cls, message: Message) -> "MessageOut":
        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            sources=[SourceOut(**source) for source in message.citations or []],
            details=AnswerDetails(**message.retrieval_info) if message.retrieval_info else None,
        )


class ConversationSummary(BaseModel):
    id: str
    title: str
    document_ids: list[str] | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_conversation(
        cls, conversation: Conversation, message_count: int
    ) -> "ConversationSummary":
        return cls(
            id=conversation.id,
            title=conversation.title,
            document_ids=conversation.document_ids,
            message_count=message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    total: int


class ChatResponse(BaseModel):
    conversation: ConversationSummary
    user_message: MessageOut
    assistant_message: MessageOut
