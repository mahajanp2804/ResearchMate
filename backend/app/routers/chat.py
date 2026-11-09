from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.dependencies import DbSession, RagDep, RetrievalDep
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
    MessageOut,
)
from app.services import chat_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
def query(
    request: ChatRequest, db: DbSession, rag: RagDep, retrieval: RetrievalDep
) -> ChatResponse:
    """Ask a question about the uploaded papers, optionally continuing a conversation.

    The answer is generated only from retrieved passages, which are returned
    as numbered sources. When the papers hold no relevant evidence the reply
    is a fixed refusal produced without calling the LLM.
    """
    conversation, user_message, assistant_message = chat_service.ask(
        db,
        rag,
        retrieval,
        question=request.question,
        conversation_id=request.conversation_id,
        document_ids=request.document_ids,
    )
    return ChatResponse(
        conversation=ConversationSummary.from_conversation(
            conversation, len(conversation.messages)
        ),
        user_message=MessageOut.from_message(user_message),
        assistant_message=MessageOut.from_message(assistant_message),
    )


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationListResponse:
    rows, total = chat_service.list_conversations(db, limit=limit, offset=offset)
    return ConversationListResponse(
        items=[ConversationSummary.from_conversation(conv, count) for conv, count in rows],
        total=total,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: DbSession) -> ConversationDetail:
    conversation = chat_service.get_conversation(db, conversation_id)
    summary = ConversationSummary.from_conversation(conversation, len(conversation.messages))
    return ConversationDetail(
        **summary.model_dump(),
        messages=[MessageOut.from_message(message) for message in conversation.messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, db: DbSession) -> Response:
    chat_service.delete_conversation(db, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
