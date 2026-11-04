from datetime import UTC

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    ActivityEvent,
    ActivityType,
    Chunk,
    Conversation,
    Document,
    IndexingStatus,
    Message,
    MessageRole,
    ProcessingStatus,
)


def make_document(file_hash: str = "a" * 64, **overrides) -> Document:
    values = {
        "filename": "paper.pdf",
        "stored_path": f"{file_hash[:8]}.pdf",
        "file_hash": file_hash,
        "file_size_bytes": 1024,
        "title": "A Paper",
        "page_count": 2,
    }
    values.update(overrides)
    return Document(**values)


def make_chunk(index: int, page: int = 1) -> Chunk:
    text = f"chunk {index} text"
    return Chunk(
        chunk_index=index,
        page_start=page,
        page_end=page,
        text=text,
        token_count=3,
        char_count=len(text),
    )


def test_document_defaults(db_session: Session) -> None:
    document = make_document()
    db_session.add(document)
    db_session.commit()

    assert len(document.id) == 32
    assert document.processing_status is ProcessingStatus.PENDING
    assert document.indexing_status is IndexingStatus.NOT_INDEXED
    assert document.extracted_metadata == {}
    assert document.upload_date.tzinfo is UTC


def test_status_is_stored_as_its_value(db_session: Session) -> None:
    db_session.add(make_document(indexing_status=IndexingStatus.INDEXED))
    db_session.commit()
    raw = db_session.connection().exec_driver_sql("SELECT indexing_status FROM documents")
    assert raw.scalar_one() == "indexed"


def test_duplicate_file_hash_is_rejected(db_session: Session) -> None:
    db_session.add(make_document())
    db_session.commit()
    db_session.add(make_document(filename="copy.pdf"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_deleting_document_cascades_to_chunks_at_database_level(db_session: Session) -> None:
    document = make_document()
    document.chunks = [make_chunk(0), make_chunk(1, page=2)]
    db_session.add(document)
    db_session.commit()

    # A bulk SQL delete bypasses the ORM cascade, proving the foreign key does it.
    db_session.connection().exec_driver_sql("DELETE FROM documents")
    db_session.commit()
    assert db_session.scalars(select(Chunk)).all() == []


def test_deleting_document_keeps_activity_history(db_session: Session) -> None:
    document = make_document()
    db_session.add(document)
    db_session.flush()
    db_session.add(
        ActivityEvent(
            event_type=ActivityType.DOCUMENT_UPLOADED,
            description="Uploaded A Paper",
            document_id=document.id,
        )
    )
    db_session.commit()

    db_session.delete(document)
    db_session.commit()
    db_session.expire_all()

    event = db_session.scalars(select(ActivityEvent)).one()
    assert event.document_id is None


def test_chunk_ids_are_never_reused(db_session: Session) -> None:
    first = make_document()
    first.chunks = [make_chunk(0), make_chunk(1)]
    db_session.add(first)
    db_session.commit()
    highest_id = max(chunk.id for chunk in first.chunks)

    db_session.delete(first)
    db_session.commit()

    second = make_document(file_hash="b" * 64)
    second.chunks = [make_chunk(0)]
    db_session.add(second)
    db_session.commit()
    # Chunk ids are FAISS vector ids, so reuse could resurrect a stale vector.
    assert second.chunks[0].id > highest_id


def test_chunk_index_unique_per_document(db_session: Session) -> None:
    document = make_document()
    document.chunks = [make_chunk(0), make_chunk(0)]
    db_session.add(document)
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_conversation_messages_are_ordered_and_cascade(db_session: Session) -> None:
    conversation = Conversation(title="FedAvg vs FedProx", document_ids=["doc-1"])
    conversation.messages = [
        Message(role=MessageRole.USER, content="What is FedProx?"),
        Message(role=MessageRole.ASSISTANT, content="FedProx adds a proximal term [S1]."),
    ]
    db_session.add(conversation)
    db_session.commit()

    assert [message.role for message in conversation.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert conversation.messages[1].citations == []

    db_session.delete(conversation)
    db_session.commit()
    assert db_session.scalars(select(Message)).all() == []
