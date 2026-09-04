from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    datetime.utcnow() is deprecated in modern Python and returns a naive
    datetime, which causes subtle comparison bugs. Always store UTC.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Shared declarative base. Every model inherits from this."""


class Document(Base):
    """One ingested source file.

    Tracks what's in the vector store so the API can list documents,
    show ingestion times, and delete a document's chunks cleanly.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Unique so re-ingesting the same file updates rather than duplicates.
    source: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    file_type: Mapped[str] = mapped_column(String(16))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    # Content hash. Lets ingestion skip files whose bytes are unchanged
    # instead of re-embedding them - the expensive part of ingestion.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


class Conversation(Base):
    """A chat session, grouping messages together."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(256), default="New conversation")
    mode: Mapped[str] = mapped_column(String(16), default="auto")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # cascade="all, delete-orphan": deleting a conversation deletes its
    # messages rather than leaving orphaned rows behind.
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    """A single turn within a conversation.

    Note: distinct from llm_providers.Message. That one is a transport
    dataclass; this one is a database row with an id and a timestamp.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)

    # Which brain produced this, recorded per message because it can
    # vary between turns in one conversation.
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
