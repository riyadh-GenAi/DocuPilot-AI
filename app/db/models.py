"""SQLAlchemy models for uploaded documents and their embedded text chunks."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_name: Mapped[str] = mapped_column(String(255), nullable=False)
    property_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    unit: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(100), default="General", nullable=False)
    note: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    issue_type: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    portfolio_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    property_code: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    tags: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    workflow_status: Mapped[str] = mapped_column(String(100), default="Active", nullable=False)
    date: Mapped[date] = mapped_column("document_date", Date, nullable=False)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(255))
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimensions), nullable=False
    )
    document: Mapped[Document] = relationship(back_populates="chunks")
