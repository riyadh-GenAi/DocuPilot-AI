"""Database engine, declarative base, and request-scoped sessions."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database() -> None:
    """Create the pgvector extension and MVP tables if they do not exist."""
    from app.db import models

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(
            bind=connection,
            tables=[models.Document.__table__, models.DocumentChunk.__table__],
        )
        columns = {
            column["name"] for column in inspect(connection).get_columns("documents")
        }
        additions = {
            "document_name": "VARCHAR(255) NOT NULL DEFAULT ''",
            "category": "VARCHAR(100) NOT NULL DEFAULT 'General'",
            "note": "VARCHAR(1000) NOT NULL DEFAULT ''",
            "storage_path": "VARCHAR(255)",
            "content_type": "VARCHAR(255)",
            "portfolio_name": "VARCHAR(200) NOT NULL DEFAULT ''",
            "property_code": "VARCHAR(100) NOT NULL DEFAULT ''",
            "tags": "VARCHAR(500) NOT NULL DEFAULT ''",
            "workflow_status": "VARCHAR(100) NOT NULL DEFAULT 'Active'",
        }
        for column_name, definition in additions.items():
            if column_name not in columns:
                connection.execute(
                    text(
                        f"ALTER TABLE documents ADD COLUMN {column_name} {definition}"
                    )
                )
        connection.execute(
            text(
                "UPDATE documents SET document_name = source_file "
                "WHERE document_name = ''"
            )
        )

        if inspect(connection).has_table("document_chunks"):
            vector_type = connection.execute(
                text("""
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public'
                      AND c.relname = 'document_chunks'
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                """)
            ).scalar()
            expected_type = f"vector({settings.embedding_dimensions})"
            if vector_type and vector_type != expected_type:
                connection.execute(text("DELETE FROM document_chunks"))
                connection.execute(
                    text(
                        "ALTER TABLE document_chunks "
                        "ALTER COLUMN embedding TYPE " + expected_type
                    )
                )


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for one API request."""
    with SessionLocal() as session:
        yield session
