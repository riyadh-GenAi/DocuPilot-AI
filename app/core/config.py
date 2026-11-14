"""Environment-backed application settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DocuPilot AI"
    debug: bool = False

    # Set PACKAGE to basic, standard, or premium.
    # It controls feature access, allowed file types, upload/document limits, and premium-only actions.
    # Basic = minimal document workflow; Standard = complete library workflow; Premium = extended capacity and management.
    package: Literal["basic", "standard", "premium"] = "standard"

    database_url: str = "postgresql+psycopg2://docupilot:docupilot@localhost:5432/docupilot_ai"
    upload_dir: Path = Path("./data/uploads")
    llm_provider: Literal["openai", "groq"] = "openai"
    llm_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    groq_api_key: str | None = None
    model_name: str = "gpt-4o-mini"
    embedding_backend: Literal["local", "provider"] = "local"
    embedding_fallback_local: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=384, ge=1)
    local_embeddings_enabled: bool = True
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_dimensions: int = Field(default=384, ge=1)
    local_model_cache_dir: Path | None = None
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    max_upload_size_mb: int = Field(default=20, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("local_model_cache_dir", mode="before")
    @classmethod
    def empty_cache_directory(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if (
            self.local_embeddings_enabled
            and self.local_embedding_dimensions != self.embedding_dimensions
        ):
            raise ValueError(
                "LOCAL_EMBEDDING_DIMENSIONS must equal EMBEDDING_DIMENSIONS"
            )
        return self


settings = Settings()
