"""Local sentence-transformer embedding provider."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import HTTPException

from app.core.config import settings

_model = None


def create_local_embeddings(texts: Sequence[str]) -> list[list[float]]:
    if not settings.local_embeddings_enabled:
        raise HTTPException(
            status_code=503,
            detail="Local embeddings are disabled. Enable LOCAL_EMBEDDINGS_ENABLED on the backend.",
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local embeddings are unavailable. Install sentence-transformers on the backend.",
        ) from exc

    global _model
    try:
        if _model is None:
            cache_kwargs = (
                {"cache_folder": str(settings.local_model_cache_dir)}
                if settings.local_model_cache_dir
                else {}
            )
            _model = SentenceTransformer(settings.local_embedding_model, **cache_kwargs)
        vectors = _model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local embedding model could not be loaded or used: "
                f"{settings.local_embedding_model}"
            ),
        ) from exc

    embeddings = [vector.astype(float).tolist() for vector in vectors]
    expected = settings.embedding_dimensions
    if any(len(vector) != expected for vector in embeddings):
        raise HTTPException(
            status_code=503,
            detail=(
                "The local embedding model returned an unexpected vector size. "
                "Update EMBEDDING_DIMENSIONS and LOCAL_EMBEDDING_DIMENSIONS consistently."
            ),
        )
    return embeddings
