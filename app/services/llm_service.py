"""Backend-only OpenAI-compatible embedding and chat calls."""

from collections.abc import Sequence

from fastapi import HTTPException
from openai import APIError, OpenAI, RateLimitError

from app.core.config import settings
from app.core.prompts import PROPERTY_MANAGEMENT_SYSTEM_PROMPT

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"


def _client() -> OpenAI:
    if settings.llm_provider == "groq":
        api_key = settings.groq_api_key
        base_url = GROQ_API_BASE_URL
    else:
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url or DEFAULT_API_BASE_URL

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=f"AI features are unavailable. Configure the {settings.llm_provider} API key on the backend.",
        )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30,
        max_retries=1,
    )


def create_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Select embedding backend from configuration while preserving provider fallback."""
    if settings.embedding_backend == "local":
        from app.services.local_embedding_service import create_local_embeddings
        return create_local_embeddings(texts)
    return _create_provider_embeddings(texts)


def _create_provider_embeddings(texts: Sequence[str]) -> list[list[float]]:
    provider_error: HTTPException | None = None
    try:
        response = _client().embeddings.create(
            model=settings.embedding_model,
            input=list(texts),
        )
        embeddings = [item.embedding for item in response.data]
        if len(embeddings) == len(texts) and all(
            len(vector) == settings.embedding_dimensions for vector in embeddings
        ):
            return embeddings
        provider_error = HTTPException(
            status_code=502,
            detail="The embedding provider returned an unexpected vector size; trying local embeddings.",
        )
    except RateLimitError as exc:
        provider_error = HTTPException(
            status_code=429,
            detail="The AI provider is rate-limiting embedding requests. Trying local embeddings.",
            headers={"Retry-After": "30"},
        )
    except APIError as exc:
        status_code = getattr(exc, "status_code", None)
        provider_error = HTTPException(
            status_code=502,
            detail=(
                f"The {settings.llm_provider} provider rejected the embeddings request "
                f"(HTTP {status_code}). Trying local embeddings."
            ),
        )
    except HTTPException as exc:
        provider_error = exc

    if settings.local_embeddings_enabled and settings.embedding_fallback_local:
        from app.services.local_embedding_service import create_local_embeddings

        try:
            return create_local_embeddings(texts)
        except HTTPException as local_error:
            if provider_error and provider_error.status_code == 429:
                raise provider_error from local_error
            raise HTTPException(
                status_code=503,
                detail=(
                    "No embedding provider is available. Provider error: "
                    f"{provider_error.detail if provider_error else 'unknown error'}. "
                    "Local model error: "
                    f"{local_error.detail}"
                ),
            ) from local_error

    raise provider_error or HTTPException(
        status_code=503,
        detail="No embedding provider or local fallback is available.",
    )


def generate_answer(
    question: str,
    context: str,
    history: Sequence[tuple[str, str]] = (),
) -> str:
    messages = [
        {
            "role": "system",
            "content": PROPERTY_MANAGEMENT_SYSTEM_PROMPT,
        },
        *(
            {"role": role, "content": content}
            for role, content in history
        ),
        {
            "role": "user",
            "content": f"Document excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]
    try:
        response = _client().chat.completions.create(
            model=settings.llm_model or settings.model_name,
            temperature=0,
            messages=messages,
        )
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="The AI provider is rate-limiting requests. Please retry shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    except APIError as exc:
        status_code = getattr(exc, "status_code", None)
        detail = (
            f"The {settings.llm_provider} provider rejected the chat request "
            f"(HTTP {status_code}); verify the configured chat model."
            if status_code is not None
            else f"The {settings.llm_provider} provider could not generate an answer."
        )
        raise HTTPException(
            status_code=502, detail=detail
        ) from exc

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise HTTPException(status_code=502, detail="The AI provider returned an empty answer.")
    return answer.strip()
