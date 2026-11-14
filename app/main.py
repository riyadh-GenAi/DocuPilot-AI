"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.core.config import settings
from app.db.database import initialize_database


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    initialize_database()
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DocuPilot AI API is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
