"""Local persistent storage for original uploaded documents."""

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import settings
from app.core.packages import ALL_SUPPORTED_EXTENSIONS


# Package access remains centralized in packages.py; storage validates the same technical allowlist.
SUPPORTED_EXTENSIONS = set(ALL_SUPPORTED_EXTENSIONS)


def save_uploaded_file(filename: str, content: bytes) -> str:
    """Save bytes under a generated name and return the relative storage path."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported document file type.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}{suffix}"
    destination = settings.upload_dir / stored_name
    try:
        destination.write_bytes(content)
    except OSError:
        destination.unlink(missing_ok=True)
        raise
    return stored_name


def _resolve_stored_path(storage_path: str | None) -> Path:
    if not storage_path or Path(storage_path).name != storage_path:
        raise FileNotFoundError("The original document file is not available.")

    root = settings.upload_dir.resolve()
    path = (root / storage_path).resolve()
    if path.parent != root:
        raise FileNotFoundError("The original document file is not available.")
    return path


def get_stored_file(storage_path: str | None) -> Path:
    """Resolve a generated storage name without allowing paths outside the library."""
    path = _resolve_stored_path(storage_path)
    if not path.is_file():
        raise FileNotFoundError("The original document file is not available.")
    return path


def remove_stored_file(storage_path: str | None) -> None:
    """Remove a stored original if it exists."""
    if storage_path is None:
        return
    path = _resolve_stored_path(storage_path)
    path.unlink(missing_ok=True)
