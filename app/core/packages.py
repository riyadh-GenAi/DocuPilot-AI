"""Central package configuration for DocuPilot AI tiers."""

from __future__ import annotations

from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel


MIME_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip", "application/octet-stream"}),
    ".doc": frozenset({"application/msword", "application/vnd.ms-word", "application/octet-stream"}),
    ".txt": frozenset({"text/plain", "application/octet-stream"}),
    ".md": frozenset({"text/markdown", "text/plain", "application/markdown", "application/octet-stream"}),
    ".rtf": frozenset({"application/rtf", "text/rtf", "text/plain", "application/msword", "application/octet-stream"}),
    ".csv": frozenset({"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain", "application/octet-stream"}),
    ".xlsx": frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/zip", "application/octet-stream"}),
    ".xls": frozenset({"application/vnd.ms-excel", "application/msexcel", "application/octet-stream"}),
    ".pptx": frozenset({"application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/zip", "application/octet-stream"}),
    ".jpg": frozenset({"image/jpeg", "image/pjpeg", "application/octet-stream"}),
    ".jpeg": frozenset({"image/jpeg", "image/pjpeg", "application/octet-stream"}),
    ".png": frozenset({"image/png", "application/octet-stream"}),
    ".tiff": frozenset({"image/tiff", "image/x-tiff", "application/octet-stream"}),
    ".tif": frozenset({"image/tiff", "image/x-tiff", "application/octet-stream"}),
    ".zip": frozenset({"application/zip", "application/x-zip-compressed", "application/octet-stream"}),
    ".dwg": frozenset({"image/vnd.dwg", "application/acad", "application/x-acad", "application/octet-stream"}),
    ".rvt": frozenset({"application/octet-stream", "application/x-rvt"}),
    ".ifc": frozenset({"application/octet-stream", "text/plain", "model/ifc", "application/x-step"}),
}

ALL_SUPPORTED_EXTENSIONS = frozenset(MIME_TYPES)


PackageName = Literal["basic", "standard", "premium"]


class PackageConfig(BaseModel):
    name: PackageName
    max_documents: int
    max_upload_mb: int
    allowed_extensions: set[str]
    features: frozenset[str]


# Package limits and file support are intentionally centralized here.
# Changing PACKAGE in .env changes feature access; it does not create a separate app.
PACKAGES: dict[PackageName, PackageConfig] = {
    "basic": PackageConfig(
        name="basic",
        max_documents=10,
        max_upload_mb=10,
        allowed_extensions={".pdf", ".docx", ".txt", ".md"},
        features=frozenset({"document_upload", "basic_qa", "basic_retrieval"}),
    ),
    "standard": PackageConfig(
        name="standard",
        max_documents=100,
        max_upload_mb=20,
        allowed_extensions={".pdf", ".docx", ".txt", ".md", ".rtf", ".csv", ".xlsx"},
        features=frozenset({
            "document_upload", "basic_qa", "basic_retrieval", "document_library",
            "metadata", "library_search", "download", "delete", "moderate_workflow",
        }),
    ),
    "premium": PackageConfig(
        name="premium",
        max_documents=500,
        max_upload_mb=50,
        allowed_extensions={
            ".pdf", ".docx", ".doc", ".txt", ".rtf", ".csv", ".xlsx", ".xls",
            ".dwg", ".rvt", ".ifc", ".jpg", ".jpeg", ".png", ".tiff", ".tif",
            ".pptx", ".zip",
        },
        features=frozenset({
            "document_upload", "basic_qa", "basic_retrieval", "document_library",
            "metadata", "library_search", "download", "delete", "moderate_workflow",
            "rich_metadata", "portfolio_organization", "advanced_filters",
            "bulk_actions", "premium_export",
        }),
    ),
}


def get_package(package_name: str) -> PackageConfig:
    normalized = package_name.strip().lower()
    if normalized not in PACKAGES:
        raise ValueError(f"Unsupported package '{package_name}'. Use basic, standard, or premium.")
    return PACKAGES[normalized]  # type: ignore[return-value]


def validate_file_for_package(
    package: PackageConfig,
    filename: str,
    content_type: str | None,
    size_bytes: int,
    effective_limit_mb: int | None = None,
) -> str:
    """Return one clear validation failure, or an empty string when valid."""
    from pathlib import Path

    if size_bytes <= 0:
        return "The selected file is empty and cannot be indexed."

    suffix = Path(filename).suffix.lower()
    if not suffix:
        return "Unsupported file format: the filename has no extension."
    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        return f"Unsupported file format: '{suffix}' is not supported by DocuPilot AI."
    if suffix not in package.allowed_extensions:
        eligible_packages = [
            candidate
            for candidate in PACKAGES.values()
            if suffix in candidate.allowed_extensions
        ]
        next_package = eligible_packages[0].name.title()
        return (
            f"Your {package.name.title()} package does not include "
            f"{suffix[1:].upper()} uploads. "
            f"Choose {next_package} to upload this file."
        )

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    allowed_mimes = MIME_TYPES.get(suffix)
    if mime and mime != "application/octet-stream" and allowed_mimes and mime not in allowed_mimes:
        return f"MIME type '{mime}' does not match the {suffix} file extension. Unsupported file format."

    upload_limit = effective_limit_mb or package.max_upload_mb
    if size_bytes > upload_limit * 1024 * 1024:
        return f"File size exceeds the {package.name.title()} package limit of {upload_limit} MB."
    return ""


def require_feature(package: PackageConfig, feature: str) -> None:
    if feature not in package.features:
        raise HTTPException(
            status_code=403,
            detail=f"The {package.name.title()} package does not include '{feature}'. Upgrade to Standard or Premium as applicable.",
        )
