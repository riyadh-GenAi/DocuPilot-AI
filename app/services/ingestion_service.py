"""Document validation, text extraction, chunking, and vector persistence."""

import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.packages import get_package, require_feature, validate_file_for_package
from sqlalchemy import func, select
from app.db.models import Document, DocumentChunk
from app.schemas import DocumentCreate
from app.services.chunking import chunk_text
from app.services.file_storage import remove_stored_file, save_uploaded_file
from app.services.llm_service import create_embeddings
from app.services.text_extraction import extract_text

logger = logging.getLogger(__name__)


async def ingest_document(
    file: UploadFile,
    metadata: DocumentCreate,
    db: Session,
) -> Document:
    package = get_package(settings.package)
    require_feature(package, "document_upload")
    filename = file.filename or metadata.source_file
    suffix = Path(filename).suffix.lower()
    current_count = db.scalar(select(func.count()).select_from(Document)) or 0
    if current_count >= package.max_documents:
        raise HTTPException(
            status_code=403,
            detail=f"The {package.name.title()} package document limit of {package.max_documents} has been reached.",
        )
    upload_limit = min(settings.max_upload_size_mb, package.max_upload_mb)
    content = await file.read(upload_limit * 1024 * 1024 + 1)
    validation_error = validate_file_for_package(
        package=package,
        filename=filename,
        content_type=file.content_type,
        size_bytes=len(content),
        effective_limit_mb=upload_limit,
    )
    if validation_error:
        if validation_error.startswith("File size"):
            status = 413
        elif "package does not include" in validation_error:
            status = 403
        elif validation_error.startswith("The selected file is empty"):
            status = 422
        else:
            status = 415
        raise HTTPException(status_code=status, detail=validation_error)

    try:
        text = extract_text(filename, content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The {suffix.lstrip('.').upper()} file could not be read or extracted. "
                "It may be damaged, encrypted, or use unsupported features. "
                "Verify that it opens correctly, then upload a readable copy."
            ),
        ) from exc
    if len(content) > upload_limit * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {package.name.title()} package limit of {upload_limit} MB.",
        )
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    embeddings = create_embeddings(chunks)
    try:
        storage_path = save_uploaded_file(filename, content)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="The document passed validation, but its original file could not be saved.",
        ) from exc

    try:
        document = Document(
            document_name=metadata.document_name or filename,
            property_name=metadata.property_name,
            unit=metadata.unit,
            document_type=metadata.document_type,
            category=metadata.category,
            note=metadata.note,
            issue_type=metadata.issue_type,
            portfolio_name=metadata.portfolio_name if package.name == "premium" else "",
            property_code=metadata.property_code if package.name == "premium" else "",
            tags=metadata.tags if package.name == "premium" else "",
            workflow_status=metadata.workflow_status if package.name == "premium" else "Active",
            date=metadata.date,
            source_file=filename,
            storage_path=storage_path,
            content_type=file.content_type or "application/octet-stream",
        )
        document.chunks = [
            DocumentChunk(chunk_index=index, content=chunk, embedding=embedding)
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        db.add(document)
        db.flush()
        db.refresh(document)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        remove_stored_file(storage_path)
        raise HTTPException(
            status_code=500, detail="The document could not be saved to the library."
        ) from exc
    except ValueError as exc:
        db.rollback()
        remove_stored_file(storage_path)
        if "NUL" in str(exc):
            raise HTTPException(
                status_code=422,
                detail=(
                    "The extracted document text contains characters that cannot be "
                    "stored. Upload a clean, text-readable copy of the document."
                ),
            ) from exc
        logger.exception("Document indexing failed during database persistence.")
        raise HTTPException(
            status_code=500,
            detail=(
                "The document could not be indexed or saved. No document was added; "
                "check the API logs or contact support before retrying."
            ),
        ) from exc
    except Exception as exc:
        db.rollback()
        remove_stored_file(storage_path)
        logger.exception("Document indexing failed during database persistence.")
        raise HTTPException(
            status_code=500,
            detail=(
                "The document could not be indexed or saved. No document was added; "
                "check the API logs or contact support before retrying."
            ),
        ) from exc
    return document
