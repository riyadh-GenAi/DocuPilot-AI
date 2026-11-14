"""HTTP endpoints for document upload, library management, and Q&A."""

from datetime import date as Date

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from io import StringIO
import csv
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Document, DocumentChunk
from app.schemas import (
    DocumentCreate,
    DocumentDetailResponse,
    DocumentResponse,
    DocumentSearchOption,
    QueryRequest,
    QueryResponse,
    ReviewRequest,
)
from app.services.file_storage import get_stored_file, remove_stored_file
from app.core.config import settings
from app.core.packages import (
    get_package,
    require_feature,
)
from app.services.ingestion_service import ingest_document
from app.services.query_service import query_document
from app.services.review import REVIEW_MODES
from app.services.text_extraction import (
    CSV_MAX_CELL_CHARS,
    CSV_MAX_COLUMNS,
    CSV_MAX_EXTRACTED_CHARS,
    CSV_MAX_ROWS,
)

router = APIRouter()


@router.get("/package")
def package_info() -> dict:
    package = get_package(settings.package)
    return {
        "name": package.name,
        "max_documents": package.max_documents,
        "max_upload_mb": package.max_upload_mb,
        "effective_upload_mb": min(settings.max_upload_size_mb, package.max_upload_mb),
        "csv_limits": {
            "rows": CSV_MAX_ROWS,
            "columns": CSV_MAX_COLUMNS,
            "cell_characters": CSV_MAX_CELL_CHARS,
            "extracted_characters": CSV_MAX_EXTRACTED_CHARS,
        },
        "allowed_extensions": sorted(package.allowed_extensions),
        "features": sorted(package.features),
    }


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    property_name: str = Form("", max_length=200),
    unit: str = Form("", max_length=100),
    document_type: str = Form("", max_length=100),
    category: str = Form("General", min_length=1, max_length=100),
    document_name: str | None = Form(None, max_length=255),
    note: str = Form("", max_length=1000),
    issue_type: str = Form("", max_length=100),
    portfolio_name: str = Form("", max_length=200),
    property_code: str = Form("", max_length=100),
    tags: str = Form("", max_length=500),
    workflow_status: str = Form("Active", max_length=100),
    date: Date | None = Form(None),
    db: Session = Depends(get_db),
) -> Document:
    filename = file.filename or ""
    if not filename or len(filename) > 255:
        raise HTTPException(
            status_code=422, detail="A filename of at most 255 characters is required."
        )
    package = get_package(settings.package)
    rich_values_used = any((portfolio_name.strip(), property_code.strip(), tags.strip(), workflow_status.strip() != "Active"))
    if rich_values_used:
        require_feature(package, "rich_metadata")

    metadata = DocumentCreate(
        document_name=(
            document_name.strip()
            if document_name and document_name.strip()
            else filename
        ),
        property_name=property_name.strip(),
        unit=unit.strip(),
        document_type=document_type.strip() or "Other",
        category=category.strip(),
        note=note.strip(),
        issue_type=issue_type.strip(),
        portfolio_name=portfolio_name.strip(),
        property_code=property_code.strip(),
        tags=tags.strip(),
        workflow_status=workflow_status.strip() or "Active",
        date=date or Date.today(),
        source_file=filename,
    )
    return await ingest_document(file, metadata, db)


@router.post("/ask", response_model=QueryResponse)
def ask_question(
    request: QueryRequest, db: Session = Depends(get_db)
) -> QueryResponse:
    package = get_package(settings.package)
    require_feature(package, "basic_qa")
    premium_filters = any((request.issue_type, request.portfolio_name, request.property_code, request.workflow_status, request.date_from, request.date_to))
    if premium_filters:
        require_feature(package, "advanced_filters")
    return query_document(request, db)


@router.post("/review", response_model=QueryResponse)
def review_project(
    request: ReviewRequest, db: Session = Depends(get_db)
) -> QueryResponse:
    package = get_package(settings.package)
    require_feature(package, "basic_qa")
    retrieval_query = REVIEW_MODES[request.review_type].retrieval_query
    if request.review_type == "compliance" and request.checklist.strip():
        retrieval_query = f"{retrieval_query} {request.checklist.strip()}"
    query = QueryRequest(
        question=retrieval_query,
        property_name=request.project_name,
    )
    return query_document(
        query,
        db,
        review_type=request.review_type,
        checklist=request.checklist,
    )


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    search: str | None = Query(default=None, max_length=200),
    property_name: str | None = Query(default=None, max_length=200),
    unit: str | None = Query(default=None, max_length=100),
    category: str | None = Query(default=None, max_length=100),
    document_type: str | None = Query(default=None, max_length=100),
    issue_type: str | None = Query(default=None, max_length=100),
    portfolio_name: str | None = Query(default=None, max_length=200),
    property_code: str | None = Query(default=None, max_length=100),
    workflow_status: str | None = Query(default=None, max_length=100),
    date_from: Date | None = Query(default=None),
    date_to: Date | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[Document]:
    package = get_package(settings.package)
    require_feature(package, "document_library")
    premium_filters = any((issue_type, portfolio_name, property_code, workflow_status, date_from, date_to))
    if premium_filters:
        require_feature(package, "advanced_filters")
    statement = select(Document).order_by(Document.uploaded_at.desc())
    filters = (
        (Document.property_name, property_name),
        (Document.unit, unit),
        (Document.category, category),
        (Document.document_type, document_type),
    )
    for column, value in filters:
        if value and value.strip():
            statement = statement.where(column.ilike(value.strip()))

    premium_filters_exact = (
        (Document.issue_type, issue_type),
        (Document.portfolio_name, portfolio_name),
        (Document.property_code, property_code),
        (Document.workflow_status, workflow_status),
    )
    for column, value in premium_filters_exact:
        if value and value.strip():
            statement = statement.where(column.ilike(value.strip()))
    if date_from:
        statement = statement.where(Document.date >= date_from)
    if date_to:
        statement = statement.where(Document.date <= date_to)

    if search and search.strip():
        pattern = f"%{search.strip()}%"
        statement = statement.where(
            or_(
                Document.document_name.ilike(pattern),
                Document.note.ilike(pattern),
                Document.source_file.ilike(pattern),
                Document.property_name.ilike(pattern),
                Document.unit.ilike(pattern),
                Document.category.ilike(pattern),
                Document.document_type.ilike(pattern),
                Document.issue_type.ilike(pattern),
                Document.portfolio_name.ilike(pattern),
                Document.property_code.ilike(pattern),
                Document.tags.ilike(pattern),
                Document.workflow_status.ilike(pattern),
                Document.chunks.any(DocumentChunk.content.ilike(pattern)),
            )
        )
    return list(db.scalars(statement).all())


@router.get(
    "/documents/search-options", response_model=list[DocumentSearchOption]
)
def list_document_search_options(
    property_name: str | None = Query(default=None, max_length=200),
    db: Session = Depends(get_db),
) -> list[DocumentSearchOption]:
    """Return minimal document labels for the package-independent Q&A selector."""
    package = get_package(settings.package)
    require_feature(package, "basic_qa")
    statement = select(
        Document.id, Document.document_name, Document.source_file
    ).order_by(Document.uploaded_at.desc())
    if property_name and property_name.strip():
        statement = statement.where(Document.property_name == property_name.strip())
    rows = db.execute(statement).all()
    return [
        DocumentSearchOption(
            id=row.id,
            document_name=row.document_name,
            source_file=row.source_file,
        )
        for row in rows
    ]


@router.get("/documents/export/csv")
def export_library_csv(db: Session = Depends(get_db)) -> StreamingResponse:
    require_feature(get_package(settings.package), "premium_export")
    documents = list(db.scalars(select(Document).order_by(Document.uploaded_at.desc())).all())
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["document_name", "portfolio_name", "property_name", "property_code", "unit", "document_type", "category", "issue_type", "tags", "workflow_status", "date"])
    for document in documents:
        writer.writerow([document.document_name, document.portfolio_name, document.property_name, document.property_code, document.unit, document.document_type, document.category, document.issue_type, document.tags, document.workflow_status, document.date.isoformat()])
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=docupilot-library.csv"})


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    require_feature(get_package(settings.package), "document_library")
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.get("/documents/{document_id}/file")
def download_document(document_id: int, db: Session = Depends(get_db)) -> FileResponse:
    require_feature(get_package(settings.package), "download")
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        path = get_stored_file(document.storage_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="The original document file is not available."
        ) from exc

    return FileResponse(
        path,
        media_type=document.content_type or "application/octet-stream",
        filename=document.source_file,
        content_disposition_type="attachment",
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db)) -> None:
    require_feature(get_package(settings.package), "delete")
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    storage_path = document.storage_path
    db.delete(document)
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="The document could not be deleted."
        ) from exc

    try:
        remove_stored_file(storage_path)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="The document record was deleted, but its file could not be removed.",
        ) from exc
