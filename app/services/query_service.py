"""Semantic document retrieval and citation-grounded answer generation."""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, DocumentChunk
from app.schemas import QueryRequest, QueryResponse, SourceCitation
from app.services.llm_service import create_embeddings, generate_answer
from app.services.review import ReviewType, build_review_prompt


def query_document(
    request: QueryRequest,
    db: Session,
    review_type: ReviewType | None = None,
    checklist: str = "",
) -> QueryResponse:
    query_vector = create_embeddings([request.question])[0]
    statement = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
        .limit(settings.retrieval_top_k)
    )
    filters = (
        (Document.property_name, request.property_name),
        (Document.unit, request.unit),
        (Document.category, request.category),
        (Document.document_type, request.document_type),
    )
    for column, value in filters:
        if value:
            statement = statement.where(column == value)
    if request.document_id is not None:
        statement = statement.where(Document.id == request.document_id)
    premium_filters = (
        (Document.issue_type, request.issue_type),
        (Document.portfolio_name, request.portfolio_name),
        (Document.property_code, request.property_code),
        (Document.workflow_status, request.workflow_status),
    )
    for column, value in premium_filters:
        if value:
            statement = statement.where(column == value)
    if request.date_from:
        statement = statement.where(Document.date >= request.date_from)
    if request.date_to:
        statement = statement.where(Document.date <= request.date_to)

    matches = db.execute(statement).all()
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                "No matching documents were found. Upload a document or adjust the "
                "metadata filters."
            ),
        )

    citations: list[SourceCitation] = []
    passages: list[str] = []
    for citation_number, (chunk, document) in enumerate(matches, start=1):
        citations.append(
            SourceCitation(
                citation=citation_number,
                document_id=document.id,
                document_name=document.document_name,
                source_file=document.source_file,
                property_name=document.property_name,
                unit=document.unit,
                document_type=document.document_type,
                category=document.category,
                issue_type=document.issue_type,
                portfolio_name=document.portfolio_name,
                property_code=document.property_code,
                tags=document.tags,
                workflow_status=document.workflow_status,
                date=document.date,
                excerpt=chunk.content[:500],
            )
        )
        passages.append(f"[{citation_number}] {chunk.content}")

    question = (
        build_review_prompt(review_type, checklist)
        if review_type is not None
        else request.question
    )
    answer = generate_answer(
        question,
        "\n\n".join(passages),
        [(message.role, message.content) for message in request.history],
    )
    return QueryResponse(answer=answer, sources=citations)
