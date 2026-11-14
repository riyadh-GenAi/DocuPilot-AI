"""Pydantic request and response models."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    document_name: str | None = Field(default=None, min_length=1, max_length=255)
    property_name: str = Field(default="", max_length=200)
    unit: str = Field(default="", max_length=100)
    document_type: str = Field(default="Other", max_length=100)
    category: str = Field(default="General", min_length=1, max_length=100)
    note: str = Field(default="", max_length=1000)
    issue_type: str = Field(default="", max_length=100)
    portfolio_name: str = Field(default="", max_length=200)
    property_code: str = Field(default="", max_length=100)
    tags: str = Field(default="", max_length=500)
    workflow_status: str = Field(default="Active", max_length=100)
    date: date
    source_file: str = Field(min_length=1, max_length=255)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_name: str
    property_name: str
    unit: str
    document_type: str
    category: str
    note: str
    issue_type: str
    portfolio_name: str
    property_code: str
    tags: str
    workflow_status: str
    date: date
    source_file: str
    uploaded_at: datetime


class DocumentDetailResponse(DocumentResponse):
    content_type: str | None


class DocumentSearchOption(BaseModel):
    id: int
    document_name: str
    source_file: str


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=10)
    property_name: str | None = Field(default=None, max_length=200)
    unit: str | None = Field(default=None, max_length=100)
    category: str | None = Field(default=None, max_length=100)
    document_type: str | None = Field(default=None, max_length=100)
    issue_type: str | None = Field(default=None, max_length=100)
    portfolio_name: str | None = Field(default=None, max_length=200)
    property_code: str | None = Field(default=None, max_length=100)
    workflow_status: str | None = Field(default=None, max_length=100)
    date_from: date | None = None
    date_to: date | None = None
    document_id: int | None = Field(default=None, gt=0)


class ReviewRequest(BaseModel):
    review_type: Literal["risk", "cost_timeline", "compliance"]
    project_name: str | None = Field(default=None, max_length=200)
    checklist: str = Field(default="", max_length=500)


class SourceCitation(BaseModel):
    citation: int
    document_id: int
    document_name: str
    source_file: str
    property_name: str
    unit: str
    document_type: str
    category: str
    issue_type: str
    portfolio_name: str
    property_code: str
    tags: str
    workflow_status: str
    date: date
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
