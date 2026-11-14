"""Helpers for selecting a stable document scope in the Q&A interface."""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


ALL_DOCUMENTS: None = None


def document_selection_options(
    documents: Sequence[Mapping[str, Any]],
) -> list[int | None]:
    identifiers = list(dict.fromkeys(int(document["id"]) for document in documents))
    return [ALL_DOCUMENTS, *identifiers]


def document_selection_label(
    document_id: int | None, documents: Sequence[Mapping[str, Any]]
) -> str:
    if document_id is None:
        return "All uploaded documents"
    document = next(
        (item for item in documents if int(item["id"]) == document_id), None
    )
    if document is None:
        return "All uploaded documents"
    name = str(document.get("document_name") or document.get("source_file") or "Document")
    filename = str(document.get("source_file") or "")
    return f"{name} ({filename})" if filename and filename != name else name


def synchronize_document_selection(
    state: MutableMapping[str, Any],
    project: str,
    documents: Sequence[Mapping[str, Any]],
) -> int | None:
    """Reset the widget value when its project or available document set changes."""
    project_key = "ask_document_project"
    selection_key = "ask_document_id"
    available_ids = {int(document["id"]) for document in documents}
    selected_id = state.get(selection_key)
    if state.get(project_key) != project or selected_id not in available_ids | {None}:
        selected_id = ALL_DOCUMENTS
        state[selection_key] = selected_id
    state[project_key] = project
    return selected_id


def question_search_scope(
    project: str, document_id: int | None, documents: Sequence[Mapping[str, Any]]
) -> str:
    document_label = document_selection_label(document_id, documents)
    if project:
        return f"{document_label} · {project}"
    return document_label
