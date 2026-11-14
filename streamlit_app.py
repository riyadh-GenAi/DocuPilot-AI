"""Streamlit frontend for the DocuPilot AI API."""

import os
from typing import Any
import requests
import streamlit as st
from app.core.chat_history import (
    MAX_CHAT_QUESTION_CHARS,
    clear_conversation,
    conversation_display_pairs,
    conversation_messages,
    describe_chat_request_error,
    scope_conversation_to_project,
    submit_chat_turn,
)
from app.core.package_client import load_package_state
from app.core.document_scope import (
    document_selection_label,
    document_selection_options,
    question_search_scope,
    synchronize_document_selection,
)
from app.core.upload_errors import describe_request_error
from app.core.upload_batch import (
    UploadOutcome,
    is_duplicate_batch,
    upload_batch_fingerprint,
    upload_selected_files,
)
from app.core.review_export import build_review_markdown, review_report_filename
from app.services.review import (
    REVIEW_MODES,
    ReviewType,
    build_empty_review_report,
    ensure_report_sections,
)


API_URL = os.getenv("API_URL", "http://localhost:8000/api").rstrip("/")
REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = (10, 180)
DOCUMENT_TYPES = [
    "Lease",
    "Maintenance record",
    "Inspection report",
    "Work order",
    "Other",
]
CATEGORIES = ["General", "Lease", "Maintenance", "Inspection", "Work order", "Other"]

PACKAGE, PACKAGE_LOAD_ERROR = load_package_state(API_URL)
PACKAGE_NAME = PACKAGE["name"].title() if PACKAGE else "Package unavailable"
PACKAGE_FEATURES = set(PACKAGE.get("features", [])) if PACKAGE else set()
ALLOWED_EXTENSIONS = (
    [item.lstrip(".") for item in PACKAGE.get("allowed_extensions", [])]
    if PACKAGE
    else []
)

st.set_page_config(page_title="DocuPilot AI", page_icon="📄", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f4f7fb; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e4eaf2; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #253247; }
.block-container { padding-top: 1.35rem; padding-bottom: 1rem; max-width: 1120px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stForm"] { border: 1px solid #e1e7ef; border-radius: 10px; padding: 0.75rem 0.85rem; background: #ffffff; box-shadow: 0 1px 4px rgba(31, 41, 55, 0.035); }
[data-testid="stExpander"] { border: 1px solid #e4eaf2; border-radius: 8px; background: #ffffff; }
[data-testid="stTabs"] [role="tablist"] { gap: 0.35rem; border-bottom: 1px solid #e6ebf2; }
[data-testid="stTabs"] button[role="tab"] { flex: 1; justify-content: center; border: 1px solid #d8e1ed; border-radius: 8px; padding: 0.75rem 0.5rem; font-size: 1.05rem; font-weight: 650; background: #ffffff; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { border-color: #3478bd; background: #eaf2fb; color: #174f84; }
button[kind="primary"] { border-radius: 7px; min-height: 40px; }
[data-testid="stAlert"] { border-radius: 8px; padding-top: 0.55rem; padding-bottom: 0.55rem; }
[data-testid="stFileUploader"] { border-radius: 8px; }
@media (max-width: 640px) {
  [data-testid="stTabs"] button[role="tab"] { padding: 0.65rem 0.25rem; font-size: 0.92rem; }
}
</style>
""", unsafe_allow_html=True)
title_col, contact_col = st.columns([3, 2])
title_col.title("DocuPilot AI")
contact_col.markdown(
    '<div style="text-align: right; padding-top: 0.8rem; font-size: 1.15rem; '
    'font-weight: 600;">Mail -&gt; RiyadhGenAi@gmail.com</div>',
    unsafe_allow_html=True,
)
st.caption("Property records, inspections, leases, and maintenance documents in one workspace.")
st.sidebar.title("Active package")
if PACKAGE is None:
    st.sidebar.warning("Package unavailable")
else:
    st.sidebar.success(f"**{PACKAGE_NAME}**")
    if PACKAGE["name"] == "basic":
        st.sidebar.caption("Upload, Q&A, and grounded citations")
    elif PACKAGE["name"] == "standard":
        st.sidebar.caption("Library management, search, downloads, and deletion")
    else:
        st.sidebar.caption("Premium metadata, advanced filters, and library export")
st.sidebar.divider()
st.sidebar.caption("Supported uploads")
format_labels = [item if item.startswith(".") else f".{item}" for item in ALLOWED_EXTENSIONS]
st.sidebar.caption(" · ".join(format_labels))
if PACKAGE:
    effective_upload_mb = PACKAGE.get(
        "effective_upload_mb", PACKAGE.get("max_upload_mb")
    )
    if effective_upload_mb:
        st.sidebar.caption(f"Maximum file size: {effective_upload_mb} MB")
    csv_limits = PACKAGE.get("csv_limits", {})
    if ".csv" in PACKAGE.get("allowed_extensions", []) and csv_limits:
        st.sidebar.caption(
            "CSV limits: "
            f"{csv_limits['rows']:,} rows, {csv_limits['columns']} columns, "
            f"{csv_limits['cell_characters']:,} characters/cell, and "
            f"{csv_limits['extracted_characters']:,} extracted characters."
        )


def show_request_error(error: requests.RequestException) -> None:
    st.error(describe_request_error(error, API_URL))


existing_projects: list[str] = []
project_list_error: requests.RequestException | None = None
if "document_library" in PACKAGE_FEATURES:
    try:
        project_response = requests.get(
            f"{API_URL}/documents", timeout=REQUEST_TIMEOUT
        )
        project_response.raise_for_status()
        existing_projects = sorted(
            {
                document["property_name"].strip()
                for document in project_response.json()
                if document.get("property_name", "").strip()
            },
            key=str.casefold,
        )
    except requests.RequestException as error:
        project_list_error = error

project_options = ["Enter a project / property"]
project_options.extend(existing_projects)
project_choice = st.selectbox(
    "Project",
    project_options,
    help="The selected project controls which documents are shown and searched.",
)
if project_choice == project_options[0]:
    project_context = st.text_input(
        "Project / property name",
        help="Uses the existing property-name field to group and filter this project's documents.",
    ).strip()
else:
    project_context = project_choice
    st.caption(
        f"Selected project controls which documents are shown and searched: "
        f"“{project_context}”."
    )
if project_list_error:
    st.caption(
        "Saved project names could not be loaded. You can still enter a project name."
    )

scope_conversation_to_project(st.session_state, project_context)

if "document_library" in PACKAGE_FEATURES:
    upload_tab, ask_tab, library_tab = st.tabs(["Upload", "Ask", "Document library"])
else:
    upload_tab, ask_tab = st.tabs(["Upload", "Ask"])
    library_tab = None

with upload_tab:
    st.subheader("Add a property document")
    if PACKAGE_LOAD_ERROR is not None:
        if isinstance(PACKAGE_LOAD_ERROR, requests.RequestException):
            package_error = describe_request_error(PACKAGE_LOAD_ERROR, API_URL)
        else:
            package_error = str(PACKAGE_LOAD_ERROR)
        st.error(
            "Unable to verify package. Uploads are disabled until the API responds "
            f"with a valid package. {package_error}"
        )
    with st.form("document_upload"):
        if project_context:
            st.caption(f"Upload will be assigned to project/property: {project_context}")
        else:
            st.caption("Select or identify a project above to group this document.")
        uploaded_files = st.file_uploader(
            "Choose a property document",
            help=(
                f"{PACKAGE_NAME.title()} package • allowed formats: "
                + ", ".join(ALLOWED_EXTENSIONS)
            ),
            disabled=PACKAGE is None,
            accept_multiple_files=True,
        )
        st.caption(
            "Scanned PDFs are not text-extracted; images require Tesseract OCR; "
            "ZIP indexing lists filenames only; CAD and legacy DOC extraction may be limited."
        )
        document_name = st.text_input("Document name (optional)")
        document_type = st.selectbox(
            "Document type (optional)",
            ["Not specified", *DOCUMENT_TYPES],
        )
        document_date = st.date_input("Document date (optional)", value=None)
        note = st.text_input("Note (optional)")
        st.caption(
            "Optional metadata below applies to every selected file. If you enter a "
            "document name, that same name is assigned to each file; leave it blank "
            "to use each filename."
        )
        upload_metadata: dict[str, str] = {"note": note.strip()}
        if project_context:
            upload_metadata["property_name"] = project_context
        if document_name.strip():
            upload_metadata["document_name"] = document_name.strip()
        if document_type != "Not specified":
            upload_metadata["document_type"] = document_type
        if document_date:
            upload_metadata["date"] = document_date.isoformat()
        batch_signature = (
            upload_batch_fingerprint(uploaded_files, upload_metadata)
            if uploaded_files
            else None
        )
        prior_batch_signatures = st.session_state.get(
            "submitted_upload_signatures", []
        )
        same_batch_was_submitted = bool(
            batch_signature is not None
            and batch_signature in prior_batch_signatures
        )
        allow_duplicate_batch = False
        if same_batch_was_submitted:
            allow_duplicate_batch = st.checkbox(
                "I intend to upload this exact batch again",
                help=(
                    "Use only when you intentionally want duplicate documents. "
                    "Check the document library before confirming."
                ),
            )
        submitted = st.form_submit_button(
            "Upload and index",
            type="primary",
            disabled=PACKAGE is None,
        )

    if submitted:
        if not uploaded_files:
            st.warning("Choose one or more documents to upload.")
        elif batch_signature is not None and is_duplicate_batch(
            batch_signature,
            prior_batch_signatures,
            confirmed=allow_duplicate_batch,
        ):
            st.warning(
                "This exact batch was already submitted. Check the document library "
                "before retrying. Confirm the duplicate-upload option only if you "
                "intentionally want to add these files again."
            )
        else:
            st.session_state.pop("upload_batch_results", None)
            prior_batch_signatures = [
                signature
                for signature in prior_batch_signatures
                if signature != batch_signature
            ]
            prior_batch_signatures.append(batch_signature)
            st.session_state["submitted_upload_signatures"] = (
                prior_batch_signatures[-32:]
            )
            file_statuses: dict[int, Any] = {}

            def start_file_upload(index: int, filename: str) -> None:
                file_statuses[index] = st.status(
                    f"{filename}: uploading, extracting text, and indexing.",
                    expanded=False,
                )

            def finish_file_upload(index: int, outcome: UploadOutcome) -> None:
                file_statuses[index].update(
                    label=(
                        f"{outcome.filename}: "
                        f"{'indexed' if outcome.succeeded else 'upload failed'}"
                    ),
                    state="complete" if outcome.succeeded else "error",
                )

            def send_one_file(selected_file: Any, metadata: dict[str, str]) -> dict[str, Any]:
                response = requests.post(
                    f"{API_URL}/documents",
                    files={
                        "file": (
                            selected_file.name,
                            selected_file.getvalue(),
                            selected_file.type or "application/octet-stream",
                        )
                    },
                    data=metadata,
                    timeout=UPLOAD_TIMEOUT,
                )
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as error:
                    raise ValueError(
                        f"{selected_file.name}: the API returned an unreadable upload response."
                    ) from error

            outcomes = upload_selected_files(
                uploaded_files,
                upload_metadata,
                send_one_file,
                lambda error: describe_request_error(error, API_URL),
                on_start=start_file_upload,
                on_finish=finish_file_upload,
            )
            st.session_state["upload_batch_results"] = {
                "signature": batch_signature,
                "outcomes": [outcome.as_dict() for outcome in outcomes],
            }

    batch_results = st.session_state.get("upload_batch_results")
    if (
        batch_signature is not None
        and batch_results
        and batch_results.get("signature") == batch_signature
    ):
        outcomes = batch_results["outcomes"]
        success_count = sum(item["succeeded"] for item in outcomes)
        failure_count = len(outcomes) - success_count
        st.info(
            f"Batch complete: {success_count} indexed; {failure_count} failed."
        )
        for outcome in outcomes:
            if outcome["succeeded"]:
                st.success(f"{outcome['filename']}: {outcome['message']}")
            else:
                st.error(f"{outcome['filename']}: {outcome['message']}")

with ask_tab:
    st.subheader("Review this project")
    review_keys: list[ReviewType] = ["risk", "cost_timeline", "compliance"]
    selected_review: ReviewType = st.selectbox(
        "Review type",
        review_keys,
        format_func=lambda key: REVIEW_MODES[key].label,
    )
    review_mode = REVIEW_MODES[selected_review]
    st.caption(f"What this review checks: {review_mode.description}")
    st.markdown(
        "\n".join(f"- {item}" for item in review_mode.checklist)
    )
    st.caption(
        "Reviews use the most relevant retrieved excerpts, not an exhaustive audit "
        "of every page. An absent finding is not proof that no issue exists."
    )
    project_checklist = ""
    if selected_review == "compliance":
        project_checklist = st.text_area(
            "Project-specific checklist or requirements (optional)",
            max_chars=500,
            help=(
                "Only requirements entered here are used to compare documents. "
                "Without a checklist, the report can organize items for verification "
                "but cannot determine compliance."
            ),
        )
    if st.button("Start a new review / conversation"):
        clear_conversation(st.session_state)

    if st.button("Generate project review", type="primary"):
        st.session_state.pop("review_report", None)
        if PACKAGE is None:
            st.error("Unable to verify the active package; the API is unavailable.")
        elif not project_context:
            st.warning("Select or identify a project before generating its review.")
        else:
            with st.status(
                f"Retrieving evidence for {project_context} — {review_mode.label.lower()}…",
                expanded=False,
            ) as review_status:
                try:
                    response = requests.post(
                        f"{API_URL}/review",
                        json={
                            "review_type": selected_review,
                            "project_name": project_context,
                            "checklist": project_checklist.strip(),
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    if response.status_code == 404:
                        try:
                            detail = response.json().get("detail", "")
                        except ValueError:
                            detail = ""
                        if detail.startswith("No matching documents were found"):
                            review = {
                                "answer": build_empty_review_report(
                                    selected_review,
                                    has_project_checklist=bool(
                                        project_checklist.strip()
                                    ),
                                ),
                                "sources": [],
                            }
                        else:
                            response.raise_for_status()
                    else:
                        response.raise_for_status()
                        review = response.json()
                    review["answer"] = ensure_report_sections(
                        review["answer"],
                        selected_review,
                        has_project_checklist=bool(project_checklist.strip()),
                    )
                    st.session_state["review_report"] = {
                        **review,
                        "review_type": selected_review,
                        "project": project_context,
                    }
                    review_status.update(
                        label="Review draft ready — confirm every finding against its evidence.",
                        state="complete",
                    )
                except requests.Timeout:
                    review_status.update(
                        label="Review request timed out.",
                        state="error",
                    )
                    st.error(
                        "The review request timed out. Check API logs and retry when "
                        "the service is responsive."
                    )
                except requests.RequestException as error:
                    review_status.update(
                        label="Review request failed.",
                        state="error",
                    )
                    show_request_error(error)
                except ValueError as error:
                    review_status.update(
                        label="Review response was invalid.",
                        state="error",
                    )
                    st.error(f"The API returned an invalid review response: {error}")

    saved_review = st.session_state.get("review_report")
    if saved_review:
        review_label = REVIEW_MODES[saved_review["review_type"]].label
        st.markdown(f"### {review_label} — {saved_review['project']}")
        st.warning(
            "AI-generated draft for human review; not a verified fact, final decision, "
            "legal advice, or proof of compliance. Retrieval is not exhaustive; check "
            "each citation against the source."
        )
        st.markdown(saved_review["answer"])
        if saved_review["sources"]:
            st.subheader("Evidence")
            for source in saved_review["sources"]:
                st.markdown(
                    f"**[{source['citation']}] {source['document_name']}** "
                    f"({source['source_file']})"
                )
                st.caption(
                    "Precise page, section, or row location is unavailable in the current index."
                )
                st.text(source["excerpt"])
        st.download_button(
            "Download Markdown review report",
            build_review_markdown(
                saved_review["project"],
                review_label,
                saved_review["answer"],
                saved_review["sources"],
            ),
            file_name=review_report_filename(
                saved_review["project"], saved_review["review_type"]
            ),
            mime="text/markdown",
            help="Downloads the report displayed above, including its evidence references.",
        )
    else:
        st.caption(
            "No completed review report is available to download yet. Generate a "
            "review for the selected project first."
        )

    st.divider()
    st.subheader("Ask a follow-up question")
    document_search_error: str | None = None
    try:
        document_response = requests.get(
            f"{API_URL}/documents/search-options",
            params={"property_name": project_context} if project_context else {},
            timeout=REQUEST_TIMEOUT,
        )
        document_response.raise_for_status()
        search_documents = document_response.json()
        if not isinstance(search_documents, list) or any(
            not isinstance(document, dict)
            or not isinstance(document.get("id"), int)
            or not isinstance(document.get("document_name"), str)
            or not isinstance(document.get("source_file"), str)
            for document in search_documents
        ):
            raise ValueError("The API returned an invalid document list.")
    except requests.RequestException as error:
        document_search_error = describe_request_error(error, API_URL)
        search_documents = []
    except ValueError:
        document_search_error = (
            "The API returned an invalid document list. Search is disabled."
        )
        search_documents = []
    if document_search_error:
        st.error(
            "Available documents could not be loaded, so search is disabled. "
            + document_search_error
        )
    elif not search_documents:
        scope_description = (
            f"project “{project_context}”" if project_context else "this workspace"
        )
        st.info(f"No uploaded documents are available for {scope_description}.")
    selected_document_id = synchronize_document_selection(
        st.session_state, project_context, search_documents
    )
    chat_history = st.session_state.get("chat_history", [])
    with st.form("document_question"):
        question = st.text_area(
            "Question",
            placeholder="What does the inspection report say about the roof?",
            max_chars=MAX_CHAT_QUESTION_CHARS,
            disabled=not search_documents or document_search_error is not None,
        )
        search_document_id = st.selectbox(
            "Search in",
            document_selection_options(search_documents),
            index=document_selection_options(search_documents).index(
                selected_document_id
            ),
            format_func=lambda document_id: document_selection_label(
                document_id, search_documents
            ),
            key="ask_document_id",
            disabled=not search_documents or document_search_error is not None,
        )
        ask_submitted = st.form_submit_button(
            "Get answer",
            type="primary",
            disabled=not search_documents or document_search_error is not None,
        )

    if ask_submitted:
        st.session_state.pop("chat_error", None)
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            payload = {"question": question.strip()}
            if project_context:
                payload["property_name"] = project_context
            if search_document_id is not None:
                payload["document_id"] = search_document_id
            payload["history"] = conversation_messages(chat_history)
            search_scope = question_search_scope(
                project_context, search_document_id, search_documents
            )
            try:
                updated_history, result = submit_chat_turn(
                    chat_history,
                    question,
                    lambda: requests.post(
                        f"{API_URL}/ask",
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    ),
                    search_scope=search_scope,
                )
                st.session_state["chat_history"] = updated_history
                st.session_state.pop("chat_error", None)
            except requests.RequestException as error:
                st.session_state["chat_error"] = {
                    "question": question.strip(),
                    "message": describe_chat_request_error(error),
                }
            except ValueError as error:
                st.session_state["chat_error"] = {
                    "question": question.strip(),
                    "message": str(error),
                }

    chat_error = st.session_state.get("chat_error")
    if chat_error:
        with st.chat_message("user"):
            st.text(chat_error["question"])
        with st.chat_message("assistant"):
            st.warning(chat_error["message"])

    for pair in conversation_display_pairs(
        st.session_state.get("chat_history", [])
    ):
        with st.chat_message("user"):
            st.text(pair["question"])
        with st.chat_message("assistant"):
            if pair.get("search_scope"):
                st.caption(f"Searched in: {pair['search_scope']}")
            st.markdown(pair["answer"])
            if pair["sources"]:
                with st.expander("Sources"):
                    for source in pair["sources"]:
                        st.markdown(
                            f"**[{source['citation']}] {source['document_name']}** — "
                            f"{source['document_type']} / {source['category']} "
                            f"({source['date']})"
                        )
                        st.caption(
                            "Precise page, section, or row location is unavailable "
                            "in the current index."
                        )
                        st.text(source["excerpt"])

with (library_tab if library_tab is not None else st.empty()):
    st.subheader("Document library")
    current_filters = st.session_state.get("library_filters", {})
    with st.form("library_search"):
        search_col1, search_col2, search_col3 = st.columns(3)
        library_search = search_col1.text_input(
            "Search name, metadata, or document text",
            value=current_filters.get("search", ""),
        )
        library_property = search_col2.text_input(
            "Property", value=current_filters.get("property_name", "")
        )
        library_unit = search_col3.text_input("Unit", value=current_filters.get("unit", ""))
        category_options = ["All categories", *CATEGORIES]
        saved_category = current_filters.get("category", "")
        library_category = search_col1.selectbox(
            "Category",
            category_options,
            index=category_options.index(saved_category)
            if saved_category in category_options
            else 0,
        )
        type_options = ["All document types", *DOCUMENT_TYPES]
        saved_type = current_filters.get("document_type", "")
        library_type = search_col2.selectbox(
            "Document type",
            type_options,
            index=type_options.index(saved_type) if saved_type in type_options else 0,
        )
        if "advanced_filters" in PACKAGE_FEATURES:
            library_issue = search_col1.text_input("Issue type")
            library_portfolio = search_col2.text_input("Portfolio")
            library_code = search_col3.text_input("Property code")
            library_status = search_col1.selectbox("Workflow status", ["", "Active", "In review", "Resolved", "Archived"])
        else:
            library_issue = library_portfolio = library_code = library_status = ""
        search_submitted = st.form_submit_button("Search library")

    if search_submitted or "library_filters" not in st.session_state:
        st.session_state["library_filters"] = {
            "search": library_search.strip(),
            "property_name": library_property.strip(),
            "unit": library_unit.strip(),
            "category": "" if library_category == "All categories" else library_category,
            "document_type": "" if library_type == "All document types" else library_type,
            "issue_type": library_issue.strip(),
            "portfolio_name": library_portfolio.strip(),
            "property_code": library_code.strip(),
            "workflow_status": library_status,
        }

    try:
        response = requests.get(
            f"{API_URL}/documents",
            params=st.session_state["library_filters"],
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        documents = response.json()
        if not documents:
            st.info("No property documents match these filters. Upload a record or adjust the search.", icon="ℹ️")
        else:
            st.caption(f"{len(documents)} property record{'s' if len(documents) != 1 else ''} found")
        if "premium_export" in PACKAGE_FEATURES:
            export_response = requests.get(f"{API_URL}/documents/export/csv", timeout=REQUEST_TIMEOUT)
            if export_response.ok:
                st.download_button("Export library as CSV", export_response.content, file_name="docupilot-library.csv", mime="text/csv")

        for document in documents:
            with st.expander(
                f"{document['document_name']} · {document['property_name']}"
            ):
                unit_label = f" · Unit {document['unit']}" if document["unit"] else ""
                st.write(
                    f"**Type:** {document['document_type']}  \n"
                    f"**Category:** {document['category']}  \n"
                    f"**Property:** {document['property_name']}{unit_label}  \n"
                    f"**Document date:** {document['date']}  \n"
                    f"**Uploaded:** {document['uploaded_at']}"
                )
                if document.get("note"):
                    st.write(f"**Note:** {document['note']}")
                if document["issue_type"]:
                    st.write(f"**Issue type:** {document['issue_type']}")
                download_key = f"download_file_{document['id']}"
                if st.button("Prepare download", key=f"prepare_{document['id']}"):
                    try:
                        file_response = requests.get(
                            f"{API_URL}/documents/{document['id']}/file",
                            timeout=REQUEST_TIMEOUT,
                        )
                        file_response.raise_for_status()
                        st.session_state[download_key] = file_response.content
                    except requests.RequestException as error:
                        show_request_error(error)

                if download_key in st.session_state:
                    st.download_button(
                        "Download original",
                        data=st.session_state[download_key],
                        file_name=document["source_file"],
                        mime=document.get("content_type") or "application/octet-stream",
                        key=f"download_{document['id']}",
                    )

                with st.form(f"delete_document_{document['id']}"):
                    confirm_delete = st.checkbox(
                        "Confirm deletion", key=f"confirm_{document['id']}"
                    )
                    delete_submitted = st.form_submit_button("Delete document")
                if delete_submitted:
                    if not confirm_delete:
                        st.warning("Confirm deletion before removing this document.")
                    else:
                        try:
                            delete_response = requests.delete(
                                f"{API_URL}/documents/{document['id']}",
                                timeout=REQUEST_TIMEOUT,
                            )
                            delete_response.raise_for_status()
                            st.session_state.pop(download_key, None)
                            st.success("Document and stored file deleted.")
                            st.rerun()
                        except requests.RequestException as error:
                            show_request_error(error)
    except requests.RequestException as error:
        show_request_error(error)
