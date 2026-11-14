"""Markdown report content and safe download filenames."""

import re
import unicodedata
from collections.abc import Mapping, Sequence


_INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_NON_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _filename_component(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = _INVALID_FILENAME_CHARACTERS.sub("-", ascii_value)
    ascii_value = _WHITESPACE.sub("-", ascii_value.strip())
    ascii_value = _NON_FILENAME.sub("-", ascii_value)
    ascii_value = re.sub(r"-{2,}", "-", ascii_value).strip(" .-_")
    component = ascii_value[:60].rstrip(" .-_")
    if not component:
        component = fallback
    if component.upper() in _WINDOWS_RESERVED_NAMES:
        component = f"project-{component}"
    return component


def review_report_filename(project: str, review_type: str) -> str:
    project_component = _filename_component(project, "project")
    review_component = _filename_component(review_type, "review")
    return f"{project_component}-{review_component}.md"


def build_review_markdown(
    project: str,
    review_label: str,
    answer: str,
    sources: Sequence[Mapping[str, object]],
) -> str:
    """Build the downloadable report from the same fields currently displayed."""
    markdown = (
        f"# {review_label} — {project}\n\n"
        "> AI-generated draft for human review. Retrieval is not exhaustive. Verify "
        "findings against the source; this is not legal advice or proof of compliance.\n\n"
        f"{answer.rstrip()}\n"
    )
    if sources:
        markdown += "\n## Evidence excerpts\n\n"
        for source in sources:
            citation = source.get("citation", "")
            document_name = str(source.get("document_name", "Source document"))
            source_file = str(source.get("source_file", ""))
            excerpt = str(source.get("excerpt", ""))
            markdown += (
                f"### [{citation}] {document_name}"
                + (f" ({source_file})" if source_file else "")
                + "\n\n"
                "Precise page, section, or row location unavailable.\n\n"
            )
            if excerpt:
                markdown += f"> {excerpt.replace(chr(10), chr(10) + '> ')}\n\n"
    return markdown
