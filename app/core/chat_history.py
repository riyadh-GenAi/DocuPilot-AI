"""Session-scoped helpers for bounded question-and-answer history."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import json
import re
from typing import Any

import requests

MAX_CHAT_PAIRS = 5
MAX_CHAT_QUESTION_CHARS = 2_000
MAX_HISTORY_MESSAGE_CHARS = 400
MAX_HISTORY_CONTENT_CHARS = 5_000
MAX_HISTORY_JSON_BYTES = 12_000
_REPORT_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CITATION = re.compile(r"\[(\d+)\]")


def _truncate_at_boundary(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    boundary = text.rfind(" ", 0, limit - 1)
    if boundary < limit // 2:
        boundary = limit - 1
    return text[:boundary].rstrip() + "…"


def _assistant_context(pair: Mapping[str, Any]) -> str:
    answer = pair["answer"].strip()
    sections = list(_REPORT_SECTION.finditer(answer))
    if sections:
        report: dict[str, str] = {}
        for index, match in enumerate(sections):
            end = sections[index + 1].start() if index + 1 < len(sections) else len(answer)
            report[match.group(1).strip().casefold()] = answer[match.end():end].strip()

        selected: list[str] = []
        for section in ("executive summary", "key costs and dates", "potential risks"):
            content = report.get(section)
            if content and content.lower().startswith("the model did not return this section"):
                continue
            if content:
                selected.append(f"{section.title()}: {_truncate_at_boundary(content, 220)}")
        summary = " ".join(selected) or _truncate_at_boundary(answer, 400)
    else:
        summary = _truncate_at_boundary(answer, 600)

    citations = set(_CITATION.findall(answer))
    source_refs = []
    for source in pair.get("sources", []):
        citation = str(source.get("citation", ""))
        if citation and citation in citations:
            name = str(source.get("document_name", "")).strip()
            source_refs.append(
                f"[{citation}] {_truncate_at_boundary(name, 55)}"
                if name
                else f"[{citation}]"
            )
    if source_refs:
        references = f"Sources: {', '.join(source_refs[:2])}."
        summary = _truncate_at_boundary(
            summary, MAX_HISTORY_MESSAGE_CHARS - len(references) - 1
        )
        return f"{summary} {references}"
    return _truncate_at_boundary(summary, MAX_HISTORY_MESSAGE_CHARS)


def clear_conversation(state: MutableMapping[str, Any]) -> None:
    state.pop("chat_history", None)
    state.pop("chat_error", None)
    state.pop("review_report", None)


def scope_conversation_to_project(
    state: MutableMapping[str, Any], project: str | None
) -> None:
    if state.get("active_project") != project:
        clear_conversation(state)
        state["active_project"] = project


def append_chat_pair(
    history: Sequence[Mapping[str, Any]],
    question: str,
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    search_scope: str | None = None,
) -> list[dict[str, Any]]:
    if not question.strip() or not answer.strip():
        raise ValueError("A completed question and answer are required.")
    pairs = [
        {
            "question": pair["question"],
            "answer": pair["answer"],
            "sources": pair.get("sources", []),
            **(
                {"search_scope": pair["search_scope"]}
                if pair.get("search_scope")
                else {}
            ),
        }
        for pair in history
        if pair.get("question", "").strip() and pair.get("answer", "").strip()
    ]
    pairs.append(
        {
            "question": question.strip(),
            "answer": answer.strip(),
            "sources": sources or [],
            **({"search_scope": search_scope} if search_scope else {}),
        }
    )
    return pairs[-MAX_CHAT_PAIRS:]


def conversation_messages(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    compact_pairs: list[list[dict[str, str]]] = []
    for pair in history[-MAX_CHAT_PAIRS:]:
        if pair.get("question", "").strip() and pair.get("answer", "").strip():
            compact_pairs.append(
                [
                    {
                        "role": "user",
                        "content": _truncate_at_boundary(
                            str(pair["question"]), MAX_HISTORY_MESSAGE_CHARS
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": _assistant_context(pair),
                    },
                ]
            )

    selected_pairs: list[list[dict[str, str]]] = []
    for pair in reversed(compact_pairs):
        candidate = [pair, *selected_pairs]
        messages = [message for selected in candidate for message in selected]
        content_chars = sum(len(message["content"]) for message in messages)
        payload_bytes = len(json.dumps(messages).encode("utf-8"))
        if (
            content_chars <= MAX_HISTORY_CONTENT_CHARS
            and payload_bytes <= MAX_HISTORY_JSON_BYTES
        ):
            selected_pairs = candidate
    return [message for pair in selected_pairs for message in pair]


def conversation_display_pairs(
    history: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return a newest-first view without mutating chronological stored history."""
    return list(reversed(history[-MAX_CHAT_PAIRS:]))


def validate_current_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Enter a question first.")
    if len(question) > MAX_CHAT_QUESTION_CHARS:
        raise ValueError(
            f"Keep the question to {MAX_CHAT_QUESTION_CHARS:,} characters or fewer."
        )
    return question


def describe_chat_request_error(error: requests.RequestException) -> str:
    response = getattr(error, "response", None)
    if response is None:
        if isinstance(error, requests.Timeout):
            return (
                "The request timed out. Your earlier conversation is still available; "
                "check the service and try again when ready."
            )
        return (
            "Could not reach the API. Your earlier conversation is still available; "
            "check the API connection and try again."
        )
    if response.status_code == 422:
        return (
            "The API could not accept this follow-up. Your earlier conversation is "
            "preserved. Try a shorter question or ask it without prior context."
        )
    if response.status_code == 429:
        return (
            "The AI provider is rate-limiting requests. Your earlier conversation is "
            "preserved; wait briefly before trying again."
        )
    return (
        f"The API could not complete this question (HTTP {response.status_code}). "
        "Your earlier conversation is preserved; check the service and try again."
    )


def submit_chat_turn(
    history: Sequence[Mapping[str, Any]],
    question: str,
    submit_request: Callable[[], requests.Response],
    search_scope: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question = validate_current_question(question)
    response = submit_request()
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise ValueError("The API returned an invalid answer.")
    if not isinstance(result.get("sources"), list):
        raise ValueError("The API returned invalid source details.")
    return append_chat_pair(
        history, question, result["answer"], result["sources"], search_scope
    ), result
