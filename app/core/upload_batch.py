"""Sequential uploads through the existing single-file endpoint."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Protocol

import requests


class SelectedFile(Protocol):
    name: str
    type: str

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class UploadOutcome:
    filename: str
    succeeded: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_duplicate_batch(
    fingerprint: str, prior_fingerprints: Sequence[str], confirmed: bool = False
) -> bool:
    return fingerprint in prior_fingerprints and not confirmed


def upload_batch_fingerprint(
    files: Sequence[SelectedFile], metadata: Mapping[str, str]
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(dict(metadata), sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    file_digests: list[bytes] = []
    for selected_file in files:
        content = selected_file.getvalue()
        file_digest = hashlib.sha256()
        file_digest.update(selected_file.name.encode("utf-8"))
        file_digest.update((selected_file.type or "").encode("utf-8"))
        file_digest.update(len(content).to_bytes(8, "big"))
        file_digest.update(content)
        file_digests.append(file_digest.digest())
    for file_digest in sorted(file_digests):
        digest.update(file_digest)
    return digest.hexdigest()


def upload_selected_files(
    files: Sequence[SelectedFile],
    metadata: Mapping[str, str],
    send_upload: Callable[[SelectedFile, dict[str, str]], Mapping[str, Any]],
    describe_error: Callable[[requests.RequestException], str],
    on_start: Callable[[int, str], None] | None = None,
    on_finish: Callable[[int, UploadOutcome], None] | None = None,
) -> list[UploadOutcome]:
    outcomes: list[UploadOutcome] = []
    for index, selected_file in enumerate(files):
        if on_start:
            on_start(index, selected_file.name)
        try:
            response = send_upload(selected_file, dict(metadata))
            if not isinstance(response, Mapping):
                raise ValueError("The API returned an invalid upload response.")
            if not response.get("id"):
                raise ValueError("The API response did not confirm a saved document.")
            display_name = str(response.get("document_name") or selected_file.name)
            outcome = UploadOutcome(
                filename=selected_file.name,
                succeeded=True,
                message=f"Added {display_name} to the document library.",
            )
        except requests.RequestException as error:
            outcome = UploadOutcome(
                filename=selected_file.name,
                succeeded=False,
                message=describe_error(error),
            )
        except ValueError as error:
            outcome = UploadOutcome(
                filename=selected_file.name,
                succeeded=False,
                message=str(error),
            )
        outcomes.append(outcome)
        if on_finish:
            on_finish(index, outcome)
    return outcomes
