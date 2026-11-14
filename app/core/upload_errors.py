"""User-facing descriptions for upload request failures."""

import requests


def _contains_socket_timeout(error: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        if isinstance(current, BaseException):
            pending.extend(current.args)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
    return False


def describe_request_error(error: requests.RequestException, api_url: str) -> str:
    response = getattr(error, "response", None)
    if response is not None:
        try:
            body = response.json()
            detail = body.get("detail", response.text) if isinstance(body, dict) else response.text
        except ValueError:
            detail = response.text
        return f"API error ({response.status_code}): {detail}"

    if isinstance(error, requests.ConnectTimeout):
        return f"Could not connect to the API at {api_url}. Check that the API service is running and reachable."
    if isinstance(error, requests.ReadTimeout):
        return (
            "The API connection was established, but document processing did not finish before "
            "the upload time limit. Check the document library before retrying; indexing may "
            "still be running."
        )
    if _contains_socket_timeout(error):
        return (
            "The API connection was interrupted while waiting for upload processing to finish. "
            "Check the document library before retrying; indexing may still be running."
        )
    if isinstance(error, requests.Timeout):
        return f"The request to the API at {api_url} timed out before it completed."
    if isinstance(error, requests.ConnectionError):
        return f"Could not connect to the API at {api_url}. Check the API service and API_URL setting."
    return f"The request to the API failed: {error}"
