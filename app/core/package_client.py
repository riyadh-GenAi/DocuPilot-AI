"""Fetch and validate the active package from the API."""

from typing import Any

import requests


def fetch_package(api_url: str) -> dict[str, Any]:
    """Fetch the current API package; never substitute a local tier on failure."""
    response = requests.get(f"{api_url.rstrip('/')}/package", timeout=10)
    response.raise_for_status()
    package = response.json()
    if not isinstance(package, dict):
        raise ValueError("The API returned an invalid package response.")
    if (
        package.get("name") not in {"basic", "standard", "premium"}
        or not isinstance(package.get("features"), list)
        or not isinstance(package.get("allowed_extensions"), list)
    ):
        raise ValueError("The API returned an incomplete package configuration.")
    return package


def load_package_state(
    api_url: str,
) -> tuple[dict[str, Any] | None, requests.RequestException | ValueError | None]:
    """Return no package on failure so callers cannot display a false tier."""
    try:
        return fetch_package(api_url), None
    except (requests.RequestException, ValueError) as error:
        return None, error
