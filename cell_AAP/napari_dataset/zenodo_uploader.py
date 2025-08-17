from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional


ZENODO_API_BASE = "https://zenodo.org/api"


class ZenodoUploadError(Exception):
    pass


def _auth_headers(pat: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {pat}"}


def create_deposition(pat: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise ZenodoUploadError("The 'requests' package is required. Please install it (e.g., pip install requests).") from e

    url = f"{ZENODO_API_BASE}/deposit/depositions"
    headers = _auth_headers(pat)
    headers["Content-Type"] = "application/json"
    payload = {"metadata": metadata}
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    if r.status_code not in (200, 201):
        raise ZenodoUploadError(f"Failed to create deposition ({r.status_code}): {r.text}")
    return r.json()


def upload_file_to_bucket(pat: str, bucket_url: str, file_path: str) -> None:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise ZenodoUploadError("The 'requests' package is required. Please install it (e.g., pip install requests).") from e

    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{file_name}",
            data=f,
            headers=_auth_headers(pat),
        )
    if r.status_code not in (200, 201):
        raise ZenodoUploadError(f"Failed to upload file to bucket ({r.status_code}): {r.text}")


def publish_deposition(pat: str, deposition_id: int) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise ZenodoUploadError("The 'requests' package is required. Please install it (e.g., pip install requests).") from e

    url = f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}/actions/publish"
    r = requests.post(url, headers=_auth_headers(pat))
    if r.status_code not in (200, 201, 202):
        raise ZenodoUploadError(f"Failed to publish deposition ({r.status_code}): {r.text}")
    return r.json()


def upload_dataset_zip(
    pat: str,
    zip_path: str,
    *,
    title: str,
    description: str,
    creators: List[Dict[str, str]],
    keywords: Optional[List[str]] = None,
    license_id: str = "cc-by-4.0",
    community_identifier: str = "cellapp_external",
    publish: bool = True,
) -> str:
    """
    Create a deposition, upload the given ZIP, assign community, and optionally publish.

    Returns the record HTML URL.
    """
    metadata: Dict[str, Any] = {
        "title": title,
        "upload_type": "dataset",
        "description": description,
        "creators": creators,
        "access_right": "open",
        "license": license_id,
        "communities": [{"identifier": community_identifier}],
    }
    if keywords:
        metadata["keywords"] = keywords

    dep = create_deposition(pat, metadata)
    bucket_url = dep.get("links", {}).get("bucket")
    if not bucket_url:
        raise ZenodoUploadError("No bucket URL returned by Zenodo.")

    upload_file_to_bucket(pat, bucket_url, zip_path)

    if publish:
        dep = publish_deposition(pat, int(dep["id"]))

    html_url = dep.get("links", {}).get("html")
    if not html_url:
        raise ZenodoUploadError("Upload succeeded but HTML link missing in response.")
    return html_url


