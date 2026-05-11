from __future__ import annotations

from typing import Any

from api_client import SyncApiClient


def fetch_gpu_info(client: SyncApiClient) -> dict[str, Any]:
    response = client.request("GET", "/gpu/info")
    payload = response.json if isinstance(response.json, dict) else {}
    return payload

