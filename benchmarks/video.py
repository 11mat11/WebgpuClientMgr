from __future__ import annotations

from typing import Iterable

from api_client import SyncApiClient
from benchmarks.common import BenchResult, utc_now_iso


def _extract_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_first_file(client: SyncApiClient) -> str | None:
    response = client.request("GET", "/video/list")
    data = response.json if isinstance(response.json, dict) else {}
    files = data.get("files") if isinstance(data, dict) else None
    if isinstance(files, list) and files:
        return str(files[0])
    return None


def run_video_benchmarks(
    client: SyncApiClient,
    backend: str,
    frame_indices: Iterable[int],
    iterations: int,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    file_name = _pick_first_file(client)
    if not file_name:
        return results
    for frame in frame_indices:
        for iteration in range(iterations):
            payload = {
                "fileName": file_name,
                "frameIndex": frame,
                "backend": backend,
            }
            response = client.request("POST", "/video/histogram", json_body=payload)
            data = response.json if isinstance(response.json, dict) else {}
            if iteration == 0:
                continue
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint="/video/histogram",
                    pipeline="video",
                    backend=backend,
                    optimized=None,
                    size_label=str(frame),
                    params=payload,
                    run_mode="sequential",
                    status=response.status,
                    gpu_duration_ms=_extract_float(data, "gpuDurationMs"),
                    backend_duration_ms=_extract_float(data, "backendDurationMs"),
                    server_duration_ms=_extract_float(data, "serverDurationMs"),
                    client_rtt_ms=response.client_rtt_ms,
                )
            )
    return results

