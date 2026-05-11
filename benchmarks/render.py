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


def run_render_benchmarks(
    client: SyncApiClient,
    counts: Iterable[int],
    backend: str,
    iterations: int,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for count in counts:
        for iteration in range(iterations):
            payload = {
                "seed": 1234,
                "count": count,
                "backend": backend,
            }
            response = client.request("POST", "/render/", json_body=payload)
            data = response.json if isinstance(response.json, dict) else {}
            if iteration == 0:
                continue
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint="/render/",
                    pipeline="render",
                    backend=backend,
                    optimized=None,
                    size_label=str(count),
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

