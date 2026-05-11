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


def run_matrix_benchmarks(
    client: SyncApiClient,
    sizes: Iterable[int],
    backend: str,
    optimized_variants: Iterable[bool],
    iterations: int,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for size in sizes:
        for optimized in optimized_variants:
            for iteration in range(iterations):
                payload = {
                    "size": size,
                    "backend": backend,
                    "inputMode": "random",
                    "optimized": optimized,
                    "randomMin": -1,
                    "randomMax": 1,
                }
                response = client.request("POST", "/matrix/multiply", json_body=payload)
                data = response.json if isinstance(response.json, dict) else {}
                if iteration == 0:
                    continue
                results.append(
                    BenchResult(
                        timestamp_utc=utc_now_iso(),
                        endpoint="/matrix/multiply",
                        pipeline="matrix",
                        backend=backend,
                        optimized=optimized,
                        size_label=str(size),
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

