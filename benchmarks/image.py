from __future__ import annotations

from typing import Iterable

from api_client import SyncApiClient
from benchmarks.common import BenchResult, utc_now_iso


FILTERS = ("gaussian", "sobel", "grayscale")


def _extract_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_image_benchmarks(
    client: SyncApiClient,
    sizes: Iterable[tuple[int, int]],
    backend: str,
    iterations: int,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for width, height in sizes:
        for image_filter in FILTERS:
            for iteration in range(iterations):
                payload = {
                    "filter": image_filter,
                    "backend": backend,
                    "inputMode": "random",
                    "width": width,
                    "height": height,
                }
                response = client.request("POST", "/image/filter", json_body=payload)
                data = response.json if isinstance(response.json, dict) else {}
                if iteration == 0:
                    continue
                results.append(
                    BenchResult(
                        timestamp_utc=utc_now_iso(),
                        endpoint="/image/filter",
                        pipeline="image",
                        backend=backend,
                        optimized=None,
                        size_label=f"{width}x{height}",
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

