from __future__ import annotations

from typing import Iterable

from api_client import SyncApiClient
from benchmarks.common import BenchResult, utc_now_iso


FILTERS = ("gaussian",)


def _extract_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_memory(payload: dict) -> tuple[float | None, float | None, float | None]:
    memory = payload.get("memory") if isinstance(payload, dict) else None
    if not isinstance(memory, dict):
        return None, None, None
    return (
        _extract_float(memory, "gpuBytes"),
        _extract_float(memory, "hostBytes"),
        _extract_float(memory, "serverRssBytes"),
    )


def run_image_benchmarks(
    client: SyncApiClient,
    sizes: Iterable[tuple[int, int]],
    backend: str,
    iterations: int,
    warmup: bool = True,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for width, height in sizes:
        for image_filter in FILTERS:
            for iteration in range(iterations):
                print(
                    f"[image] {width}x{height} backend={backend} filter={image_filter} iter={iteration + 1}/{iterations}"
                )
                payload = {
                    "filter": image_filter,
                    "backend": backend,
                    "inputMode": "random",
                    "width": width,
                    "height": height,
                }
                response = client.request("POST", "/image/filter", json_body=payload)
                data = response.json if isinstance(response.json, dict) else {}
                mem_gpu, mem_host, mem_rss = _extract_memory(data)
                if response.status != 200:
                    error_msg = data.get(
                        "message", data.get("error", "Brak szczegółów")
                    )
                    print(
                        f"\033[91m[image] Błąd: {response.status} - {error_msg} (size={width}x{height} backend={backend})\033[0m"
                    )
                    continue
                if warmup and iteration == 0:
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
                        memory_gpu_bytes=mem_gpu,
                        memory_host_bytes=mem_host,
                        memory_server_rss_bytes=mem_rss,
                    )
                )
    return results
