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


def _extract_memory(payload: dict) -> tuple[float | None, float | None, float | None]:
    memory = payload.get("memory") if isinstance(payload, dict) else None
    if not isinstance(memory, dict):
        return None, None, None
    return (
        _extract_float(memory, "gpuBytes"),
        _extract_float(memory, "hostBytes"),
        _extract_float(memory, "serverRssBytes"),
    )


def run_matrix_benchmarks(
    client: SyncApiClient,
    sizes: Iterable[int],
    backend: str,
    optimized_variants: Iterable[bool],
    iterations: int,
    warmup: bool = True,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for size in sizes:
        for optimized in optimized_variants:
            for iteration in range(iterations):
                print(
                    f"[matrix] size={size} backend={backend} optimized={optimized} iter={iteration + 1}/{iterations}"
                )
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
                mem_gpu, mem_host, mem_rss = _extract_memory(data)
                if response.status != 200:
                    print(f"[matrix] status={response.status} size={size} backend={backend}")
                    continue
                if warmup and iteration == 0:
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
                        memory_gpu_bytes=mem_gpu,
                        memory_host_bytes=mem_host,
                        memory_server_rss_bytes=mem_rss,
                    )
                )
    return results
