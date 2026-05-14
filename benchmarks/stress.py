from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable

from api_client import ApiResponse, AsyncApiClient
from benchmarks.common import BenchResult, utc_now_iso


@dataclass(frozen=True)
class StressTarget:
    method: str
    path: str
    payload: dict[str, Any] | None
    pipeline: str
    backend: str | None
    optimized: bool | None
    size_label: str


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


def _default_targets(
    backends: list[str],
    render_backends: list[str],
    profile: str,
    matrix_sizes: Iterable[int],
    image_sizes: Iterable[tuple[int, int]],
    render_counts: Iterable[int],
) -> list[StressTarget]:
    is_load = profile == "load"
    targets: list[StressTarget] = []
    for backend in backends:
        for matrix_size in matrix_sizes:
            for optimized in (False, True):
                targets.append(
                    StressTarget(
                        method="POST",
                        path="/matrix/multiply",
                        payload={
                            "size": matrix_size,
                            "backend": backend,
                            "inputMode": "random",
                            "optimized": optimized,
                            "randomMin": -1,
                            "randomMax": 1,
                        },
                        pipeline="matrix",
                        backend=backend,
                        optimized=optimized,
                        size_label=str(matrix_size),
                    )
                )
        for width, height in image_sizes:
            targets.append(
                StressTarget(
                    method="POST",
                    path="/image/filter",
                    payload={
                        "filter": "gaussian",
                        "backend": backend,
                        "inputMode": "random",
                        "width": width,
                        "height": height,
                    },
                    pipeline="image",
                    backend=backend,
                    optimized=None,
                    size_label=f"{width}x{height}",
                )
            )

    for backend in render_backends:
        for render_count in render_counts:
            targets.append(
                StressTarget(
                    method="POST",
                    path="/render/",
                    payload={
                        "seed": 1234,
                        "count": render_count,
                        "backend": backend,
                    },
                    pipeline="render",
                    backend=backend,
                    optimized=None,
                    size_label=str(render_count),
                )
            )
    return targets


async def _hit_target(client: AsyncApiClient, target: StressTarget) -> ApiResponse:
    return await client.request(target.method, target.path, json_body=target.payload)


async def _run_one(
    client: AsyncApiClient,
    semaphore: asyncio.Semaphore,
    target: StressTarget,
    results: list[BenchResult],
    run_mode: str,
) -> None:
    async with semaphore:
        response = await _hit_target(client, target)
    data = response.json if isinstance(response.json, dict) else {}
    if response.status != 200:
        error_msg = (
            response.json.get("message", response.json.get("error", "Brak szczegółów"))
            if response.json
            else "Brak odpowiedzi JSON"
        )
        print(
            f"\033[91m[{run_mode}] Błąd {target.path}: {response.status} - {error_msg}\033[0m"
        )
        return
    mem_gpu, mem_host, mem_rss = _extract_memory(data)
    results.append(
        BenchResult(
            timestamp_utc=utc_now_iso(),
            endpoint=target.path,
            pipeline=target.pipeline,
            backend=target.backend,
            optimized=target.optimized,
            size_label=target.size_label,
            params=target.payload or {},
            run_mode=run_mode,
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


async def run_stress_benchmarks(
    base_url: str,
    concurrency: int,
    total_requests: int,
    backends: list[str],
    render_backends: list[str],
    profile: str,
    matrix_sizes: Iterable[int],
    image_sizes: Iterable[tuple[int, int]],
    render_counts: Iterable[int],
) -> list[BenchResult]:
    targets = _default_targets(
        backends=backends,
        render_backends=render_backends,
        profile=profile,
        matrix_sizes=matrix_sizes,
        image_sizes=image_sizes,
        render_counts=render_counts,
    )
    run_mode = "load" if profile == "load" else "stress"
    results: list[BenchResult] = []
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with AsyncApiClient(base_url) as client:
        async with asyncio.TaskGroup() as tg:
            for idx in range(total_requests):
                target = targets[idx % len(targets)]
                tg.create_task(_run_one(client, semaphore, target, results, run_mode))
    return results
