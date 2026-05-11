from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

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


def _default_targets(use_cuda: bool) -> list[StressTarget]:
    backends = ["webgpu"] + (["cuda"] if use_cuda else [])
    targets: list[StressTarget] = []
    for backend in backends:
        for optimized in (False, True):
            targets.append(
                StressTarget(
                    method="POST",
                    path="/matrix/multiply",
                    payload={
                        "size": 256,
                        "backend": backend,
                        "inputMode": "random",
                        "optimized": optimized,
                        "randomMin": -1,
                        "randomMax": 1,
                    },
                    pipeline="matrix",
                    backend=backend,
                    optimized=optimized,
                    size_label="256",
                )
            )
        targets.append(
            StressTarget(
                method="POST",
                path="/image/filter",
                payload={
                    "filter": "gaussian",
                    "backend": backend,
                    "inputMode": "random",
                    "width": 1280,
                    "height": 720,
                },
                pipeline="image",
                backend=backend,
                optimized=None,
                size_label="1280x720",
            )
        )

    render_backends = ["webgpu-render", "webgpu-compute"] + (["cuda"] if use_cuda else [])
    for backend in render_backends:
        targets.append(
            StressTarget(
                method="POST",
                path="/render/",
                payload={
                    "seed": 1234,
                    "count": 1500,
                    "backend": backend,
                },
                pipeline="render",
                backend=backend,
                optimized=None,
                size_label="1500",
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
) -> None:
    async with semaphore:
        response = await _hit_target(client, target)
    data = response.json if isinstance(response.json, dict) else {}
    results.append(
        BenchResult(
            timestamp_utc=utc_now_iso(),
            endpoint=target.path,
            pipeline=target.pipeline,
            backend=target.backend,
            optimized=target.optimized,
            size_label=target.size_label,
            params=target.payload or {},
            run_mode="stress",
            status=response.status,
            gpu_duration_ms=_extract_float(data, "gpuDurationMs"),
            backend_duration_ms=_extract_float(data, "backendDurationMs"),
            server_duration_ms=_extract_float(data, "serverDurationMs"),
            client_rtt_ms=response.client_rtt_ms,
        )
    )


async def run_stress_benchmarks(
    base_url: str,
    concurrency: int,
    total_requests: int,
    use_cuda: bool,
) -> list[BenchResult]:
    targets = _default_targets(use_cuda)
    results: list[BenchResult] = []
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with AsyncApiClient(base_url) as client:
        async with asyncio.TaskGroup() as tg:
            for idx in range(total_requests):
                target = targets[idx % len(targets)]
                tg.create_task(_run_one(client, semaphore, target, results))
    return results
