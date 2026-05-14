from __future__ import annotations

import asyncio
from typing import Iterable

from api_client import SyncApiClient, AsyncApiClient
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


def _build_payload(size: int, backend: str, optimized: bool) -> dict:
    return {
        "size": size,
        "backend": backend,
        "inputMode": "random",
        "optimized": optimized,
        "randomMin": -1,
        "randomMax": 1,
    }


async def _run_concurrency_batch(
    client: AsyncApiClient,
    payload: dict,
    concurrency: int,
) -> list:
    responses = []
    async with asyncio.TaskGroup() as tg:
        for _ in range(concurrency):
            tg.create_task(_collect_response(client, payload, responses))
    return responses


async def _collect_response(
    client: AsyncApiClient, payload: dict, responses: list
) -> None:
    response = await client.request("POST", "/matrix/multiply", json_body=payload)
    responses.append(response)


async def _run_matrix_concurrency_async(
    base_url: str,
    backend: str,
    size: int,
    optimized: bool,
    iterations: int,
    concurrency_levels: Iterable[int],
) -> list[BenchResult]:
    results: list[BenchResult] = []
    async with AsyncApiClient(base_url) as client:
        for concurrency in concurrency_levels:
            concurrency = max(1, int(concurrency))
            for iteration in range(iterations):
                print(
                    f"[matrix-concurrency] size={size} backend={backend} optimized={optimized} "
                    f"concurrency={concurrency} iter={iteration + 1}/{iterations}"
                )
                payload = _build_payload(size, backend, optimized)
                responses = await _run_concurrency_batch(client, payload, concurrency)
                for response in responses:
                    data = response.json if isinstance(response.json, dict) else {}
                    if response.status != 200:
                        error_msg = (
                            response.json.get(
                                "message", response.json.get("error", "Brak szczegółów")
                            )
                            if response.json
                            else "Brak odpowiedzi JSON"
                        )
                        print(
                            f"\033[91mBłąd matrix (współbieżność): {response.status} - {error_msg}\033[0m"
                        )
                        continue
                    mem_gpu, mem_host, mem_rss = _extract_memory(data)
                    results.append(
                        BenchResult(
                            timestamp_utc=utc_now_iso(),
                            endpoint="/matrix/multiply",
                            pipeline="matrix-concurrency",
                            backend=backend,
                            optimized=optimized,
                            size_label=str(concurrency),
                            params={
                                **payload,
                                "concurrency": concurrency,
                                "iteration": iteration + 1,
                            },
                            run_mode="concurrency",
                            status=response.status,
                            gpu_duration_ms=_extract_float(data, "gpuDurationMs"),
                            backend_duration_ms=_extract_float(
                                data, "backendDurationMs"
                            ),
                            server_duration_ms=_extract_float(data, "serverDurationMs"),
                            client_rtt_ms=response.client_rtt_ms,
                            memory_gpu_bytes=mem_gpu,
                            memory_host_bytes=mem_host,
                            memory_server_rss_bytes=mem_rss,
                        )
                    )
    return results


def _extract_message(payload: dict) -> str | None:
    message = payload.get("message") if isinstance(payload, dict) else None
    if message is None:
        return None
    return str(message)


def _print_error(
    label: str, status: int, message: str | None, details: str = ""
) -> None:
    detail = f" {details}" if details else ""
    message_text = f" message={message}" if message else ""
    print(f"\x1b[31m[{label}] status={status}{detail}{message_text}\x1b[0m")


def run_matrix_concurrency_benchmarks(
    base_url: str,
    backend: str,
    size: int,
    optimized: bool,
    iterations: int,
    concurrency_levels: Iterable[int],
) -> list[BenchResult]:
    return asyncio.run(
        _run_matrix_concurrency_async(
            base_url=base_url,
            backend=backend,
            size=size,
            optimized=optimized,
            iterations=iterations,
            concurrency_levels=concurrency_levels,
        )
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
                    error_msg = (
                        response.json.get(
                            "message", response.json.get("error", "Brak szczegółów")
                        )
                        if response.json
                        else "Brak odpowiedzi JSON"
                    )
                    print(
                        f"\033[91mBłąd matrix: {response.status} - {error_msg}\033[0m"
                    )
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
