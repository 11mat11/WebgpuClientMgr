from __future__ import annotations

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


def _record_simple_call(
    client: SyncApiClient,
    method: str,
    endpoint: str,
    payload: dict | None,
    pipeline: str,
    run_mode: str,
    iterations: int,
    warmup: bool,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for iteration in range(iterations):
        response = client.request(method, endpoint, json_body=payload)
        data = response.json if isinstance(response.json, dict) else {}
        if warmup and iteration == 0:
            continue
        results.append(
            BenchResult(
                timestamp_utc=utc_now_iso(),
                endpoint=endpoint,
                pipeline=pipeline,
                backend=None,
                optimized=None,
                size_label="na",
                params=payload or {},
                run_mode=run_mode,
                status=response.status,
                gpu_duration_ms=_extract_float(data, "gpuDurationMs"),
                backend_duration_ms=_extract_float(data, "backendDurationMs"),
                server_duration_ms=_extract_float(data, "serverDurationMs"),
                client_rtt_ms=response.client_rtt_ms,
            )
        )
    return results


def run_system_benchmarks(
    client: SyncApiClient,
    iterations: int,
    warmup: bool = True,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    results += _record_simple_call(
        client=client,
        method="GET",
        endpoint="/health",
        payload=None,
        pipeline="system_health",
        run_mode="sequential",
        iterations=iterations,
        warmup=warmup,
    )
    results += _record_simple_call(
        client=client,
        method="GET",
        endpoint="/gpu/info",
        payload=None,
        pipeline="system_gpu_info",
        run_mode="sequential",
        iterations=iterations,
        warmup=warmup,
    )
    results += _record_simple_call(
        client=client,
        method="GET",
        endpoint="/gpu/test",
        payload=None,
        pipeline="system_gpu_test",
        run_mode="sequential",
        iterations=iterations,
        warmup=warmup,
    )
    results += _record_simple_call(
        client=client,
        method="GET",
        endpoint="/gpu/stress/",
        payload=None,
        pipeline="system_gpu_stress_list",
        run_mode="sequential",
        iterations=iterations,
        warmup=warmup,
    )
    results += _record_simple_call(
        client=client,
        method="GET",
        endpoint="/video/list",
        payload=None,
        pipeline="system_video_list",
        run_mode="sequential",
        iterations=iterations,
        warmup=warmup,
    )
    return results


def run_gpu_stress_alloc_benchmarks(
    client: SyncApiClient,
    iterations: int,
    warmup: bool = True,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    payload = {"durationSec": 2, "targetMb": 256}
    for iteration in range(iterations):
        start_response = client.request("POST", "/gpu/stress/start", json_body=payload)
        start_data = start_response.json if isinstance(start_response.json, dict) else {}
        if not (warmup and iteration == 0):
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint="/gpu/stress/start",
                    pipeline="system_gpu_stress_start",
                    backend=None,
                    optimized=None,
                    size_label="256mb",
                    params=payload,
                    run_mode="sequential",
                    status=start_response.status,
                    gpu_duration_ms=_extract_float(start_data, "gpuDurationMs"),
                    backend_duration_ms=_extract_float(start_data, "backendDurationMs"),
                    server_duration_ms=_extract_float(start_data, "serverDurationMs"),
                    client_rtt_ms=start_response.client_rtt_ms,
                )
            )
        stress_id = start_data.get("id") if isinstance(start_data, dict) else None
        if stress_id:
            delete_response = client.request("DELETE", f"/gpu/stress/{stress_id}")
            delete_data = delete_response.json if isinstance(delete_response.json, dict) else {}
            if not (warmup and iteration == 0):
                results.append(
                    BenchResult(
                        timestamp_utc=utc_now_iso(),
                        endpoint="/gpu/stress/{id}",
                        pipeline="system_gpu_stress_delete",
                        backend=None,
                        optimized=None,
                        size_label="256mb",
                        params={"id": stress_id},
                        run_mode="sequential",
                        status=delete_response.status,
                        gpu_duration_ms=_extract_float(delete_data, "gpuDurationMs"),
                        backend_duration_ms=_extract_float(delete_data, "backendDurationMs"),
                        server_duration_ms=_extract_float(delete_data, "serverDurationMs"),
                        client_rtt_ms=delete_response.client_rtt_ms,
                    )
                )
    return results

