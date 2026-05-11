from __future__ import annotations

from api_client import SyncApiClient
from benchmarks.common import BenchResult, utc_now_iso


MLP_INPUT_SIZE = 128 * 128
CNN_INPUT_SIZE = 3 * 128 * 128


def _extract_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_models_loaded(client: SyncApiClient, use_cuda: bool) -> None:
    payload = {"webgpu": True, "cuda": use_cuda}
    client.request("POST", "/ai/load", json_body=payload)


def run_ai_benchmarks(
    client: SyncApiClient,
    backend: str,
    iterations: int,
    use_cuda: bool,
) -> list[BenchResult]:
    _ensure_models_loaded(client, use_cuda)
    results: list[BenchResult] = []
    for iteration in range(iterations):
        mlp_payload = {
            "backend": backend,
            "input": [0.01] * MLP_INPUT_SIZE,
        }
        mlp_response = client.request("POST", "/ai/predict/mlp", json_body=mlp_payload)
        mlp_data = mlp_response.json if isinstance(mlp_response.json, dict) else {}
        if iteration != 0:
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint="/ai/predict/mlp",
                    pipeline="ai_mlp",
                    backend=backend,
                    optimized=None,
                    size_label=str(MLP_INPUT_SIZE),
                    params={},
                    run_mode="sequential",
                    status=mlp_response.status,
                    gpu_duration_ms=_extract_float(mlp_data, "gpuDurationMs"),
                    backend_duration_ms=_extract_float(mlp_data, "backendDurationMs"),
                    server_duration_ms=_extract_float(mlp_data, "serverDurationMs"),
                    client_rtt_ms=mlp_response.client_rtt_ms,
                )
            )

        cnn_payload = {
            "backend": backend,
            "input": [0.01] * CNN_INPUT_SIZE,
        }
        cnn_response = client.request("POST", "/ai/predict/cnn", json_body=cnn_payload)
        cnn_data = cnn_response.json if isinstance(cnn_response.json, dict) else {}
        if iteration != 0:
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint="/ai/predict/cnn",
                    pipeline="ai_cnn",
                    backend=backend,
                    optimized=None,
                    size_label=str(CNN_INPUT_SIZE),
                    params={},
                    run_mode="sequential",
                    status=cnn_response.status,
                    gpu_duration_ms=_extract_float(cnn_data, "gpuDurationMs"),
                    backend_duration_ms=_extract_float(cnn_data, "backendDurationMs"),
                    server_duration_ms=_extract_float(cnn_data, "serverDurationMs"),
                    client_rtt_ms=cnn_response.client_rtt_ms,
                )
            )
    return results
