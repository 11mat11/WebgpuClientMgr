from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from array import array

from api_client import SyncApiClient
from benchmarks.common import BenchResult, utc_now_iso


MLP_INPUT_SIZE = 128 * 128
CNN_INPUT_SIZE = 3 * 128 * 128
AI_QUICK_REPEATS = 1
AI_FULL_REPEATS = 12


@dataclass(frozen=True)
class AiSample:
    label: int
    values: list[float]
    source: str


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


def _backend_flags(backend: str) -> dict[str, bool]:
    return {
        "webgpu": backend == "webgpu",
        "cuda": backend == "cuda",
    }


def check_ai_status(client: SyncApiClient, model: str, backend: str) -> dict:
    response = client.request("GET", "/ai/status")
    payload = response.json if isinstance(response.json, dict) else {}
    return payload if isinstance(payload, dict) else {}


def load_ai_model(client: SyncApiClient, model: str, backend: str) -> None:
    payload = {"model": model, **_backend_flags(backend)}
    response = client.request("POST", "/ai/load", json_body=payload)
    if response.status != 200:
        err_data = response.json if isinstance(response.json, dict) else {}
        error_msg = err_data.get('message', err_data.get('error', 'Brak szczegółów'))
        print(f"\033[91m[ai] Błąd LOAD: {response.status} - {error_msg} (model={model} backend={backend})\033[0m")
        return


def unload_ai_model(client: SyncApiClient, model: str, backend: str) -> None:
    payload = {"model": model, **_backend_flags(backend)}
    response = client.request("POST", "/ai/unload", json_body=payload)
    if response.status != 200:
        err_data = response.json if isinstance(response.json, dict) else {}
        error_msg = err_data.get('message', err_data.get('error', 'Brak szczegółów'))
        print(f"\033[91m[ai] Błąd UNLOAD: {response.status} - {error_msg} (model={model} backend={backend})\033[0m")
        return


def _ai_test_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "ai_test"


def _parse_label(file_name: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(file_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _load_float_array(file_path: Path, expected_len: int) -> list[float] | None:
    payload = file_path.read_bytes()
    values = array("f")
    values.frombytes(payload)
    if len(values) != expected_len:
        print(f"[ai] skip {file_path.name}: len={len(values)} expected={expected_len}")
        return None
    return list(values)


def _collect_samples(
    folder: Path,
    pattern: re.Pattern[str],
    expected_len: int,
    quick: bool,
    repeats: int,
) -> list[AiSample]:
    samples: list[AiSample] = []
    by_label: dict[int, Path] = {}
    for file_path in sorted(folder.glob("*.bin")):
        label = _parse_label(file_path.name, pattern)
        if label is None:
            continue
        if quick:
            if label not in by_label:
                by_label[label] = file_path
        else:
            by_label.setdefault(label, None)
            values = _load_float_array(file_path, expected_len)
            if values is None:
                continue
            samples.append(AiSample(label=label, values=values, source=file_path.name))

    if quick:
        for label in sorted(by_label.keys()):
            file_path = by_label[label]
            if file_path is None:
                continue
            values = _load_float_array(file_path, expected_len)
            if values is None:
                continue
            samples.append(AiSample(label=label, values=values, source=file_path.name))

    if repeats > 1:
        samples = samples * repeats
    return samples


def load_ai_samples(model: str, quick: bool) -> list[AiSample]:
    root = _ai_test_dir()
    if model == "mlp":
        folder = root / "benchmark_samples"
        pattern = re.compile(r"digit_(\d+)_sample_(\d+)\.bin")
        expected_len = MLP_INPUT_SIZE
    elif model == "cnn":
        folder = root / "cifar10_benchmark"
        pattern = re.compile(r"class_(\d+)_sample_(\d+)\.bin")
        expected_len = CNN_INPUT_SIZE
    else:
        raise ValueError(f"Unknown AI model: {model}")

    repeats = AI_QUICK_REPEATS if quick else AI_FULL_REPEATS
    return _collect_samples(folder, pattern, expected_len, quick=quick, repeats=repeats)


def count_ai_samples(model: str, quick: bool) -> int:
    root = _ai_test_dir()
    if model == "mlp":
        folder = root / "benchmark_samples"
        pattern = re.compile(r"digit_(\d+)_sample_(\d+)\.bin")
    elif model == "cnn":
        folder = root / "cifar10_benchmark"
        pattern = re.compile(r"class_(\d+)_sample_(\d+)\.bin")
    else:
        return 0

    labels: set[int] = set()
    total_files = 0
    for file_path in folder.glob("*.bin"):
        label = _parse_label(file_path.name, pattern)
        if label is None:
            continue
        total_files += 1
        labels.add(label)

    if quick:
        return len(labels)
    return total_files * AI_FULL_REPEATS


def _extract_prediction(data: dict) -> int | None:
    if not isinstance(data, dict):
        return None
    value = data.get("prediction")
    if value is None:
        value = data.get("predictionLabel")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_ai_inference_benchmarks(
    client: SyncApiClient,
    backend: str,
    model: str,
    samples: list[AiSample],
    iterations: int,
    warmup: bool = True,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    if model == "mlp":
        endpoint = "/ai/predict/mlp"
        pipeline = "ai_mlp"
    elif model == "cnn":
        endpoint = "/ai/predict/cnn"
        pipeline = "ai_cnn"
    else:
        raise ValueError(f"Unknown AI model: {model}")

    for iteration in range(iterations):
        for sample in samples:
            print(
                f"[ai] model={model} backend={backend} iter={iteration + 1}/{iterations} label={sample.label}"
            )
            payload = {"backend": backend, "input": sample.values}
            response = client.request("POST", endpoint, json_body=payload)
            data = response.json if isinstance(response.json, dict) else {}
            mem_gpu, mem_host, mem_rss = _extract_memory(data)
            if response.status != 200:
                error_msg = response.json.get('message', response.json.get('error',
                                                                           'Brak szczegółów')) if response.json else 'Brak odpowiedzi JSON'
                print(f"\033[91mBłąd predict AI: {response.status} - {error_msg}\033[0m")
                continue
            if warmup and iteration == 0:
                continue
            predicted = _extract_prediction(data)
            accuracy = None
            if predicted is not None:
                accuracy = 1.0 if predicted == sample.label else 0.0
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint=endpoint,
                    pipeline=pipeline,
                    backend=backend,
                    optimized=None,
                    size_label=str(sample.label),
                    params={"expectedLabel": sample.label, "source": sample.source},
                    run_mode="sequential",
                    status=response.status,
                    gpu_duration_ms=_extract_float(data, "gpuDurationMs"),
                    backend_duration_ms=_extract_float(data, "backendDurationMs"),
                    server_duration_ms=_extract_float(data, "serverDurationMs"),
                    client_rtt_ms=response.client_rtt_ms,
                    memory_gpu_bytes=mem_gpu,
                    memory_host_bytes=mem_host,
                    memory_server_rss_bytes=mem_rss,
                    accuracy=accuracy,
                )
            )
    return results
