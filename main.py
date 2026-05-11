from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Final

from api_client import SyncApiClient
from benchmarks.ai import run_ai_benchmarks
from benchmarks.gpu_info import fetch_gpu_info
from benchmarks.image import run_image_benchmarks
from benchmarks.matrix import run_matrix_benchmarks
from benchmarks.render import run_render_benchmarks
from benchmarks.stress import run_stress_benchmarks
from benchmarks.video import run_video_benchmarks
from config import AppConfig, load_config
from reporters import plot_summary, write_csv

DEFAULT_RESULTS_DIR: Final[Path] = Path("results")
DEFAULT_ITERATIONS: Final[int] = 20
DEFAULT_STRESS_CONCURRENCY: Final[int] = 32
DEFAULT_STRESS_REQUESTS: Final[int] = 200


@dataclass(frozen=True)
class ArgsConfig(AppConfig):
    mode: str
    target: str | None
    iterations: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="webgpu-bench",
        description="Client-side benchmark runner for WebGPU/CUDA REST API.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "single", "full", "stress"],
        default="quick",
        help="Benchmark mode (quick/single/full/stress).",
    )
    parser.add_argument(
        "--target",
        help="Pipeline name for --mode single (matrix/image/video/ai/render).",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CUDA off regardless of USE_CUDA env flag.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Iterations per parameter combo (first is warmup).",
    )
    return parser


def _sanitize_token(value: str) -> str:
    cleaned = []
    for ch in value.strip().lower():
        cleaned.append(ch if ch.isalnum() else "_")
    result = "".join(cleaned).strip("_")
    return result or "unknown"


def _derive_results_filename(gpu_info: dict, use_cuda: bool) -> str:
    vendor = _sanitize_token(str(gpu_info.get("vendor", "unknown")))
    arch = _sanitize_token(str(gpu_info.get("architecture", "unknown")))
    backend = _sanitize_token(str(gpu_info.get("backend", "unknown")))
    cuda_label = "cuda" if use_cuda else "no_cuda"
    date = datetime.utcnow().strftime("%Y%m%d")
    return f"results_{vendor}_{arch}_{backend}_{cuda_label}_{date}.csv"


def _select_backends(use_cuda: bool, webgpu_label: str = "webgpu") -> list[str]:
    backends = [webgpu_label]
    if use_cuda:
        backends.append("cuda")
    return backends


def _select_render_backends(use_cuda: bool) -> list[str]:
    backends = ["webgpu-render", "webgpu-compute"]
    if use_cuda:
        backends.append("cuda")
    return backends


def _run_quick(client: SyncApiClient, config: ArgsConfig) -> list:
    results = []
    for backend in _select_backends(config.use_cuda):
        results += run_matrix_benchmarks(
            client,
            sizes=[256],
            backend=backend,
            optimized_variants=[False, True],
            iterations=2,
        )
        results += run_image_benchmarks(
            client,
            sizes=[(640, 360)],
            backend=backend,
            iterations=2,
        )
        results += run_video_benchmarks(
            client,
            backend=backend,
            frame_indices=[0],
            iterations=2,
        )
        results += run_ai_benchmarks(
            client,
            backend=backend,
            iterations=2,
            use_cuda=config.use_cuda,
        )
    for backend in _select_render_backends(config.use_cuda):
        results += run_render_benchmarks(
            client,
            counts=[1000],
            backend=backend,
            iterations=2,
        )
    return results


def _run_single(client: SyncApiClient, config: ArgsConfig) -> list:
    match config.target:
        case "matrix":
            results = []
            for backend in _select_backends(config.use_cuda):
                results += run_matrix_benchmarks(
                    client,
                    sizes=[256, 512, 1024, 2048],
                    backend=backend,
                    optimized_variants=[False, True],
                    iterations=config.iterations,
                )
            return results
        case "image":
            results = []
            for backend in _select_backends(config.use_cuda):
                results += run_image_benchmarks(
                    client,
                    sizes=[(640, 360), (1280, 720), (1920, 1080)],
                    backend=backend,
                    iterations=config.iterations,
                )
            return results
        case "video":
            results = []
            for backend in _select_backends(config.use_cuda):
                results += run_video_benchmarks(
                    client,
                    backend=backend,
                    frame_indices=[0, 50, 100],
                    iterations=config.iterations,
                )
            return results
        case "ai":
            results = []
            for backend in _select_backends(config.use_cuda):
                results += run_ai_benchmarks(
                    client,
                    backend=backend,
                    iterations=config.iterations,
                    use_cuda=config.use_cuda,
                )
            return results
        case "render":
            results = []
            for backend in _select_render_backends(config.use_cuda):
                results += run_render_benchmarks(
                    client,
                    counts=[1000, 2000, 4000],
                    backend=backend,
                    iterations=config.iterations,
                )
            return results
        case _:
            raise ValueError(f"Unknown target: {config.target}")


def _run_full(client: SyncApiClient, config: ArgsConfig) -> list:
    results = []
    results += _run_single(client, replace(config, target="matrix"))
    results += _run_single(client, replace(config, target="image"))
    results += _run_single(client, replace(config, target="video"))
    results += _run_single(client, replace(config, target="ai"))
    results += _run_single(client, replace(config, target="render"))
    results += asyncio.run(
        run_stress_benchmarks(
            config.server_url,
            concurrency=DEFAULT_STRESS_CONCURRENCY,
            total_requests=DEFAULT_STRESS_REQUESTS,
            use_cuda=config.use_cuda,
        )
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "single" and not args.target:
        parser.error("--target is required when --mode single is used.")
        return 2

    config = load_config(force_no_cuda=args.no_cuda)
    config = ArgsConfig(
        server_url=config.server_url,
        use_cuda=config.use_cuda,
        mode=args.mode,
        target=args.target,
        iterations=args.iterations,
    )

    client = SyncApiClient(config.server_url)
    try:
        gpu_info = fetch_gpu_info(client)
        output_name = _derive_results_filename(gpu_info, config.use_cuda)
        output_path = DEFAULT_RESULTS_DIR / output_name

        if config.mode == "stress":
            results = asyncio.run(
                run_stress_benchmarks(
                    config.server_url,
                    concurrency=DEFAULT_STRESS_CONCURRENCY,
                    total_requests=DEFAULT_STRESS_REQUESTS,
                    use_cuda=config.use_cuda,
                )
            )
        elif config.mode == "quick":
            results = _run_quick(client, config)
        elif config.mode == "single":
            results = _run_single(client, config)
        elif config.mode == "full":
            results = _run_full(client, config)
        else:
            parser.error(f"Unknown mode: {config.mode}")
            return 2

        write_csv(results, output_path)
        plot_summary(results, output_path.parent)
        print(f"Saved results to {output_path}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
