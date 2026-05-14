from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from api_client import SyncApiClient
from benchmarks.ai import (
    check_ai_status,
    count_ai_samples,
    load_ai_model,
    load_ai_samples,
    run_ai_inference_benchmarks,
    unload_ai_model,
)
from benchmarks.gpu_info import fetch_gpu_info
from benchmarks.image import run_image_benchmarks
from benchmarks.matrix import (
    run_matrix_benchmarks,
    run_matrix_concurrency_benchmarks,
)
from benchmarks.render import run_render_benchmarks
from benchmarks.stress import run_stress_benchmarks
from benchmarks.video import pick_first_video_file, run_video_benchmarks
from config import AppConfig, load_config
from reporters import write_reports

DEFAULT_RESULTS_DIR: Final[Path] = Path("results")
DEFAULT_ITERATIONS: Final[int] = 30
DEFAULT_STRESS_CONCURRENCY: Final[int] = 64
DEFAULT_STRESS_REQUESTS: Final[int] = 1000
DEFAULT_STRESS_XL_REQUESTS: Final[int] = 200
DEFAULT_LOAD_CONCURRENCY: Final[int] = 16
DEFAULT_LOAD_REQUESTS: Final[int] = 100
DEFAULT_MATRIX_CONCURRENCY_MAX: Final[int] = 8
MATRIX_SIZES: Final[list[int]] = [256, 500, 512, 1000, 1024, 2048, 3000, 4096, 5000, 8192, 10000, 12000]
MATRIX_SIZES_QUICK: Final[list[int]] = [256, 512]
IMAGE_SIZES: Final[list[tuple[int, int]]] = [
    (320, 180),    # 180p
    (640, 360),    # 360p
    (960, 540),    # 540p (qHD)
    (1280, 720),   # 720p (HD)
    (1600, 900),   # 900p (HD+)
    (1920, 1080),  # 1080p (Full HD)
    (2560, 1440),  # 1440p (QHD / 2.5K)
    (3840, 2160),  # 4K (UHD)
    (5120, 2880),  # 5K
    (6144, 3456),  # 6K
    (7680, 4320),  # 8K (UHD-2)
]
IMAGE_SIZES_QUICK: Final[list[tuple[int, int]]] = [(320, 180), (640, 360)]
VIDEO_FRAMES: Final[list[int]] = list(range(20))
VIDEO_FRAMES_QUICK: Final[list[int]] = [0, 1, 2]
VIDEO_QUALITIES: Final[list[str]] = ["1080p", "720p", "480p", "160p"]
VIDEO_QUALITIES_QUICK: Final[list[str]] = ["480p", "160p"]
RENDER_COUNTS: Final[list[int]] = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
RENDER_COUNTS_QUICK: Final[list[int]] = [500, 1000]

FILTER_COUNT: Final[int] = 1
AI_MODELS: Final[list[str]] = ["cnn", "mlp"]
STRESS_SUBDIR: Final[str] = "stress"
STRESS_XL_SUBDIR: Final[str] = "stressxl"


@dataclass(frozen=True)
class ArgsConfig(AppConfig):
    mode: str
    target: str | None
    iterations: int
    load_test: bool
    load_concurrency: int
    load_requests: int
    quick_max: bool
    stress_xl: bool
    matrix_concurrency_max: int


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
    parser.add_argument(
        "--quick-max",
        action="store_true",
        help="In quick mode, run only the maximum sizes to verify upper bounds.",
    )
    parser.add_argument(
        "--load-test",
        action="store_true",
        help="Run large parallel requests to probe server under load.",
    )
    parser.add_argument(
        "--load-concurrency",
        type=int,
        default=DEFAULT_LOAD_CONCURRENCY,
        help="Concurrency for --load-test (parallel requests).",
    )
    parser.add_argument(
        "--load-requests",
        type=int,
        default=DEFAULT_LOAD_REQUESTS,
        help="Total requests for --load-test.",
    )
    parser.add_argument(
        "--matrix-concurrency-max",
        type=int,
        default=DEFAULT_MATRIX_CONCURRENCY_MAX,
        help="Max rownoleglych requestow dla testu duzych macierzy (1..N).",
    )
    parser.add_argument(
        "--stress-xl",
        action="store_true",
        help="Run extreme stress (full sizes, fewer requests) into stressxl/ (only with --mode stress).",
    )
    return parser


def _sanitize_token(value: str) -> str:
    cleaned = []
    for ch in value.strip().lower():
        cleaned.append(ch if ch.isalnum() else "_")
    result = "".join(cleaned).strip("_")
    return result or "unknown"


def _derive_results_dirname(gpu_info: dict, use_cuda: bool) -> str:
    vendor = _sanitize_token(str(gpu_info.get("vendor", "unknown")))
    arch = _sanitize_token(str(gpu_info.get("architecture", "unknown")))
    backend = _sanitize_token(str(gpu_info.get("backend", "unknown")))
    cuda_label = "cuda" if use_cuda else "no_cuda"
    date = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"results_{vendor}_{arch}_{backend}_{cuda_label}_{date}"


def _select_backends(use_cuda: bool, webgpu_label: str = "webgpu") -> list[str]:
    backends=[]
    if use_cuda:
        backends.append("cuda")
    backends.append(webgpu_label)
    return backends



def _select_render_backends(use_cuda: bool) -> list[str]:
    backends = []
    if use_cuda:
        backends.append("cuda")
    backends += ["webgpu-render", "webgpu-compute"]
    return backends


def _select_ai_backends(use_cuda: bool) -> list[str]:
    return ["cuda", "webgpu"] if use_cuda else ["webgpu"]


def _pick_values(values: list, use_max_only: bool) -> list:
    return [values[-1]] if use_max_only else values


def _half_list(values: list) -> list:
    if not values:
        return values
    return values[: max(1, len(values) // 2)]


def _matrix_concurrency_levels(max_concurrency: int, use_max_only: bool) -> list[int]:
    max_concurrency = max(1, max_concurrency)
    if use_max_only:
        return [max_concurrency]
    return list(range(1, max_concurrency + 1))


def _write_stage(results: list, output_dir: Path) -> None:
    if results:
        write_reports(results, output_dir)


def _log_progress(label: str, completed: int, planned_total: int | None) -> None:
    if planned_total is None or planned_total <= 0:
        return
    percent = (completed / planned_total) * 100.0
    print(f"[{label}] progress={percent:.1f}% ({completed}/{planned_total})")


def _run_matrix_pipeline(
    client: SyncApiClient,
    config: ArgsConfig,
    iterations: int,
    warmup: bool,
    sizes: list[int],
    use_max_only: bool,
    planned_total: int | None = None,
) -> list:
    results = []
    for backend in _select_backends(config.use_cuda):
        print(f"[matrix] backend={backend} sizes={sizes} optimized=[False, True]")
        results += run_matrix_benchmarks(
            client,
            sizes=_pick_values(sizes, use_max_only),
            backend=backend,
            optimized_variants=[False, True],
            iterations=iterations,
            warmup=warmup,
        )
    return results


def _run_matrix_concurrency_pipeline(
    config: ArgsConfig,
    iterations: int,
    concurrency_levels: list[int],
) -> list:
    if not concurrency_levels:
        return []
    results = []
    size = MATRIX_SIZES[-1]
    optimized = True
    for backend in _select_backends(config.use_cuda):
        results += run_matrix_concurrency_benchmarks(
            config.server_url,
            backend=backend,
            size=size,
            optimized=optimized,
            iterations=iterations,
            concurrency_levels=concurrency_levels,
        )
    return results


def _run_image_pipeline(
    client: SyncApiClient,
    config: ArgsConfig,
    iterations: int,
    warmup: bool,
    sizes: list[tuple[int, int]],
    use_max_only: bool,
    planned_total: int | None = None,
) -> list:
    results = []
    for backend in _select_backends(config.use_cuda):
        print(f"[image] backend={backend} sizes={sizes}")
        results += run_image_benchmarks(
            client,
            sizes=_pick_values(sizes, use_max_only),
            backend=backend,
            iterations=iterations,
            warmup=warmup,
        )
    return results


def _run_video_pipeline(
    client: SyncApiClient,
    config: ArgsConfig,
    iterations: int,
    warmup: bool,
    frame_indices: list[int],
    stream_qualities: list[str],
    stream_frames: int,
    use_max_only: bool,
    planned_total: int | None = None,
) -> list:
    results = []
    file_name = pick_first_video_file(client)
    if not file_name:
        print("[video] no files available; skipping video benchmarks")
        return results
    frame_indices = _pick_values(frame_indices, use_max_only)
    stream_qualities = _pick_values(stream_qualities, use_max_only)
    for backend in _select_backends(config.use_cuda):
        print(
            f"[video] backend={backend} file={file_name} frames={frame_indices} qualities={stream_qualities}"
        )
        results += run_video_benchmarks(
            client,
            config.server_url,
            backend=backend,
            frame_indices=frame_indices,
            iterations=iterations,
            warmup=warmup,
            stream_qualities=stream_qualities,
            stream_frames=stream_frames,
            file_name=file_name,
        )
    return results


def _run_ai_pipeline(
    client: SyncApiClient,
    config: ArgsConfig,
    quick_samples: bool,
    planned_total: int | None = None,
) -> list:
    results = []
    for backend in _select_ai_backends(config.use_cuda):
        for model in ("cnn", "mlp"):
            samples = load_ai_samples(model, quick=quick_samples)
            print(f"[ai] samples={len(samples)} model={model} backend={backend}")
            print(f"[ai] load model={model} backend={backend}")
            load_ai_model(client, model=model, backend=backend)
            check_ai_status(client, model=model, backend=backend)
            print(f"[ai] run model={model} backend={backend}")
            results += run_ai_inference_benchmarks(
                client,
                backend=backend,
                model=model,
                samples=samples,
                iterations=1,
                warmup=False,
            )
            print(f"[ai] unload model={model} backend={backend}")
            unload_ai_model(client, model=model, backend=backend)
            check_ai_status(client, model=model, backend=backend)
    return results


def _run_render_pipeline(
    client: SyncApiClient,
    config: ArgsConfig,
    iterations: int,
    warmup: bool,
    counts: list[int],
    use_max_only: bool,
    planned_total: int | None = None,
) -> list:
    results = []
    for backend in _select_render_backends(config.use_cuda):
        print(f"[render] backend={backend} counts={counts}")
        results += run_render_benchmarks(
            client,
            counts=_pick_values(counts, use_max_only),
            backend=backend,
            iterations=iterations,
            warmup=warmup,
        )
    return results


def _run_all(
    client: SyncApiClient,
    config: ArgsConfig,
    iterations: int,
    warmup: bool,
    use_max_only: bool,
) -> list:
    results = []
    results += _run_matrix_pipeline(
        client,
        config,
        iterations=iterations,
        warmup=warmup,
        sizes=MATRIX_SIZES,
        use_max_only=use_max_only,
    )
    results += _run_image_pipeline(
        client,
        config,
        iterations=iterations,
        warmup=warmup,
        sizes=IMAGE_SIZES,
        use_max_only=use_max_only,
    )
    results += _run_video_pipeline(
        client,
        config,
        iterations=iterations,
        warmup=warmup,
        frame_indices=VIDEO_FRAMES,
        stream_qualities=VIDEO_QUALITIES,
        stream_frames=1000,
        use_max_only=use_max_only,
    )
    results += _run_ai_pipeline(client, config, quick_samples=False)
    results += _run_render_pipeline(
        client,
        config,
        iterations=iterations,
        warmup=warmup,
        counts=RENDER_COUNTS,
        use_max_only=use_max_only,
    )
    return results


def _run_quick(
    client: SyncApiClient,
    config: ArgsConfig,
    planned_total: int | None,
    output_dir: Path,
) -> list:
    if config.quick_max:
        matrix_sizes = MATRIX_SIZES
        image_sizes = IMAGE_SIZES
        video_frames = VIDEO_FRAMES
        video_qualities = VIDEO_QUALITIES
        render_counts = RENDER_COUNTS
        stream_frames = 20
        use_max_only = True
        concurrency_levels = _matrix_concurrency_levels(config.matrix_concurrency_max, True)
    else:
        matrix_sizes = MATRIX_SIZES_QUICK
        image_sizes = IMAGE_SIZES_QUICK
        video_frames = VIDEO_FRAMES_QUICK
        video_qualities = VIDEO_QUALITIES_QUICK
        render_counts = RENDER_COUNTS_QUICK
        stream_frames = 20
        use_max_only = False
        concurrency_levels = [2]

    completed_total = 0
    results = []
    results += _run_matrix_pipeline(
        client,
        config,
        iterations=1,
        warmup=False,
        sizes=matrix_sizes,
        use_max_only=use_max_only,
    )
    completed_total += len(results)
    _log_progress("matrix", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_image_pipeline(
        client,
        config,
        iterations=1,
        warmup=False,
        sizes=image_sizes,
        use_max_only=use_max_only,
    )
    completed_total += len(results)
    _log_progress("image", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_video_pipeline(
        client,
        config,
        iterations=1,
        warmup=False,
        frame_indices=video_frames,
        stream_qualities=video_qualities,
        stream_frames=stream_frames,
        use_max_only=use_max_only,
    )
    completed_total += len(results)
    _log_progress("video", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_ai_pipeline(client, config, quick_samples=True)
    completed_total += len(results)
    _log_progress("ai", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_render_pipeline(
        client,
        config,
        iterations=1,
        warmup=False,
        counts=render_counts,
        use_max_only=use_max_only,
    )
    completed_total += len(results)
    _log_progress("render", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_matrix_concurrency_pipeline(
        config,
        iterations=1,
        concurrency_levels=concurrency_levels,
    )
    completed_total += len(results)
    _log_progress("matrix-concurrency", completed_total, planned_total)
    _write_stage(results, output_dir / "concurrency")
    return []


def _run_single(client: SyncApiClient, config: ArgsConfig, output_dir: Path, planned_total: int | None) -> list:
    match config.target:
        case "matrix":
            results = _run_matrix_pipeline(
                client,
                config,
                iterations=config.iterations,
                warmup=True,
                sizes=MATRIX_SIZES,
                use_max_only=False,
            )
            _log_progress("matrix", len(results), planned_total)
            _write_stage(results, output_dir)
            results = _run_matrix_concurrency_pipeline(
                config,
                iterations=config.iterations,
                concurrency_levels=_matrix_concurrency_levels(config.matrix_concurrency_max, False),
            )
            _log_progress("matrix-concurrency", len(results), planned_total)
            _write_stage(results, output_dir / "concurrency")
            return []
        case "image":
            results = _run_image_pipeline(
                client,
                config,
                iterations=config.iterations,
                warmup=True,
                sizes=IMAGE_SIZES,
                use_max_only=False,
            )
            _log_progress("image", len(results), planned_total)
            _write_stage(results, output_dir)
            return []
        case "video":
            results = _run_video_pipeline(
                client,
                config,
                iterations=config.iterations,
                warmup=True,
                frame_indices=VIDEO_FRAMES,
                stream_qualities=VIDEO_QUALITIES,
                stream_frames=1000,
                use_max_only=False,
            )
            _log_progress("video", len(results), planned_total)
            _write_stage(results, output_dir)
            return []
        case "ai":
            results = _run_ai_pipeline(client, config, quick_samples=False)
            _log_progress("ai", len(results), planned_total)
            _write_stage(results, output_dir)
            return []
        case "render":
            results = _run_render_pipeline(
                client,
                config,
                iterations=config.iterations,
                warmup=True,
                counts=RENDER_COUNTS,
                use_max_only=False,
            )
            _log_progress("render", len(results), planned_total)
            _write_stage(results, output_dir)
            return []
        case _:
            raise ValueError(f"Unknown target: {config.target}")


def _run_full(client: SyncApiClient, config: ArgsConfig, output_dir: Path, planned_total: int | None) -> list:
    completed_total = 0
    results = []
    results += _run_matrix_pipeline(
        client,
        config,
        iterations=config.iterations,
        warmup=True,
        sizes=MATRIX_SIZES,
        use_max_only=False,
    )
    completed_total += len(results)
    _log_progress("matrix", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_image_pipeline(
        client,
        config,
        iterations=config.iterations,
        warmup=True,
        sizes=IMAGE_SIZES,
        use_max_only=False,
    )
    completed_total += len(results)
    _log_progress("image", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_video_pipeline(
        client,
        config,
        iterations=config.iterations,
        warmup=True,
        frame_indices=VIDEO_FRAMES,
        stream_qualities=VIDEO_QUALITIES,
        stream_frames=1000,
        use_max_only=False,
    )
    completed_total += len(results)
    _log_progress("video", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_ai_pipeline(client, config, quick_samples=False)
    completed_total += len(results)
    _log_progress("ai", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_render_pipeline(
        client,
        config,
        iterations=config.iterations,
        warmup=True,
        counts=RENDER_COUNTS,
        use_max_only=False,
    )
    completed_total += len(results)
    _log_progress("render", completed_total, planned_total)
    _write_stage(results, output_dir)
    results = []
    results += _run_matrix_concurrency_pipeline(
        config,
        iterations=config.iterations,
        concurrency_levels=_matrix_concurrency_levels(config.matrix_concurrency_max, False),
    )
    completed_total += len(results)
    _log_progress("matrix-concurrency", completed_total, planned_total)
    _write_stage(results, output_dir / "concurrency")
    return []


def _plan_iterations(iterations: int, warmup: bool) -> int:
    return max(0, iterations - 1) if warmup else iterations


def _planned_total_for_target(
    target: str,
    iterations: int,
    warmup: bool,
    use_cuda: bool,
    sizes: list[int] | list[tuple[int, int]] | None = None,
    frames: list[int] | None = None,
    qualities: list[str] | None = None,
    stream_frames: int | None = None,
    use_max_only: bool = False,
    ai_quick: bool = False,
) -> int:
    iter_count = _plan_iterations(iterations, warmup)
    if target == "matrix":
        backends = len(_select_backends(use_cuda))
        size_count = len(_pick_values(sizes or [], use_max_only))
        return size_count * backends * 2 * iter_count
    if target == "matrix-concurrency":
        backends = len(_select_backends(use_cuda))
        levels = sizes or []
        return backends * iter_count * sum(int(level) for level in levels)
    if target == "image":
        backends = len(_select_backends(use_cuda))
        size_count = len(_pick_values(sizes or [], use_max_only))
        return size_count * backends * FILTER_COUNT * iter_count
    if target == "video":
        backends = len(_select_backends(use_cuda))
        frame_count = len(_pick_values(frames or [], use_max_only))
        quality_count = len(_pick_values(qualities or [], use_max_only))
        stream_frame_count = stream_frames or 0
        histogram_total = frame_count * backends * iter_count
        stream_total = backends * (quality_count * (stream_frame_count + 1))
        return histogram_total + stream_total
    if target == "ai":
        backends = len(_select_ai_backends(use_cuda))
        return backends * sum(count_ai_samples(model, quick=ai_quick) for model in AI_MODELS)
    if target == "render":
        backends = len(_select_render_backends(use_cuda))
        count_total = len(_pick_values(sizes or [], use_max_only))
        return count_total * backends * iter_count
    return 0


def _planned_total(config: ArgsConfig) -> int:
    if config.mode == "stress":
        planned = DEFAULT_STRESS_XL_REQUESTS if config.stress_xl else DEFAULT_STRESS_REQUESTS
    elif config.mode == "quick":
        if config.quick_max:
            planned = _planned_total_for_target(
                "matrix",
                1,
                False,
                config.use_cuda,
                sizes=MATRIX_SIZES,
                use_max_only=True,
            )
            planned += _planned_total_for_target(
                "matrix-concurrency",
                1,
                False,
                config.use_cuda,
                sizes=_matrix_concurrency_levels(config.matrix_concurrency_max, True),
            )
            planned += _planned_total_for_target(
                "image",
                1,
                False,
                config.use_cuda,
                sizes=IMAGE_SIZES,
                use_max_only=True,
            )
            planned += _planned_total_for_target(
                "video",
                1,
                False,
                config.use_cuda,
                frames=VIDEO_FRAMES,
                qualities=VIDEO_QUALITIES,
                stream_frames=20,
                use_max_only=True,
            )
            planned += _planned_total_for_target(
                "ai",
                1,
                False,
                config.use_cuda,
                ai_quick=True,
            )
            planned += _planned_total_for_target(
                "render",
                1,
                False,
                config.use_cuda,
                sizes=RENDER_COUNTS,
                use_max_only=True,
            )

        else:
            planned = _planned_total_for_target(
                "matrix",
                1,
                False,
                config.use_cuda,
                sizes=MATRIX_SIZES_QUICK,
            )
            planned += _planned_total_for_target(
                "matrix-concurrency",
                1,
                False,
                config.use_cuda,
                sizes=[2],
            )
            planned += _planned_total_for_target(
                "image",
                1,
                False,
                config.use_cuda,
                sizes=IMAGE_SIZES_QUICK,
            )
            planned += _planned_total_for_target(
                "video",
                1,
                False,
                config.use_cuda,
                frames=VIDEO_FRAMES_QUICK,
                qualities=VIDEO_QUALITIES_QUICK,
                stream_frames=20,
            )
            planned += _planned_total_for_target(
                "ai",
                1,
                False,
                config.use_cuda,
                ai_quick=True,
            )
            planned += _planned_total_for_target(
                "render",
                1,
                False,
                config.use_cuda,
                sizes=RENDER_COUNTS_QUICK,
            )

    elif config.mode == "single":
        target = config.target or ""
        if target == "matrix":
            planned = _planned_total_for_target(
                "matrix",
                config.iterations,
                True,
                config.use_cuda,
                sizes=MATRIX_SIZES,
            )
            planned += _planned_total_for_target(
                "matrix-concurrency",
                config.iterations,
                False,
                config.use_cuda,
                sizes=_matrix_concurrency_levels(config.matrix_concurrency_max, False),
            )
        elif target == "image":
            planned = _planned_total_for_target(
                "image",
                config.iterations,
                True,
                config.use_cuda,
                sizes=IMAGE_SIZES,
            )
        elif target == "video":
            planned = _planned_total_for_target(
                "video",
                config.iterations,
                True,
                config.use_cuda,
                frames=VIDEO_FRAMES,
                qualities=VIDEO_QUALITIES,
                stream_frames=1000,
            )
        elif target == "ai":
            planned = _planned_total_for_target(
                "ai",
                config.iterations,
                True,
                config.use_cuda,
                ai_quick=False,
            )
        elif target == "render":
            planned = _planned_total_for_target(
                "render",
                config.iterations,
                True,
                config.use_cuda,
                sizes=RENDER_COUNTS,
            )
        else:
            planned = 0

    else:
        planned = _planned_total_for_target(
            "matrix",
            config.iterations,
            True,
            config.use_cuda,
            sizes=MATRIX_SIZES,
        )
        planned += _planned_total_for_target(
            "matrix-concurrency",
            config.iterations,
            False,
            config.use_cuda,
            sizes=_matrix_concurrency_levels(config.matrix_concurrency_max, False),
        )
        planned += _planned_total_for_target(
            "image",
            config.iterations,
            True,
            config.use_cuda,
            sizes=IMAGE_SIZES,
        )
        planned += _planned_total_for_target(
            "video",
            config.iterations,
            True,
            config.use_cuda,
            frames=VIDEO_FRAMES,
            qualities=VIDEO_QUALITIES,
            stream_frames=1000,
        )
        planned += _planned_total_for_target(
            "ai",
            config.iterations,
            True,
            config.use_cuda,
            ai_quick=False,
        )
        planned += _planned_total_for_target(
            "render",
            config.iterations,
            True,
            config.use_cuda,
            sizes=RENDER_COUNTS,
        )

    if config.load_test:
        planned += config.load_requests
    return planned


def _run_stress_sequence(
    config: ArgsConfig,
    output_dir: Path,
    profile: str,
    matrix_sizes: list[int],
    image_sizes: list[tuple[int, int]],
    render_counts: list[int],
    total_requests: int,
    subdir: str,
) -> None:
    stress_dir = output_dir / subdir
    if config.use_cuda:
        stages = [
            ("cuda", ["cuda"], ["cuda"]),
            ("webgpu", ["webgpu"], ["webgpu-render", "webgpu-compute"]),
        ]
    else:
        stages = [("webgpu", ["webgpu"], ["webgpu-render", "webgpu-compute"])]

    for label, backends, render_backends in stages:
        print(f"[stress] stage={label} profile={profile} requests={total_requests}")
        results = asyncio.run(
            run_stress_benchmarks(
                config.server_url,
                concurrency=DEFAULT_STRESS_CONCURRENCY,
                total_requests=total_requests,
                backends=backends,
                render_backends=render_backends,
                profile=profile,
                matrix_sizes=matrix_sizes,
                image_sizes=image_sizes,
                render_counts=render_counts,
            )
        )
        _write_stage(results, stress_dir)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "single" and not args.target:
        parser.error("--target is required when --mode single is used.")
        return 2
    if args.stress_xl and args.mode != "stress":
        parser.error("--stress-xl is only valid with --mode stress.")
        return 2

    config = load_config(force_no_cuda=args.no_cuda)
    config = ArgsConfig(
        server_url=config.server_url,
        use_cuda=config.use_cuda,
        mode=args.mode,
        target=args.target,
        iterations=args.iterations,
        load_test=args.load_test,
        load_concurrency=args.load_concurrency,
        load_requests=args.load_requests,
        quick_max=args.quick_max,
        stress_xl=args.stress_xl,
        matrix_concurrency_max=args.matrix_concurrency_max,
    )

    client = SyncApiClient(config.server_url)
    try:
        gpu_info = fetch_gpu_info(client)
        output_name = _derive_results_dirname(gpu_info, config.use_cuda)
        output_dir = DEFAULT_RESULTS_DIR / output_name
        planned_total = _planned_total(config)

        if config.mode == "stress":
            if config.stress_xl:
                _run_stress_sequence(
                    config,
                    output_dir,
                    profile="stress",
                    matrix_sizes=MATRIX_SIZES,
                    image_sizes=IMAGE_SIZES,
                    render_counts=RENDER_COUNTS,
                    total_requests=DEFAULT_STRESS_XL_REQUESTS,
                    subdir=STRESS_XL_SUBDIR,
                )
            else:
                _run_stress_sequence(
                    config,
                    output_dir,
                    profile="stress",
                    matrix_sizes=_half_list(MATRIX_SIZES),
                    image_sizes=_half_list(IMAGE_SIZES),
                    render_counts=_half_list(RENDER_COUNTS),
                    total_requests=DEFAULT_STRESS_REQUESTS,
                    subdir=STRESS_SUBDIR,
                )
            results = []
        elif config.mode == "quick":
            results = _run_quick(client, config, planned_total, output_dir)
        elif config.mode == "single":
            results = _run_single(client, config, output_dir, planned_total)
        elif config.mode == "full":
            results = _run_full(client, config, output_dir, planned_total)
        else:
            parser.error(f"Unknown mode: {config.mode}")
            return 2

        if config.mode == "full":
            _run_stress_sequence(
                config,
                output_dir,
                profile="stress",
                matrix_sizes=_half_list(MATRIX_SIZES),
                image_sizes=_half_list(IMAGE_SIZES),
                render_counts=_half_list(RENDER_COUNTS),
                total_requests=DEFAULT_STRESS_REQUESTS,
                subdir=STRESS_SUBDIR,
            )

        if config.load_test:
            print(
                f"[load] concurrency={config.load_concurrency} requests={config.load_requests}"
            )
            _run_stress_sequence(
                config,
                output_dir,
                profile="load",
                matrix_sizes=_half_list(MATRIX_SIZES),
                image_sizes=_half_list(IMAGE_SIZES),
                render_counts=_half_list(RENDER_COUNTS),
                total_requests=config.load_requests,
                subdir=STRESS_SUBDIR,
            )


        print(f"Saved results to {output_dir}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
