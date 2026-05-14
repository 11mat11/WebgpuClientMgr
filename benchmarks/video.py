from __future__ import annotations

import asyncio
import json
import ssl
import time
from typing import Iterable
from urllib.parse import urlparse

import websockets

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


def pick_first_video_file(client: SyncApiClient) -> str | None:
    response = client.request("GET", "/video/list")
    data = response.json if isinstance(response.json, dict) else {}
    files = data.get("files") if isinstance(data, dict) else None
    if isinstance(files, list) and files:
        selected = str(files[0])
        print(f"[video] list_status={response.status} files={len(files)} selected={selected}")
        return selected
    print(f"[video] list_status={response.status} files=0 payload={data}")
    return None


def _build_ws_url(base_url: str) -> tuple[str, ssl.SSLContext | None]:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    ssl_context = None
    if scheme == "wss":
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    return f"{scheme}://{netloc}/video/stream", ssl_context


def run_video_histogram_benchmarks(
    client: SyncApiClient,
    backend: str,
    frame_indices: Iterable[int],
    iterations: int,
    file_name: str,
    warmup: bool = True,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for frame in frame_indices:
        for iteration in range(iterations):
            print(
                f"[video-hist] frame={frame} backend={backend} iter={iteration + 1}/{iterations}"
            )
            payload = {
                "fileName": file_name,
                "frameIndex": frame,
                "backend": backend,
            }
            response = client.request("POST", "/video/histogram", json_body=payload)
            data = response.json if isinstance(response.json, dict) else {}
            mem_gpu, mem_host, mem_rss = _extract_memory(data)
            if response.status != 200:
                error_msg = data.get('message', data.get('error', 'Brak szczegółów'))
                print(
                    f"\033[91m[video_hist] Błąd: {response.status} - {error_msg} (file={file_name} frame={frame} backend={backend})\033[0m")
                continue
            if warmup and iteration == 0:
                continue
            results.append(
                BenchResult(
                    timestamp_utc=utc_now_iso(),
                    endpoint="/video/histogram",
                    pipeline="video_histogram",
                    backend=backend,
                    optimized=None,
                    size_label=str(frame),
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


async def _run_stream_for_quality(
        ws_url: str,
        file_name: str,
        backend: str,
        quality: str,
        frame_count: int,
        ssl_context: ssl.SSLContext | None,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    last_frame_time: float | None = None

    try:
        async with websockets.connect(ws_url, max_size=None, ssl=ssl_context) as websocket:
            print(
                f"[video-stream] backend={backend} quality={quality} frames={frame_count}"
            )
            select_payload = {
                "action": "select",
                "fileName": file_name,
                "backend": backend,
                "quality": quality,
                "compress": False,
            }
            await websocket.send(json.dumps(select_payload))
            try:
                while True:
                    message = await websocket.recv()
                    payload = json.loads(message)

                    if isinstance(payload, dict) and payload.get("error"):
                        print(
                            f"\033[91m[video_stream] Błąd z serwera: {payload.get('error')} (quality={quality})\033[0m")
                        return results

                    if isinstance(payload, dict) and payload.get("type") == "selected":
                        mem_gpu, mem_host, mem_rss = _extract_memory(payload)
                        results.append(
                            BenchResult(
                                timestamp_utc=utc_now_iso(),
                                endpoint="/video/stream",
                                pipeline="video_stream_init",
                                backend=backend,
                                optimized=None,
                                size_label=quality,
                                params={"fileName": file_name, "quality": quality},
                                run_mode="stream",
                                status=200,
                                gpu_duration_ms=None,
                                backend_duration_ms=None,
                                server_duration_ms=None,
                                client_rtt_ms=0.0,
                                memory_gpu_bytes=mem_gpu,
                                memory_host_bytes=mem_host,
                                memory_server_rss_bytes=mem_rss,
                                gpu_init_ms=_extract_float(payload, "gpuInitTimeMs"),
                            )
                        )
                        break

                frames_collected = 0
                while frames_collected < frame_count:
                    message = await websocket.recv()
                    payload = json.loads(message)

                    if isinstance(payload, dict) and payload.get("error"):
                        print(f"\033[91m[video_stream] Błąd w trakcie streamu: {payload.get('error')}\033[0m")
                        break

                    if not isinstance(payload, dict) or payload.get("type") != "frame":
                        continue
                    payload.pop("frameDataBase64", None)
                    now = time.perf_counter()
                    if last_frame_time is None:
                        frame_rtt_ms = 0.0
                    else:
                        frame_rtt_ms = (now - last_frame_time) * 1000.0
                    last_frame_time = now
                    mem_gpu, mem_host, mem_rss = _extract_memory(payload)
                    results.append(
                        BenchResult(
                            timestamp_utc=utc_now_iso(),
                            endpoint="/video/stream",
                            pipeline="video_stream",
                            backend=backend,
                            optimized=None,
                            size_label=quality,
                            params={"fileName": file_name, "quality": quality},
                            run_mode="stream",
                            status=200,
                            gpu_duration_ms=_extract_float(payload, "gpuDurationMs"),
                            backend_duration_ms=_extract_float(payload, "backendDurationMs"),
                            server_duration_ms=_extract_float(payload, "serverDurationMs"),
                            client_rtt_ms=frame_rtt_ms,
                            time_between_frames_ms=frame_rtt_ms,
                            memory_gpu_bytes=mem_gpu,
                            memory_host_bytes=mem_host,
                            memory_server_rss_bytes=mem_rss,
                        )
                    )
                    frames_collected += 1
            finally:
                try:
                    await websocket.send(json.dumps({"action": "stop"}))
                    while True:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        payload = json.loads(message)
                        if isinstance(payload, dict) and payload.get("type") == "stopped":
                            break
                except Exception:
                    pass
                await websocket.close()

    except Exception as e:
        print(f"\033[91m[video_stream] Błąd zerwania połączenia WebSocket: {e} (quality={quality})\033[0m")

    return results


async def run_video_stream_benchmarks(
    base_url: str,
    backend: str,
    file_name: str,
    qualities: Iterable[str],
    frame_count: int,
) -> list[BenchResult]:
    ws_url, ssl_context = _build_ws_url(base_url)
    results: list[BenchResult] = []
    for quality in qualities:
        results += await _run_stream_for_quality(
            ws_url=ws_url,
            file_name=file_name,
            backend=backend,
            quality=quality,
            frame_count=frame_count,
            ssl_context=ssl_context,
        )
    return results


def run_video_benchmarks(
    client: SyncApiClient,
    base_url: str,
    backend: str,
    frame_indices: Iterable[int],
    iterations: int,
    warmup: bool,
    stream_qualities: Iterable[str],
    stream_frames: int,
    file_name: str,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    results += run_video_histogram_benchmarks(
        client=client,
        backend=backend,
        frame_indices=frame_indices,
        iterations=iterations,
        file_name=file_name,
        warmup=warmup,
    )
    results += asyncio.run(
        run_video_stream_benchmarks(
            base_url=base_url,
            backend=backend,
            file_name=file_name,
            qualities=stream_qualities,
            frame_count=stream_frames,
        )
    )
    return results

