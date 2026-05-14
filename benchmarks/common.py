from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class BenchResult:
    timestamp_utc: str
    endpoint: str
    pipeline: str
    backend: str | None
    optimized: bool | None
    size_label: str
    params: dict[str, Any]
    run_mode: str
    status: int
    gpu_duration_ms: float | None
    backend_duration_ms: float | None
    server_duration_ms: float | None
    client_rtt_ms: float
    time_between_frames_ms: float | None = None
    memory_gpu_bytes: float | None = None
    memory_host_bytes: float | None = None
    memory_server_rss_bytes: float | None = None
    gpu_init_ms: float | None = None
    accuracy: float | None = None


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

