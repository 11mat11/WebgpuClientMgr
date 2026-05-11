from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns

from benchmarks.common import BenchResult


RESULT_COLUMNS = [
    "timestamp_utc",
    "endpoint",
    "pipeline",
    "backend",
    "optimized",
    "size_label",
    "params",
    "run_mode",
    "status",
    "gpu_duration_ms",
    "backend_duration_ms",
    "server_duration_ms",
    "client_rtt_ms",
]


def write_csv(results: Iterable[BenchResult], output_path: Path) -> None:
    rows = [asdict(result) for result in results]
    df = pd.DataFrame(rows)
    if not df.empty:
        ordered = [col for col in RESULT_COLUMNS if col in df.columns]
        extra = [col for col in df.columns if col not in ordered]
        df = df[ordered + extra]
        df = df.sort_values(["pipeline", "backend", "size_label", "timestamp_utc"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    if metric not in df.columns:
        return
    plot_df = df.copy()
    if "optimized" in plot_df.columns:
        plot_df["optimized_label"] = plot_df["optimized"].fillna("na").astype(str)
        style = "optimized_label"
    else:
        style = None
    chart = sns.lineplot(
        data=plot_df,
        x="size_label",
        y=metric,
        hue="pipeline",
        style=style,
        marker="o",
    )
    fig = chart.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png")
    fig.clf()


def plot_summary(results: Iterable[BenchResult], output_dir: Path) -> None:
    rows = [asdict(result) for result in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    _plot_metric(df, "backend_duration_ms", output_dir)
    _plot_metric(df, "gpu_duration_ms", output_dir)
    _plot_metric(df, "client_rtt_ms", output_dir)
