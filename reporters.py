from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns
import json

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
    "time_between_frames_ms",
    "memory_gpu_bytes",
    "memory_host_bytes",
    "memory_server_rss_bytes",
    "gpu_init_ms",
    "accuracy",
]


def _slugify(value: str) -> str:
    cleaned = []
    for ch in value.strip().lower():
        cleaned.append(ch if ch.isalnum() else "_")
    result = "".join(cleaned).strip("_")
    return result or "unknown"


def _ordered_df(results: Iterable[BenchResult]) -> pd.DataFrame:
    rows = [asdict(result) for result in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ordered = [col for col in RESULT_COLUMNS if col in df.columns]
    extra = [col for col in df.columns if col not in ordered]
    df = df[ordered + extra]
    sort_cols = [col for col in ["endpoint", "pipeline", "backend", "size_label", "timestamp_utc"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df


def _choose_hue(plot_df: pd.DataFrame) -> str | None:
    if "backend" in plot_df.columns:
        backends = plot_df["backend"].dropna().unique()
        if len(backends) > 1:
            return "backend"
    if "pipeline" in plot_df.columns:
        pipelines = plot_df["pipeline"].dropna().unique()
        if len(pipelines) > 1:
            return "pipeline"
    return None


def _size_label_key(value: object) -> tuple:
    if value is None:
        return (2, "")
    text = str(value)
    if "x" in text:
        parts = text.split("x", maxsplit=1)
        try:
            return (0, int(parts[0]), int(parts[1]))
        except (TypeError, ValueError):
            return (1, text)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


def _plot_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    if metric not in df.columns or df.empty:
        return
    plot_df = df.copy()
    if "size_label" in plot_df.columns:
        plot_df = plot_df.sort_values(
            by="size_label",
            key=lambda series: series.map(_size_label_key),
        )
    hue = _choose_hue(plot_df)
    chart = sns.lineplot(
        data=plot_df,
        x="size_label",
        y=metric,
        hue=hue,
        marker="o",
    )
    fig = chart.get_figure()
    fig.tight_layout()
    fig.savefig(output_path)
    fig.clf()


def _normalize_params(df: pd.DataFrame) -> pd.DataFrame:
    if "params" not in df.columns:
        return df
    normalized = df.copy()
    def _to_key(value: object) -> str:
        if isinstance(value, dict):
            try:
                return json.dumps(value, sort_keys=True)
            except (TypeError, ValueError):
                return str(value)
        return str(value)
    normalized["params"] = normalized["params"].apply(_to_key)
    return normalized


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        "gpu_duration_ms",
        "backend_duration_ms",
        "server_duration_ms",
        "client_rtt_ms",
        "time_between_frames_ms",
        "memory_gpu_bytes",
        "memory_host_bytes",
        "memory_server_rss_bytes",
        "gpu_init_ms",
        "accuracy",
    ]
    id_cols = [
        col
        for col in [
            "endpoint",
            "pipeline",
            "backend",
            "optimized",
            "size_label",
            "params",
            "run_mode",
            "status",
        ]
        if col in df.columns
    ]
    grouped = df.groupby(id_cols, dropna=False, as_index=False)
    agg_map = {col: ["mean", "std"] for col in metric_cols if col in df.columns}
    aggregated = grouped.agg(agg_map)
    aggregated.columns = [
        f"{col}_{stat}" if stat else col for col, stat in aggregated.columns
    ]
    for col in metric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in aggregated.columns:
            aggregated[col] = aggregated[mean_col]
            aggregated.drop(columns=[mean_col], inplace=True)
        if std_col in aggregated.columns:
            aggregated[f"{col}_plus_minus"] = aggregated[std_col]
            aggregated.drop(columns=[std_col], inplace=True)
    return aggregated


def write_reports(results: Iterable[BenchResult], output_dir: Path, planned_total: int | None = None) -> None:
    df = _ordered_df(results)
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    grouped = df.groupby("endpoint", dropna=False)
    for endpoint, endpoint_df in grouped:
        endpoint_label = _slugify(str(endpoint))
        endpoint_dir = output_dir / endpoint_label
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        run_modes = endpoint_df["run_mode"].dropna().unique().tolist() if "run_mode" in endpoint_df.columns else []
        include_run_mode = len(run_modes) > 1
        run_mode_groups = endpoint_df.groupby("run_mode", dropna=False) if include_run_mode else [(None, endpoint_df)]

        for run_mode, run_df in run_mode_groups:
            run_suffix = f"_{run_mode}" if include_run_mode and run_mode is not None else ""
            if "optimized" in run_df.columns:
                opt_values = run_df["optimized"].dropna().unique().tolist()
            else:
                opt_values = []
            if not opt_values:
                opt_groups = [(None, run_df)]
            else:
                opt_groups = [(opt, run_df[run_df["optimized"] == opt]) for opt in opt_values]

            for opt_value, opt_df in opt_groups:
                opt_suffix = ""
                if opt_value is True:
                    opt_suffix = "_optimized_true"
                elif opt_value is False:
                    opt_suffix = "_optimized_false"

                csv_path = endpoint_dir / f"tabelki{run_suffix}{opt_suffix}.csv"
                export_df = _normalize_params(opt_df)
                if csv_path.exists():
                    existing_df = pd.read_csv(csv_path)
                    existing_df = _normalize_params(existing_df)
                    export_df = pd.concat([existing_df, export_df], ignore_index=True)
                export_df = _aggregate_metrics(export_df)
                export_df = export_df.dropna(axis=1, how="all")
                export_df.to_csv(csv_path, index=False)

                _plot_metric(opt_df, "backend_duration_ms", endpoint_dir / f"backend_duration_ms{run_suffix}{opt_suffix}.png")
                _plot_metric(opt_df, "gpu_duration_ms", endpoint_dir / f"gpu_duration_ms{run_suffix}{opt_suffix}.png")
                _plot_metric(opt_df, "client_rtt_ms", endpoint_dir / f"client_rtt_ms{run_suffix}{opt_suffix}.png")
