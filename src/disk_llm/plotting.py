"""Offline plotting helpers for Disk-LLM benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .optional import require_plotly


def generate_plots(results_dir: str | Path) -> dict[str, Path]:
    results_path = Path(results_dir)
    summary_rows = _load_csv(results_path / "benchmark_summary.csv")
    timeline_rows = _load_csv(results_path / "memory_timeline.csv")
    metadata = _load_metadata(results_path / "benchmark_metadata.json")

    plots_dir = results_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = plots_dir / "tokens_per_second.png"
    latency_path = plots_dir / "first_token_latency.png"
    timeline_path = plots_dir / "rss_timeline.png"
    logical_mapped_path = plots_dir / "logical_mapped.png"
    markdown_path = plots_dir / "comparison_summary.md"

    _plot_grouped_metric(summary_rows, metric_key="tokens_per_second_mean", error_key="tokens_per_second_stdev", ylabel="Generated tokens / second", title="Disk-LLM throughput vs. baseline", output_path=tokens_path, metadata=metadata)
    _plot_line_metric(summary_rows, metric_key="first_token_seconds_mean", ylabel="First-token latency (s)", title="First-token latency by prompt length", output_path=latency_path, metadata=metadata)
    _plot_line_metric(summary_rows, metric_key="logical_bytes_mapped_mb_mean", ylabel="Logical Mapped (MB)", title="Total logical bytes mapped by prompt length", output_path=logical_mapped_path, metadata=metadata)
    _plot_timeline(timeline_rows, output_path=timeline_path, metadata=metadata)
    _write_markdown_summary(summary_rows, metadata, markdown_path)

    return {
        "tokens_per_second": tokens_path,
        "first_token_latency": latency_path,
        "rss_timeline": timeline_path,
        "logical_mapped_mb": logical_mapped_path,
        "comparison_summary": markdown_path,
    }


def _plot_grouped_metric(summary_rows: list[dict[str, Any]], *, metric_key: str, error_key: str, ylabel: str, title: str, output_path: Path, metadata: dict[str, Any]) -> None:
    go, _ = require_plotly()
    max_new_tokens_values = sorted({int(row["max_new_tokens"]) for row in summary_rows})
    prompt_tokens = sorted({int(row["prompt_tokens"]) for row in summary_rows})
    backends = _ordered_backends(summary_rows)
    colors = {"disk_llm": "#00d2ff", "hf_cpu": "#ff8a00"}  # Vibrant modern palette

    for max_new_tokens in max_new_tokens_values:
        fig = go.Figure()
        subset = [row for row in summary_rows if int(row["max_new_tokens"]) == max_new_tokens]
        for backend in backends:
            backend_rows = {int(row["prompt_tokens"]): row for row in subset if row["backend"] == backend}
            values = [_maybe_float(backend_rows.get(pt, {}).get(metric_key)) for pt in prompt_tokens]
            errors = [_maybe_float(backend_rows.get(pt, {}).get(error_key)) for pt in prompt_tokens]
            
            # Filter None to avoid broken bars
            clean_values = [v if v is not None else 0 for v in values]
            clean_errors = [e if e is not None else 0 for e in errors]

            fig.add_trace(go.Bar(
                name=_backend_label(backend, summary_rows),
                x=[str(p) for p in prompt_tokens],
                y=clean_values,
                error_y=dict(type='data', array=clean_errors, visible=True, thickness=1.5, color='rgba(255,255,255,0.7)'),
                marker_color=colors.get(backend, "#5b7083"),
                marker_line=dict(width=1.5, color='rgba(255, 255, 255, 0.2)'),
            ))

        fig.update_layout(
            title=f"<b>{title}</b><br><sup>(max_new_tokens={max_new_tokens})</sup>",
            xaxis_title="Prompt tokens",
            yaxis_title=ylabel,
            barmode='group',
            template='plotly_dark',
            plot_bgcolor='rgba(15, 15, 20, 1)',
            paper_bgcolor='rgba(15, 15, 20, 1)',
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Generate for the last max_new_tokens map if there are multiple
        # (Assuming we only have 1 value for max_new_tokens usually based on original behavior)
        fig.write_image(str(output_path), scale=2.5)


def _plot_line_metric(summary_rows: list[dict[str, Any]], *, metric_key: str, ylabel: str, title: str, output_path: Path, metadata: dict[str, Any]) -> None:
    go, _ = require_plotly()
    max_new_tokens_values = sorted({int(row["max_new_tokens"]) for row in summary_rows})
    backends = _ordered_backends(summary_rows)
    colors = {"disk_llm": "#00f2fe", "hf_cpu": "#f093fb"}

    for max_new_tokens in max_new_tokens_values:
        fig = go.Figure()
        subset = [row for row in summary_rows if int(row["max_new_tokens"]) == max_new_tokens]
        for backend in backends:
            backend_rows = sorted((row for row in subset if row["backend"] == backend), key=lambda row: int(row["prompt_tokens"]))
            x_vals = [str(int(row["prompt_tokens"])) for row in backend_rows]
            y_vals = [_maybe_float(row.get(metric_key)) for row in backend_rows]

            fig.add_trace(go.Scatter(
                name=_backend_label(backend, summary_rows),
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=colors.get(backend, "#5b7083"), width=3, shape='spline', smoothing=0.3),
                marker=dict(size=10, line=dict(width=2, color='rgba(255,255,255,0.8)')),
            ))

        fig.update_layout(
            title=f"<b>{title}</b><br><sup>(max_new_tokens={max_new_tokens})</sup>",
            xaxis_title="Prompt tokens",
            yaxis_title=ylabel,
            template='plotly_dark',
            plot_bgcolor='rgba(15, 15, 20, 1)',
            paper_bgcolor='rgba(15, 15, 20, 1)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', type='category'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.write_image(str(output_path), scale=2.5)


def _plot_timeline(timeline_rows: list[dict[str, Any]], *, output_path: Path, metadata: dict[str, Any]) -> None:
    go, _ = require_plotly()
    if not timeline_rows:
        fig = go.Figure()
        fig.add_annotation(text="No memory timeline data recorded.", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(template='plotly_dark', xaxis=dict(visible=False), yaxis=dict(visible=False))
        fig.write_image(str(output_path), scale=2)
        return

    focus_prompt_tokens = max(int(row["prompt_tokens"]) for row in timeline_rows)
    focus_max_new_tokens = max(int(row["max_new_tokens"]) for row in timeline_rows)
    candidates = [row for row in timeline_rows if int(row["prompt_tokens"]) == focus_prompt_tokens and int(row["max_new_tokens"]) == focus_max_new_tokens]
    
    selected_run_ids: dict[str, tuple[int, str]] = {}
    for row in candidates:
        backend = row["backend"]
        run_index = int(row["run_index"])
        current = selected_run_ids.get(backend)
        if current is None or run_index >= current[0]:
            selected_run_ids[backend] = (run_index, row["run_id"])

    colors = {"disk_llm": "#00f2fe", "hf_cpu": "#f093fb"}
    fig = go.Figure()
    
    for backend, (_, run_id) in sorted(selected_run_ids.items()):
        series = sorted((row for row in candidates if row["run_id"] == run_id), key=lambda row: int(row["sample_index"]))
        x_vals = [_maybe_float(row.get("elapsed_seconds")) for row in series]
        y_vals = [_maybe_float(row.get("rss_mb")) for row in series]
        
        fig.add_trace(go.Scatter(
            name=_backend_label(backend, candidates),
            x=x_vals,
            y=y_vals,
            mode='lines',
            fill='tozeroy',  # Area chart makes timeline look incredible
            fillcolor=colors.get(backend, "#5b7083").replace(')', ', 0.15)').replace('rgb', 'rgba') if '#' not in colors.get(backend, "") else None,
            line=dict(color=colors.get(backend, "#5b7083"), width=2.5),
        ))

    fig.update_layout(
        title=f"<b>RSS memory over time</b><br><sup>(prompt_tokens={focus_prompt_tokens}, max_new_tokens={focus_max_new_tokens})</sup>",
        xaxis_title="Elapsed seconds",
        yaxis_title="RSS (MB)",
        template='plotly_dark',
        plot_bgcolor='rgba(15, 15, 20, 1)',
        paper_bgcolor='rgba(15, 15, 20, 1)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_image(str(output_path), scale=2.5)


def _write_markdown_summary(summary_rows: list[dict[str, Any]], metadata: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"- Manifest: `{metadata.get('manifest_path', 'unknown')}`",
        f"- Runs per case: `{metadata.get('runs', 'unknown')}`",
        f"- Warmup runs per case: `{metadata.get('warmup_runs', 'unknown')}`",
        "",
        "| Backend | Prompt Tokens | Max New Tokens | Mean Tokens/s | Mean First Token (s) | Mean Peak RSS (MB) | Mean IO Read (MB) | Mean Logical Mapped (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (int(item["prompt_tokens"]), int(item["max_new_tokens"]), item["backend"])):
        lines.append("| {backend} | {prompt_tokens} | {max_new_tokens} | {tps} | {latency} | {rss} | {io_read} | {mapped} |".format(backend=row["backend_label"], prompt_tokens=row["prompt_tokens"], max_new_tokens=row["max_new_tokens"], tps=_format_number(row.get("tokens_per_second_mean")), latency=_format_number(row.get("first_token_seconds_mean")), rss=_format_number(row.get("rss_mb_peak_mean")), io_read=_format_number(row.get("io_read_mb_mean")), mapped=_format_number(row.get("logical_bytes_mapped_mb_mean"))))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_backends(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        if row["backend"] not in ordered:
            ordered.append(row["backend"])
    return ordered


def _backend_label(backend: str, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row["backend"] == backend:
            return row.get("backend_label", backend)
    return backend


def _figure_title(base_title: str, metadata: dict[str, Any]) -> str:
    manifest_path = metadata.get("manifest_path")
    return base_title if not manifest_path else f"{base_title}\n{Path(manifest_path).name}"


def _maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _format_number(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    return f"{float(value):.3f}"
