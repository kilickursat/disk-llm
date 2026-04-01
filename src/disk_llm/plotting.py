"""Offline plotting helpers for Disk-LLM benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .optional import require_matplotlib_pyplot


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
    markdown_path = plots_dir / "comparison_summary.md"

    _plot_grouped_metric(summary_rows, metric_key="tokens_per_second_mean", error_key="tokens_per_second_stdev", ylabel="Generated tokens / second", title="Disk-LLM throughput vs. baseline", output_path=tokens_path, metadata=metadata)
    _plot_line_metric(summary_rows, metric_key="first_token_seconds_mean", ylabel="First-token latency (s)", title="First-token latency by prompt length", output_path=latency_path, metadata=metadata)
    _plot_timeline(timeline_rows, output_path=timeline_path, metadata=metadata)
    _write_markdown_summary(summary_rows, metadata, markdown_path)

    return {
        "tokens_per_second": tokens_path,
        "first_token_latency": latency_path,
        "rss_timeline": timeline_path,
        "comparison_summary": markdown_path,
    }


def _plot_grouped_metric(summary_rows: list[dict[str, Any]], *, metric_key: str, error_key: str, ylabel: str, title: str, output_path: Path, metadata: dict[str, Any]) -> None:
    plt = require_matplotlib_pyplot()
    max_new_tokens_values = sorted({int(row["max_new_tokens"]) for row in summary_rows})
    prompt_tokens = sorted({int(row["prompt_tokens"]) for row in summary_rows})
    backends = _ordered_backends(summary_rows)
    colors = {"disk_llm": "#1f5aa6", "hf_cpu": "#d98e04"}

    figure, axes = plt.subplots(len(max_new_tokens_values), 1, figsize=(10, max(4.5, 4.2 * len(max_new_tokens_values))), squeeze=False)
    for axis, max_new_tokens in zip(axes.flatten(), max_new_tokens_values):
        subset = [row for row in summary_rows if int(row["max_new_tokens"]) == max_new_tokens]
        base_positions = list(range(len(prompt_tokens)))
        width = 0.8 / max(len(backends), 1)
        for backend_index, backend in enumerate(backends):
            backend_rows = {int(row["prompt_tokens"]): row for row in subset if row["backend"] == backend}
            positions = [position - 0.4 + width / 2 + backend_index * width for position in base_positions]
            values = [_maybe_float(backend_rows.get(prompt_token, {}).get(metric_key)) for prompt_token in prompt_tokens]
            errors = [_maybe_float(backend_rows.get(prompt_token, {}).get(error_key)) for prompt_token in prompt_tokens]
            axis.bar(positions, values, width=width, yerr=errors, label=_backend_label(backend, summary_rows), color=colors.get(backend, "#5b7083"), capsize=4, edgecolor="white", linewidth=0.7)
        axis.set_title(f"{title} (max_new_tokens={max_new_tokens})", fontsize=12, fontweight="bold")
        axis.set_xticks(base_positions)
        axis.set_xticklabels([str(value) for value in prompt_tokens])
        axis.set_xlabel("Prompt tokens")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, linestyle=":")
        axis.legend(frameon=False)
    figure.suptitle(_figure_title(title, metadata), fontsize=14, fontweight="bold", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_line_metric(summary_rows: list[dict[str, Any]], *, metric_key: str, ylabel: str, title: str, output_path: Path, metadata: dict[str, Any]) -> None:
    plt = require_matplotlib_pyplot()
    max_new_tokens_values = sorted({int(row["max_new_tokens"]) for row in summary_rows})
    backends = _ordered_backends(summary_rows)
    colors = {"disk_llm": "#1f5aa6", "hf_cpu": "#d98e04"}

    figure, axes = plt.subplots(len(max_new_tokens_values), 1, figsize=(10, max(4.5, 4.0 * len(max_new_tokens_values))), squeeze=False)
    for axis, max_new_tokens in zip(axes.flatten(), max_new_tokens_values):
        subset = [row for row in summary_rows if int(row["max_new_tokens"]) == max_new_tokens]
        for backend in backends:
            backend_rows = sorted((row for row in subset if row["backend"] == backend), key=lambda row: int(row["prompt_tokens"]))
            axis.plot([int(row["prompt_tokens"]) for row in backend_rows], [_maybe_float(row.get(metric_key)) for row in backend_rows], marker="o", linewidth=2.2, markersize=6, label=_backend_label(backend, summary_rows), color=colors.get(backend, "#5b7083"))
        axis.set_title(f"{title} (max_new_tokens={max_new_tokens})", fontsize=12, fontweight="bold")
        axis.set_xlabel("Prompt tokens")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linestyle=":")
        axis.legend(frameon=False)
    figure.suptitle(_figure_title(title, metadata), fontsize=14, fontweight="bold", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_timeline(timeline_rows: list[dict[str, Any]], *, output_path: Path, metadata: dict[str, Any]) -> None:
    plt = require_matplotlib_pyplot()
    if not timeline_rows:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.text(0.5, 0.5, "No memory timeline data recorded.", ha="center", va="center")
        axis.axis("off")
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
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

    colors = {"disk_llm": "#1f5aa6", "hf_cpu": "#d98e04"}
    figure, axis = plt.subplots(figsize=(10, 5))
    for backend, (_, run_id) in sorted(selected_run_ids.items()):
        series = sorted((row for row in candidates if row["run_id"] == run_id), key=lambda row: int(row["sample_index"]))
        axis.plot([_maybe_float(row.get("elapsed_seconds")) for row in series], [_maybe_float(row.get("rss_mb")) for row in series], linewidth=2.2, label=_backend_label(backend, candidates), color=colors.get(backend, "#5b7083"))
    axis.set_title(f"RSS over time (prompt_tokens={focus_prompt_tokens}, max_new_tokens={focus_max_new_tokens})", fontsize=12, fontweight="bold")
    axis.set_xlabel("Elapsed seconds")
    axis.set_ylabel("RSS (MB)")
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(frameon=False)
    figure.suptitle(_figure_title("Memory timeline", metadata), fontsize=14, fontweight="bold", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


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
