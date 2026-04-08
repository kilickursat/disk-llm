"""Export public benchmark plots for README and GitHub Pages."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BACKGROUND = "#07111f"
PANEL = "#0f1b33"
GRID = "#203252"
TEXT = "#f2f7ff"
MUTED = "#a8c1e8"
DISK_LLM = "#52e1ff"
HF_CPU = "#ff8c42"
PALETTE = {"Disk-LLM": DISK_LLM, "HF CPU": HF_CPU}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Benchmark result directory containing CSV outputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported plot images.")
    parser.add_argument("--prefix", required=True, help="Filename prefix for the generated plot set.")
    parser.add_argument("--title-prefix", default="Disk-LLM vs HF CPU", help="Prefix used in plot titles.")
    return parser.parse_args()


def configure_theme() -> None:
    sns.set_theme(style="darkgrid")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "text.color": TEXT,
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": BACKGROUND,
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )


def _apply_common(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=16, pad=16)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _label_bars(ax: plt.Axes, fmt: str) -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, padding=3, color=TEXT, fontsize=9)


def _plot_summary_metric(df: pd.DataFrame, *, metric: str, ylabel: str, title: str, output_path: Path, label_fmt: str, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    sns.barplot(
        data=df,
        x="prompt_tokens",
        y=metric,
        hue="backend_label",
        palette=PALETTE,
        ax=ax,
    )
    _apply_common(ax, title=title, ylabel=ylabel)
    if log_scale:
        ax.set_yscale("log")
    _label_bars(ax, label_fmt)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_logical_mapped(df: pd.DataFrame, *, title: str, output_path: Path) -> None:
    logical = df[df["backend_label"] == "Disk-LLM"].copy()
    logical["logical_gib"] = logical["logical_bytes_mapped_mb_mean"] / 1024.0
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    sns.barplot(data=logical, x="prompt_tokens", y="logical_gib", color=DISK_LLM, ax=ax)
    ax.set_title(title, loc="left", fontsize=16, pad=16)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Logical bytes mapped (GiB)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:,.0f}" for value in container.datavalues], padding=3, color=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _downsample(group: pd.DataFrame, limit: int = 1200) -> pd.DataFrame:
    if len(group) <= limit:
        return group
    step = max(len(group) // limit, 1)
    return group.iloc[::step]


def _plot_rss_timeline(df: pd.DataFrame, *, title: str, output_path: Path) -> None:
    timeline = df.copy()
    timeline["prompt_tokens"] = timeline["prompt_tokens"].astype(int)
    timeline["rss_gib"] = timeline["rss_mb"] / 1024.0
    sampled_groups = []
    for _, group in timeline.groupby(["backend_label", "prompt_tokens", "run_id"], sort=False):
        sampled_groups.append(_downsample(group))
    sampled = pd.concat(sampled_groups, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    sns.lineplot(
        data=sampled,
        x="elapsed_seconds",
        y="rss_gib",
        hue="backend_label",
        style="prompt_tokens",
        palette=PALETTE,
        linewidth=2.3,
        ax=ax,
    )
    ax.set_title(title, loc="left", fontsize=16, pad=16)
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("RSS (GiB)")
    ax.legend(frameon=False, labelcolor=TEXT, title=None)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_theme()

    result_dir = args.result_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(result_dir / "benchmark_summary.csv")
    timeline = pd.read_csv(result_dir / "memory_timeline.csv")

    summary["prompt_tokens"] = summary["prompt_tokens"].astype(int)
    summary["tokens_per_second_mean"] = summary["tokens_per_second_mean"].astype(float)
    summary["first_token_seconds_mean"] = summary["first_token_seconds_mean"].astype(float)
    summary["rss_mb_peak_mean"] = summary["rss_mb_peak_mean"].astype(float) / 1024.0
    summary["logical_bytes_mapped_mb_mean"] = pd.to_numeric(summary["logical_bytes_mapped_mb_mean"], errors="coerce")

    prefix = args.prefix
    title_prefix = args.title_prefix
    _plot_summary_metric(
        summary,
        metric="tokens_per_second_mean",
        ylabel="Generated tokens / second",
        title=f"{title_prefix} | Throughput",
        output_path=output_dir / f"{prefix}-throughput.png",
        label_fmt="{:.3f}",
    )
    _plot_summary_metric(
        summary,
        metric="first_token_seconds_mean",
        ylabel="First token latency (s, log scale)",
        title=f"{title_prefix} | First-token latency",
        output_path=output_dir / f"{prefix}-first-token-latency.png",
        label_fmt="{:.1f}",
        log_scale=True,
    )
    _plot_logical_mapped(
        summary,
        title=f"{title_prefix} | Logical bytes mapped",
        output_path=output_dir / f"{prefix}-logical-mapped.png",
    )
    _plot_rss_timeline(
        timeline,
        title=f"{title_prefix} | RSS timeline",
        output_path=output_dir / f"{prefix}-rss-timeline.png",
    )


if __name__ == "__main__":
    main()
