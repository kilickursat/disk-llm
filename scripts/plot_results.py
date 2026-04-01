"""Generate comparison plots from benchmark CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from disk_llm.exceptions import DiskLLMError
from disk_llm.plotting import generate_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate standard Disk-LLM benchmark plots.")
    parser.add_argument("results_dir", help="Directory containing benchmark_runs.csv, memory_timeline.csv, and benchmark_summary.csv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_paths = generate_plots(args.results_dir)
    except DiskLLMError as exc:
        print(f"disk-llm plots: {exc}", file=sys.stderr)
        return 2
    print(f"Plot artifacts written to {(Path(args.results_dir) / 'plots').resolve()}")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
