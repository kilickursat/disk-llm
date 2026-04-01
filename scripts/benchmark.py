"""Run repeated Disk-LLM benchmark cases and save CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from disk_llm.benchmarking import parse_int_list, parse_name_list, run_benchmark_suite, write_benchmark_artifacts
from disk_llm.exceptions import DiskLLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible Disk-LLM benchmark cases.")
    parser.add_argument("manifest_path", help="Packed Disk-LLM manifest.json path.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Text prompt to tokenize.")
    prompt_group.add_argument("--prompt-file", help="Path to a UTF-8 text file used as the base prompt.")
    prompt_group.add_argument("--prompt-ids", help="Comma-separated token ids used as the base prompt.")
    parser.add_argument("--tokenizer", help="Tokenizer path for --prompt or --prompt-file. Defaults to manifest.source_dir.")
    parser.add_argument("--prompt-lengths", default="8,64,256,512", help="Comma-separated prompt lengths to benchmark.")
    parser.add_argument("--max-new-tokens", default="16", help="Comma-separated max_new_tokens cases.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per case.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmup runs per case before recording metrics.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backends", default="disk_llm", help="Comma-separated backends: disk_llm,hf_cpu")
    parser.add_argument("--hf-model", help="Path to a Hugging Face source snapshot for the hf_cpu baseline.")
    parser.add_argument("--sample-interval-ms", type=float, default=25.0, help="RSS/IO sampling interval in milliseconds.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote-code model/tokenizer adapters for Hugging Face.")
    parser.add_argument("--output-dir", default="benchmark-results", help="Directory for CSV artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark_suite(
            args.manifest_path,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            prompt_ids=args.prompt_ids,
            tokenizer_path=args.tokenizer,
            prompt_lengths=parse_int_list(args.prompt_lengths),
            max_new_tokens_values=parse_int_list(args.max_new_tokens),
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            backends=parse_name_list(args.backends),
            hf_model_path=args.hf_model,
            sample_interval_seconds=args.sample_interval_ms / 1000.0,
            trust_remote_code=args.trust_remote_code,
        )
        paths = write_benchmark_artifacts(report, args.output_dir)
    except DiskLLMError as exc:
        print(f"disk-llm benchmark: {exc}", file=sys.stderr)
        return 2

    print(f"Benchmark artifacts written to {Path(args.output_dir).resolve()}")
    print(f"Runs CSV: {paths['runs']}")
    print(f"Timeline CSV: {paths['timeline']}")
    print(f"Summary CSV: {paths['summary']}")
    print(f"Metadata JSON: {paths['metadata']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
