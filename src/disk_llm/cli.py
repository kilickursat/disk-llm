"""CLI entrypoints for Disk-LLM."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import time

from .converter import convert_model
from .demo import launch_demo
from .exceptions import DiskLLMError
from .inspect import inspect_packed_manifest, inspect_source_dir, render_inspection
from .manifest import PackedModelManifest
from .optional import require_auto_tokenizer
from .runtime import DiskLLMTextModel, TelemetryRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="disk-llm", description="Disk-backed LLM research kit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert a safetensors snapshot into Disk-LLM layout.")
    convert.add_argument("source_dir")
    convert.add_argument("output_dir")
    convert.add_argument("--family", default="qwen3.5")
    convert.add_argument("--variant", default="9b")
    convert.add_argument("--include-vision", action="store_true")
    convert.add_argument("--align-bytes", type=int, default=64)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--json", action="store_true", dest="json_output")

    inspect = subparsers.add_parser("inspect", help="Inspect a source snapshot or a packed manifest.")
    source_group = inspect.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-dir")
    source_group.add_argument("--manifest")
    inspect.add_argument("--include-vision", action="store_true")
    inspect.add_argument("--json", action="store_true", dest="json_output")

    generate = subparsers.add_parser("generate", help="Generate tokens from a packed Disk-LLM model.")
    _add_runtime_args(generate)
    generate.add_argument("--show-telemetry", action="store_true")

    bench = subparsers.add_parser("bench", help="Run simple generation benchmarks.")
    _add_runtime_args(bench)
    bench.add_argument("--runs", type=int, default=1)
    bench.add_argument("--json", action="store_true", dest="json_output")

    demo = subparsers.add_parser("demo", help="Launch the optional Gradio demo.")
    demo.add_argument("manifest_path")
    demo.add_argument("--tokenizer", required=True)
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=7860)

    return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest_path")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-ids")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int)


def _parse_prompt_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resolve_prompt_ids(args, manifest: PackedModelManifest) -> list[int]:
    if args.prompt_ids:
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
        if not prompt_ids:
            raise DiskLLMError("No token ids were parsed from --prompt-ids.")
        return prompt_ids

    tokenizer_path = args.tokenizer or manifest.source_dir
    AutoTokenizer = require_auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    return list(tokenizer.encode(args.prompt, add_special_tokens=False))


def command_convert(args) -> int:
    result = convert_model(
        args.source_dir,
        args.output_dir,
        family=args.family,
        variant=args.variant,
        text_only=not args.include_vision,
        align_bytes=args.align_bytes,
        overwrite=args.overwrite,
    )
    payload = {
        "manifest_path": str(result.manifest_path.resolve()),
        "tensor_count": len(result.manifest.tensors),
        "shard_count": len(result.manifest.shards),
        "skipped_tensor_count": len(result.manifest.skipped_tensors),
    }
    if args.json_output:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Manifest written to {payload['manifest_path']}")
        print(f"Packed tensors: {payload['tensor_count']}")
        print(f"Shards: {payload['shard_count']}")
        print(f"Skipped tensors: {payload['skipped_tensor_count']}")
    return 0


def command_inspect(args) -> int:
    if args.source_dir:
        summary = inspect_source_dir(args.source_dir, text_only=not args.include_vision)
    else:
        summary = inspect_packed_manifest(args.manifest)
    if args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        print(render_inspection(summary))
    return 0


def command_generate(args) -> int:
    manifest = PackedModelManifest.from_path(args.manifest_path)
    prompt_ids = _resolve_prompt_ids(args, manifest)
    model = DiskLLMTextModel.from_manifest(args.manifest_path)
    telemetry = TelemetryRecorder(prompt_tokens=len(prompt_ids))
    generated_ids, telemetry_payload = model.generate_token_ids(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        telemetry=telemetry,
    )

    if args.prompt:
        tokenizer_path = args.tokenizer or manifest.source_dir
        AutoTokenizer = require_auto_tokenizer()
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(text)
    else:
        print(",".join(str(token_id) for token_id in generated_ids))

    if args.show_telemetry:
        print(json.dumps(telemetry_payload, indent=2))
    return 0


def command_bench(args) -> int:
    manifest = PackedModelManifest.from_path(args.manifest_path)
    prompt_ids = _resolve_prompt_ids(args, manifest)
    model = DiskLLMTextModel.from_manifest(args.manifest_path)

    runs: list[dict[str, object]] = []
    for index in range(args.runs):
        telemetry = TelemetryRecorder(prompt_tokens=len(prompt_ids))
        started = time.perf_counter()
        generated_ids, telemetry_payload = model.generate_token_ids(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=None if args.seed is None else args.seed + index,
            telemetry=telemetry,
        )
        run_elapsed = time.perf_counter() - started
        runs.append(
            {
                "run": index,
                "elapsed_seconds": run_elapsed,
                "generated_token_count": len(generated_ids),
                "telemetry": telemetry_payload,
            }
        )

    summary = {
        "runs": runs,
        "avg_elapsed_seconds": sum(run["elapsed_seconds"] for run in runs) / len(runs),
        "avg_first_token_seconds": sum(
            (run["telemetry"].get("first_token_seconds") or 0.0) for run in runs
        )
        / len(runs),
    }

    if args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Runs: {len(runs)}")
        print(f"Average elapsed: {summary['avg_elapsed_seconds']:.6f}s")
        print(f"Average first token: {summary['avg_first_token_seconds']:.6f}s")
    return 0


def command_demo(args) -> int:
    launch_demo(
        args.manifest_path,
        tokenizer_path=args.tokenizer,
        host=args.host,
        port=args.port,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "convert":
            return command_convert(args)
        if args.command == "inspect":
            return command_inspect(args)
        if args.command == "generate":
            return command_generate(args)
        if args.command == "bench":
            return command_bench(args)
        if args.command == "demo":
            return command_demo(args)
    except DiskLLMError as exc:
        print(f"disk-llm: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
