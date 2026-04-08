"""Run the full Disk-LLM Qwen benchmark workflow on Modal."""

from __future__ import annotations

# --- Windows console encoding fix ---
# Rich's legacy Windows renderer writes Unicode symbols that the default
# cp1252 code page cannot encode, causing UnicodeEncodeError.  Force UTF-8
# for all console I/O before any other imports touch the console.
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass  # Already reconfigured or not a TextIOWrapper
# --- end fix ---

import argparse
import json
from pathlib import Path
import re
import shlex
from typing import Any

from modal.app import App
from modal.image import Image
from modal.output import enable_output
from modal.secret import Secret
from modal.volume import Volume


APP_NAME = "disk-llm-qwen-benchmark"
DEFAULT_VOLUME_NAME = "disk-llm-benchmarks"
VOLUME_ROOT = Path("/vol")
MODELS_ROOT = VOLUME_ROOT / "models"
PACKED_ROOT = VOLUME_ROOT / "packed"
RESULTS_ROOT = VOLUME_ROOT / "results"


def _parse_local_env(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'").strip('"')
        values[key] = value
    return values


def _hf_secret_from_env() -> Secret:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_values = _parse_local_env(".env")
        hf_token = env_values.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN was not found in .env. Add it locally before running the Modal workflow.")
    return Secret.from_dict({"HF_TOKEN": hf_token})


volume = Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["hf", "bench"])
    .pip_install(
        "huggingface_hub>=0.31.0",
        "hf_xet>=1.1.0",
        "torch>=2.6.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("disk_llm")
)

app = App(APP_NAME)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    cpu=16.0,
    memory=98304,
    volumes={str(VOLUME_ROOT): volume},
    secrets=[_hf_secret_from_env()],
)
def run_qwen_benchmark(
    *,
    repo_id: str = "Qwen/Qwen3.5-9B",
    revision: str = "main",
    prompt: str = "Explain disk-backed inference in one paragraph.",
    prompt_lengths: list[int] | None = None,
    max_new_tokens_values: list[int] | None = None,
    runs: int = 3,
    warmup_runs: int = 0,
    backends: list[str] | None = None,
    hf_dtype: str = "float32",
    trust_remote_code: bool = True,
    run_label: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime, timezone
    import shutil

    from huggingface_hub import HfApi, snapshot_download

    from disk_llm.benchmarking import run_benchmark_suite, write_benchmark_artifacts
    from disk_llm.converter import convert_model
    from disk_llm.inspect import inspect_packed_manifest, inspect_source_dir
    from disk_llm.plotting import generate_plots

    prompt_lengths = prompt_lengths or [8, 64, 256, 512]
    max_new_tokens_values = max_new_tokens_values or [16]
    backends = backends or ["disk_llm", "hf_cpu"]

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the Modal runtime environment.")

    model_slug = repo_id.replace("/", "--")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    effective_label = run_label or timestamp

    source_dir = MODELS_ROOT / model_slug / revision
    packed_dir = PACKED_ROOT / model_slug / effective_label
    results_dir = RESULTS_ROOT / model_slug / effective_label
    for path in (source_dir, packed_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    model_info = api.model_info(repo_id, revision=revision)
    resolved_sha = getattr(model_info, "sha", None)

    print(f"[modal] downloading or reusing {repo_id}@{revision}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(source_dir),
        resume_download=True,
    )
    volume.commit()

    print("[modal] inspecting source snapshot")
    source_summary = inspect_source_dir(source_dir, text_only=True)
    _write_json(results_dir / "source_inspection.json", source_summary)

    print("[modal] converting source snapshot into Disk-LLM layout")
    shutil.rmtree(packed_dir, ignore_errors=True)
    packed_dir.mkdir(parents=True, exist_ok=True)
    conversion = convert_model(
        source_dir,
        packed_dir,
        overwrite=True,
        notes=[
            f"repo_id={repo_id}",
            f"requested_revision={revision}",
            f"resolved_sha={resolved_sha}",
        ],
    )
    packed_summary = inspect_packed_manifest(conversion.manifest_path)
    _write_json(results_dir / "packed_inspection.json", packed_summary)

    print("[modal] running repeated benchmarks")
    report = run_benchmark_suite(
        conversion.manifest_path,
        prompt=prompt,
        tokenizer_path=source_dir,
        prompt_lengths=prompt_lengths,
        max_new_tokens_values=max_new_tokens_values,
        runs=runs,
        warmup_runs=warmup_runs,
        backends=backends,
        hf_model_path=source_dir,
        hf_dtype=hf_dtype,
        trust_remote_code=trust_remote_code,
    )
    benchmark_paths = write_benchmark_artifacts(report, results_dir)

    print("[modal] generating plots and summaries")
    plot_paths = generate_plots(results_dir)
    insight_path = results_dir / "benchmark_insights.md"
    insight_path.write_text(
        _build_insights_markdown(
            summary_rows=report.summary_rows,
            repo_id=repo_id,
            revision=revision,
            resolved_sha=resolved_sha,
            source_summary=source_summary,
            packed_summary=packed_summary,
            hf_dtype=hf_dtype,
        ),
        encoding="utf-8",
    )

    final_report = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_sha": resolved_sha,
        "source_dir": str(source_dir),
        "packed_dir": str(packed_dir),
        "results_dir": str(results_dir),
        "volume_results_path": _volume_path(results_dir),
        "manifest_path": str(conversion.manifest_path),
        "benchmark_paths": {name: str(path) for name, path in benchmark_paths.items()},
        "plot_paths": {name: str(path) for name, path in plot_paths.items()},
        "insight_path": str(insight_path),
        "backends": backends,
        "hf_dtype": hf_dtype,
        "prompt_lengths": prompt_lengths,
        "max_new_tokens_values": max_new_tokens_values,
        "runs": runs,
        "warmup_runs": warmup_runs,
    }
    _write_json(results_dir / "modal_run_report.json", final_report)
    volume.commit()
    return final_report


def run_local(
    repo_id: str = "Qwen/Qwen3.5-9B",
    revision: str = "main",
    prompt: str = "Explain disk-backed inference in one paragraph.",
    prompt_lengths: str = "8 64 256 512",
    max_new_tokens: str = "16",
    runs: int = 3,
    warmup_runs: int = 0,
    backends: str = "disk_llm,hf_cpu",
    hf_dtype: str = "float32",
    run_label: str = "",
    local_report_path: str = "",
) -> None:
    with enable_output():
        with app.run():
            result = run_qwen_benchmark.remote(
                repo_id=repo_id,
                revision=revision,
                prompt=prompt,
                prompt_lengths=_parse_int_csv(prompt_lengths),
                max_new_tokens_values=_parse_int_csv(max_new_tokens),
                runs=runs,
                warmup_runs=warmup_runs,
                backends=_parse_name_csv(backends),
                hf_dtype=hf_dtype,
                run_label=run_label or None,
            )
    if local_report_path:
        report_path = Path(local_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    mounted_results_dir = result["results_dir"]
    volume_results_path = result.get("volume_results_path") or _normalize_volume_results_path(mounted_results_dir)
    quoted_remote_path = shlex.quote(volume_results_path)
    print()
    print("Modal workflow finished.")
    print(f"Results live in Volume '{DEFAULT_VOLUME_NAME}' under {volume_results_path}")
    print(f"Mounted inside the Modal container at {mounted_results_dir}")
    print("To copy them back locally later, run:")
    print(f"  modal volume get {DEFAULT_VOLUME_NAME} {quoted_remote_path} ./modal-results")


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part) for part in re.split(r"[\s,;]+", raw.strip()) if part]


def _parse_name_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_volume_results_path(raw: str) -> str:
    if raw.startswith("/vol/"):
        return raw[len("/vol") :]
    return raw


def _volume_path(path: Path) -> str:
    return "/" + path.relative_to(VOLUME_ROOT).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_insights_markdown(
    *,
    summary_rows: list[dict[str, Any]],
    repo_id: str,
    revision: str,
    resolved_sha: str | None,
    source_summary: dict[str, Any],
    packed_summary: dict[str, Any],
    hf_dtype: str,
) -> str:
    lines = [
        "# Disk-LLM Modal Benchmark Insights",
        "",
        f"- Model: `{repo_id}`",
        f"- Requested revision: `{revision}`",
        f"- Resolved SHA: `{resolved_sha or 'unknown'}`",
        f"- HF CPU dtype: `{hf_dtype}`",
        f"- Source tensors discovered: `{source_summary.get('tensor_count', 'unknown')}`",
        f"- Packed tensors: `{packed_summary.get('tensor_count', 'unknown')}`",
        f"- Packed shards: `{packed_summary.get('shard_count', 'unknown')}`",
        "",
    ]

    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = (int(row["prompt_tokens"]), int(row["max_new_tokens"]))
        grouped.setdefault(key, {})[row["backend"]] = row

    if not grouped:
        lines.append("No benchmark summary rows were produced.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Key Comparisons",
            "",
        ]
    )
    for (prompt_tokens, max_new_tokens), backend_rows in sorted(grouped.items()):
        disk = backend_rows.get("disk_llm")
        hf = backend_rows.get("hf_cpu")
        lines.append(f"### Prompt {prompt_tokens} / Generate {max_new_tokens}")
        if disk is None:
            lines.append("- Disk-LLM data is missing for this case.")
            lines.append("")
            continue

        lines.append(
            "- Disk-LLM mean throughput: `{}` tok/s; first-token latency: `{}` s; peak RSS: `{}` MB.".format(
                _fmt(disk.get("tokens_per_second_mean")),
                _fmt(disk.get("first_token_seconds_mean")),
                _fmt(disk.get("rss_mb_peak_mean")),
            )
        )
        if hf is None:
            lines.append("- HF CPU baseline data is missing for this case.")
            lines.append("")
            continue
        lines.append(
            "- HF CPU mean throughput: `{}` tok/s; first-token latency: `{}` s; peak RSS: `{}` MB.".format(
                _fmt(hf.get("tokens_per_second_mean")),
                _fmt(hf.get("first_token_seconds_mean")),
                _fmt(hf.get("rss_mb_peak_mean")),
            )
        )
        throughput_ratio = _ratio(disk.get("tokens_per_second_mean"), hf.get("tokens_per_second_mean"))
        latency_ratio = _ratio(disk.get("first_token_seconds_mean"), hf.get("first_token_seconds_mean"))
        rss_ratio = _ratio(disk.get("rss_mb_peak_mean"), hf.get("rss_mb_peak_mean"))
        lines.append(
            "- Relative to HF CPU, Disk-LLM throughput ratio is `{}`, first-token latency ratio is `{}`, and peak RSS ratio is `{}`.".format(
                _fmt(throughput_ratio),
                _fmt(latency_ratio),
                _fmt(rss_ratio),
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Disk-LLM is most compelling if it maintains a materially lower RAM ceiling while staying within an acceptable throughput band versus the full-RAM HF baseline.",
            "- Modal Volume-backed results should be described as remote-volume benchmarks, not bare-metal local-SSD benchmarks.",
            "- If parity or quality looks weak despite reasonable throughput, the next likely target is the delta-block path and exact tensor-name coverage in the runtime adapter.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    return f"{float(value):.3f}"


def _ratio(left: Any, right: Any) -> float | None:
    if left in ("", None) or right in ("", None):
        return None
    right_value = float(right)
    if right_value == 0:
        return None
    return float(left) / right_value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Disk-LLM Qwen benchmark workflow on Modal.")
    parser.add_argument("--repo-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt", default="Explain disk-backed inference in one paragraph.")
    parser.add_argument("--prompt-lengths", default="8 64 256 512")
    parser.add_argument("--max-new-tokens", default="16")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--backends", default="disk_llm,hf_cpu")
    parser.add_argument("--hf-dtype", default="float32")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--local-report-path", default="")
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_local(
        repo_id=args.repo_id,
        revision=args.revision,
        prompt=args.prompt,
        prompt_lengths=args.prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        backends=args.backends,
        hf_dtype=args.hf_dtype,
        run_label=args.run_label,
        local_report_path=args.local_report_path,
    )
