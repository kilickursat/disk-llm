"""Research benchmark helpers for Disk-LLM and optional CPU baselines."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import threading
import time
from typing import Any, Sequence

from .exceptions import DependencyMissingError, DiskLLMError
from .manifest import PackedModelManifest
from .optional import require_auto_model_for_causal_lm, require_auto_tokenizer, require_psutil
from .runtime import DiskLLMTextModel, TelemetryRecorder

MB = 1024 * 1024
RUN_FIELDNAMES = [
    "run_id", "backend", "backend_label", "prompt_label", "prompt_tokens", "max_new_tokens",
    "run_index", "run_phase", "seed", "temperature", "top_p", "generated_tokens",
    "elapsed_seconds", "first_token_seconds", "tokens_per_second", "rss_mb_start",
    "rss_mb_peak", "rss_mb_end", "rss_delta_mb", "io_read_mb", "io_write_mb",
    "logical_bytes_mapped_mb", "tensors_touched", "layer_count", "notes",
]
TIMELINE_FIELDNAMES = [
    "run_id", "backend", "backend_label", "prompt_label", "prompt_tokens", "max_new_tokens",
    "run_index", "sample_index", "elapsed_seconds", "rss_mb", "io_read_mb", "io_write_mb",
]
SUMMARY_FIELDNAMES = [
    "backend", "backend_label", "prompt_label", "prompt_tokens", "max_new_tokens", "run_count",
    "elapsed_seconds_mean", "elapsed_seconds_min", "elapsed_seconds_max", "elapsed_seconds_stdev",
    "first_token_seconds_mean", "first_token_seconds_min", "first_token_seconds_max",
    "first_token_seconds_stdev", "tokens_per_second_mean", "tokens_per_second_min",
    "tokens_per_second_max", "tokens_per_second_stdev", "generated_tokens_mean",
    "rss_mb_peak_mean", "rss_mb_peak_min", "rss_mb_peak_max", "rss_mb_peak_stdev",
    "io_read_mb_mean", "logical_bytes_mapped_mb_mean",
]


@dataclass(frozen=True)
class PromptCase:
    label: str
    token_ids: list[int]

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class BenchmarkReport:
    metadata: dict[str, Any]
    run_rows: list[dict[str, Any]] = field(default_factory=list)
    timeline_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_name_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def dedupe_preserve_order(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def expand_prompt_ids(base_prompt_ids: Sequence[int], target_length: int) -> list[int]:
    if target_length <= 0:
        raise DiskLLMError("Prompt lengths must be positive integers.")
    if not base_prompt_ids:
        raise DiskLLMError("At least one prompt token is required to build benchmark cases.")
    repeats = (target_length + len(base_prompt_ids) - 1) // len(base_prompt_ids)
    return (list(base_prompt_ids) * repeats)[:target_length]


def build_prompt_cases(base_prompt_ids: Sequence[int], prompt_lengths: Sequence[int]) -> list[PromptCase]:
    lengths = dedupe_preserve_order(int(length) for length in prompt_lengths) or [len(base_prompt_ids)]
    return [PromptCase(label=f"tokens_{length:04d}", token_ids=expand_prompt_ids(base_prompt_ids, length)) for length in lengths]


def resolve_prompt_ids(
    *,
    manifest: PackedModelManifest,
    prompt: str | None,
    prompt_file: str | Path | None,
    prompt_ids: str | None,
    tokenizer_path: str | Path | None,
    trust_remote_code: bool = False,
) -> list[int]:
    if prompt_ids:
        values = parse_int_list(prompt_ids)
        if values:
            return values
        raise DiskLLMError("No token ids were parsed from --prompt-ids.")
    prompt_text = prompt
    if prompt_file:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8")
    if prompt_text is None:
        raise DiskLLMError("Provide one of --prompt, --prompt-file, or --prompt-ids.")
    AutoTokenizer = require_auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or manifest.source_dir), trust_remote_code=trust_remote_code)
    values = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    if not values:
        raise DiskLLMError("The prompt produced zero tokens after tokenization.")
    return values


class ProcessSampler:
    def __init__(self, interval_seconds: float = 0.025):
        self.interval_seconds = max(float(interval_seconds), 0.001)
        self.samples: list[dict[str, float | None]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._baseline_io: dict[str, int] | None = None
        self._process = None
        self.available = True
        try:
            psutil = require_psutil()
            self._process = psutil.Process(os.getpid())
        except DependencyMissingError:
            self.available = False

    def start(self) -> None:
        if not self.available or self._process is None:
            return
        self._started_at = time.perf_counter()
        self._baseline_io = self._safe_io()
        self._record_sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not self.available or self._process is None:
            return {"rss_mb_start": None, "rss_mb_peak": None, "rss_mb_end": None, "rss_delta_mb": None, "io_read_mb": None, "io_write_mb": None, "timeline": []}
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._record_sample()
        rss_values = [sample["rss_mb"] for sample in self.samples if sample["rss_mb"] is not None]
        final_io = self._safe_io()
        io_read_mb = None
        io_write_mb = None
        if self._baseline_io is not None and final_io is not None:
            io_read_mb = (final_io["read_bytes"] - self._baseline_io["read_bytes"]) / MB
            io_write_mb = (final_io["write_bytes"] - self._baseline_io["write_bytes"]) / MB
        rss_start = rss_values[0] if rss_values else None
        rss_end = rss_values[-1] if rss_values else None
        return {
            "rss_mb_start": rss_start,
            "rss_mb_peak": max(rss_values) if rss_values else None,
            "rss_mb_end": rss_end,
            "rss_delta_mb": None if rss_start is None or rss_end is None else rss_end - rss_start,
            "io_read_mb": io_read_mb,
            "io_write_mb": io_write_mb,
            "timeline": list(self.samples),
        }

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._record_sample()

    def _record_sample(self) -> None:
        io_counters = self._safe_io()
        baseline = self._baseline_io or {"read_bytes": 0, "write_bytes": 0}
        try:
            rss_mb = self._process.memory_info().rss / MB
        except Exception:  # pragma: no cover
            rss_mb = None
        self.samples.append(
            {
                "elapsed_seconds": max(time.perf_counter() - self._started_at, 0.0),
                "rss_mb": rss_mb,
                "io_read_mb": None if io_counters is None else (io_counters["read_bytes"] - baseline["read_bytes"]) / MB,
                "io_write_mb": None if io_counters is None else (io_counters["write_bytes"] - baseline["write_bytes"]) / MB,
            }
        )

    def _safe_io(self) -> dict[str, int] | None:
        try:
            counters = self._process.io_counters()
        except Exception:  # pragma: no cover
            return None
        return {"read_bytes": int(getattr(counters, "read_bytes", 0)), "write_bytes": int(getattr(counters, "write_bytes", 0))}


class DiskLLMBenchmarkBackend:
    name = "disk_llm"
    label = "Disk-LLM"

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.model: DiskLLMTextModel | None = None

    def load(self) -> None:
        self.model = DiskLLMTextModel.from_manifest(self.manifest_path)

    def run(self, prompt_ids: Sequence[int], *, max_new_tokens: int, temperature: float, top_p: float, seed: int | None) -> dict[str, Any]:
        if self.model is None:
            self.load()
        telemetry = TelemetryRecorder(prompt_tokens=len(prompt_ids))
        generated_ids, telemetry_payload = self.model.generate_token_ids(prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed, telemetry=telemetry)
        return {
            "generated_tokens": len(generated_ids),
            "elapsed_seconds": telemetry_payload.get("elapsed_seconds"),
            "first_token_seconds": telemetry_payload.get("first_token_seconds"),
            "tokens_per_second": telemetry_payload.get("tokens_per_second"),
            "logical_bytes_mapped_mb": telemetry_payload.get("logical_bytes_mapped", 0) / MB,
            "tensors_touched": telemetry_payload.get("tensors_touched"),
            "layer_count": len(telemetry_payload.get("layer_times", {})),
            "notes": [],
        }

    def close(self) -> None:
        self.model = None
        gc.collect()


class HuggingFaceCPUBenchmarkBackend:
    name = "hf_cpu"
    label = "HF CPU"

    def __init__(self, model_path: str | Path, *, trust_remote_code: bool = False):
        self.model_path = Path(model_path)
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.torch = None
        self.eos_token_id: int | None = None

    def load(self) -> None:
        AutoModelForCausalLM = require_auto_model_for_causal_lm()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissingError("PyTorch is required for the Hugging Face CPU baseline. Install a CPU build alongside `pip install -e .[hf]`.") from exc
        self.torch = torch
        try:
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), torch_dtype=torch.float32, trust_remote_code=self.trust_remote_code)
        except Exception as exc:  # pragma: no cover
            raise DiskLLMError(f"Failed to load Hugging Face model from {self.model_path}: {exc}") from exc
        self.model.to("cpu")
        self.model.eval()
        self.eos_token_id = getattr(self.model.config, "eos_token_id", None)

    def run(self, prompt_ids: Sequence[int], *, max_new_tokens: int, temperature: float, top_p: float, seed: int | None) -> dict[str, Any]:
        if self.model is None or self.torch is None:
            self.load()
        torch = self.torch
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        started = time.perf_counter()
        first_token_seconds = None
        generated_tokens = 0
        with torch.no_grad():
            input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device="cpu")
            outputs = self.model(input_ids=input_ids, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            for _ in range(max_new_tokens):
                token_id = self._sample_token(logits, temperature=temperature, top_p=top_p, generator=generator)
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started
                generated_tokens += 1
                if self.eos_token_id is not None and token_id == self.eos_token_id:
                    break
                outputs = self.model(input_ids=torch.tensor([[token_id]], dtype=torch.long, device="cpu"), past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
        elapsed = time.perf_counter() - started
        return {"generated_tokens": generated_tokens, "elapsed_seconds": elapsed, "first_token_seconds": first_token_seconds, "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0, "logical_bytes_mapped_mb": None, "tensors_touched": None, "layer_count": None, "notes": ["float32_cpu_reference"]}

    def _sample_token(self, logits, *, temperature: float, top_p: float, generator) -> int:
        torch = self.torch
        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return int(sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1, generator=generator)).item())
        return int(torch.multinomial(probs, 1, generator=generator).item())

    def close(self) -> None:
        self.model = None
        self.torch = None
        gc.collect()


def create_backend(backend_name: str, *, manifest_path: str | Path, hf_model_path: str | Path | None, trust_remote_code: bool):
    normalized = backend_name.strip().lower()
    if normalized == "disk_llm":
        return DiskLLMBenchmarkBackend(manifest_path)
    if normalized == "hf_cpu":
        if hf_model_path is None:
            raise DiskLLMError("The hf_cpu backend requires --hf-model or a manifest with a valid source snapshot.")
        return HuggingFaceCPUBenchmarkBackend(hf_model_path, trust_remote_code=trust_remote_code)
    raise DiskLLMError(f"Unknown backend: {backend_name}")


def aggregate_run_rows(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (row["backend"], row["backend_label"], row["prompt_label"], row["prompt_tokens"], row["max_new_tokens"])
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    numeric_fields = ["elapsed_seconds", "first_token_seconds", "tokens_per_second", "generated_tokens", "rss_mb_peak", "io_read_mb", "logical_bytes_mapped_mb"]
    for key in sorted(grouped, key=lambda item: (item[0], item[3], item[4])):
        rows = grouped[key]
        summary: dict[str, Any] = {"backend": key[0], "backend_label": key[1], "prompt_label": key[2], "prompt_tokens": key[3], "max_new_tokens": key[4], "run_count": len(rows)}
        for field_name in numeric_fields:
            values = [float(row[field_name]) for row in rows if row.get(field_name) is not None]
            summary[f"{field_name}_mean"] = statistics.fmean(values) if values else None
            if field_name != "generated_tokens":
                summary[f"{field_name}_min"] = min(values) if values else None
                summary[f"{field_name}_max"] = max(values) if values else None
                summary[f"{field_name}_stdev"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        summary_rows.append(summary)
    return summary_rows


def run_benchmark_suite(
    manifest_path: str | Path,
    *,
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
    prompt_ids: str | None = None,
    tokenizer_path: str | Path | None = None,
    prompt_lengths: Sequence[int] | None = None,
    max_new_tokens_values: Sequence[int] | None = None,
    runs: int = 3,
    warmup_runs: int = 0,
    temperature: float = 0.0,
    top_p: float = 0.95,
    seed: int | None = 0,
    backends: Sequence[str] | None = None,
    hf_model_path: str | Path | None = None,
    sample_interval_seconds: float = 0.025,
    trust_remote_code: bool = False,
) -> BenchmarkReport:
    manifest = PackedModelManifest.from_path(manifest_path)
    if runs <= 0:
        raise DiskLLMError("Benchmark runs must be >= 1.")
    if warmup_runs < 0:
        raise DiskLLMError("Warmup runs must be >= 0.")
    base_prompt_ids = resolve_prompt_ids(manifest=manifest, prompt=prompt, prompt_file=prompt_file, prompt_ids=prompt_ids, tokenizer_path=tokenizer_path, trust_remote_code=trust_remote_code)
    prompt_cases = build_prompt_cases(base_prompt_ids, prompt_lengths or [len(base_prompt_ids)])
    max_tokens_cases = dedupe_preserve_order(int(value) for value in (max_new_tokens_values or [16]))
    if any(value <= 0 for value in max_tokens_cases):
        raise DiskLLMError("All max_new_tokens values must be positive integers.")
    backend_names = dedupe_preserve_order(backends or ["disk_llm"])
    if not backend_names:
        raise DiskLLMError("At least one benchmark backend must be selected.")
    resolved_hf_model_path = hf_model_path or manifest.source_dir
    run_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    for backend_name in backend_names:
        backend = create_backend(backend_name, manifest_path=manifest_path, hf_model_path=resolved_hf_model_path, trust_remote_code=trust_remote_code)
        backend.load()
        try:
            for prompt_case in prompt_cases:
                for max_new_tokens in max_tokens_cases:
                    for warmup_index in range(warmup_runs):
                        backend.run(prompt_case.token_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=None if seed is None else seed - (warmup_index + 1))
                    for run_index in range(runs):
                        sampler = ProcessSampler(interval_seconds=sample_interval_seconds)
                        sampler.start()
                        try:
                            result = backend.run(prompt_case.token_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=None if seed is None else seed + run_index)
                        finally:
                            resource_metrics = sampler.stop()
                        run_id = f"{backend.name}|{prompt_case.prompt_tokens}|{max_new_tokens}|{run_index}"
                        run_rows.append(
                            {
                                "run_id": run_id,
                                "backend": backend.name,
                                "backend_label": backend.label,
                                "prompt_label": prompt_case.label,
                                "prompt_tokens": prompt_case.prompt_tokens,
                                "max_new_tokens": max_new_tokens,
                                "run_index": run_index,
                                "run_phase": "initial" if run_index == 0 and warmup_runs == 0 else "steady_state",
                                "seed": None if seed is None else seed + run_index,
                                "temperature": temperature,
                                "top_p": top_p,
                                "generated_tokens": result.get("generated_tokens"),
                                "elapsed_seconds": result.get("elapsed_seconds"),
                                "first_token_seconds": result.get("first_token_seconds"),
                                "tokens_per_second": result.get("tokens_per_second"),
                                "rss_mb_start": resource_metrics["rss_mb_start"],
                                "rss_mb_peak": resource_metrics["rss_mb_peak"],
                                "rss_mb_end": resource_metrics["rss_mb_end"],
                                "rss_delta_mb": resource_metrics["rss_delta_mb"],
                                "io_read_mb": resource_metrics["io_read_mb"],
                                "io_write_mb": resource_metrics["io_write_mb"],
                                "logical_bytes_mapped_mb": result.get("logical_bytes_mapped_mb"),
                                "tensors_touched": result.get("tensors_touched"),
                                "layer_count": result.get("layer_count"),
                                "notes": " | ".join(result.get("notes", [])),
                            }
                        )
                        for sample_index, sample in enumerate(resource_metrics["timeline"]):
                            timeline_rows.append({"run_id": run_id, "backend": backend.name, "backend_label": backend.label, "prompt_label": prompt_case.label, "prompt_tokens": prompt_case.prompt_tokens, "max_new_tokens": max_new_tokens, "run_index": run_index, "sample_index": sample_index, **sample})
        finally:
            backend.close()
    summary_rows = aggregate_run_rows(run_rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "source_dir": manifest.source_dir,
        "tokenizer_path": None if tokenizer_path is None else str(Path(tokenizer_path).resolve()),
        "hf_model_path": None if resolved_hf_model_path is None else str(Path(resolved_hf_model_path).resolve()),
        "prompt_lengths": [case.prompt_tokens for case in prompt_cases],
        "max_new_tokens": list(max_tokens_cases),
        "runs": int(runs),
        "warmup_runs": int(warmup_runs),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "backends": list(backend_names),
        "sample_interval_seconds": float(sample_interval_seconds),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "notes": [
            "Prompt cases are built by cycling the base prompt token ids to the requested lengths.",
            "Repeated runs show initial versus steady-state behavior inside one process; OS page cache is not forcibly dropped.",
        ],
    }
    return BenchmarkReport(metadata=metadata, run_rows=run_rows, timeline_rows=timeline_rows, summary_rows=summary_rows)


def write_benchmark_artifacts(report: BenchmarkReport, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "benchmark_metadata.json"
    runs_path = output_path / "benchmark_runs.csv"
    timeline_path = output_path / "memory_timeline.csv"
    summary_path = output_path / "benchmark_summary.csv"
    metadata_path.write_text(json.dumps(report.metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(runs_path, RUN_FIELDNAMES, report.run_rows)
    _write_csv(timeline_path, TIMELINE_FIELDNAMES, report.timeline_rows)
    _write_csv(summary_path, SUMMARY_FIELDNAMES, report.summary_rows)
    return {"metadata": metadata_path, "runs": runs_path, "timeline": timeline_path, "summary": summary_path}


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})
