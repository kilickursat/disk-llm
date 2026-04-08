<p align="center">
  <img src="logo.png" alt="Disk-LLM logo" width="360">
</p>

# Disk-LLM

[![Modal Benchmark](https://github.com/kilickursat/disk-llm/actions/workflows/modal-benchmark.yml/badge.svg)](https://github.com/kilickursat/disk-llm/actions/workflows/modal-benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Status: Research Snapshot](https://img.shields.io/badge/status-research%20snapshot-C67A2B)](#current-validated-qwen-baseline)

Disk-LLM is an inspectable disk-backed LLM research kit built around one idea: if model weights are going to stream from disk, that path should stay explicit, measurable, and understandable.

Instead of treating checkpoint files as the runtime layout, Disk-LLM repacks text weights into layer-oriented memmap shards, runs a native NumPy CPU path, and exports telemetry about what the runtime actually touched. The project website is published at [kilickursat.github.io/disk-llm](https://kilickursat.github.io/disk-llm/).

## Current Validated Qwen Baseline

The latest fully validated Modal result bundle is tracked in [`modal-results-postfix/qwen35-9b-postfix-v3`](modal-results-postfix/qwen35-9b-postfix-v3).

This is the current honest baseline for `Qwen/Qwen3.5-9B` on the repo's native NumPy memmap path:

- run label: `qwen35-9b-postfix-v3`
- requested revision: `main`
- resolved SHA: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- packed tensors: `427`
- packed shards: `34`
- packed footprint: `16.68 GiB`
- executed layers: `32`
- benchmark shape: prompt lengths `8` and `128`, generate `2`, `runs = 1`, `warmup_runs = 0`

This is not a win report. It is the current validated full-model research snapshot.

<table>
  <tr>
    <td><img src="docs/assets/qwen35-postfix-v3-throughput.png" alt="Qwen v3 throughput plot"></td>
    <td><img src="docs/assets/qwen35-postfix-v3-first-token-latency.png" alt="Qwen v3 first-token latency plot"></td>
  </tr>
  <tr>
    <td><img src="docs/assets/qwen35-postfix-v3-logical-mapped.png" alt="Qwen v3 logical mapped bytes plot"></td>
    <td><img src="docs/assets/qwen35-postfix-v3-rss-timeline.png" alt="Qwen v3 RSS timeline plot"></td>
  </tr>
</table>

## Qwen v3 Comparison

| Prompt tokens | Backend | Tokens/s | First token (s) | Peak RSS (MB) | Logical mapped (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | Disk-LLM | 0.0147 | 114.217 | 24463.44 | 170780.32 |
| 8 | HF CPU | 0.1359 | 6.525 | 21981.11 | - |
| 128 | Disk-LLM | 0.00130 | 1520.855 | 24480.34 | 2220144.13 |
| 128 | HF CPU | 0.0789 | 17.074 | 21981.11 | - |

What this means right now:

- the full Qwen path is now genuinely exercised end-to-end
- the current validated postfix baseline is slower than HF CPU and uses more RSS on this Modal setup
- the logical mapped metric remains useful as a storage-facing signal, but it should not be read as resident RAM

## Legacy Note

The earlier checked-in bundle at [`modal-results-postfix/qwen35-9b-modal-cpu-postfix`](modal-results-postfix/qwen35-9b-modal-cpu-postfix) is now best read as a legacy pre-guard artifact, not the current baseline.

Its `benchmark_runs.csv` reported `layer_count = 0` and `tensors_touched = 3` for Disk-LLM, so it should not be used as the headline comparison for the current branch.

## Why Disk-LLM Exists

Disk-LLM is most compelling when it is judged as a research system rather than a generic inference stack.

- **Layout-first conversion:** checkpoints are repacked into layer-oriented shard files so the runtime layout is deliberate and inspectable.
- **Memmap-native CPU path:** the project keeps disk-backed behavior visible instead of hiding it inside a larger serving engine.
- **Storage-facing telemetry:** logical bytes mapped, tensors touched, first-token latency, and per-layer timings make the benchmark story auditable.
- **Benchmark honesty guards:** the repo now rejects zero-layer benchmark exports instead of letting misleading rows become polished figures.
- **Remote reproducibility:** the Modal workflow can inspect, pack, benchmark, and archive large-model artifacts without forcing a full local download.

## Quick Start

### 1. Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[hf,demo,test,bench]
```

For Hugging Face parity or CPU-baseline benchmarks, make sure a CPU PyTorch build is available in the environment.

### 2. Inspect a source snapshot

```bash
disk-llm inspect --source-dir /path/to/Qwen3.5-9B
```

### 3. Convert it into Disk-LLM layout

```bash
disk-llm convert /path/to/Qwen3.5-9B ./packed-qwen35
```

### 4. Inspect the packed manifest

```bash
disk-llm inspect --manifest ./packed-qwen35/manifest.json
```

### 5. Generate from the packed model

```bash
disk-llm generate ./packed-qwen35/manifest.json --prompt "Explain disk-backed inference in one paragraph."
```

### 6. Run repeated benchmark cases

```bash
python scripts/benchmark.py ./packed-qwen35/manifest.json \
  --prompt "Explain disk-backed inference in one paragraph." \
  --tokenizer /path/to/Qwen3.5-9B \
  --backends disk_llm,hf_cpu \
  --hf-model /path/to/Qwen3.5-9B \
  --prompt-lengths 8,64,256,512 \
  --max-new-tokens 16 \
  --runs 3 \
  --output-dir ./benchmark-results/qwen35-cpu
```

Outputs:

- `benchmark_runs.csv`
- `benchmark_summary.csv`
- `memory_timeline.csv`
- `benchmark_metadata.json`

### 7. Generate plots

```bash
python scripts/plot_results.py ./benchmark-results/qwen35-cpu
```

### 8. Keep the model off your local machine

If you want to run the full workflow remotely on Modal, use the runbook in [`docs/modal_remote_run.md`](docs/modal_remote_run.md).

Helper wrappers:

- `scripts/run_modal_qwen35_9b.sh`
- `scripts/run_modal_qwen35_9b.ps1`

## What Gets Packed

The default v1 converter targets the text-only path:

- `model.embed_tokens.*`
- `model.layers.<n>.*`
- `model.language_model.layers.<n>.*`
- `model.norm.*`
- `model.language_model.norm.*`
- `lm_head.*`

Known multimodal tensors such as `visual.*` are skipped and recorded in the manifest.

Weights are copied into layer-oriented shards:

- `embeddings/embeddings.bin`
- `layers/layer_000.bin`
- `layers/layer_001.bin`
- `...`
- `final/final.bin`

Each tensor receives a manifest entry with:

- shard path
- byte offset
- byte length
- source file
- dtype
- shape
- tensor checksum

## Telemetry

Every runtime call can emit:

- logical bytes mapped
- tensors touched
- per-layer wall time
- first-token latency
- generated token count
- tokens per second

The benchmark scripts extend that with repeated-run CSVs, RSS sampling via `psutil`, Markdown summaries, and plot generation.

## Roadmap

- update the README and GitHub Pages from real validated result bundles, even when the current numbers are unfavorable
- optimize the HF CPU image path so the Modal baseline stops pulling an oversized CPU-unfriendly Torch stack
- apply only small Disk-LLM runtime tweaks that preserve the project's native NumPy memmap identity
- rerun the Qwen postfix baseline as `v4`
- rerun the matching prefetch experiment after the postfix baseline is improved
- expand to additional model families such as Gemma and GLM once the Qwen path is stable
- use those later results for the stronger publication pass

## Development

Run the stdlib test suite:

```bash
python -m unittest discover -v
```

The repo is designed to stay importable even when optional dependencies are missing, which makes it easier to inspect the converter, manifest flow, and CLI without first downloading a full inference stack.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

High-value contributions include:

- new tensor-name adapters
- runtime correctness tests against reference implementations
- benchmark datasets and published result bundles
- better inspection for hybrid block layouts
- tokenizer and chat-template integration improvements
