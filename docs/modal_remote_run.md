# Modal Remote Run

Use this workflow on another machine or server that can reach Modal and Hugging Face. Do not copy your `.env` into Git.

## Prerequisites

- Clone this repo on the target machine.
- Create a local `.env` in the repo root containing:

```env
HF_TOKEN=...
```

- Authenticate the target machine to your Modal workspace before running the benchmark:

```bash
modal setup
```

or provide `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment.

## Exact One-Command Run

From the repo root, run exactly one of these:

### Linux / macOS

```bash
bash scripts/run_modal_qwen35_9b.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_qwen35_9b.ps1
```

## Equivalent Direct Python Command

If you prefer not to use the wrapper scripts, this is the exact command they run:

```bash
python scripts/modal_qwen_benchmark.py \
  --repo-id Qwen/Qwen3.5-9B \
  --revision main \
  --prompt "Explain disk-backed inference in one paragraph." \
  --prompt-lengths "8 64 256 512" \
  --max-new-tokens 16 \
  --runs 3 \
  --warmup-runs 0 \
  --backends disk_llm,hf_cpu \
  --hf-dtype float32 \
  --run-label qwen35-9b-modal-cpu
```

## What The Remote Run Does

- downloads `Qwen/Qwen3.5-9B` inside Modal, not to your local machine
- stores the source snapshot, packed shards, and outputs in the Modal Volume `disk-llm-benchmarks`
- runs `inspect`, `convert`, repeated benchmarks, plots, and an insight summary

## Result Location In Modal Volume

The fixed run label in the wrapper scripts is:

```text
qwen35-9b-modal-cpu
```

So the expected results path in the Modal Volume is:

```text
/results/Qwen--Qwen3.5-9B/qwen35-9b-modal-cpu
```

Use the Volume path above with `modal volume get`. The mounted container path is `/vol/results/...`, but the CLI fetch command expects the Volume-relative `/results/...` form.

## Pull Results Back Later

After the remote run finishes, pull the artifacts back locally with:

```bash
modal volume get disk-llm-benchmarks /results/Qwen--Qwen3.5-9B/qwen35-9b-modal-cpu ./modal-results
```

Expected artifacts include:

- `benchmark_runs.csv`
- `benchmark_summary.csv`
- `memory_timeline.csv`
- `benchmark_metadata.json`
- `plots/tokens_per_second.png` or `plots/tokens_per_second.html` when static export falls back
- `plots/first_token_latency.png` or `plots/first_token_latency.html` when static export falls back
- `plots/rss_timeline.png` or `plots/rss_timeline.html` when static export falls back
- `plots/comparison_summary.md`
- `benchmark_insights.md`
- `source_inspection.json`
- `packed_inspection.json`
- `modal_run_report.json`
