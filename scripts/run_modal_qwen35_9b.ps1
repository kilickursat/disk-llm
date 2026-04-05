$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

python scripts/modal_qwen_benchmark.py `
  --repo-id Qwen/Qwen3.5-9B `
  --revision main `
  --prompt "Explain disk-backed inference in one paragraph." `
  --prompt-lengths 8,128 `
  --max-new-tokens 2 `
  --runs 1 `
  --warmup-runs 0 `
  --backends disk_llm `
  --hf-dtype float32 `
  --run-label qwen35-9b-modal-cpu
