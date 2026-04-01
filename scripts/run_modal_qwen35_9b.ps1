$ErrorActionPreference = "Stop"

python scripts/modal_qwen_benchmark.py `
  --repo-id Qwen/Qwen3.5-9B `
  --revision main `
  --prompt "Explain disk-backed inference in one paragraph." `
  --prompt-lengths 8,64,256,512 `
  --max-new-tokens 16 `
  --runs 3 `
  --warmup-runs 0 `
  --backends disk_llm,hf_cpu `
  --hf-dtype float32 `
  --run-label qwen35-9b-modal-cpu
