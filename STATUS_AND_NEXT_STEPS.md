# Disk-LLM Status And Next Steps

Date: 2026-04-06

This note is a project handoff/status snapshot covering:
- what has been completed so far
- what was fixed in the codebase
- what remains blocked
- what should happen next

## 1. What has been completed so far

The repo has already received several important updates before this handoff:

- benchmark harness and plotting workflow were added
- Modal remote benchmark runner and runbook were added
- README and GitHub Pages were refreshed multiple times
- SVG layout/rendering issues in the showroom assets were fixed
- README-only audit figures were revised

Recent relevant commits:

- `6110ae4` Fix Qwen audit path and refresh project showroom
- `6f6d858` Fix SVG layout overflow in README and site
- `7f6460a` Harden SVG layouts for GitHub rendering
- `cc5ef0c` Upgrade Qwen audit figures and showroom copy
- `4a5f96a` Refresh README Qwen audit charts
- `b9732f8` Implement Qwen runtime audit fixes

## 2. What was fixed in the Qwen runtime audit pass

The main technical problem was that the archived real Qwen Modal run was not a trustworthy final result.
The archived artifact showed:

- `layer_count = 0`
- `tensors_touched = 3`
- only `disk_llm` rows in the CSV bundle

That meant the old run was not exercising the real transformer stack correctly.

The codebase was then updated to address the audit findings:

### Runtime/config fixes

Files:

- `src/disk_llm/runtime/config.py`
- `src/disk_llm/runtime/kernels.py`
- `src/disk_llm/runtime/model.py`
- `src/disk_llm/converter.py`

Implemented changes:

- nested `text_config` fields are now used correctly by the runtime
- Qwen full-attention now handles:
  - doubled `q_proj`
  - gated attention output
  - `q_norm` / `k_norm`
  - fixed `head_dim`
  - partial RoPE via `partial_rotary_factor`
- a native token-by-token `linear_attention` path was added for the NumPy runtime
- linear attention now keeps:
  - conv cache state
  - recurrent state
- runtime benchmark guardrails were already present and now remain aligned with the stricter audit path
- prefetch is now treated as an explicit experiment rather than silent baseline behavior

### Prefetch separation

Important boundary change:

- prefetch is no longer treated as model semantics from packed/source config
- runtime prefetch is now env-scoped
- the intended experiment flag is:
  - `DISK_LLM_EXPERIMENT_LAYER_PREFETCH=1`

This keeps baseline and experiment runs separate.

### Test coverage updates

Files:

- `tests/helpers.py`
- `tests/test_converter.py`
- `tests/test_runtime_toy_model.py`
- `tests/test_qwen_runtime_regressions.py`

Coverage added/updated:

- nested Qwen config parsing
- Qwen-style full-attention toy coverage
- linear-attention toy coverage
- prefetch behavior is now asserted as env-scoped, not manifest-scoped

## 3. Current benchmark/result status

What we have:

- archived pre-fix Modal artifacts in `modal-results/`
- code that is now much closer to the correct Qwen runtime behavior
- remote benchmark runner in `scripts/modal_qwen_benchmark.py`

What we do not yet have:

- a new validated post-fix Modal benchmark result bundle
- a new post-fix Disk-LLM vs HF CPU comparison CSV bundle
- final post-fix README and webpage figures based on new validated evidence

So the current published visuals/story still need a future refresh once the new remote run succeeds.

## 4. Local environment work completed in this session

We explicitly did not download the Qwen model to this local PC.

Completed locally:

- verified `.env` exists in the project root
- verified `HF_TOKEN` is present in `.env`
- verified `.env` is ignored by Git via `.gitignore`
- created a dedicated conda environment:
  - `disk-llm-modal`

Environment goal:

- use this env only to drive Modal and local runtime validation
- do not store model weights locally

Packages installed into that env:

- `numpy`
- `ml_dtypes`
- Modal client package

Important packaging note:

- `conda-forge::modal` turned out to be the unrelated `modAL` package, not the Modal.com SDK
- that was corrected by replacing it with `modal-client`

## 5. Where we left off

The next planned step was:

1. authenticate Modal locally
2. submit the post-fix remote smoke benchmark
3. validate the new result bundle
4. refresh README and GitHub Pages with real post-fix evidence

However, we hit two local blockers on this machine:

### Blocker A: Modal network path is still unavailable from this machine

Confirmed checks:

- `Test-NetConnection api.modal.com -Port 443` returned `False`
- `nslookup api.modal.com` timed out
- `Invoke-WebRequest https://api.modal.com` could not connect

As a result:

- `modal setup` could not complete
- `~/.modal.toml` was not created
- the remote benchmark could not be submitted from this machine

### Blocker B: local Python commands showed unstable Windows behavior

After the new environment was created, some direct `python.exe` invocations behaved inconsistently and the user observed unknown software exceptions.
Because the immediate goal is the remote Modal run, local Windows runtime debugging was intentionally stopped instead of spending more time on a machine-specific issue.

## 6. What should happen next

### Preferred next step

Resume from a machine or server that can actually reach Modal and authenticate successfully.

Use the `disk-llm-modal` environment pattern, then:

1. authenticate Modal
2. run a baseline post-fix smoke benchmark
3. run the prefetch experiment separately
4. run a larger benchmark sweep if the smoke test looks correct

### Exact sequence intended for the next execution

Baseline run:

```powershell
conda activate disk-llm-modal
modal setup

python scripts/modal_qwen_benchmark.py `
  --repo-id Qwen/Qwen3.5-9B `
  --revision main `
  --prompt "Explain disk-backed inference in one paragraph." `
  --prompt-lengths 8,128 `
  --max-new-tokens 2 `
  --runs 1 `
  --warmup-runs 0 `
  --backends disk_llm,hf_cpu `
  --hf-dtype float32 `
  --run-label qwen35-9b-modal-cpu-postfix
```

Separate prefetch experiment:

```powershell
$env:DISK_LLM_EXPERIMENT_LAYER_PREFETCH="1"

python scripts/modal_qwen_benchmark.py `
  --repo-id Qwen/Qwen3.5-9B `
  --revision main `
  --prompt "Explain disk-backed inference in one paragraph." `
  --prompt-lengths 8,128 `
  --max-new-tokens 2 `
  --runs 1 `
  --warmup-runs 0 `
  --backends disk_llm,hf_cpu `
  --hf-dtype float32 `
  --run-label qwen35-9b-modal-cpu-prefetch-exp

Remove-Item Env:DISK_LLM_EXPERIMENT_LAYER_PREFETCH
```

### After the smoke run succeeds

Do this next:

- inspect the new `modal_run_report.json`
- inspect the new CSV bundle
- confirm `layer_count` is nonzero and layer execution is real
- compare `disk_llm` against `hf_cpu`
- regenerate final visualizations and comparison figures
- update README and GitHub Pages using the new validated result set

## 7. Guardrails / constraints to keep

- do not download the Qwen model to this local PC
- keep using remote Modal storage/compute for the real run
- do not commit `.env`
- do not commit tokens or secrets
- keep prefetch labeled as an experiment, not a baseline
- do not overstate the old archived `modal-results` bundle as final proof

## 8. Bottom line

The repo is in a much better technical state now than the archived pre-fix benchmark period.
The missing piece is no longer the audit code path itself.
The missing piece is a successful post-fix Modal rerun from a machine/network that can authenticate to Modal and submit the job.
