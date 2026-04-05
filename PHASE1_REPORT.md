# Disk-LLM Phase 1 — Fix Report

**Date:** 2026-04-05  
**Status:** ✅ All blockers resolved — ready to commit & push

---

## What Was Phase 1?

Phase 1 ("Modal Benchmark Implementation") required:
1. ✅ Review existing modal scripts
2. ✅ Verify `.env` has `HF_TOKEN` configured
3. ✅ Verify Modal remote run documentation exists
4. ❌ Install Modal SDK and authenticate → **was actually already done**
5. ❌ Test Modal benchmark run (dry-run validation) → **was blocked by two bugs**

---

## Root Causes of Phase 1 Failure

### Bug 1: Missing `ml-dtypes` dependency (5 test failures)

**Symptom:** All runtime tests failed with:
```
ModuleNotFoundError: No module named 'ml_dtypes'
→ DependencyMissingError: NumPy and ml-dtypes are required for runtime commands.
```

**Root Cause:** The previous session added `ml-dtypes>=0.4.0` to `pyproject.toml` and updated `optional.py` to import it, but **never ran `pip install ml-dtypes`** on the local machine. The dependency was declared but not installed.

**Failing tests (all 5):**
- `test_apply_rope_preserves_shape`
- `test_grouped_query_attention_step_returns_expected_shape`
- `test_rms_norm_matches_manual_formula`
- `test_sampling_greedy_path`
- `test_attention_only_toy_model_generates`

**Fix:** `pip install ml-dtypes>=0.4.0`

---

### Bug 2: Windows Unicode crash in Modal benchmark (script crash)

**Symptom:** Running `modal_qwen_benchmark.py` crashed immediately with:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
in position 0: character maps to <undefined>
```

**Root Cause:** Modal SDK uses the `rich` library for console output. On Windows with legacy code page `cp1252`, Rich's `_win32_console.py` renderer tried to write a `✓` (U+2713 CHECK MARK) character which cannot be encoded in cp1252. The script never got past Modal's initialization banner.

**Fix:** Added a Windows console encoding fix at the top of `modal_qwen_benchmark.py`:
- Sets `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` environment variables
- Reconfigures `sys.stdout` and `sys.stderr` to use UTF-8 with `errors="replace"`
- Updated `run_modal_qwen35_9b.ps1` to also set `[Console]::OutputEncoding` and `chcp 65001`

---

## All Changes Made (staged for commit)

| File | Change |
|------|--------|
| `pyproject.toml` | Added `ml-dtypes>=0.4.0` to core dependencies |
| `src/disk_llm/optional.py` | Import `ml_dtypes` alongside numpy; updated error message |
| `src/disk_llm/layout.py` | Added `model.language_model.*` tensor name patterns for multimodal checkpoints (Qwen3.5) |
| `src/disk_llm/runtime/model.py` | Added `model.language_model.*` fallback tensor names for embed, norm, attention, and MLP layers |
| `scripts/modal_qwen_benchmark.py` | Added Windows UTF-8 console fix; improved HF_TOKEN lookup order (env var → .env file) |
| `scripts/run_modal_qwen35_9b.ps1` | Added UTF-8 console encoding; reduced default params for faster first-run validation |

---

## Verification Results

### Test Suite: 12/12 PASSED ✅
```
tests/test_benchmarking.py    2/2 PASSED
tests/test_cli.py             3/3 PASSED
tests/test_converter.py       1/1 PASSED
tests/test_runtime_kernels.py 4/4 PASSED
tests/test_runtime_toy_model.py 1/1 PASSED
tests/test_safetensors_io.py  1/1 PASSED
```

### Modal SDK: Authenticated ✅
- Profile: `kilickursat`
- Version: `1.1.1`

### Script Dry-Run: `--help` completes without crash ✅

### HF_TOKEN: Present in `.env` ✅ (gitignored)

---

## What's Next (Phase 2+)

After this commit, the remaining work from `nextstep.md` is:

1. **Run the actual Modal benchmark** remotely against `Qwen/Qwen3.5-9B`
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_qwen35_9b.ps1
   ```
2. **Build measurement infrastructure** — `scripts/benchmark.py` and `scripts/plot_results.py`
3. **Add HF baseline comparison** — run identical prompts through HuggingFace and record side-by-side metrics
4. **Document findings** in README and generate publication-quality plots
