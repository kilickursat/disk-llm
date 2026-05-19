# Disk-LLM Revised Implementation Plan

**Project:** `kilickursat/disk-llm`  
**Purpose:** Make the Disk-LLM performance-improvement roadmap implementable by Codex/GPT-5.5 or a local developer.  
**Status:** Revised from the earlier suggestion set to correct unsafe assumptions, preserve Disk-LLM's NumPy/memmap research identity, and introduce staged validation.

---

## 0. Executive summary

Disk-LLM should not jump directly into many experimental optimizations at once. The safest and most useful path is a staged implementation that keeps the current baseline reproducible while adding opt-in improvements behind clear flags.

The recommended first milestone is:

> **Disk-LLM v5: layout and runtime efficiency pass**

This milestone should include:

- manifest format versioning,
- converter-time QKV fusion,
- converter-time gate/up MLP fusion,
- FP16 storage with FP32 compute as the first dtype experiment,
- offset alignment,
- budgeted tensor caching,
- bounded prefetching,
- better telemetry,
- benchmark metadata that records every experiment setting,
- README and GitHub Pages updates based on real validated runs.

This milestone should **not** include full multi-request serving, int4/int8 kernels, llama.cpp integration, or major Numba rewrites. Those belong in later phases after profiling shows where the bottleneck remains.

---

## 1. Source facts and constraints

This plan is written against the current public repository structure and documentation.

Disk-LLM currently presents itself as an inspectable disk-backed LLM research kit. Its README says the project repacks text weights into layer-oriented memmap shards, runs a native NumPy CPU path, and exports telemetry about what tensors the runtime touched.[^readme]

The current published Qwen3.5-9B v4 baseline reports 427 packed tensors, 34 packed shards, a 16.68 GiB packed footprint, and 32 executed layers. The same README reports that Disk-LLM is slower than the HF CPU reference: for prompt length 8, Disk-LLM reports 0.0183 tokens/s versus 0.1646 tokens/s for HF CPU; for prompt length 128, Disk-LLM reports 0.00170 tokens/s versus 0.0795 tokens/s for HF CPU.[^readme]

The architecture documentation says Disk-LLM has three major layers: source inspection, packed layout, and an experimental runtime. The packed layout writes tensors into `embeddings/embeddings.bin`, `layers/layer_XXX.bin`, and `final/final.bin`, while the runtime uses `MemmapTensorStore`, `TelemetryRecorder`, and `DiskLLMTextModel`.[^arch]

The runtime currently processes prompt tokens one by one in `generate_token_ids`, then calls `forward_step` for each token. The attention path retrieves Q, K, V, and O projection tensors separately and performs separate dot products. The MLP path retrieves gate, up, and down projection tensors separately.[^runtime]

NumPy's `memmap` creates an array-like memory map to a binary file, allowing access to small disk segments without reading an entire file into memory. NumPy also notes that `memmap` is an `ndarray` subclass and can have unpleasant interactions with some operations.[^numpy_memmap]

The project status document states that prefetch is intentionally an explicit experiment, controlled by `DISK_LLM_EXPERIMENT_LAYER_PREFETCH=1`, not a silent baseline feature.[^status]

---

## 2. Non-negotiable implementation rules

These rules prevent the most likely implementation mistakes.

### Rule 1: Do not fuse tensors at runtime

Fusion must happen during conversion.

Bad runtime pattern:

```python
# Do not do this inside _attention_step.
fused_qkv = np.concatenate([q_proj, k_proj, v_proj], axis=0)
out = hidden @ fused_qkv.T
```

This allocates and copies very large matrices during every token and every layer. It will likely make performance worse.

Correct pattern:

```python
# During conversion only.
fused_qkv = np.concatenate([q_proj, k_proj, v_proj], axis=0)
write_tensor("...fused_qkv.weight", fused_qkv)
```

```python
# During inference only.
fused_qkv = store.get("...fused_qkv.weight")
out = hidden @ fused_qkv.T
```

### Rule 2: Do not split fused QKV into equal thirds

Q, K, and V often have different row counts because grouped-query attention uses fewer KV heads than Q heads. Some Qwen-style variants also have doubled Q projections for gates.

Bad pattern:

```python
q, k, v = np.split(out, 3)
```

Correct pattern:

```python
q_rows = meta["q_rows"]
k_rows = meta["k_rows"]
v_rows = meta["v_rows"]

q_out = out[:q_rows]
k_out = out[q_rows:q_rows + k_rows]
v_out = out[q_rows + k_rows:q_rows + k_rows + v_rows]
```

### Rule 3: Do not fuse MLP down projection with gate/up

For a SwiGLU-style MLP, gate and up are parallel, but down happens after a nonlinearity.

Correct:

```python
gate = hidden @ W_gate.T
up = hidden @ W_up.T
intermediate = silu(gate) * up
output = intermediate @ W_down.T
```

Safe fusion:

```python
fused_gate_up = concatenate([W_gate, W_up], axis=0)
gate_up = hidden @ fused_gate_up.T
gate, up = split_by_rows(gate_up, gate_rows, up_rows)
intermediate = silu(gate) * up
output = intermediate @ W_down.T
```

Unsafe idea:

```python
# Do not do this.
fused_gate_up_down = concatenate([W_gate, W_up, W_down], axis=0)
```

### Rule 4: Do not assume FP16 compute is faster on CPU

FP16 storage may reduce disk I/O. FP16 CPU matmul may or may not be faster depending on BLAS, CPU instruction support, and NumPy build. The first dtype milestone should be:

> FP16 storage, FP32 compute.

### Rule 5: Do not cache huge tensors by default

Caching is valuable, but it must be budgeted. The embedding matrix and LM head can be very large. Cache them only behind explicit flags or within a memory budget.

### Rule 6: Do not spawn unlimited prefetch threads

The current code starts a thread for the next layer. Replace this with a bounded executor and an in-flight set.

---

## 3. Recommended milestone sequence

### Phase map

| Phase | Theme | Priority |
|---|---:|---:|
| 0 | profiling and guardrails | P0 |
| 1 | experiment config | P0 |
| 2 | manifest v2 | P0 |
| 3 | projection fusion | P1 |
| 4 | FP16 storage | P1 |
| 5 | budgeted cache | P1 |
| 6 | bounded prefetch | P1 |
| 7 | prefill vectorization | P2 |
| 8 | shard alignment/grouping | P2 |
| 9 | advanced experiments | P3 |

Do not implement Phase 7 before Phase 3. Prefill vectorization is much easier and more useful after fused QKV and fused gate/up exist.

---

## 4. Phase 0: establish profiling, correctness, and reproducibility

### Objective

Before changing the layout or runtime, add metrics that identify where time is spent. The goal is to stop guessing whether the bottleneck is disk I/O, BLAS, Python overhead, page faults, prefetch contention, or LM-head projection.

### Files likely touched

- `src/disk_llm/runtime/telemetry.py`
- `src/disk_llm/runtime/model.py`
- `scripts/benchmark.py`
- `scripts/modal_qwen_benchmark.py`
- `scripts/plot_results.py`
- `tests/test_benchmarking.py`
- `tests/test_runtime_toy_model.py`

### Metrics to add

Add these fields to telemetry summaries and benchmark metadata:

- `prefill_seconds`
- `decode_seconds`
- `final_norm_seconds`
- `lm_head_seconds`
- `layer_seconds_total`
- `layer_seconds_by_layer`
- `dot_calls_total`
- `dot_seconds_total`
- `dot_seconds_by_kind`
- `tensor_get_calls_total`
- `tensor_get_calls_by_name`
- `logical_mapped_bytes_by_tensor`
- `cache_hits`
- `cache_misses`
- `cache_resident_bytes`
- `prefetch_submitted`
- `prefetch_completed`
- `prefetch_skipped_duplicate`
- `prefetch_seconds_total`
- `major_page_faults_start`
- `major_page_faults_end`
- `minor_page_faults_start`
- `minor_page_faults_end`

On Windows, page fault counters may be unavailable. Use `None` rather than failing.

### Suggested telemetry API

```python
class TelemetryRecorder:
    def __init__(self, prompt_tokens: int = 0):
        self.events = []
        self.counters = collections.Counter()
        self.durations = collections.defaultdict(float)
        self.tensor_bytes = collections.Counter()
        self.tensor_gets = collections.Counter()

    @contextmanager
    def time_block(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] += time.perf_counter() - start

    def record_dot(self, kind: str, seconds: float, left_shape, right_shape):
        self.counters["dot_calls_total"] += 1
        self.durations["dot_seconds_total"] += seconds
        self.durations[f"dot_seconds.{kind}"] += seconds
        self.events.append({
            "event": "dot",
            "kind": kind,
            "seconds": seconds,
            "left_shape": tuple(left_shape),
            "right_shape": tuple(right_shape),
        })

    def record_tensor_get(self, name: str, nbytes: int):
        self.tensor_gets[name] += 1
        self.tensor_bytes[name] += nbytes
```

### Dot timing wrapper

Use this wrapper in the runtime before larger refactors:

```python
def timed_dot(np, left, right, *, kind: str, telemetry):
    start = time.perf_counter()
    out = np.dot(left, right)
    seconds = time.perf_counter() - start
    telemetry.record_dot(kind, seconds, left.shape, right.shape)
    return out
```

Then replace:

```python
q_proj_out = np.dot(hidden, q_proj.T)
```

with:

```python
q_proj_out = timed_dot(np, hidden, q_proj.T, kind="q_proj", telemetry=telemetry)
```

### Page fault helper

```python
def read_page_faults():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "minor": getattr(usage, "ru_minflt", None),
            "major": getattr(usage, "ru_majflt", None),
        }
    except Exception:
        return {"minor": None, "major": None}
```

### Cold versus warm cache benchmarks

Add benchmark labels:

- `cold_process`
- `warm_process`
- `warm_runtime`

Definitions:

- `cold_process`: new Python process, model object recreated.
- `warm_process`: same Python process, model recreated.
- `warm_runtime`: same `DiskLLMTextModel` object reused.

Do not attempt to drop the OS page cache from normal user scripts. It is privileged and not portable. Use process-level separation instead.

### Acceptance criteria

Phase 0 is complete when:

- existing tests pass,
- benchmark CSV still includes previous fields,
- new metadata JSON includes runtime configuration and profiling fields,
- `layer_count` remains nonzero for real model runs,
- toy model tests continue to pass,
- benchmark output documents whether the run is cold or warm.

---

## 5. Phase 1: introduce a structured experiment configuration

### Objective

Centralize all optimization flags in one object so benchmark outputs can be interpreted later.

### Files likely touched

- `src/disk_llm/runtime/config.py`
- `src/disk_llm/cli.py`
- `scripts/benchmark.py`
- `scripts/modal_qwen_benchmark.py`
- `src/disk_llm/converter.py`

### New dataclass

Create a new file:

```text
src/disk_llm/experiment_config.py
```

Suggested dataclass:

```python
@dataclass(frozen=True)
class DiskLLMExperimentConfig:
    manifest_version: int = 1

    # Layout features
    fuse_qkv: bool = False
    fuse_gate_up: bool = False
    align_bytes: int = 1
    group_layers: int = 1

    # Storage dtype
    weight_storage_dtype: str = "float32"
    compute_dtype: str = "float32"

    # Runtime features
    cache_budget_mb: int = 0
    cache_small_tensors: bool = False
    cache_embedding_rows: bool = False
    cache_lm_head: bool = False

    # Prefetch
    prefetch_mode: str = "none"   # none, threadpool, madvise, touch
    prefetch_distance: int = 1
    prefetch_workers: int = 1

    # Prefill
    prefill_mode: str = "token"   # token, chunked
    prefill_chunk_size: int = 1

    # Advanced
    use_numba: bool = False
    backend: str = "numpy_memmap"

    @classmethod
    def from_env(cls):
        return cls(
            fuse_qkv=os.getenv("DISK_LLM_FUSE_QKV") == "1",
            fuse_gate_up=os.getenv("DISK_LLM_FUSE_GATE_UP") == "1",
            cache_budget_mb=int(os.getenv("DISK_LLM_CACHE_BUDGET_MB", "0")),
            prefetch_mode=os.getenv("DISK_LLM_PREFETCH_MODE", "none"),
            prefetch_distance=int(os.getenv("DISK_LLM_PREFETCH_DISTANCE", "1")),
        )

    def to_jsonable(self):
        return dataclasses.asdict(self)
```

### CLI flags

Add only stable flags to CLI initially:

```bash
--weight-storage-dtype float32|float16
--compute-dtype float32
--fuse-qkv
--fuse-gate-up
--align-bytes 4096
--cache-budget-mb 2048
--prefetch-mode none|threadpool|madvise|touch
--prefetch-distance 1
--prefetch-workers 1
```

Keep experimental flags available through environment variables if CLI churn is undesirable.

### Metadata output

Every benchmark run should write:

```json
{
  "disk_llm_experiment_config": {
    "fuse_qkv": true,
    "fuse_gate_up": true,
    "weight_storage_dtype": "float16",
    "compute_dtype": "float32",
    "cache_budget_mb": 2048,
    "prefetch_mode": "threadpool"
  }
}
```

### Acceptance criteria

Phase 1 is complete when:

- config can be created from defaults,
- config can be created from CLI flags,
- config can be created from environment variables,
- benchmark metadata includes the full config,
- old workflows still run with default config.

---

## 6. Phase 2: manifest v2 with backward compatibility

### Objective

Add new metadata needed for fused tensors, dtype policy, quantization metadata, and alignment without breaking existing v1 manifests.

### Files likely touched

- `src/disk_llm/manifest.py`
- `src/disk_llm/converter.py`
- `src/disk_llm/runtime/memmap.py`
- `tests/test_converter.py`
- `tests/test_qwen_runtime_regressions.py`

### Manifest versioning

Add a top-level field:

```json
{
  "manifest_version": 2,
  "family": "qwen3.5",
  "variant": "9b",
  "config": {},
  "layout_features": {
    "fuse_qkv": true,
    "fuse_gate_up": true,
    "align_bytes": 4096,
    "group_layers": 1,
    "weight_storage_dtype": "float16"
  },
  "tensors": {},
  "fused_groups": {}
}
```

Existing v1 manifests should load as:

```python
manifest_version = data.get("manifest_version", 1)
```

### Tensor entry extension

Extend each tensor entry with optional fields:

```json
{
  "name": "model.layers.0.self_attn.fused_qkv.weight",
  "shard": "layers/layer_000.bin",
  "offset": 1048576,
  "nbytes": 18874368,
  "dtype": "float16",
  "numpy_dtype": "float16",
  "shape": [9216, 4096],
  "source_file": "model-00001-of-000xx.safetensors",
  "checksum": "...",
  "alignment": 4096,
  "cache_hint": "large_sequential"
}
```

### Fused group metadata

Add a separate fused-groups block rather than overloading tensor names only:

```json
{
  "fused_groups": {
    "model.layers.0.self_attn.fused_qkv.weight": {
      "kind": "qkv",
      "members": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight"
      ],
      "row_splits": {
        "q": 8192,
        "k": 1024,
        "v": 1024
      },
      "has_q_gate": true,
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      "attention_head_dim": 128
    },
    "model.layers.0.mlp.fused_gate_up.weight": {
      "kind": "gate_up",
      "members": [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight"
      ],
      "row_splits": {
        "gate": 12288,
        "up": 12288
      }
    }
  }
}
```

### Loader behavior

Runtime helper:

```python
def has_fused_group(self, name: str, kind: str | None = None) -> bool:
    group = self.manifest.fused_groups.get(name)
    if group is None:
        return False
    if kind is not None and group.kind != kind:
        return False
    return True
```

### Backward compatibility

For v1 manifest:

- `manifest_version` defaults to 1,
- `layout_features` defaults to all false,
- `fused_groups` defaults to empty,
- runtime follows old separate tensor path.

### Acceptance criteria

Phase 2 is complete when:

- existing v1 manifests still load,
- new v2 toy manifests load,
- fused group metadata is validated,
- missing row split fields raise a clear error,
- inspect command displays manifest version and layout features.

---

## 7. Phase 3: converter-time projection fusion

### Objective

Reduce dot-call count and improve sequential disk access by fusing projections during conversion.

### Files likely touched

- `src/disk_llm/converter.py`
- `src/disk_llm/layout.py`
- `src/disk_llm/manifest.py`
- `src/disk_llm/runtime/model.py`
- `tests/test_converter.py`
- `tests/test_runtime_toy_model.py`
- `tests/test_qwen_runtime_regressions.py`

---

### 7.1 QKV fusion

#### Converter algorithm

For each attention layer:

1. Resolve Q, K, and V tensor names using existing candidate logic.
2. Load raw arrays from the source checkpoint.
3. Verify all tensors have the same input width.
4. Concatenate along axis 0.
5. Write the fused tensor once.
6. Add `fused_groups` metadata.
7. Optionally skip writing separate Q/K/V tensor entries.

Pseudocode:

```python
def try_fuse_qkv(layer_idx, tensor_reader, shard_writer, manifest, config):
    q_name = resolve_required([
        f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight",
    ])
    k_name = resolve_required([...])
    v_name = resolve_required([...])

    q = tensor_reader.read(q_name)
    k = tensor_reader.read(k_name)
    v = tensor_reader.read(v_name)

    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ConversionError("Q/K/V input dimensions do not match")

    fused = np.concatenate([q, k, v], axis=0)
    fused_name = canonical_fused_qkv_name(layer_idx, q_name)

    offset, nbytes = shard_writer.write_aligned(fused, align_bytes=config.align_bytes)

    manifest.add_tensor(
        name=fused_name,
        shard=shard_writer.relative_path,
        offset=offset,
        nbytes=nbytes,
        dtype=str(fused.dtype),
        numpy_dtype=str(fused.dtype),
        shape=list(fused.shape),
        cache_hint="large_sequential",
    )

    manifest.add_fused_group(
        name=fused_name,
        kind="qkv",
        members=[q_name, k_name, v_name],
        row_splits={"q": q.shape[0], "k": k.shape[0], "v": v.shape[0]},
        extra={
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "attention_head_dim": config.attention_head_dim,
            "q_rows": q.shape[0],
            "k_rows": k.shape[0],
            "v_rows": v.shape[0],
        },
    )
```

#### Runtime algorithm

In `_attention_step`, first check for fused QKV. If present, use it. Otherwise, fall back to the current separate Q/K/V path.

Pseudocode:

```python
def _attention_step(...):
    fused_info = self._resolve_fused_qkv(layer_idx)
    if fused_info is not None:
        q_proj_out, key, value = self._attention_qkv_fused(
            layer_idx,
            hidden,
            fused_info=fused_info,
            telemetry=telemetry,
        )
    else:
        q_proj_out, key, value = self._attention_qkv_separate(
            layer_idx,
            hidden,
            telemetry=telemetry,
        )

    query, gate = self._shape_query_from_q_projection(layer_idx, q_proj_out)
    key = self._shape_key(layer_idx, key)
    value = self._shape_value(layer_idx, value)
    ...
```

Fused projection:

```python
def _attention_qkv_fused(self, layer_idx, hidden, *, fused_info, telemetry):
    np = require_numpy()
    fused = self.store.get(fused_info.tensor_name, telemetry)

    out = timed_dot(
        np,
        hidden.astype(self.compute_dtype, copy=False),
        fused.T.astype(self.compute_dtype, copy=False),
        kind="fused_qkv",
        telemetry=telemetry,
    )

    q_rows = fused_info.row_splits["q"]
    k_rows = fused_info.row_splits["k"]
    v_rows = fused_info.row_splits["v"]

    q = out[:q_rows]
    k = out[q_rows:q_rows + k_rows]
    v = out[q_rows + k_rows:q_rows + k_rows + v_rows]
    return q, k, v
```

#### Correctness tests

Add a tiny attention-only toy model test:

1. Build a v1 manifest with separate Q/K/V.
2. Build a v2 manifest with fused QKV using the same weights.
3. Run one prompt through both.
4. Assert logits are close.

Example:

```python
def test_fused_qkv_matches_separate_attention_toy_model():
    separate = build_toy_model(fuse_qkv=False)
    fused = build_toy_model(fuse_qkv=True)

    prompt = [1, 2, 3]
    out_a, _ = separate.generate_token_ids(prompt, max_new_tokens=1, temperature=0)
    out_b, _ = fused.generate_token_ids(prompt, max_new_tokens=1, temperature=0)

    assert out_a == out_b
```

For stronger validation, expose a `forward_logits_for_prompt` helper and compare logits using `np.testing.assert_allclose`.

---

### 7.2 Gate/up fusion for MLP

#### Converter algorithm

For each MLP layer:

1. Resolve gate and up projection tensor names.
2. Verify same input width.
3. Concatenate gate and up along axis 0.
4. Write `fused_gate_up.weight`.
5. Keep `down_proj.weight` separate.
6. Record `row_splits` for gate and up.

Pseudocode:

```python
def try_fuse_gate_up(layer_idx, tensor_reader, shard_writer, manifest):
    gate_name = resolve_required([...gate candidates...])
    up_name = resolve_required([...up candidates...])
    down_name = resolve_required([...down candidates...])

    gate = tensor_reader.read(gate_name)
    up = tensor_reader.read(up_name)

    if gate.shape[1] != up.shape[1]:
        raise ConversionError("gate/up input dimensions do not match")

    fused = np.concatenate([gate, up], axis=0)
    fused_name = canonical_fused_gate_up_name(layer_idx, gate_name)

    offset, nbytes = shard_writer.write_aligned(fused)

    manifest.add_tensor(...)
    manifest.add_fused_group(
        name=fused_name,
        kind="gate_up",
        members=[gate_name, up_name],
        row_splits={"gate": gate.shape[0], "up": up.shape[0]},
    )

    # down remains separate
    write_or_reference_tensor(down_name)
```

#### Runtime algorithm

```python
def _mlp_step(self, layer_idx, hidden, *, telemetry):
    fused_info = self._resolve_fused_gate_up(layer_idx)
    if fused_info is not None:
        fused = self.store.get(fused_info.tensor_name, telemetry)
        gate_up = timed_dot(np, hidden, fused.T, kind="fused_gate_up", telemetry=telemetry)

        gate_rows = fused_info.row_splits["gate"]
        up_rows = fused_info.row_splits["up"]

        gate_values = gate_up[:gate_rows]
        up_values = gate_up[gate_rows:gate_rows + up_rows]
    else:
        gate = self._get_tensor([...])
        up = self._get_tensor([...])
        gate_values = timed_dot(np, hidden, gate.T, kind="mlp_gate", telemetry=telemetry)
        up_values = timed_dot(np, hidden, up.T, kind="mlp_up", telemetry=telemetry)

    fused_activation = silu(gate_values) * up_values

    down = self._get_tensor([...])
    return timed_dot(np, fused_activation, down.T, kind="mlp_down", telemetry=telemetry)
```

#### Correctness tests

Add tests comparing separate MLP and fused gate/up MLP on the same toy model.

Acceptance criterion:

```python
np.testing.assert_allclose(logits_separate, logits_fused, rtol=1e-4, atol=1e-5)
```

Use looser tolerance only for FP16 storage.

---

## 8. Phase 4: FP16 storage with FP32 compute

### Objective

Reduce disk footprint and disk read volume without depending on CPU FP16 matmul performance.

### Files likely touched

- `src/disk_llm/converter.py`
- `src/disk_llm/runtime/memmap.py`
- `src/disk_llm/runtime/model.py`
- `src/disk_llm/manifest.py`
- `tests/test_converter.py`
- `tests/test_runtime_toy_model.py`

### Converter behavior

Add converter option:

```bash
disk-llm convert /path/to/model ./packed-model --weight-storage-dtype float16
```

Valid initial values:

- `float32`
- `float16`

Do not implement int8 in this phase.

Pseudocode:

```python
def cast_for_storage(array, storage_dtype):
    if storage_dtype == "float32":
        return array.astype(np.float32, copy=False)
    if storage_dtype == "float16":
        return array.astype(np.float16)
    raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
```

Apply before writing each tensor or fused tensor:

```python
array_to_write = cast_for_storage(array, experiment_config.weight_storage_dtype)
shard_writer.write_aligned(array_to_write)
```

### Runtime behavior

Default compute policy:

```python
weight_for_compute = weight.astype(np.float32, copy=False)
hidden_for_compute = hidden.astype(np.float32, copy=False)
out = hidden_for_compute @ weight_for_compute.T
```

Warning: for `np.memmap` with `float16`, `astype(np.float32, copy=False)` may still allocate because dtype changes. That is acceptable for Phase 4 because correctness comes first. Benchmark will reveal whether the smaller disk read offsets the conversion cost.

### Optional layer-level casting helper

```python
def as_compute_array(array, compute_dtype="float32"):
    np = require_numpy()
    dtype = np.dtype(compute_dtype)
    if array.dtype == dtype:
        return array
    return array.astype(dtype, copy=False)
```

### Correctness tests

Compare float32 storage and float16 storage on a toy model:

```python
def test_float16_storage_close_to_float32_storage():
    fp32 = build_toy_model(weight_storage_dtype="float32")
    fp16 = build_toy_model(weight_storage_dtype="float16")

    logits32 = fp32.forward_logits([1, 2, 3])
    logits16 = fp16.forward_logits([1, 2, 3])

    np.testing.assert_allclose(logits32, logits16, rtol=5e-2, atol=5e-2)
```

For generated tokens, greedy output may occasionally differ if logits are close. Prefer logits comparison.

### Acceptance criteria

Phase 4 is complete when:

- float32 behavior is unchanged,
- float16 packed footprint is smaller,
- manifest records dtype correctly,
- runtime loads float16 memmaps correctly,
- correctness test passes with documented tolerance,
- benchmark metadata reports storage dtype and compute dtype.

---

## 9. Phase 5: budgeted tensor caching

### Objective

Reduce repeated disk reads for hot small tensors while preserving the disk-backed nature of the project.

### Files likely touched

- `src/disk_llm/runtime/memmap.py`
- `src/disk_llm/runtime/model.py`
- `src/disk_llm/runtime/telemetry.py`
- `src/disk_llm/manifest.py`
- `tests/test_runtime_toy_model.py`

### Cache policy

Initial cache modes:

- `none`
- `small`
- `small_plus_embedding_rows`
- `explicit_lm_head`

Recommended default:

```text
none
```

Recommended experimental config:

```text
cache_budget_mb = 2048
cache_small_tensors = true
cache_embedding_rows = true
cache_lm_head = false
```

### Tensor categories

Small tensors to cache:

- layer norm weights,
- q_norm weights,
- k_norm weights,
- small gates,
- scalar tensors,
- config-derived small arrays.

Potentially large tensors to avoid by default:

- embedding matrix,
- LM head,
- MLP projections,
- attention projections.

### MemmapTensorStore design

```python
class MemmapTensorStore:
    def __init__(self, manifest, *, base_dir, cache_budget_bytes=0, cache_policy=None):
        self.manifest = manifest
        self.base_dir = Path(base_dir)
        self._mmap_cache = {}
        self._resident_cache = {}
        self._resident_bytes = 0
        self._cache_budget_bytes = cache_budget_bytes
        self._lock = threading.RLock()

    def get(self, tensor_name, telemetry=None):
        mmap_array = self._get_mmap(tensor_name)
        entry = self.manifest.tensors[tensor_name]

        if self._should_resident_cache(tensor_name, entry):
            return self._get_resident_copy(tensor_name, mmap_array, telemetry)

        if telemetry is not None:
            telemetry.record_cache_event(tensor_name, hit=False, resident=False)
        return mmap_array
```

Resident cache helper:

```python
def _get_resident_copy(self, tensor_name, mmap_array, telemetry):
    with self._lock:
        if tensor_name in self._resident_cache:
            if telemetry:
                telemetry.record_cache_event(tensor_name, hit=True, resident=True)
            return self._resident_cache[tensor_name]

        nbytes = mmap_array.nbytes
        if self._resident_bytes + nbytes > self._cache_budget_bytes:
            if telemetry:
                telemetry.record_cache_event(tensor_name, hit=False, resident=False, skipped="budget")
            return mmap_array

        array = np.array(mmap_array, copy=True)
        self._resident_cache[tensor_name] = array
        self._resident_bytes += nbytes

        if telemetry:
            telemetry.record_cache_event(tensor_name, hit=False, resident=True, bytes=nbytes)
        return array
```

### Embedding row cache

Do not cache full embedding matrix by default. Cache rows by token ID:

```python
class EmbeddingRowCache:
    def __init__(self, max_rows=4096):
        self.cache = OrderedDict()
        self.max_rows = max_rows

    def get_row(self, embed_weight, token_id):
        if token_id in self.cache:
            row = self.cache.pop(token_id)
            self.cache[token_id] = row
            return row
        row = np.asarray(embed_weight[int(token_id)]).copy()
        self.cache[token_id] = row
        if len(self.cache) > self.max_rows:
            self.cache.popitem(last=False)
        return row
```

Runtime use:

```python
if self.config.cache_embedding_rows:
    hidden = self.embedding_row_cache.get_row(embed_weight, token_id)
else:
    hidden = np.asarray(embed_weight[int(token_id)])
```

### LM head cache

LM head cache should be explicit only:

```bash
DISK_LLM_CACHE_LM_HEAD=1
```

Telemetry must report RAM cost:

```json
{
  "lm_head_cached": true,
  "lm_head_cache_bytes": 1234567890
}
```

### Acceptance criteria

Phase 5 is complete when:

- cache can be disabled fully,
- cache budget is enforced,
- small tensors can be cached,
- embedding row cache works independently of full embedding cache,
- LM head is never cached unless explicitly requested,
- telemetry reports cache hits, misses, resident bytes, and skipped cache attempts.

---

## 10. Phase 6: bounded prefetch and OS hints

### Objective

Replace ad hoc per-layer thread spawning with a bounded, measurable, opt-in prefetch subsystem.

### Files likely touched

- `src/disk_llm/runtime/model.py`
- `src/disk_llm/runtime/memmap.py`
- `src/disk_llm/runtime/prefetch.py` (new)
- `src/disk_llm/runtime/telemetry.py`
- `tests/test_qwen_runtime_regressions.py`

### New module

Create:

```text
src/disk_llm/runtime/prefetch.py
```

### Prefetcher design

```python
class LayerPrefetcher:
    def __init__(self, store, *, max_workers=1, mode="touch", telemetry=None):
        self.store = store
        self.mode = mode
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.inflight = set()
        self.completed = set()
        self.lock = threading.Lock()
        self.telemetry = telemetry

    def submit_layer(self, layer_idx: int):
        with self.lock:
            if layer_idx in self.inflight or layer_idx in self.completed:
                self._record("skipped_duplicate", layer_idx)
                return
            self.inflight.add(layer_idx)

        future = self.executor.submit(self._prefetch_layer, layer_idx)
        future.add_done_callback(lambda f: self._mark_done(layer_idx, f))
        self._record("submitted", layer_idx)

    def _prefetch_layer(self, layer_idx: int):
        names = self.store.names_for_layer(layer_idx)
        for name in names:
            arr = self.store.get(name)
            if self.mode == "madvise":
                self._madvise(arr)
            elif self.mode == "touch":
                self._touch_pages(arr)

    def shutdown(self):
        self.executor.shutdown(wait=True)
```

### Store helper

Add a layer index helper to `MemmapTensorStore`:

```python
def names_for_layer(self, layer_idx: int) -> list[str]:
    prefixes = [
        f"model.layers.{layer_idx}.",
        f"model.language_model.layers.{layer_idx}.",
    ]
    return [name for name in self.manifest.tensors if any(name.startswith(p) for p in prefixes)]
```

For fused tensor names, ensure they follow layer prefixes or add them to a layer index during manifest loading.

### OS hint implementation

```python
def _madvise(self, arr):
    try:
        import mmap
        mm = getattr(arr, "_mmap", None)
        if mm is not None and hasattr(mm, "madvise"):
            mm.madvise(mmap.MADV_WILLNEED)
            return True
    except Exception:
        return False
    return False
```

### Page-touch fallback

```python
def _touch_pages(self, arr, page_size=4096):
    try:
        view = arr.view("uint8")
        _ = view[::page_size].sum()
        return True
    except Exception:
        return False
```

### Runtime integration

Replace current direct thread creation with:

```python
if self.prefetcher is not None:
    target_layer = layer_idx + self.experiment_config.prefetch_distance
    if target_layer < num_hidden_layers:
        self.prefetcher.submit_layer(target_layer)
```

### Important benchmark separation

Do not compare prefetch and non-prefetch without labeling them. Use run labels such as:

- `qwen35-v5-baseline-no-prefetch`
- `qwen35-v5-threadpool-prefetch-d1-w1`
- `qwen35-v5-madvise-prefetch-d1`

### Acceptance criteria

Phase 6 is complete when:

- no unbounded thread creation remains,
- prefetch is disabled by default,
- prefetch mode appears in metadata,
- duplicate prefetch requests are skipped,
- telemetry reports submitted/completed/skipped counts,
- prefetch tests remain env-scoped, not manifest-scoped.

---

## 11. Phase 7: prompt prefill vectorization

### Objective

Avoid streaming the same layer weights once per prompt token. Process prompt chunks as matrices.

### Why this matters

The current prompt path loops over `prompt_ids` and calls `forward_step` per token. For long prompts, this means the same model weights are repeatedly touched. The 128-token baseline is therefore much worse than the 8-token baseline.

### Files likely touched

- `src/disk_llm/runtime/model.py`
- `src/disk_llm/runtime/kernels.py`
- `src/disk_llm/runtime/telemetry.py`
- `tests/test_runtime_toy_model.py`

### Scope

Implement prefill for standard full-attention blocks first. Keep fallback behavior for:

- delta blocks,
- linear attention blocks,
- hybrid blocks not yet vectorized.

### Runtime API

Add:

```python
def forward_prefill(self, prompt_ids: Sequence[int], *, cache, telemetry):
    ...
```

Use it in `generate_token_ids`:

```python
if self.experiment_config.prefill_mode == "chunked" and self._can_prefill_chunked():
    logits = self.forward_prefill(prompt_ids, cache=cache, telemetry=telemetry)
else:
    logits = None
    for position, token_id in enumerate(prompt_ids):
        logits = self.forward_step(token_id, position=position, cache=cache, telemetry=telemetry)
```

### Prefill algorithm

```python
def forward_prefill(self, prompt_ids, *, cache, telemetry):
    np = require_numpy()
    positions = np.arange(len(prompt_ids), dtype=np.int64)

    embed_weight = self._get_tensor(self._EMBED_TENSOR_CANDIDATES, telemetry=telemetry)
    hidden = np.asarray(embed_weight[np.asarray(prompt_ids, dtype=np.int64)])
    # hidden shape: [T, hidden_size]

    for layer_idx in range(self.config.num_hidden_layers):
        hidden = self._forward_layer_prefill(
            layer_idx,
            hidden,
            positions=positions,
            cache=cache[layer_idx],
            telemetry=telemetry,
        )

    norm_weight = self._get_tensor(self._FINAL_NORM_TENSOR_CANDIDATES, telemetry=telemetry)
    hidden = self._apply_hidden_norm(hidden, norm_weight)

    last_hidden = hidden[-1]
    lm_head = self._get_tensor(self._LM_HEAD_TENSOR_CANDIDATES, telemetry=telemetry)
    logits = timed_dot(np, last_hidden, lm_head.T, kind="lm_head", telemetry=telemetry)
    return logits
```

### Attention prefill sketch

This is more complex than single-token attention because each token attends causally to previous tokens.

```python
def _attention_prefill(self, layer_idx, hidden_matrix, *, positions, cache, telemetry):
    # hidden_matrix: [T, hidden_size]
    qkv = hidden_matrix @ fused_qkv.T
    q, k, v = split_qkv_matrix(qkv, row_splits)

    # q: [T, num_q_heads, head_dim]
    # k: [T, num_kv_heads, head_dim]
    # v: [T, num_kv_heads, head_dim]
    q = apply_rope_batch(q, positions)
    k = apply_rope_batch(k, positions)

    # Store all prompt keys/values in cache for decode.
    cache.keys.extend(list(k))
    cache.values.extend(list(v))

    # Compute causal attention for prompt tokens.
    outputs = []
    for t in range(T):
        k_hist = k[:t+1]
        v_hist = v[:t+1]
        out_t = grouped_query_attention_step(q[t], k_hist, v_hist, scale=scale)
        outputs.append(out_t)

    attn_out = np.stack(outputs, axis=0)
    attn_out = attn_out.reshape(T, -1)
    return attn_out @ o_proj.T
```

This loop still has a Python loop over prompt positions. That is acceptable for the first chunked prefill milestone if it reduces repeated weight streaming. Later, it can be optimized with a batched causal attention kernel.

### Chunking

For long prompts, use chunks:

```python
for chunk_start in range(0, len(prompt_ids), chunk_size):
    chunk = prompt_ids[chunk_start:chunk_start + chunk_size]
    logits = self.forward_prefill_chunk(chunk, start_position=chunk_start, cache=cache)
```

Suggested default:

```text
prefill_chunk_size = 16
```

Benchmark chunk sizes:

- 1
- 4
- 8
- 16
- 32

### Acceptance criteria

Phase 7 is complete when:

- token-by-token path remains default,
- chunked prefill is opt-in,
- chunked prefill produces close logits on toy full-attention model,
- fallback works for unsupported block types,
- benchmark metadata reports prefill mode and chunk size,
- prompt length 128 improves versus token path without increasing RSS unacceptably.

---

## 12. Phase 8: shard alignment and optional layer grouping

### Objective

Improve disk and page behavior without changing model math.

### Files likely touched

- `src/disk_llm/converter.py`
- `src/disk_llm/manifest.py`
- `src/disk_llm/inspect.py`
- `tests/test_converter.py`

### Offset alignment

Align tensor writes to at least 4096 bytes.

```python
def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 1:
        return offset
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)
```

Shard writer:

```python
def write_aligned(self, array, alignment=4096):
    current = self.file.tell()
    aligned = align_offset(current, alignment)
    padding = aligned - current
    if padding:
        self.file.write(b"\0" * padding)
    offset = self.file.tell()
    array.tofile(self.file)
    return offset, array.nbytes
```

Manifest:

```json
{
  "offset": 8192,
  "alignment": 4096
}
```

### Layer grouping

Start with conservative group sizes:

- `1` current behavior,
- `2` small grouping,
- `4` experimental.

Do not start with one giant model file.

Shard naming:

```text
layers/layers_000_001.bin
layers/layers_002_003.bin
```

Helper:

```python
def shard_name_for_layer(layer_idx, group_layers):
    group_start = (layer_idx // group_layers) * group_layers
    group_end = group_start + group_layers - 1
    if group_layers == 1:
        return f"layers/layer_{layer_idx:03d}.bin"
    return f"layers/layers_{group_start:03d}_{group_end:03d}.bin"
```

### Inspect output

Update `disk-llm inspect --manifest` to show:

```text
manifest_version: 2
weight_storage_dtype: float16
align_bytes: 4096
group_layers: 2
fused_qkv_layers: 32
fused_gate_up_layers: 32
```

### Acceptance criteria

Phase 8 is complete when:

- default grouping remains 1,
- align-bytes can be set to 1 or 4096,
- manifest offsets are correct,
- checksums still validate,
- grouped shards load correctly,
- inspect command reports layout accurately.

---

## 13. Phase 9: advanced experiments after v5

These ideas are valuable but should not block v5.

### 13.1 INT8 quantization

Only implement after FP16 storage is validated.

Recommended design:

- per-row or blockwise scales,
- no full FP32 materialization if possible,
- separate correctness tests,
- explicit quality/perplexity checks.

Avoid this naive runtime pattern for large matrices:

```python
weight = qweight.astype(np.float32) * scale
out = hidden @ weight.T
```

It can allocate a full FP32 matrix every token.

### 13.2 Numba kernels

Use only after profiling shows Python kernels are material. Candidate functions:

- RoPE,
- softmax,
- grouped query attention inner loop,
- recurrent state updates,
- page-touch prefetch loop.

Do not expect Numba to speed up NumPy BLAS dot calls directly.

### 13.3 PyTorch backend

Useful as a comparison backend, not the core identity.

Possible backend names:

- `disk_llm_numpy`
- `disk_llm_torch_from_numpy`
- `hf_cpu`

### 13.4 llama.cpp comparison

Treat as an external baseline or export target. Do not make it a core Disk-LLM runtime until the NumPy/memmap path is fully measured.

---

## 14. Benchmark plan

### Core benchmark matrix

Use a small matrix first, then a full remote run.

| Variant | Fusion | Dtype | Prefetch | Cache |
|---|---:|---:|---:|---:|
| baseline | none | fp32 | none | none |
| v5-a | qkv | fp32 | none | none |
| v5-b | qkv+mlp | fp32 | none | none |
| v5-c | qkv+mlp | fp16 | none | none |
| v5-d | qkv+mlp | fp16 | threadpool | none |
| v5-e | qkv+mlp | fp16 | threadpool | budgeted |
| v5-f | qkv+mlp | fp16 | threadpool | budgeted |

For `v5-f`, enable chunked prefill only if Phase 7 is complete.

### Prompt lengths

Use:

- 8
- 32
- 128
- 512

For early smoke tests, use 8 and 128 only.

### Generation lengths

Use:

- 2 for smoke tests,
- 16 for full tests,
- optionally 64 for decode-heavy tests.

### Metrics

Report:

- tokens/s,
- first-token latency,
- prompt prefill seconds,
- decode seconds,
- peak RSS,
- logical mapped MB,
- major page faults,
- minor page faults,
- dot calls,
- dot seconds,
- tensor get calls,
- cache resident MB,
- prefetch counts,
- layer count,
- tensors touched.

### Guardrails

A benchmark row is invalid if:

- `layer_count == 0`,
- `tensors_touched` is suspiciously low,
- `generated_token_count == 0` unless EOS was generated and recorded,
- runtime config is missing,
- model SHA/revision is missing,
- manifest version is missing,
- backend name is missing.

### PowerShell smoke commands

Baseline:

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
  --run-label qwen35-v5-baseline
```

Fused FP16 experiment:

```powershell
$env:DISK_LLM_FUSE_QKV="1"
$env:DISK_LLM_FUSE_GATE_UP="1"
$env:DISK_LLM_WEIGHT_STORAGE_DTYPE="float16"
$env:DISK_LLM_COMPUTE_DTYPE="float32"
$env:DISK_LLM_ALIGN_BYTES="4096"

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
  --run-label qwen35-v5-fused-fp16

Remove-Item Env:DISK_LLM_FUSE_QKV
Remove-Item Env:DISK_LLM_FUSE_GATE_UP
Remove-Item Env:DISK_LLM_WEIGHT_STORAGE_DTYPE
Remove-Item Env:DISK_LLM_COMPUTE_DTYPE
Remove-Item Env:DISK_LLM_ALIGN_BYTES
```

Prefetch experiment:

```powershell
$env:DISK_LLM_FUSE_QKV="1"
$env:DISK_LLM_FUSE_GATE_UP="1"
$env:DISK_LLM_WEIGHT_STORAGE_DTYPE="float16"
$env:DISK_LLM_PREFETCH_MODE="threadpool"
$env:DISK_LLM_PREFETCH_DISTANCE="1"
$env:DISK_LLM_PREFETCH_WORKERS="1"

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
  --run-label qwen35-v5-fused-fp16-prefetch
```

---

## 15. Testing plan

### Unit tests

Add or update:

```text
tests/test_manifest_v2.py
tests/test_converter_fusion.py
tests/test_runtime_fused_qkv.py
tests/test_runtime_fused_gate_up.py
tests/test_runtime_dtype.py
tests/test_runtime_cache.py
tests/test_runtime_prefetch.py
tests/test_benchmark_metadata.py
```

### Test categories

#### Manifest tests

- v1 manifest loads,
- v2 manifest loads,
- missing fused row splits fail clearly,
- unsupported dtype fails clearly,
- layout features are parsed correctly.

#### Converter tests

- QKV fusion writes one tensor,
- row splits are correct,
- gate/up fusion writes one tensor,
- down projection remains separate,
- alignment padding produces aligned offsets,
- grouped shard path is correct.

#### Runtime tests

- fused QKV logits match separate QKV logits,
- fused gate/up logits match separate gate/up logits,
- FP16 storage runs and gives close logits,
- cache budget is enforced,
- embedding row cache works,
- prefetcher does not submit duplicate layers,
- prefetcher uses bounded worker count.

#### Benchmark tests

- metadata contains experiment config,
- invalid zero-layer rows are rejected,
- cache and prefetch metrics are present,
- cold/warm label is present.

### CI command

```bash
python -m unittest discover -v
```

Add optional slow tests:

```bash
DISK_LLM_RUN_SLOW_TESTS=1 python -m unittest discover -v
```

---

## 16. GitHub Pages and README update plan

### Principles

Do not overstate improvements. Keep the site honest. Disk-LLM should be framed as:

- inspectable,
- disk-backed,
- measurable,
- experimental,
- improving through validated runs.

### Pages to update

Likely files:

```text
README.md
docs/index.md
docs/architecture.md
docs/modal_remote_run.md
docs/assets/*
```

### New result card

Create a result card for each validated run:

```markdown
## Qwen3.5-9B v5 Fused FP16 Experiment

- run label: `qwen35-v5-fused-fp16`
- manifest version: `2`
- packed tensors: `<value>`
- packed shards: `<value>`
- packed footprint: `<value>`
- fusion: `qkv + gate/up`
- storage dtype: `float16`
- compute dtype: `float32`
- prefetch: `none`
- cache budget: `0 MB`
```

### Charts to generate

Use the existing plotting script or extend it for:

- throughput by prompt length,
- first-token latency by prompt length,
- prefill/decode split,
- dot seconds by kind,
- logical mapped MB,
- peak RSS,
- page faults,
- cache resident MB,
- prefetch events.

### Recommended README wording

Use wording like:

> This run improves the native NumPy/memmap path under the tested Modal setup, but it should not be interpreted as a general inference win. Disk-LLM remains a research system for inspectable disk-backed execution.

Avoid wording like:

> Disk-LLM is faster than Hugging Face.

unless the exact benchmark proves it and the environment is clearly stated.

---

## 17. Suggested PR breakdown

Do not submit one giant PR. Use small PRs.

### PR 1: profiling and metadata

Files:

- telemetry,
- benchmark scripts,
- metadata output,
- tests.

Acceptance:

- no behavior change,
- richer benchmark output.

### PR 2: experiment config

Files:

- new config dataclass,
- CLI/env parsing,
- metadata integration.

Acceptance:

- all defaults preserve existing behavior.

### PR 3: manifest v2

Files:

- manifest loader/writer,
- inspect command,
- tests.

Acceptance:

- v1 backward compatible,
- v2 toy manifests load.

### PR 4: QKV fusion

Files:

- converter,
- runtime,
- tests.

Acceptance:

- fused and separate logits match on toy model.

### PR 5: gate/up fusion

Files:

- converter,
- runtime,
- tests.

Acceptance:

- down remains separate,
- logits match.

### PR 6: FP16 storage

Files:

- converter dtype option,
- runtime dtype handling,
- tests.

Acceptance:

- packed footprint drops,
- toy logits close.

### PR 7: cache and prefetch

Files:

- cache subsystem,
- prefetch subsystem,
- telemetry,
- tests.

Acceptance:

- disabled by default,
- bounded behavior.

### PR 8: prefill vectorization

Files:

- runtime prefill path,
- kernels,
- tests.

Acceptance:

- opt-in,
- full-attention toy parity.

### PR 9: docs and site refresh

Files:

- README,
- docs,
- plots,
- result bundles.

Acceptance:

- figures generated from validated output,
- no overclaims.

---

## 18. Codex/GPT-5.5 task prompts

Use these task prompts one at a time. Do not ask Codex to implement all phases at once.

### Task prompt 1: telemetry

```text
Implement Phase 0 of docs/disk_llm_revised_implementation_plan.md. Add richer telemetry for dot calls, tensor gets, cache metrics, prefill/decode timing, LM-head timing, page fault counters when available, and benchmark metadata. Preserve current behavior. Add unit tests. Do not change model math.
```

### Task prompt 2: experiment config

```text
Implement Phase 1. Add DiskLLMExperimentConfig with defaults that preserve current behavior. Parse relevant env vars and optional CLI flags. Ensure benchmark_metadata.json records the full config. Add tests.
```

### Task prompt 3: manifest v2

```text
Implement Phase 2. Add manifest_version support and v2 fused_groups metadata while keeping v1 manifests backward compatible. Update inspect output. Add tests for v1 and v2 manifests.
```

### Task prompt 4: QKV fusion

```text
Implement Phase 3.1 only. Add converter-time QKV fusion behind DISK_LLM_FUSE_QKV or a converter flag. Store row splits in manifest v2. Update runtime to use fused QKV if present and fall back otherwise. Do not split QKV into equal thirds. Add toy parity tests.
```

### Task prompt 5: gate/up fusion

```text
Implement Phase 3.2 only. Add converter-time gate/up fusion for MLP. Do not fuse down_proj. Store row splits. Update runtime to use fused gate/up if present. Add toy parity tests.
```

### Task prompt 6: FP16 storage

```text
Implement Phase 4. Add float16 storage support with float32 compute. Do not implement int8. Ensure manifest records dtype. Add converter and runtime tests.
```

### Task prompt 7: cache and prefetch

```text
Implement Phases 5 and 6. Add budgeted resident cache, embedding row cache, and bounded layer prefetcher. Keep disabled by default. Add telemetry and tests.
```

### Task prompt 8: prefill

```text
Implement Phase 7 as an opt-in chunked prefill path for standard full-attention toy models first. Fall back for unsupported block kinds. Add correctness tests and benchmark metadata.
```

---

## 19. Risks and mitigations

### Risk: fused tensor names break model-family adapters

Mitigation:

- keep original member names in `fused_groups`,
- use canonical fused names only internally,
- fall back to separate tensors when fused names are absent.

### Risk: FP16 storage changes generated tokens

Mitigation:

- compare logits, not just sampled tokens,
- use deterministic greedy sampling,
- document tolerance,
- keep float32 as default.

### Risk: cache hides disk behavior

Mitigation:

- cache disabled by default,
- report cache resident bytes,
- label cache experiments separately.

### Risk: prefetch improves warm cache but hurts cold cache

Mitigation:

- separate cold and warm labels,
- record prefetch metrics,
- benchmark with and without prefetch.

### Risk: prefill vectorization changes semantics

Mitigation:

- implement only for attention blocks first,
- fallback for unsupported blocks,
- compare logits against token-by-token path.

### Risk: benchmarks overclaim

Mitigation:

- keep README language conservative,
- publish raw CSVs,
- include exact run config,
- include HF CPU baseline.

---

## 20. Definition of done for Disk-LLM v5

Disk-LLM v5 is ready to publish when all of the following are true:

- `python -m unittest discover -v` passes.
- v1 manifests still work.
- v2 manifests include layout features and fused group metadata.
- fused QKV passes toy parity tests.
- fused gate/up passes toy parity tests.
- FP16 storage passes toy tolerance tests.
- cache is budgeted and disabled by default.
- prefetch is bounded and disabled by default.
- every benchmark row includes experiment config.
- zero-layer benchmark rows are rejected.
- a fresh Modal run validates nonzero executed layers.
- README and GitHub Pages are updated with raw CSVs and plots.
- Claims are limited to the exact validated run.

---

## 21. Reference links

[^readme]: Disk-LLM README, current validated baseline and project description: https://raw.githubusercontent.com/kilickursat/disk-llm/main/README.md

[^arch]: Disk-LLM architecture documentation: https://raw.githubusercontent.com/kilickursat/disk-llm/main/docs/architecture.md

[^runtime]: Disk-LLM runtime model implementation: https://raw.githubusercontent.com/kilickursat/disk-llm/main/src/disk_llm/runtime/model.py

[^status]: Disk-LLM status and next steps, including prefetch separation: https://raw.githubusercontent.com/kilickursat/disk-llm/main/STATUS_AND_NEXT_STEPS.md

[^numpy_memmap]: NumPy `memmap` documentation: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
