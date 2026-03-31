<p align="center">
  <img src="logo.png" alt="Disk-LLM logo" width="420">
</p>

# Disk-LLM

Disk-LLM is an inspectable disk-backed LLM research kit. The project is built around two ideas:

1. pack weights into layer-oriented memmap shards instead of treating raw checkpoints as the final runtime layout
2. make the runtime observable, so you can see which tensors are mapped, how long each layer took, and what the generation loop is doing

This repository is intentionally opinionated. It is not trying to beat mature inference engines like `llama.cpp`, `vLLM`, or `SGLang` on production throughput. The v1 goal is narrower and more tangible:

Run short text-only generations from a disk-packed model layout on CPU while exposing live telemetry for researchers and contributors.

A branded static landing page is included at [`site/index.html`](site/index.html), and the local demo UI now uses the project logo directly.

## Current status

- `convert`: implemented
- `inspect`: implemented
- `bench`: implemented
- `generate`: implemented
- `demo`: implemented as an optional Gradio wrapper
- Qwen 3.5 text runtime: experimental adapter scaffold

The converter and manifest flow are solid enough to start experimenting today. The NumPy runtime is intentionally marked experimental because Qwen 3.5 uses a newer hybrid architecture and exact tensor-name coverage should be validated against a real checkpoint snapshot.

## Why this project exists

The original project idea focused on `numpy.memmap`, but `safetensors` already gives you zero-copy and lazy-loading benefits. Disk-LLM becomes interesting when it adds something new:

- layer-packed disk layout
- model inspection before conversion
- text-only subgraph filtering for multimodal checkpoints
- runtime telemetry
- a compact codebase that contributors can actually read

## Project structure

```text
disk-llm/
├─ src/disk_llm/
│  ├─ cli.py
│  ├─ converter.py
│  ├─ inspect.py
│  ├─ layout.py
│  ├─ manifest.py
│  ├─ optional.py
│  ├─ safetensors_io.py
│  └─ runtime/
│     ├─ config.py
│     ├─ kernels.py
│     ├─ memmap.py
│     ├─ model.py
│     └─ telemetry.py
├─ docs/
│  └─ architecture.md
├─ tests/
└─ pyproject.toml
```

## Quick start

### 1. Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[hf,demo,test]
```

### 2. Inspect a Hugging Face snapshot

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

### 5. Generate with token ids

```bash
disk-llm generate ./packed-qwen35/manifest.json --prompt-ids 1,2,3 --max-new-tokens 8 --show-telemetry
```

### 6. Generate with text

If the source snapshot contains tokenizer files and `transformers` is installed:

```bash
disk-llm generate ./packed-qwen35/manifest.json --prompt "Explain disk-backed inference in one paragraph."
```

### 7. Launch the demo UI

```bash
disk-llm demo ./packed-qwen35/manifest.json --tokenizer /path/to/Qwen3.5-9B
```

## What gets packed

The default v1 converter targets the text-only path:

- `model.embed_tokens.*`
- `model.layers.<n>.*`
- `model.norm.*`
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

These metrics are intentionally approximate on CPU because page-cache behavior is owned by the OS, but they are still useful for comparative experiments.

## Development

Run the stdlib test suite:

```bash
python -m unittest discover -v
```

The repository is designed to remain importable even when optional dependencies are missing. That allows contributors to inspect the CLI, manifest flow, and converter logic without first downloading the full inference stack.

## Roadmap

- validate tensor-name coverage against a real `Qwen/Qwen3.5-9B` snapshot
- harden the Qwen 3.5 text adapter against exact hybrid block definitions
- add parity checks against a reference backend when `transformers` is installed
- deepen telemetry with cache-specific metrics and disk-fault sampling
- add focused docs for custom adapters and alternate packing strategies

## Open source contribution

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Good first contributions include:

- new tensor-name adapters
- improved block-layout inspection
- benchmark datasets and reproducible reports
- runtime correctness tests against reference implementations
- tokenizer and chat-template integration improvements
