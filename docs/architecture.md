# Architecture

Disk-LLM has three primary layers.

## 1. Source inspection

`disk_llm.safetensors_io` parses the published safetensors header format directly. That gives the project a lightweight, dependency-minimal inspection path that works before NumPy, PyTorch, or Hugging Face tooling is installed.

`disk_llm.inspect` then combines:

- `config.json`
- safetensors tensor metadata
- Disk-LLM packing heuristics

This makes it possible to answer:

- how many tensors exist
- how large the checkpoint is
- which tensors belong to the text path
- how the converter will shard them

## 2. Packed layout

The converter rewrites selected tensor byte ranges into layer-oriented shard files:

- `embeddings/embeddings.bin`
- `layers/layer_XXX.bin`
- `final/final.bin`

The manifest records:

- tensor name
- shard
- byte offset
- byte size
- dtype
- shape
- source file
- tensor checksum

This layout keeps tensors that are commonly accessed together physically close together, which is the main practical difference between Disk-LLM and simply reading the original safetensors snapshot directly.

## 3. Experimental runtime

The runtime is intentionally small and transparent.

- `MemmapTensorStore` lazily opens `numpy.memmap` views from the manifest
- `TelemetryRecorder` tracks logical bytes mapped and layer timing
- `DiskLLMTextModel` implements a text-only decoder loop with hybrid block support

The runtime currently focuses on:

- short prompts
- batch size 1
- CPU experiments
- visibility into what was touched and when

## Important limitation

Qwen 3.5 uses a newer hybrid architecture and exact tensor naming can vary by upstream integration. The converter and manifest flow are stable, but the runtime adapter should still be validated against a real `Qwen/Qwen3.5-9B` snapshot before being treated as a correctness reference.
