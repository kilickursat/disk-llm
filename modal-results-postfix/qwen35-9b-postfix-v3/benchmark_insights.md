# Disk-LLM Modal Benchmark Insights

- Model: `Qwen/Qwen3.5-9B`
- Requested revision: `main`
- Resolved SHA: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- HF CPU dtype: `float32`
- Source tensors discovered: `775`
- Packed tensors: `427`
- Packed shards: `34`

## Key Comparisons

### Prompt 8 / Generate 2
- Disk-LLM mean throughput: `0.015` tok/s; first-token latency: `114.217` s; peak RSS: `24463.441` MB.
- HF CPU mean throughput: `0.136` tok/s; first-token latency: `6.525` s; peak RSS: `21981.113` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.108`, first-token latency ratio is `17.504`, and peak RSS ratio is `1.113`.

### Prompt 128 / Generate 2
- Disk-LLM mean throughput: `0.001` tok/s; first-token latency: `1520.855` s; peak RSS: `24480.340` MB.
- HF CPU mean throughput: `0.079` tok/s; first-token latency: `17.074` s; peak RSS: `21981.113` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.016`, first-token latency ratio is `89.072`, and peak RSS ratio is `1.114`.

## Interpretation Notes

- Disk-LLM is most compelling if it maintains a materially lower RAM ceiling while staying within an acceptable throughput band versus the full-RAM HF baseline.
- Modal Volume-backed results should be described as remote-volume benchmarks, not bare-metal local-SSD benchmarks.
- If parity or quality looks weak despite reasonable throughput, the next likely target is the delta-block path and exact tensor-name coverage in the runtime adapter.
