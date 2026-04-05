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
- Disk-LLM mean throughput: `0.204` tok/s; first-token latency: `7.906` s; peak RSS: `11263.941` MB.
- HF CPU baseline data is missing for this case.

### Prompt 128 / Generate 2
- Disk-LLM mean throughput: `0.016` tok/s; first-token latency: `126.610` s; peak RSS: `11264.902` MB.
- HF CPU baseline data is missing for this case.

## Interpretation Notes

- Disk-LLM is most compelling if it maintains a materially lower RAM ceiling while staying within an acceptable throughput band versus the full-RAM HF baseline.
- Modal Volume-backed results should be described as remote-volume benchmarks, not bare-metal local-SSD benchmarks.
- If parity or quality looks weak despite reasonable throughput, the next likely target is the delta-block path and exact tensor-name coverage in the runtime adapter.
