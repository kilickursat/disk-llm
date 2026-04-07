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
- Disk-LLM mean throughput: `0.104` tok/s; first-token latency: `16.212` s; peak RSS: `11266.543` MB.
- HF CPU mean throughput: `0.117` tok/s; first-token latency: `6.990` s; peak RSS: `21957.195` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.890`, first-token latency ratio is `2.319`, and peak RSS ratio is `0.513`.

### Prompt 128 / Generate 2
- Disk-LLM mean throughput: `0.010` tok/s; first-token latency: `197.267` s; peak RSS: `11268.930` MB.
- HF CPU mean throughput: `0.068` tok/s; first-token latency: `18.740` s; peak RSS: `21957.195` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.147`, first-token latency ratio is `10.526`, and peak RSS ratio is `0.513`.

## Interpretation Notes

- Disk-LLM is most compelling if it maintains a materially lower RAM ceiling while staying within an acceptable throughput band versus the full-RAM HF baseline.
- Modal Volume-backed results should be described as remote-volume benchmarks, not bare-metal local-SSD benchmarks.
- If parity or quality looks weak despite reasonable throughput, the next likely target is the delta-block path and exact tensor-name coverage in the runtime adapter.
