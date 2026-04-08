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
- Disk-LLM mean throughput: `0.018` tok/s; first-token latency: `88.501` s; peak RSS: `21873.832` MB.
- HF CPU mean throughput: `0.165` tok/s; first-token latency: `4.773` s; peak RSS: `19403.957` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.111`, first-token latency ratio is `18.541`, and peak RSS ratio is `1.127`.

### Prompt 128 / Generate 2
- Disk-LLM mean throughput: `0.002` tok/s; first-token latency: `1157.395` s; peak RSS: `21885.977` MB.
- HF CPU mean throughput: `0.080` tok/s; first-token latency: `17.142` s; peak RSS: `19407.145` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.021`, first-token latency ratio is `67.517`, and peak RSS ratio is `1.128`.

## Interpretation Notes

- Disk-LLM is most compelling if it maintains a materially lower RAM ceiling while staying within an acceptable throughput band versus the full-RAM HF baseline.
- Modal Volume-backed results should be described as remote-volume benchmarks, not bare-metal local-SSD benchmarks.
- If parity or quality looks weak despite reasonable throughput, the next likely target is the delta-block path and exact tensor-name coverage in the runtime adapter.
