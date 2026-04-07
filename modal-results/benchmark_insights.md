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
- Disk-LLM mean throughput: `0.153` tok/s; first-token latency: `10.388` s; peak RSS: `11266.043` MB.
- HF CPU mean throughput: `0.125` tok/s; first-token latency: `6.117` s; peak RSS: `21943.293` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `1.221`, first-token latency ratio is `1.698`, and peak RSS ratio is `0.513`.

### Prompt 128 / Generate 2
- Disk-LLM mean throughput: `0.012` tok/s; first-token latency: `158.646` s; peak RSS: `11268.035` MB.
- HF CPU mean throughput: `0.076` tok/s; first-token latency: `18.310` s; peak RSS: `21943.293` MB.
- Relative to HF CPU, Disk-LLM throughput ratio is `0.162`, first-token latency ratio is `8.664`, and peak RSS ratio is `0.514`.

## Interpretation Notes

- Disk-LLM is most compelling if it maintains a materially lower RAM ceiling while staying within an acceptable throughput band versus the full-RAM HF baseline.
- Modal Volume-backed results should be described as remote-volume benchmarks, not bare-metal local-SSD benchmarks.
- If parity or quality looks weak despite reasonable throughput, the next likely target is the delta-block path and exact tensor-name coverage in the runtime adapter.
