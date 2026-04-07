# Benchmark Summary

- Manifest: `/__modal/volumes/vo-oHh6Gg8uG2RRqiwJljz00y/packed/Qwen--Qwen3.5-9B/qwen35-9b-modal-cpu-prefetch-exp/manifest.json`
- Runs per case: `1`
- Warmup runs per case: `0`

| Backend | Prompt Tokens | Max New Tokens | Mean Tokens/s | Mean First Token (s) | Mean Peak RSS (MB) | Mean IO Read (MB) | Mean Logical Mapped (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Disk-LLM | 8 | 2 | 0.104 | 16.212 | 11266.543 | n/a | 38800.078 |
| HF CPU | 8 | 2 | 0.117 | 6.990 | 21957.195 | n/a | n/a |
| Disk-LLM | 128 | 2 | 0.010 | 197.267 | 11268.930 | n/a | 504401.016 |
| HF CPU | 128 | 2 | 0.068 | 18.740 | 21957.195 | n/a | n/a |
