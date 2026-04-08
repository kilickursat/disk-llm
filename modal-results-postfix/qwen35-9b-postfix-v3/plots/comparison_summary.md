# Benchmark Summary

- Manifest: `/__modal/volumes/vo-oHh6Gg8uG2RRqiwJljz00y/packed/Qwen--Qwen3.5-9B/qwen35-9b-postfix-v3/manifest.json`
- Runs per case: `1`
- Warmup runs per case: `0`

| Backend | Prompt Tokens | Max New Tokens | Mean Tokens/s | Mean First Token (s) | Mean Peak RSS (MB) | Mean IO Read (MB) | Mean Logical Mapped (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Disk-LLM | 8 | 2 | 0.015 | 114.217 | 24463.441 | n/a | 170780.317 |
| HF CPU | 8 | 2 | 0.136 | 6.525 | 21981.113 | n/a | n/a |
| Disk-LLM | 128 | 2 | 0.001 | 1520.855 | 24480.340 | n/a | 2220144.126 |
| HF CPU | 128 | 2 | 0.079 | 17.074 | 21981.113 | n/a | n/a |
