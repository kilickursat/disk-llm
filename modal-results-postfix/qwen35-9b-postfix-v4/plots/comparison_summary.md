# Benchmark Summary

- Manifest: `/__modal/volumes/vo-oHh6Gg8uG2RRqiwJljz00y/packed/Qwen--Qwen3.5-9B/qwen35-9b-postfix-v4/manifest.json`
- Runs per case: `1`
- Warmup runs per case: `0`

| Backend | Prompt Tokens | Max New Tokens | Mean Tokens/s | Mean First Token (s) | Mean Peak RSS (MB) | Mean IO Read (MB) | Mean Logical Mapped (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Disk-LLM | 8 | 2 | 0.018 | 88.501 | 21873.832 | n/a | 170780.317 |
| HF CPU | 8 | 2 | 0.165 | 4.773 | 19403.957 | n/a | n/a |
| Disk-LLM | 128 | 2 | 0.002 | 1157.395 | 21885.977 | n/a | 2220144.126 |
| HF CPU | 128 | 2 | 0.080 | 17.142 | 19407.145 | n/a | n/a |
