# Benchmark Summary

- Manifest: `/__modal/volumes/vo-oHh6Gg8uG2RRqiwJljz00y/packed/Qwen--Qwen3.5-9B/qwen35-9b-modal-cpu-postfix/manifest.json`
- Runs per case: `1`
- Warmup runs per case: `0`

| Backend | Prompt Tokens | Max New Tokens | Mean Tokens/s | Mean First Token (s) | Mean Peak RSS (MB) | Mean IO Read (MB) | Mean Logical Mapped (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Disk-LLM | 8 | 2 | 0.153 | 10.388 | 11266.043 | n/a | 38800.078 |
| HF CPU | 8 | 2 | 0.125 | 6.117 | 21943.293 | n/a | n/a |
| Disk-LLM | 128 | 2 | 0.012 | 158.646 | 11268.035 | n/a | 504401.016 |
| HF CPU | 128 | 2 | 0.076 | 18.310 | 21943.293 | n/a | n/a |
