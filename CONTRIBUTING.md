# Contributing to Disk-LLM

Thanks for considering a contribution.

Disk-LLM is intentionally narrow: we care more about inspectability, reproducibility, and code readability than benchmark-chasing. If you open a PR, optimize for clarity first.

## Ground rules

- keep the runtime observable
- prefer small, reviewable changes
- document new public flags and manifest fields
- avoid hiding important behavior behind magical defaults
- keep optional dependencies optional whenever possible

## Local setup

```bash
pip install -e .[hf,demo,test]
python -m unittest discover -v
```

## What we review closely

- manifest compatibility
- converter determinism
- runtime telemetry changes
- tensor-name assumptions for model adapters
- test coverage for new CLI behavior

## Pull request checklist

- add or update tests
- update `README.md` when CLI or workflow changes
- update `docs/architecture.md` if the manifest or runtime contract changes
- keep error messages actionable

## Suggested contribution areas

- Qwen 3.5 adapter hardening
- packed-layout alternatives
- benchmark scenarios
- tokenizer integration
- cache instrumentation
- docs for new checkpoint families
