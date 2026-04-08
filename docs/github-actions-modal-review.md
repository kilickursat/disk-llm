# GitHub Actions Modal Runbook

This workflow is a review-first path for running the Modal benchmark from GitHub-hosted infrastructure instead of the local company network path.

## Why this version is conservative

- Manual trigger only via `workflow_dispatch`
- Minimal workflow permission: `contents: read`
- Uses a dedicated GitHub environment named `modal-benchmark`
- Reads secrets from GitHub secrets, not from `.env`
- Keeps the experimental prefetch path opt-in and separate from the baseline
- Uploads local logs and any produced result folders as workflow artifacts

## Required GitHub environment secrets

Add these three secrets to the `modal-benchmark` environment:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`

Recommended: keep them as environment secrets rather than broad repository secrets.

## Recommended first run

1. Open `Actions -> Modal Benchmark -> Run workflow`.
2. Leave the defaults as they are for the first smoke test.
3. Keep `enable_prefetch = false` for the baseline run.
4. Review the uploaded artifact bundle after the job finishes.

## Review checklist before a larger run

- Confirm `scripts/modal_qwen_benchmark.py` is still the correct entrypoint.
- Confirm the default prompt lengths and token count are appropriate for a smoke test.
- Confirm the runner should remain `ubuntu-latest`.
- Confirm the artifact directories this workflow collects still match the script outputs.
- Confirm you want `Qwen/Qwen3.5-9B` and `main` as the default benchmark target.

## What this workflow does not do

- It does not commit or use `.env`.
- It does not download Qwen to your local PC.
- It does not run automatically on every push.
- It does not treat prefetch as part of the baseline claim.

## Likely next improvement

If the first manual run succeeds, the next safe upgrade is adding a follow-up workflow or job that stages the produced CSV and figure bundle into a cleaner artifact layout for easier publication.
