The main runbook is modal_remote_run.md. The helper one-command launchers are run_modal_qwen35_9b.sh and run_modal_qwen35_9b.ps1. The remote workflow itself is in modal_qwen_benchmark.py.

Exact later-run command saved in the project:

Linux/macOS: bash scripts/run_modal_qwen35_9b.sh
Windows PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_qwen35_9b.ps1
I also updated .gitignore so .env and .env.* stay out of Git. No .env or token content was committed.

Pushed to origin/main:

Commit: 169bbe4
Message: Add Modal remote benchmark runner and runbook
When you run this on the other machine, the result artifacts are designed to end up in the Modal Volume path documented in modal_remote_run.md, and then we can pull them back and package the final benchmark results into the repo.
