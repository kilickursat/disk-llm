Right now

Install the optional stack so the skipped runtime tests can run: pip install -e .[hf,demo,test]
Download a real Qwen/Qwen3.5-9B snapshot locally and run:
disk-llm inspect --source-dir ...
disk-llm convert ...
Use that real snapshot to tighten the tensor-name adapter in model.py and confirm the block layout assumptions from layout.py

After that

Add parity checks against a reference backend so one prompt produces comparable logits/token choices
Harden the converter/manifest flow around real Qwen configs in converter.py
Run real benchmarks and document findings in README.md
If you want a public project site, publish site/index.html with GitHub Pages
