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




-------------------------------------------
Updated notes

Good question — let me be direct about what you'll get vs. what needs to be built.
What you get today (out of the box)
Running the parity test gives you pass/fail assertions only — no numbers printed, no charts, no comparison tables. The bench CLI command emits raw telemetry JSON to stdout. That's it. Nothing is automatically saved, visualized, or compared against a baseline.
What's actually missing for a useful research result
There are four layers of missing work:

Benchmark harness — right now disk-llm bench runs once and prints JSON. You need repeated runs (warm/cold cache), controlled prompt lengths (8, 64, 256, 512 tokens), and controlled max_new_tokens values to get statistically meaningful numbers. None of that scaffolding exists.
A reference to compare against — disk-llm has no opponent. You need to run the same prompts through HF AutoModelForCausalLM (float32, CPU) and record its tokens/sec and memory footprint side-by-side. Otherwise the disk-llm numbers mean nothing on their own.
Memory / IO instrumentation — the whole point of the memmap approach is that it pages tensors from disk on demand rather than loading the full model into RAM. But you're not currently measuring RSS (resident set size), page fault counts, or actual bytes read from disk vs. bytes in RAM at any moment. Without that, you can't demonstrate the core claim.
Visualization — nothing generates plots. You'd need to produce at minimum: a tokens/sec bar chart (disk-llm vs. HF baseline), a RAM usage timeline, and a first-token-latency vs. prompt-length curve.

Concrete next steps in order
The logical sequence is:
First, run disk-llm inspect --source-dir /path/to/Qwen3.5-9B --json to validate that the tensor names in the real checkpoint match the adapter's expectations. This will expose any gaps before you commit time to a full conversion.
Second, run disk-llm convert /path/to/Qwen3.5-9B /output/qwen35-packed and confirm the manifest looks sane — correct block_kinds, no unexpected skipped tensors, reasonable shard sizes.
Third, run the parity tests. If test_argmax_token_matches fails but correlation is above 0.5, the model is computing something reasonable but the delta-block path (currently a stub falling back to attention) is producing drift. That's your first real research finding to document.
Fourth — and this is the work that actually needs to be built — write a scripts/benchmark.py that runs both disk-llm and HF with identical prompts, captures tokens/sec, first_token_latency_s, and rss_mb (via psutil), and saves results to a CSV. Then a separate scripts/plot_results.py that reads that CSV and produces publication-quality figures using matplotlib.
Fifth, document the delta-block tensor names once you've seen the real checkpoint, replace the fallback stub in _delta_block(), and re-run parity.
The honest picture
The research value of disk-llm is specifically the claim: "you can run inference on a model that doesn't fit in RAM by streaming weight tensors from disk via memmap, with measurable throughput and a quantified RAM ceiling." Right now the machinery to make that claim exists but the measurement infrastructure to prove it doesn't. The benchmarks, the RAM timeline, and the HF comparison are what turn this from a working prototype into a result you can put in a paper or README. That's the real next milestone.
Want me to build scripts/benchmark.py and scripts/plot_results.py now so everything is ready to run the moment you have the checkpoint?

-------------------------------------------------------------------------------
