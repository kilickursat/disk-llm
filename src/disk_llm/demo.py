"""Optional Gradio demo wrapper."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from importlib.resources import files

from .optional import require_auto_tokenizer, require_gradio
from .runtime import DiskLLMTextModel

DEMO_CSS = """
:root {
  --disk-ink: #0f2d57;
  --disk-blue: #1e88d8;
  --disk-cyan: #30d8d0;
  --disk-surface: #f6fbff;
}

body {
  background:
    radial-gradient(circle at top right, rgba(48, 216, 208, 0.20), transparent 32%),
    radial-gradient(circle at left center, rgba(30, 136, 216, 0.14), transparent 28%),
    linear-gradient(180deg, #ffffff 0%, #eef7ff 100%);
}

.gradio-container {
  max-width: 1180px !important;
}

.disk-shell {
  position: relative;
  overflow: hidden;
}

.disk-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(15, 45, 87, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15, 45, 87, 0.05) 1px, transparent 1px);
  background-size: 72px 72px;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.8), transparent);
  pointer-events: none;
}

.disk-hero {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(260px, 420px);
  gap: 2rem;
  align-items: center;
  min-height: 58vh;
  padding: 2rem 0 1rem;
}

.disk-kicker {
  display: inline-block;
  margin-bottom: 0.9rem;
  padding: 0.35rem 0.8rem;
  border: 1px solid rgba(15, 45, 87, 0.12);
  border-radius: 999px;
  color: var(--disk-ink);
  background: rgba(255, 255, 255, 0.72);
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.disk-hero h1 {
  margin: 0;
  color: var(--disk-ink);
  font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
  font-size: clamp(2.6rem, 6vw, 5.4rem);
  line-height: 0.95;
  letter-spacing: -0.05em;
}

.disk-hero p {
  max-width: 34rem;
  margin: 1.15rem 0 0;
  color: rgba(15, 45, 87, 0.86);
  font-size: 1.05rem;
  line-height: 1.7;
}

.disk-hero-mark {
  display: flex;
  justify-content: center;
}

.disk-hero-mark img {
  width: min(100%, 420px);
  filter: drop-shadow(0 36px 48px rgba(20, 77, 120, 0.18));
  animation: diskFloat 5.5s ease-in-out infinite;
}

.disk-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin: 0.4rem 0 2.2rem;
}

.disk-stat {
  padding: 1rem 1.1rem;
  border-top: 1px solid rgba(15, 45, 87, 0.15);
  color: rgba(15, 45, 87, 0.82);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.66), rgba(255, 255, 255, 0.2));
}

.disk-stat strong {
  display: block;
  margin-bottom: 0.3rem;
  color: var(--disk-ink);
  font-size: 1.05rem;
}

.disk-note {
  margin: 0 0 1rem;
  color: rgba(15, 45, 87, 0.72);
  font-size: 0.92rem;
  line-height: 1.6;
}

@keyframes diskFloat {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
}

@media (max-width: 900px) {
  .disk-hero {
    grid-template-columns: 1fr;
    min-height: auto;
    padding-top: 1rem;
  }

  .disk-strip {
    grid-template-columns: 1fr;
  }
}
"""


def _logo_data_uri() -> str:
    asset = files("disk_llm").joinpath("assets/logo.png")
    encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def launch_demo(
    manifest_path: str | Path,
    *,
    tokenizer_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    gr = require_gradio()
    AutoTokenizer = require_auto_tokenizer()

    model = DiskLLMTextModel.from_manifest(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    logo_uri = _logo_data_uri()

    def generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids: list[int] = []
        telemetry_payload: dict[str, object] = {}
        for event in model.stream_generate_token_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            generated_ids.append(int(event["token_id"]))
            telemetry_payload = dict(event["telemetry"])
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            yield decoded, json.dumps(telemetry_payload, indent=2)

    with gr.Blocks(title="Disk-LLM Research Kit", css=DEMO_CSS) as app:
        gr.HTML(
            f"""
            <section class="disk-shell">
              <div class="disk-hero">
                <div>
                  <div class="disk-kicker">Disk-backed Research Kit</div>
                  <h1>Observe the model, not just the output.</h1>
                  <p>
                    Pack checkpoints into layer shards, inspect their tensor layout,
                    and stream short generations with telemetry that makes CPU-side
                    experimentation tangible.
                  </p>
                </div>
                <div class="disk-hero-mark">
                  <img src="{logo_uri}" alt="Disk-LLM logo">
                </div>
              </div>
              <div class="disk-strip">
                <div class="disk-stat">
                  <strong>Inspect</strong>
                  Read configs, count tensors, and preview the text-only subgraph before conversion.
                </div>
                <div class="disk-stat">
                  <strong>Pack</strong>
                  Re-layout weights into memmap-friendly shards with offsets and checksums.
                </div>
                <div class="disk-stat">
                  <strong>Measure</strong>
                  Watch first-token latency, layer timings, and logical bytes mapped per run.
                </div>
              </div>
            </section>
            """
        )
        gr.HTML(
            '<p class="disk-note">The Qwen 3.5 runtime path is experimental and best suited for short CPU-side research loops while the adapter is hardened against real checkpoints.</p>'
        )
        prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Ask Disk-LLM something...")
        with gr.Row():
            max_new_tokens = gr.Slider(1, 128, value=32, step=1, label="Max new tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        output = gr.Textbox(label="Generated text", lines=12)
        telemetry = gr.Code(label="Telemetry", language="json")
        prompt.submit(generate, [prompt, max_new_tokens, temperature, top_p], [output, telemetry])

    app.queue().launch(server_name=host, server_port=port)
