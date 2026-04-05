"""Packing heuristics and lightweight Qwen 3.5 layout helpers."""

from __future__ import annotations

import re
from typing import Any

LAYER_PATTERN = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.")


def is_text_tensor(name: str) -> bool:
    """Return True when a tensor belongs to the text-only path."""
    text_prefixes = (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
        "model.language_model.embed_tokens.",
        "model.language_model.layers.",
        "model.language_model.norm.",
        "lm_head.",
    )
    return name.startswith(text_prefixes)


def classify_tensor_group(name: str) -> str:
    """Map a source tensor name to a packed shard path."""
    if name.startswith("model.embed_tokens.") or name.startswith("model.language_model.embed_tokens."):
        return "embeddings/embeddings.bin"
    if name.startswith("model.norm.") or name.startswith("model.language_model.norm.") or name.startswith("lm_head."):
        return "final/final.bin"
    match = LAYER_PATTERN.match(name)
    if match:
        layer_id = int(match.group(1))
        return f"layers/layer_{layer_id:03d}.bin"
    return "misc/misc.bin"


def default_block_kinds(num_hidden_layers: int) -> list[str]:
    """Use the published repeating Qwen 3.5 text pattern when nothing more precise is known."""
    if num_hidden_layers <= 0:
        return []
    pattern = ("delta", "delta", "delta", "attention")
    return [pattern[index % len(pattern)] for index in range(num_hidden_layers)]


def derive_block_kinds(config: dict[str, Any]) -> list[str]:
    """Derive hybrid block kinds from config when possible, otherwise use the v1 default."""
    for key in ("block_kinds", "layer_types", "hybrid_block_types"):
        value = config.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [item.lower() for item in value]
    num_hidden_layers = int(config.get("num_hidden_layers", 0))
    return default_block_kinds(num_hidden_layers)


def build_pack_plan(
    tensor_names: list[str],
    *,
    text_only: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Build a tensor-to-shard mapping and a list of skipped tensors."""
    kept: dict[str, str] = {}
    skipped: list[str] = []
    for name in sorted(tensor_names):
        if text_only and not is_text_tensor(name):
            skipped.append(name)
            continue
        kept[name] = classify_tensor_group(name)
    return kept, skipped
