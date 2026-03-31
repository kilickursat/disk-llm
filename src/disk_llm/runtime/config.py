"""Runtime configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..layout import derive_block_kinds
from ..manifest import PackedModelManifest


@dataclass(frozen=True)
class TextModelConfig:
    """Minimal runtime configuration for the v1 text path."""

    family: str
    variant: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    block_kinds: tuple[str, ...]
    delta_num_heads: int | None = None
    delta_head_dim: int | None = None
    attention_head_dim: int | None = None

    def block_kind(self, layer_idx: int) -> str:
        if not self.block_kinds:
            return "attention"
        return self.block_kinds[layer_idx % len(self.block_kinds)]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        family: str = "qwen3.5",
        variant: str = "9b",
    ) -> "TextModelConfig":
        block_kinds = tuple(derive_block_kinds(data))
        return cls(
            family=family,
            variant=variant,
            vocab_size=int(data.get("vocab_size", 0)),
            hidden_size=int(data.get("hidden_size", 0)),
            intermediate_size=int(data.get("intermediate_size", data.get("ffn_hidden_size", 0))),
            num_hidden_layers=int(data.get("num_hidden_layers", 0)),
            num_attention_heads=int(data.get("num_attention_heads", 1)),
            num_key_value_heads=int(data.get("num_key_value_heads", data.get("num_kv_heads", 1))),
            rms_norm_eps=float(data.get("rms_norm_eps", data.get("layer_norm_epsilon", 1e-6))),
            rope_theta=float(data.get("rope_theta", 1_000_000.0)),
            max_position_embeddings=int(data.get("max_position_embeddings", 0)),
            bos_token_id=_maybe_int(data.get("bos_token_id")),
            eos_token_id=_maybe_int(data.get("eos_token_id")),
            pad_token_id=_maybe_int(data.get("pad_token_id")),
            block_kinds=block_kinds,
            delta_num_heads=_maybe_int(data.get("delta_num_heads")),
            delta_head_dim=_maybe_int(data.get("delta_head_dim")),
            attention_head_dim=_maybe_int(data.get("attention_head_dim")),
        )

    @classmethod
    def from_manifest(cls, manifest: PackedModelManifest) -> "TextModelConfig":
        return cls.from_dict(manifest.config, family=manifest.family, variant=manifest.variant)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
