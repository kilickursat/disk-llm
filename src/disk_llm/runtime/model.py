"""Experimental NumPy text runtime backed by packed memmap shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..exceptions import RuntimeShapeError
from ..manifest import PackedModelManifest
from ..optional import require_numpy
from .config import TextModelConfig
from .kernels import (
    apply_rope_single,
    attention_scale,
    gated_delta_step,
    grouped_query_attention_step,
    repeat_kv_heads,
    reshape_heads,
    rms_norm,
    sample_from_logits,
    swiglu,
)
from .memmap import MemmapTensorStore
from .telemetry import TelemetryRecorder


@dataclass
class LayerCache:
    """Per-layer generation cache."""

    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    delta_state: Any | None = None


class DiskLLMTextModel:
    """Experimental text-only runtime for packed Disk-LLM models."""

    def __init__(
        self,
        manifest: PackedModelManifest,
        *,
        manifest_path: str | Path,
    ):
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.store = MemmapTensorStore(manifest, base_dir=self.manifest_path.parent)
        self.config = TextModelConfig.from_manifest(manifest)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "DiskLLMTextModel":
        manifest = PackedModelManifest.from_path(manifest_path)
        return cls(manifest, manifest_path=manifest_path)

    def empty_cache(self) -> list[LayerCache]:
        return [LayerCache() for _ in range(self.config.num_hidden_layers)]

    def generate_token_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int | None = None,
        telemetry: TelemetryRecorder | None = None,
    ) -> tuple[list[int], dict[str, object]]:
        np = require_numpy()
        if not prompt_ids:
            raise RuntimeShapeError("prompt_ids must contain at least one token.")

        telemetry = telemetry or TelemetryRecorder(prompt_tokens=len(prompt_ids))
        rng = np.random.default_rng(seed)
        cache = self.empty_cache()

        logits = None
        for position, token_id in enumerate(prompt_ids):
            logits = self.forward_step(token_id, position=position, cache=cache, telemetry=telemetry)
        if logits is None:
            raise RuntimeShapeError("Prompt processing did not produce logits.")

        generated: list[int] = []
        for offset in range(max_new_tokens):
            token_id = sample_from_logits(
                logits,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
            )
            if offset == 0:
                telemetry.mark_first_token()
            telemetry.record_generated_token()
            generated.append(token_id)
            if self.config.eos_token_id is not None and token_id == self.config.eos_token_id:
                break
            logits = self.forward_step(
                token_id,
                position=len(prompt_ids) + offset,
                cache=cache,
                telemetry=telemetry,
            )
        return generated, telemetry.summary()

    def stream_generate_token_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> Iterable[dict[str, object]]:
        telemetry = TelemetryRecorder(prompt_tokens=len(prompt_ids))
        generated, _ = self.generate_token_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            telemetry=telemetry,
        )
        for index, token_id in enumerate(generated):
            yield {
                "event": "token",
                "token_id": token_id,
                "index": index,
                "telemetry": telemetry.summary(),
            }

    def forward_step(
        self,
        token_id: int,
        *,
        position: int,
        cache: list[LayerCache],
        telemetry: TelemetryRecorder,
    ):
        np = require_numpy()
        embed_weight = self._get_tensor(
            [
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
            ],
            telemetry=telemetry,
        )
        hidden = np.asarray(embed_weight[int(token_id)])

        import threading
        for layer_idx in range(self.config.num_hidden_layers):
            layer_name = f"layer_{layer_idx:03d}"
            
            # Spawn OS page fault scout for the next layer in the background
            if layer_idx + 1 < self.config.num_hidden_layers:
                threading.Thread(target=self._prefetch_layer, args=(layer_idx + 1,), daemon=True).start()
                
            with telemetry.time_layer(layer_name):
                hidden = self._forward_layer(layer_idx, hidden, position=position, cache=cache[layer_idx], telemetry=telemetry)

        norm_weight = self._get_tensor(
            [
                "model.norm.weight",
                "model.language_model.norm.weight",
                "model.final_layernorm.weight",
            ],
            telemetry=telemetry,
        )
        hidden = rms_norm(hidden, norm_weight, eps=self.config.rms_norm_eps)

        if self.store.has("lm_head.weight"):
            lm_head = self.store.get("lm_head.weight", telemetry)
        else:
            lm_head = self._get_tensor(
                [
                    "model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                ],
                telemetry=telemetry,
            )
        return np.dot(hidden, lm_head.T)

    def _forward_layer(
        self,
        layer_idx: int,
        hidden,
        *,
        position: int,
        cache: LayerCache,
        telemetry: TelemetryRecorder,
    ):
        input_norm = self._get_tensor(
            [
                f"model.layers.{layer_idx}.input_layernorm.weight",
                f"model.language_model.layers.{layer_idx}.input_layernorm.weight",
                f"model.layers.{layer_idx}.pre_attention_layernorm.weight",
                f"model.layers.{layer_idx}.attention_norm.weight",
            ],
            telemetry=telemetry,
        )
        normed = rms_norm(hidden, input_norm, eps=self.config.rms_norm_eps)

        block_kind = self.config.block_kind(layer_idx)
        if block_kind == "delta":
            block_output = self._delta_step(layer_idx, normed, cache=cache, telemetry=telemetry)
        else:
            block_output = self._attention_step(
                layer_idx,
                normed,
                position=position,
                cache=cache,
                telemetry=telemetry,
            )
        hidden = hidden + block_output

        ffn_norm = self._get_tensor(
            [
                f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight",
                f"model.layers.{layer_idx}.pre_ff_layernorm.weight",
                f"model.layers.{layer_idx}.ffn_norm.weight",
            ],
            telemetry=telemetry,
        )
        normed_ffn = rms_norm(hidden, ffn_norm, eps=self.config.rms_norm_eps)
        hidden = hidden + self._mlp_step(layer_idx, normed_ffn, telemetry=telemetry)
        return hidden

    def _attention_step(self, layer_idx: int, hidden, *, position: int, cache: LayerCache, telemetry: TelemetryRecorder):
        np = require_numpy()
        q_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight",
                f"model.layers.{layer_idx}.attn.q_proj.weight",
                f"model.layers.{layer_idx}.attention.q_proj.weight",
            ],
            telemetry=telemetry,
        )
        k_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight",
                f"model.layers.{layer_idx}.attn.k_proj.weight",
                f"model.layers.{layer_idx}.attention.k_proj.weight",
            ],
            telemetry=telemetry,
        )
        v_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight",
                f"model.layers.{layer_idx}.attn.v_proj.weight",
                f"model.layers.{layer_idx}.attention.v_proj.weight",
            ],
            telemetry=telemetry,
        )
        o_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight",
                f"model.layers.{layer_idx}.attn.o_proj.weight",
                f"model.layers.{layer_idx}.attention.o_proj.weight",
            ],
            telemetry=telemetry,
        )

        query = np.dot(hidden, q_proj.T)
        key = np.dot(hidden, k_proj.T)
        value = np.dot(hidden, v_proj.T)

        query = reshape_heads(query, num_heads=self.config.num_attention_heads)
        kv_heads = max(self.config.num_key_value_heads, 1)
        key = reshape_heads(key, num_heads=kv_heads)
        value = reshape_heads(value, num_heads=kv_heads)
        head_dim = query.shape[-1]

        query = apply_rope_single(query, position=position, theta=self.config.rope_theta)
        key = apply_rope_single(key, position=position, theta=self.config.rope_theta)

        key = repeat_kv_heads(key, target_heads=self.config.num_attention_heads)
        value = repeat_kv_heads(value, target_heads=self.config.num_attention_heads)

        cache.keys.append(np.asarray(key))
        cache.values.append(np.asarray(value))

        key_history = np.stack(cache.keys, axis=0)
        value_history = np.stack(cache.values, axis=0)
        context = grouped_query_attention_step(
            query,
            key_history,
            value_history,
            scale=attention_scale(head_dim),
        )
        return np.dot(context.reshape(-1), o_proj.T)

    def _delta_step(self, layer_idx: int, hidden, *, cache: LayerCache, telemetry: TelemetryRecorder):
        np = require_numpy()
        q_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.delta_net.q_proj.weight",
                f"model.layers.{layer_idx}.delta.q_proj.weight",
            ],
            telemetry=telemetry,
        )
        k_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.delta_net.k_proj.weight",
                f"model.layers.{layer_idx}.delta.k_proj.weight",
            ],
            telemetry=telemetry,
        )
        v_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.delta_net.v_proj.weight",
                f"model.layers.{layer_idx}.delta.v_proj.weight",
            ],
            telemetry=telemetry,
        )
        a_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.delta_net.a_proj.weight",
                f"model.layers.{layer_idx}.delta.a_proj.weight",
                f"model.layers.{layer_idx}.delta_net.alpha_proj.weight",
            ],
            telemetry=telemetry,
        )
        b_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.delta_net.b_proj.weight",
                f"model.layers.{layer_idx}.delta.b_proj.weight",
                f"model.layers.{layer_idx}.delta_net.beta_proj.weight",
                f"model.layers.{layer_idx}.delta_net.dt_proj.weight",
            ],
            telemetry=telemetry,
        )
        o_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.delta_net.o_proj.weight",
                f"model.layers.{layer_idx}.delta.o_proj.weight",
            ],
            telemetry=telemetry,
        )

        num_heads = self.config.delta_num_heads or self.config.num_attention_heads
        query = reshape_heads(np.dot(hidden, q_proj.T), num_heads=num_heads)
        key = reshape_heads(np.dot(hidden, k_proj.T), num_heads=num_heads)
        value = reshape_heads(np.dot(hidden, v_proj.T), num_heads=num_heads)
        alpha = reshape_heads(np.dot(hidden, a_proj.T), num_heads=num_heads)
        beta = reshape_heads(np.dot(hidden, b_proj.T), num_heads=num_heads)

        alpha = 1.0 / (1.0 + np.exp(-alpha))
        beta = 1.0 / (1.0 + np.exp(-beta))

        output, cache.delta_state = gated_delta_step(
            query,
            key,
            value,
            alpha,
            beta,
            cache.delta_state,
        )
        return np.dot(output.reshape(-1), o_proj.T)

    def _mlp_step(self, layer_idx: int, hidden, *, telemetry: TelemetryRecorder):
        gate = self._get_tensor(
            [
                f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight",
                f"model.layers.{layer_idx}.feed_forward.gate_proj.weight",
            ],
            telemetry=telemetry,
        )
        up = self._get_tensor(
            [
                f"model.layers.{layer_idx}.mlp.up_proj.weight",
                f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight",
                f"model.layers.{layer_idx}.feed_forward.up_proj.weight",
            ],
            telemetry=telemetry,
        )
        down = self._get_tensor(
            [
                f"model.layers.{layer_idx}.mlp.down_proj.weight",
                f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight",
                f"model.layers.{layer_idx}.feed_forward.down_proj.weight",
            ],
            telemetry=telemetry,
        )
        return swiglu(hidden, gate, up, down)

    def _get_tensor(self, candidates: Sequence[str], *, telemetry: TelemetryRecorder):
        for name in candidates:
            if self.store.has(name):
                return self.store.get(name, telemetry)
        raise RuntimeShapeError(
            "Could not resolve any tensor name from candidates: " + ", ".join(candidates)
        )

    def _prefetch_layer(self, layer_idx: int):
        """
        Runs in a background thread. Touches every 4KB page of the target 
        layer's memmap arrays natively in C-space to trick the OS into 
        fetching it directly from the NVMe SSD into system cache before 
        the main thread actually needs it.
        """
        prefix1 = f"model.layers.{layer_idx}."
        prefix2 = f"model.language_model.layers.{layer_idx}."
        
        for name, mmap_tensor in self.store._cache.items():
            if name.startswith(prefix1) or name.startswith(prefix2):
                # .view("uint8") provides byte-level math
                # [::4096] skips exactly 1 OS page per iter
                # .sum() runs entirely in numpy C-backend, releasing the Python GIL
                _ = mmap_tensor.view("uint8")[::4096].sum()
