"""Experimental NumPy text runtime backed by packed memmap shards."""

from __future__ import annotations

import threading
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
    depthwise_causal_conv1d_update,
    gated_delta_step,
    grouped_query_attention_step,
    l2norm,
    qwen3next_rms_norm,
    recurrent_gated_delta_step,
    repeat_kv_heads,
    rms_norm_gated,
    reshape_heads,
    rms_norm,
    sample_from_logits,
    sigmoid,
    softplus,
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
    linear_conv_state: Any | None = None
    linear_recurrent_state: Any | None = None


class DiskLLMTextModel:
    """Experimental text-only runtime for packed Disk-LLM models."""

    _EMBED_TENSOR_CANDIDATES = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    )
    _FINAL_NORM_TENSOR_CANDIDATES = (
        "model.norm.weight",
        "model.language_model.norm.weight",
        "model.final_layernorm.weight",
    )
    _LM_HEAD_TENSOR_CANDIDATES = ("lm_head.weight",) + _EMBED_TENSOR_CANDIDATES

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
        self._resolved_tensor_names: dict[tuple[str, ...], str] = {}
        self._optional_tensor_names: dict[tuple[str, ...], str | None] = {}

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
            self._EMBED_TENSOR_CANDIDATES,
            telemetry=telemetry,
        )
        hidden = np.asarray(embed_weight[int(token_id)])

        enable_prefetch = self.config.enable_prefetch
        num_hidden_layers = self.config.num_hidden_layers
        for layer_idx in range(num_hidden_layers):
            layer_name = f"layer_{layer_idx:03d}"

            # Spawn OS page fault scout for the next layer in the background.
            if enable_prefetch and layer_idx + 1 < num_hidden_layers:
                threading.Thread(
                    target=self._prefetch_layer,
                    args=(layer_idx + 1,),
                    daemon=True,
                ).start()

            with telemetry.time_layer(layer_name):
                hidden = self._forward_layer(
                    layer_idx,
                    hidden,
                    position=position,
                    cache=cache[layer_idx],
                    telemetry=telemetry,
                )

        norm_weight = self._get_tensor(
            self._FINAL_NORM_TENSOR_CANDIDATES,
            telemetry=telemetry,
        )
        hidden = self._apply_hidden_norm(hidden, norm_weight)

        lm_head = self._get_tensor(
            self._LM_HEAD_TENSOR_CANDIDATES,
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
        normed = self._apply_hidden_norm(hidden, input_norm)

        block_kind = self.config.block_kind(layer_idx)
        if block_kind == "delta":
            block_output = self._delta_step(layer_idx, normed, cache=cache, telemetry=telemetry)
        elif block_kind == "linear_attention":
            block_output = self._linear_attention_step(
                layer_idx,
                normed,
                cache=cache,
                telemetry=telemetry,
            )
        elif block_kind == "attention":
            block_output = self._attention_step(
                layer_idx,
                normed,
                position=position,
                cache=cache,
                telemetry=telemetry,
            )
        else:
            raise RuntimeShapeError(
                f"Unsupported block kind {block_kind!r} for layer {layer_idx}."
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
        normed_ffn = self._apply_hidden_norm(hidden, ffn_norm)
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
        q_norm_weight = self._maybe_get_tensor(
            [
                f"model.layers.{layer_idx}.self_attn.q_norm.weight",
                f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight",
            ],
            telemetry=telemetry,
        )
        k_norm_weight = self._maybe_get_tensor(
            [
                f"model.layers.{layer_idx}.self_attn.k_norm.weight",
                f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight",
            ],
            telemetry=telemetry,
        )

        attention_head_dim = self.config.attention_head_dim or self.config.head_dim
        if attention_head_dim is None:
            attention_head_dim = max(q_proj.shape[0] // max(self.config.num_attention_heads, 1), 1)
        q_proj_out = np.dot(hidden, q_proj.T)
        expected_q_with_gate = self.config.num_attention_heads * attention_head_dim * 2
        expected_q = self.config.num_attention_heads * attention_head_dim
        if q_proj_out.shape[-1] == expected_q_with_gate:
            query_flat, gate_flat = np.split(q_proj_out, 2)
            query = query_flat.reshape(self.config.num_attention_heads, attention_head_dim)
            gate = gate_flat.reshape(self.config.num_attention_heads, attention_head_dim)
        elif q_proj_out.shape[-1] == expected_q:
            query = q_proj_out.reshape(self.config.num_attention_heads, attention_head_dim)
            gate = None
        else:
            raise RuntimeShapeError(
                f"Unexpected q_proj output size {q_proj_out.shape[-1]} for layer {layer_idx}. "
                f"Expected {expected_q} or {expected_q_with_gate}."
            )

        key = np.dot(hidden, k_proj.T)
        value = np.dot(hidden, v_proj.T)

        kv_heads = max(self.config.num_key_value_heads, 1)
        expected_kv = kv_heads * attention_head_dim
        if key.shape[-1] != expected_kv or value.shape[-1] != expected_kv:
            raise RuntimeShapeError(
                f"Unexpected KV projection size for layer {layer_idx}. "
                f"Expected {expected_kv}, found key={key.shape[-1]} value={value.shape[-1]}."
            )

        key = key.reshape(kv_heads, attention_head_dim)
        value = value.reshape(kv_heads, attention_head_dim)

        if q_norm_weight is not None:
            query = qwen3next_rms_norm(query, q_norm_weight, eps=self.config.rms_norm_eps)
        if k_norm_weight is not None:
            key = qwen3next_rms_norm(key, k_norm_weight, eps=self.config.rms_norm_eps)

        head_dim = query.shape[-1]

        query = apply_rope_single(
            query,
            position=position,
            theta=self.config.rope_theta,
            rotary_fraction=self.config.partial_rotary_factor,
        )
        key = apply_rope_single(
            key,
            position=position,
            theta=self.config.rope_theta,
            rotary_fraction=self.config.partial_rotary_factor,
        )

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
        output = context.reshape(-1)
        if gate is not None:
            output = output * sigmoid(gate).reshape(-1)
        return np.dot(output, o_proj.T)

    def _linear_attention_step(self, layer_idx: int, hidden, *, cache: LayerCache, telemetry: TelemetryRecorder):
        np = require_numpy()
        num_k_heads = self.config.linear_num_key_heads or self.config.num_attention_heads
        num_v_heads = self.config.linear_num_value_heads or num_k_heads
        head_k_dim = self.config.linear_key_head_dim or self.config.attention_head_dim
        head_v_dim = self.config.linear_value_head_dim or head_k_dim
        if head_k_dim is None or head_v_dim is None:
            raise RuntimeShapeError(
                f"Missing linear attention head dimensions for layer {layer_idx}."
            )

        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim

        qkv_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight",
            ],
            telemetry=telemetry,
        )
        z_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.in_proj_z.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight",
            ],
            telemetry=telemetry,
        )
        a_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.in_proj_a.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a.weight",
            ],
            telemetry=telemetry,
        )
        b_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.in_proj_b.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b.weight",
            ],
            telemetry=telemetry,
        )
        conv_weight = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.conv1d.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.conv1d.weight",
            ],
            telemetry=telemetry,
        )
        dt_bias = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.dt_bias",
                f"model.language_model.layers.{layer_idx}.linear_attn.dt_bias",
            ],
            telemetry=telemetry,
        )
        a_log = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.A_log",
                f"model.language_model.layers.{layer_idx}.linear_attn.A_log",
            ],
            telemetry=telemetry,
        )
        norm_weight = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.norm.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.norm.weight",
            ],
            telemetry=telemetry,
        )
        out_proj = self._get_tensor(
            [
                f"model.layers.{layer_idx}.linear_attn.out_proj.weight",
                f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight",
            ],
            telemetry=telemetry,
        )

        mixed_qkv = np.dot(hidden, qkv_proj.T)
        if mixed_qkv.shape[-1] != conv_dim:
            raise RuntimeShapeError(
                f"Unexpected linear_attn qkv size {mixed_qkv.shape[-1]} for layer {layer_idx}; expected {conv_dim}."
            )
        mixed_qkv, cache.linear_conv_state = depthwise_causal_conv1d_update(
            mixed_qkv,
            cache.linear_conv_state,
            conv_weight,
            activation="silu",
        )

        query = mixed_qkv[:key_dim].reshape(num_k_heads, head_k_dim)
        key = mixed_qkv[key_dim : key_dim * 2].reshape(num_k_heads, head_k_dim)
        value = mixed_qkv[key_dim * 2 :].reshape(num_v_heads, head_v_dim)
        gate = np.dot(hidden, z_proj.T).reshape(num_v_heads, head_v_dim)
        beta = 1.0 / (1.0 + np.exp(-np.dot(hidden, b_proj.T)))
        a = np.dot(hidden, a_proj.T)
        g = -np.exp(np.asarray(a_log, dtype=np.float32)) * softplus(
            np.asarray(a, dtype=np.float32) + np.asarray(dt_bias, dtype=np.float32)
        )

        # Reshape to head-structure if they are channel-wise
        if beta.size == num_v_heads * head_v_dim:
            beta = beta.reshape(num_v_heads, head_v_dim)
        if g.size == num_v_heads * head_v_dim:
            g = g.reshape(num_v_heads, head_v_dim)

        if num_v_heads % num_k_heads != 0:
            raise RuntimeShapeError(
                f"linear_attention head ratio mismatch in layer {layer_idx}: "
                f"{num_v_heads} value heads vs {num_k_heads} key heads."
            )
        repeat_factor = num_v_heads // num_k_heads
        if repeat_factor > 1:
            query = np.repeat(query, repeat_factor, axis=0)
            key = np.repeat(key, repeat_factor, axis=0)

        core_output, cache.linear_recurrent_state = recurrent_gated_delta_step(
            query,
            key,
            value,
            g,
            beta,
            cache.linear_recurrent_state,
            use_qk_l2norm=True,
        )
        core_output = rms_norm_gated(
            core_output,
            norm_weight,
            gate,
            eps=self.config.rms_norm_eps,
        )
        return np.dot(core_output.reshape(-1), out_proj.T)

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

    def _resolve_optional_tensor_name(self, candidates: Sequence[str]) -> str | None:
        key = tuple(candidates)
        resolved = self._resolved_tensor_names.get(key)
        if resolved is not None:
            return resolved
        if key in self._optional_tensor_names:
            return self._optional_tensor_names[key]
        for name in key:
            if self.store.has(name):
                self._resolved_tensor_names[key] = name
                self._optional_tensor_names[key] = name
                return name
        self._optional_tensor_names[key] = None
        return None

    def _resolve_tensor_name(self, candidates: Sequence[str]) -> str:
        resolved = self._resolve_optional_tensor_name(candidates)
        if resolved is not None:
            return resolved
        raise RuntimeShapeError(
            "Could not resolve any tensor name from candidates: " + ", ".join(candidates)
        )

    def _get_tensor(self, candidates: Sequence[str], *, telemetry: TelemetryRecorder):
        return self.store.get(self._resolve_tensor_name(candidates), telemetry)

    def _maybe_get_tensor(self, candidates: Sequence[str], *, telemetry: TelemetryRecorder):
        resolved = self._resolve_optional_tensor_name(candidates)
        if resolved is None:
            return None
        return self.store.get(resolved, telemetry)

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

    def _apply_hidden_norm(self, hidden, weight):
        if self.config.use_qwen3_next_norms:
            return qwen3next_rms_norm(hidden, weight, eps=self.config.rms_norm_eps)
        return rms_norm(hidden, weight, eps=self.config.rms_norm_eps)
