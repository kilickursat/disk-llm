"""NumPy kernels for the experimental text runtime."""

from __future__ import annotations

import math
from typing import Any

from ..optional import require_numpy


def softmax(values, *, axis: int = -1):
    np = require_numpy()
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def silu(values):
    np = require_numpy()
    return values / (1.0 + np.exp(-values))


def rms_norm(hidden, weight, *, eps: float):
    np = require_numpy()
    hidden = np.asarray(hidden)
    variance = np.mean(hidden * hidden, axis=-1, keepdims=True)
    return (hidden / np.sqrt(variance + eps)) * weight


def apply_rope_single(values, *, position: int, theta: float):
    np = require_numpy()
    head_dim = values.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    angles = position * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    output = np.empty_like(values)
    output[..., 0::2] = rotated_even
    output[..., 1::2] = rotated_odd
    return output


def grouped_query_attention_step(query, key_history, value_history, *, scale: float):
    np = require_numpy()
    scores = np.einsum("hd,thd->ht", query, key_history) * scale
    weights = softmax(scores, axis=-1)
    return np.einsum("ht,thd->hd", weights, value_history)


def gated_delta_step(query, key, value, alpha, beta, state):
    np = require_numpy()
    if state is None:
        state = np.zeros_like(value)
    new_state = beta * state + alpha * key * value
    output = query * new_state
    return output, new_state


def swiglu(hidden, gate_weight, up_weight, down_weight):
    np = require_numpy()
    gate_values = np.dot(hidden, gate_weight.T)
    up_values = np.dot(hidden, up_weight.T)
    fused = silu(gate_values) * up_values
    return np.dot(fused, down_weight.T)


def sample_from_logits(
    logits,
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
    rng: Any | None = None,
) -> int:
    np = require_numpy()
    if temperature <= 0:
        return int(np.argmax(logits))

    rng = rng or np.random.default_rng()
    scaled = logits / max(temperature, 1e-6)
    probs = softmax(scaled)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    keep_mask = cumulative <= top_p
    if not np.any(keep_mask):
        keep_mask[0] = True
    filtered_indices = sorted_indices[keep_mask]
    filtered_probs = sorted_probs[keep_mask]
    filtered_probs = filtered_probs / filtered_probs.sum()
    return int(rng.choice(filtered_indices, p=filtered_probs))


def reshape_heads(vector, *, num_heads: int):
    np = require_numpy()
    if vector.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Cannot split hidden dimension {vector.shape[-1]} across {num_heads} heads."
        )
    head_dim = vector.shape[-1] // num_heads
    return np.asarray(vector).reshape(num_heads, head_dim)


def repeat_kv_heads(values, *, target_heads: int):
    np = require_numpy()
    current_heads = values.shape[0]
    if current_heads == target_heads:
        return values
    if target_heads % current_heads != 0:
        raise ValueError(
            f"Cannot repeat {current_heads} KV heads to target head count {target_heads}."
        )
    repeats = target_heads // current_heads
    return np.repeat(values, repeats, axis=0)


def attention_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
