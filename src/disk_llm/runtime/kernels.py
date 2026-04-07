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


def sigmoid(values):
    np = require_numpy()
    values = np.asarray(values)
    return 1.0 / (1.0 + np.exp(-values))


def softplus(values):
    np = require_numpy()
    values = np.asarray(values)
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)


def _ensure_broadcastable(values, weight):
    np = require_numpy()
    weight = np.asarray(weight)
    if weight.shape != values.shape and weight.size == values.size:
        return weight.reshape(values.shape)
    return weight


def rms_norm(hidden, weight, *, eps: float):
    np = require_numpy()
    hidden = np.asarray(hidden)
    weight = _ensure_broadcastable(hidden, weight)
    variance = np.mean(hidden * hidden, axis=-1, keepdims=True)
    return (hidden / np.sqrt(variance + eps)) * weight


def apply_rope_single(values, *, position: int, theta: float, rotary_fraction: float = 1.0):
    np = require_numpy()
    values = np.asarray(values)
    head_dim = values.shape[-1]
    rotary_dim = int(head_dim * rotary_fraction)
    if rotary_dim <= 0:
        return values
    if rotary_dim > head_dim:
        rotary_dim = head_dim
    if rotary_dim % 2 != 0:
        rotary_dim -= 1
    if rotary_dim <= 0:
        return values
    if rotary_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2) / rotary_dim))
    angles = position * inv_freq
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=0)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=0)
    rotary_values = values[..., :rotary_dim]
    pass_values = values[..., rotary_dim:]
    first_half = rotary_values[..., : rotary_dim // 2]
    second_half = rotary_values[..., rotary_dim // 2 :]
    rotated = np.concatenate(
        [
            (first_half * cos[: rotary_dim // 2]) - (second_half * sin[: rotary_dim // 2]),
            (second_half * cos[rotary_dim // 2 :]) + (first_half * sin[rotary_dim // 2 :]),
        ],
        axis=-1,
    )
    output = np.empty_like(values)
    output[..., :rotary_dim] = rotated
    output[..., rotary_dim:] = pass_values
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


def l2norm(values, *, axis: int = -1, eps: float = 1e-6):
    np = require_numpy()
    values = np.asarray(values)
    inv_norm = 1.0 / np.sqrt(np.sum(values * values, axis=axis, keepdims=True) + eps)
    return values * inv_norm


def qwen3next_rms_norm(values, weight, *, eps: float):
    np = require_numpy()
    values = np.asarray(values)
    weight = _ensure_broadcastable(values, weight)
    variance = np.mean(values * values, axis=-1, keepdims=True)
    normalized = values / np.sqrt(variance + eps)
    return normalized * (1.0 + weight)


def rms_norm_gated(values, weight, gate, *, eps: float):
    np = require_numpy()
    values = np.asarray(values)
    weight = _ensure_broadcastable(values, weight)
    gate = _ensure_broadcastable(values, gate)
    variance = np.mean(values * values, axis=-1, keepdims=True)
    normalized = values / np.sqrt(variance + eps)
    return weight * normalized * silu(gate)


def depthwise_causal_conv1d_update(hidden_states, state, weight, *, activation: str = "silu"):
    np = require_numpy()
    hidden_states = np.asarray(hidden_states)
    kernels = np.asarray(weight)
    if kernels.ndim == 3:
        kernels = kernels[:, 0, :]
    kernel_size = kernels.shape[-1]
    state_width = max(kernel_size - 1, 0)
    if state is None:
        state = np.zeros((hidden_states.shape[0], state_width), dtype=hidden_states.dtype)
    if state_width > 0:
        window = np.concatenate([state, hidden_states[:, None]], axis=-1)
        new_state = window[:, -state_width:]
    else:
        window = hidden_states[:, None]
        new_state = np.zeros((hidden_states.shape[0], 0), dtype=hidden_states.dtype)
    output = np.sum(window * kernels, axis=-1)
    if activation == "silu":
        output = silu(output)
    return output, new_state


def recurrent_gated_delta_step(query, key, value, g, beta, state, *, use_qk_l2norm: bool = True):
    np = require_numpy()
    query = np.asarray(query)
    key = np.asarray(key)
    value = np.asarray(value)
    if use_qk_l2norm:
        query = l2norm(query, axis=-1, eps=1e-6)
        key = l2norm(key, axis=-1, eps=1e-6)
    scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * scale

    query_f32 = query.astype(np.float32, copy=False)
    key_f32 = key.astype(np.float32, copy=False)
    value_f32 = value.astype(np.float32, copy=False)
    g_f32 = np.exp(np.asarray(g, dtype=np.float32))
    beta_f32 = np.asarray(beta, dtype=np.float32)

    # State is (H, D_k, D_v). 
    # g applies to each state element.
    # beta applies to the kv update.
    if g_f32.ndim == 1:
        g_f32 = g_f32[:, None, None]
    elif g_f32.ndim == 2:
        g_f32 = g_f32[:, :, None]

    if beta_f32.ndim == 1:
        beta_f32 = beta_f32[:, None]
    # If 2D (H, D_v), it stays (H, D_v) which matches (value - memory) -> (H, D_v)

    if state is None:
        state = np.zeros(
            (query.shape[0], query.shape[-1], value.shape[-1]),
            dtype=np.float32,
        )
    else:
        state = np.asarray(state, dtype=np.float32)

    state = state * g_f32
    kv_memory = np.sum(state * key_f32[:, :, None], axis=-2)
    delta = (value_f32 - kv_memory) * beta_f32
    state = state + key_f32[:, :, None] * delta[:, None, :]
    output = np.sum(state * query_f32[:, :, None], axis=-2).astype(value.dtype, copy=False)
    return output, state
