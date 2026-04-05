"""Helpers for lazily importing optional dependencies."""

from __future__ import annotations

from .exceptions import DependencyMissingError


def require_numpy():
    """Import NumPy only when a runtime path truly needs it."""
    try:
        import numpy as np
        import ml_dtypes  # Registers bfloat16 and float8 datatypes with numpy
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "NumPy and ml-dtypes are required for runtime commands. Install them with `pip install -e .`."
        ) from exc
    return np


def require_gradio():
    """Import Gradio only for the demo command."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "Gradio is required for the demo UI. Install it with `pip install -e .[demo]`."
        ) from exc
    return gr


def require_psutil():
    """Import psutil only when benchmark telemetry needs process metrics."""
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "psutil is required for benchmark RSS and IO sampling. Install it with `pip install -e .[bench]`."
        ) from exc
    return psutil


def require_matplotlib_pyplot():
    """Import matplotlib only for offline plot generation."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "matplotlib is required for plot generation. Install it with `pip install -e .[bench]`."
        ) from exc
    return plt


def require_auto_tokenizer():
    """Import Hugging Face tokenizer support only when text prompts are used."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "Transformers is required for text prompts. Install it with `pip install -e .[hf]`."
        ) from exc
    return AutoTokenizer


def require_auto_model_for_causal_lm():
    """Import Hugging Face causal language-model support on demand."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "Transformers is required for the Hugging Face baseline. Install it with `pip install -e .[hf]`."
        ) from exc
    return AutoModelForCausalLM
