"""Helpers for lazily importing optional dependencies."""

from __future__ import annotations

from .exceptions import DependencyMissingError


def require_numpy():
    """Import NumPy only when a runtime path truly needs it."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "NumPy is required for runtime commands. Install it with `pip install -e .`."
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


def require_auto_tokenizer():
    """Import Hugging Face tokenizer support only when text prompts are used."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise DependencyMissingError(
            "Transformers is required for text prompts. Install it with `pip install -e .[hf]`."
        ) from exc
    return AutoTokenizer
