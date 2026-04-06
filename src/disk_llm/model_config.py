"""Helpers for extracting the text-runtime config from source or packed models."""

from __future__ import annotations

from typing import Any


def normalized_text_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten nested text_config values onto the top level when present."""
    if not isinstance(config, dict):
        return {}
    normalized = dict(config)
    nested = config.get("text_config")
    if isinstance(nested, dict):
        normalized.setdefault("parent_model_type", normalized.get("model_type"))
        normalized.update(nested)
        normalized.setdefault("text_model_type", nested.get("model_type"))
    return normalized


def nested_config_value(config: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Read a value from a config, preferring nested text_config when available."""
    normalized = normalized_text_config(config)
    return normalized.get(key, default)
