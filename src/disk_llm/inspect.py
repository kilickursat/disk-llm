"""Model inspection helpers for source snapshots and packed manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .layout import build_pack_plan, derive_block_kinds
from .manifest import PackedModelManifest, validate_manifest_files
from .safetensors_io import discover_safetensors_files, format_bytes, read_config_json, read_safetensors_header


def inspect_source_dir(source_dir: str | Path, *, text_only: bool = True) -> dict[str, Any]:
    """Inspect a source model snapshot before conversion."""
    source_path = Path(source_dir)
    config = read_config_json(source_path)
    files = discover_safetensors_files(source_path)
    tensors: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for file_path in files:
        parsed = read_safetensors_header(file_path)
        total_bytes += parsed.total_tensor_bytes
        for name, tensor in parsed.tensors.items():
            tensors[name] = {
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "nbytes": tensor.nbytes,
                "source_file": file_path.name,
            }

    kept, skipped = build_pack_plan(list(tensors), text_only=text_only)
    return {
        "kind": "source",
        "source_dir": str(source_path.resolve()),
        "family": config.get("model_type", "unknown"),
        "architectures": config.get("architectures", []),
        "tensor_count": len(tensors),
        "kept_tensor_count": len(kept),
        "skipped_tensor_count": len(skipped),
        "total_tensor_bytes": total_bytes,
        "total_tensor_bytes_human": format_bytes(total_bytes),
        "num_hidden_layers": int(config.get("num_hidden_layers", 0)),
        "block_kinds": derive_block_kinds(config),
        "files": [path.name for path in files],
        "config": config,
        "tensors": tensors,
        "skipped_tensors": skipped,
    }


def inspect_packed_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Inspect a packed Disk-LLM manifest."""
    manifest_file = Path(manifest_path)
    manifest = PackedModelManifest.from_path(manifest_file)
    validation_errors = validate_manifest_files(manifest, base_dir=manifest_file.parent)
    return {
        "kind": "packed",
        "manifest_path": str(manifest_file.resolve()),
        "family": manifest.family,
        "variant": manifest.variant,
        "text_only": manifest.text_only,
        "tensor_count": len(manifest.tensors),
        "shard_count": len(manifest.shards),
        "total_bytes": manifest.total_bytes(),
        "total_bytes_human": format_bytes(manifest.total_bytes()),
        "layer_ids": manifest.layer_ids(),
        "skipped_tensor_count": len(manifest.skipped_tensors),
        "layout_strategy": manifest.layout_strategy,
        "block_kinds": manifest.config.get("block_kinds", []),
        "validation_errors": validation_errors,
    }


def render_inspection(summary: dict[str, Any]) -> str:
    """Render an inspection summary for the CLI."""
    if summary["kind"] == "source":
        lines = [
            f"Source directory: {summary['source_dir']}",
            f"Model family: {summary['family']}",
            f"Architectures: {', '.join(summary['architectures']) or 'unknown'}",
            f"Tensor files: {len(summary['files'])}",
            f"Tensors discovered: {summary['tensor_count']}",
            f"Tensors kept: {summary['kept_tensor_count']}",
            f"Tensors skipped: {summary['skipped_tensor_count']}",
            f"Total bytes: {summary['total_tensor_bytes_human']}",
        ]
        if summary["block_kinds"]:
            preview = ", ".join(summary["block_kinds"][:8])
            lines.append(f"Block kinds preview: {preview}")
        return "\n".join(lines)

    lines = [
        f"Manifest: {summary['manifest_path']}",
        f"Family: {summary['family']} ({summary['variant']})",
        f"Text-only: {summary['text_only']}",
        f"Packed tensors: {summary['tensor_count']}",
        f"Shards: {summary['shard_count']}",
        f"Total bytes: {summary['total_bytes_human']}",
        f"Layers: {len(summary['layer_ids'])}",
    ]
    if summary["validation_errors"]:
        lines.append(f"Validation errors: {len(summary['validation_errors'])}")
    else:
        lines.append("Validation errors: 0")
    return "\n".join(lines)
