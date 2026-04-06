"""Source-to-packed conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .exceptions import ConversionError
from .inspect import inspect_source_dir
from .layout import build_pack_plan, derive_block_kinds
from .manifest import PackedModelManifest, ShardEntry, TensorEntry, validate_manifest_files
from .model_config import normalized_text_config
from .safetensors_io import discover_safetensors_files, read_config_json, read_safetensors_header, copy_tensor_bytes


@dataclass(frozen=True)
class ConversionResult:
    """Conversion output."""

    manifest_path: Path
    manifest: PackedModelManifest


def _align_offset(offset: int, align_bytes: int) -> int:
    if align_bytes <= 0:
        return offset
    return offset + ((-offset) % align_bytes)


def convert_model(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    family: str = "qwen3.5",
    variant: str = "9b",
    text_only: bool = True,
    align_bytes: int = 64,
    overwrite: bool = False,
    notes: list[str] | None = None,
) -> ConversionResult:
    """Convert a source safetensors snapshot into a packed Disk-LLM layout."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"

    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise ConversionError(
            f"Output directory is not empty: {output_path}. Use overwrite=True to continue."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    source_summary = inspect_source_dir(source_path, text_only=text_only)
    source_config = read_config_json(source_path)
    text_config = normalized_text_config(source_config)
    files = {
        file_path.name: read_safetensors_header(file_path)
        for file_path in discover_safetensors_files(source_path)
    }

    pack_plan, skipped_tensors = build_pack_plan(list(source_summary["tensors"]), text_only=text_only)
    if not pack_plan:
        raise ConversionError("No tensors were selected for packing.")

    handles: dict[str, object] = {}
    shard_hashers: dict[str, hashlib._Hash] = {}
    shard_sizes: dict[str, int] = {}
    shard_tensor_counts: dict[str, int] = {}
    tensors: dict[str, TensorEntry] = {}

    for tensor_name in sorted(pack_plan):
        source_file_name = source_summary["tensors"][tensor_name]["source_file"]
        safetensors_file = files[source_file_name]
        tensor_header = safetensors_file.tensors[tensor_name]
        shard_name = pack_plan[tensor_name]
        shard_path = output_path / shard_name
        shard_path.parent.mkdir(parents=True, exist_ok=True)

        if shard_name not in handles:
            handles[shard_name] = shard_path.open("wb")
            shard_hashers[shard_name] = hashlib.sha256()
            shard_sizes[shard_name] = 0
            shard_tensor_counts[shard_name] = 0

        handle = handles[shard_name]
        current_offset = shard_sizes[shard_name]
        aligned_offset = _align_offset(current_offset, align_bytes)
        padding = aligned_offset - current_offset
        if padding:
            zeros = b"\x00" * padding
            handle.write(zeros)
            shard_hashers[shard_name].update(zeros)
            shard_sizes[shard_name] += padding

        tensor_offset = shard_sizes[shard_name]
        tensor_digest = copy_tensor_bytes(safetensors_file, tensor_name, handle)

        begin, end = tensor_header.absolute_byte_range(safetensors_file.header_length)
        with safetensors_file.path.open("rb") as source_handle:
            source_handle.seek(begin)
            remaining = end - begin
            while remaining > 0:
                chunk = source_handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ConversionError(f"Unexpected EOF while hashing {tensor_name}.")
                shard_hashers[shard_name].update(chunk)
                remaining -= len(chunk)

        shard_sizes[shard_name] += tensor_header.nbytes
        shard_tensor_counts[shard_name] += 1
        tensors[tensor_name] = TensorEntry(
            name=tensor_name,
            shard=shard_name,
            offset=tensor_offset,
            nbytes=tensor_header.nbytes,
            dtype=tensor_header.dtype,
            shape=list(tensor_header.shape),
            group=shard_name.split("/")[0],
            source_file=source_file_name,
            sha256=tensor_digest,
            numpy_dtype=tensor_header.numpy_dtype,
        )

    for handle in handles.values():
        handle.close()

    manifest = PackedModelManifest(
        format_version=1,
        family=family,
        variant=variant,
        text_only=text_only,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_dir=str(source_path.resolve()),
        layout_strategy="layer_prefix_v1",
        config={
            **source_config,
            **{
                key: value
                for key, value in text_config.items()
                if key not in ("architectures", "text_config", "enable_prefetch", "disk_llm_enable_prefetch")
            },
            "block_kinds": derive_block_kinds(source_config),
        },
        tensors=tensors,
        shards={
            shard_name: ShardEntry(
                path=shard_name,
                size_bytes=shard_sizes[shard_name],
                tensor_count=shard_tensor_counts[shard_name],
                sha256=shard_hashers[shard_name].hexdigest(),
            )
            for shard_name in sorted(handles)
        },
        skipped_tensors=skipped_tensors,
        notes=list(notes or []),
    )

    errors = validate_manifest_files(manifest, base_dir=output_path)
    if errors:
        raise ConversionError("Packed manifest validation failed:\n" + "\n".join(errors))

    manifest.write(manifest_path)
    return ConversionResult(manifest_path=manifest_path, manifest=manifest)
