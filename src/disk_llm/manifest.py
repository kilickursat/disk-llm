"""Manifest dataclasses and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .exceptions import ManifestError


@dataclass(frozen=True)
class TensorEntry:
    """One packed tensor inside a shard."""

    name: str
    shard: str
    offset: int
    nbytes: int
    dtype: str
    shape: list[int]
    group: str
    source_file: str
    sha256: str
    numpy_dtype: str | None = None


@dataclass(frozen=True)
class ShardEntry:
    """A packed shard file on disk."""

    path: str
    size_bytes: int
    tensor_count: int
    sha256: str


@dataclass
class PackedModelManifest:
    """Top-level manifest for a packed Disk-LLM model."""

    format_version: int
    family: str
    variant: str
    text_only: bool
    created_at: str
    source_dir: str
    layout_strategy: str
    config: dict[str, Any]
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    shards: dict[str, ShardEntry] = field(default_factory=dict)
    skipped_tensors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "family": self.family,
            "variant": self.variant,
            "text_only": self.text_only,
            "created_at": self.created_at,
            "source_dir": self.source_dir,
            "layout_strategy": self.layout_strategy,
            "config": self.config,
            "tensors": {name: asdict(entry) for name, entry in sorted(self.tensors.items())},
            "shards": {name: asdict(entry) for name, entry in sorted(self.shards.items())},
            "skipped_tensors": sorted(self.skipped_tensors),
            "notes": list(self.notes),
        }

    def write(self, path: str | Path) -> Path:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PackedModelManifest":
        tensors = {
            name: TensorEntry(**entry)
            for name, entry in data.get("tensors", {}).items()
        }
        shards = {
            name: ShardEntry(**entry)
            for name, entry in data.get("shards", {}).items()
        }
        return cls(
            format_version=int(data["format_version"]),
            family=str(data["family"]),
            variant=str(data["variant"]),
            text_only=bool(data["text_only"]),
            created_at=str(data["created_at"]),
            source_dir=str(data["source_dir"]),
            layout_strategy=str(data["layout_strategy"]),
            config=dict(data.get("config", {})),
            tensors=tensors,
            shards=shards,
            skipped_tensors=[str(item) for item in data.get("skipped_tensors", [])],
            notes=[str(item) for item in data.get("notes", [])],
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "PackedModelManifest":
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise ManifestError(f"Manifest not found: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Manifest is not valid JSON: {manifest_path}") from exc
        return cls.from_dict(payload)

    def layer_ids(self) -> list[int]:
        values: set[int] = set()
        for name in self.tensors:
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                try:
                    values.add(int(parts[2]))
                except ValueError:
                    continue
        return sorted(values)

    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.shards.values())


def validate_manifest_files(
    manifest: PackedModelManifest,
    *,
    base_dir: str | Path,
) -> list[str]:
    """Validate packed files on disk against a manifest."""
    errors: list[str] = []
    base_path = Path(base_dir)

    for shard_name, shard in manifest.shards.items():
        shard_path = base_path / shard.path
        if not shard_path.exists():
            errors.append(f"Missing shard file for {shard_name}: {shard_path}")
            continue
        size = shard_path.stat().st_size
        if size != shard.size_bytes:
            errors.append(
                f"Shard size mismatch for {shard_name}: expected {shard.size_bytes}, found {size}"
            )

    per_shard_offsets: dict[str, list[tuple[int, int, str]]] = {}
    for tensor_name, tensor in manifest.tensors.items():
        per_shard_offsets.setdefault(tensor.shard, []).append(
            (tensor.offset, tensor.offset + tensor.nbytes, tensor_name)
        )

    for shard_name, spans in per_shard_offsets.items():
        spans.sort(key=lambda item: item[0])
        shard_entry = manifest.shards.get(shard_name)
        shard_size = shard_entry.size_bytes if shard_entry else None
        previous_end = -1
        for begin, end, tensor_name in spans:
            if begin < previous_end:
                errors.append(f"Tensor overlap detected in {shard_name}: {tensor_name}")
            if shard_size is not None and end > shard_size:
                errors.append(
                    f"Tensor {tensor_name} extends past shard boundary in {shard_name}"
                )
            previous_end = max(previous_end, end)
    return errors
