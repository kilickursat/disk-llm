"""Manifest-backed memmap tensor store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import RuntimeShapeError
from ..manifest import PackedModelManifest
from ..optional import require_numpy
from .telemetry import TelemetryRecorder


class MemmapTensorStore:
    """Lazily materialize NumPy memmaps from a Disk-LLM manifest."""

    def __init__(self, manifest: PackedModelManifest, *, base_dir: str | Path):
        self.manifest = manifest
        self.base_dir = Path(base_dir)
        self._cache: dict[str, Any] = {}

    def has(self, tensor_name: str) -> bool:
        return tensor_name in self.manifest.tensors

    def names(self) -> list[str]:
        return sorted(self.manifest.tensors)

    def get(self, tensor_name: str, telemetry: TelemetryRecorder | None = None):
        np = require_numpy()
        if tensor_name not in self.manifest.tensors:
            raise RuntimeShapeError(f"Tensor not found in manifest: {tensor_name}")
        if tensor_name not in self._cache:
            entry = self.manifest.tensors[tensor_name]
            if entry.numpy_dtype is None:
                raise RuntimeShapeError(
                    f"Tensor {tensor_name} uses unsupported dtype {entry.dtype} for NumPy runtime."
                )
            shard_path = self.base_dir / entry.shard
            self._cache[tensor_name] = np.memmap(
                shard_path,
                dtype=np.dtype(entry.numpy_dtype),
                mode="r",
                offset=entry.offset,
                shape=tuple(entry.shape),
            )
        if telemetry is not None:
            telemetry.record_tensor_map(tensor_name, self.manifest.tensors[tensor_name].nbytes)
        return self._cache[tensor_name]
