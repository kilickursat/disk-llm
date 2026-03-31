"""Small safetensors reader utilities built on the published file format."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

from .exceptions import SafetensorsFormatError

SAFETENSORS_DTYPE_BYTE_SIZES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}

SAFETENSORS_TO_NUMPY_DTYPE: dict[str, str] = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "BF16": "bfloat16",
    "F16": "float16",
    "I16": "int16",
    "U16": "uint16",
    "F32": "float32",
    "I32": "int32",
    "U32": "uint32",
    "F64": "float64",
    "I64": "int64",
    "U64": "uint64",
}


@dataclass(frozen=True)
class TensorHeader:
    """Header metadata for one tensor in a safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def itemsize(self) -> int:
        try:
            return SAFETENSORS_DTYPE_BYTE_SIZES[self.dtype]
        except KeyError as exc:
            raise SafetensorsFormatError(f"Unsupported safetensors dtype: {self.dtype}") from exc

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def numpy_dtype(self) -> str | None:
        return SAFETENSORS_TO_NUMPY_DTYPE.get(self.dtype)

    def absolute_byte_range(self, header_length: int) -> tuple[int, int]:
        data_start = 8 + header_length
        begin, end = self.data_offsets
        return data_start + begin, data_start + end


@dataclass(frozen=True)
class SafetensorsFile:
    """A parsed safetensors header."""

    path: Path
    header_length: int
    metadata: dict[str, str]
    tensors: dict[str, TensorHeader]

    @property
    def data_offset(self) -> int:
        return 8 + self.header_length

    @property
    def total_tensor_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())


def format_bytes(num_bytes: int) -> str:
    """Render byte counts in a compact human-readable form."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _validate_offsets(tensor: TensorHeader) -> None:
    begin, end = tensor.data_offsets
    if begin < 0 or end < 0 or end < begin:
        raise SafetensorsFormatError(
            f"Invalid data offsets for tensor {tensor.name!r}: {tensor.data_offsets}"
        )
    if end - begin != tensor.nbytes:
        raise SafetensorsFormatError(
            f"Byte size mismatch for tensor {tensor.name!r}: "
            f"header span={end - begin}, inferred={tensor.nbytes}"
        )


def read_safetensors_header(path: str | Path) -> SafetensorsFile:
    """Parse the published safetensors header without third-party dependencies."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise SafetensorsFormatError(f"{file_path} is too small to be a safetensors file.")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise SafetensorsFormatError(f"{file_path} ended before the full header was read.")
    try:
        payload = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(f"Invalid safetensors header in {file_path}.") from exc

    metadata = payload.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict):
        raise SafetensorsFormatError(f"Invalid __metadata__ block in {file_path}.")

    tensors: dict[str, TensorHeader] = {}
    for name, info in payload.items():
        if not isinstance(info, dict):
            raise SafetensorsFormatError(f"Tensor entry {name!r} in {file_path} is not an object.")
        try:
            tensor = TensorHeader(
                name=name,
                dtype=str(info["dtype"]),
                shape=tuple(int(value) for value in info["shape"]),
                data_offsets=(int(info["data_offsets"][0]), int(info["data_offsets"][1])),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SafetensorsFormatError(
                f"Malformed tensor entry {name!r} in {file_path}."
            ) from exc
        _validate_offsets(tensor)
        tensors[name] = tensor

    return SafetensorsFile(
        path=file_path,
        header_length=header_length,
        metadata={str(key): str(value) for key, value in metadata.items()},
        tensors=tensors,
    )


def discover_safetensors_files(source_dir: str | Path) -> list[Path]:
    """Find safetensors files, respecting Hugging Face index ordering when present."""
    source_path = Path(source_dir)
    index_path = source_path / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        if isinstance(weight_map, dict):
            ordered = sorted({str(name) for name in weight_map.values()})
            files = [source_path / name for name in ordered]
            return [path for path in files if path.exists()]
    return sorted(source_path.glob("*.safetensors"))


def read_config_json(source_dir: str | Path) -> dict[str, Any]:
    """Load a source-model config if one is present."""
    config_path = Path(source_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def copy_tensor_bytes(
    safetensors_file: SafetensorsFile,
    tensor_name: str,
    destination_handle,
    *,
    buffer_size: int = 1024 * 1024,
) -> str:
    """Copy one tensor's raw bytes into a packed shard and return its SHA256 digest."""
    if tensor_name not in safetensors_file.tensors:
        raise SafetensorsFormatError(
            f"Tensor {tensor_name!r} is not present in {safetensors_file.path}."
        )
    tensor = safetensors_file.tensors[tensor_name]
    begin, end = tensor.absolute_byte_range(safetensors_file.header_length)
    remaining = end - begin
    digest = hashlib.sha256()
    with safetensors_file.path.open("rb") as source_handle:
        source_handle.seek(begin)
        while remaining > 0:
            chunk = source_handle.read(min(buffer_size, remaining))
            if not chunk:
                raise SafetensorsFormatError(
                    f"Unexpected end of file while copying tensor {tensor_name!r}."
                )
            destination_handle.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()
