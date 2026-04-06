"""Test helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import struct
from typing import Any
from uuid import uuid4


STRUCT_FORMATS = {
    "BOOL": "?",
    "U8": "B",
    "I8": "b",
    "F16": "e",
    "I16": "h",
    "U16": "H",
    "F32": "f",
    "I32": "i",
    "U32": "I",
    "F64": "d",
    "I64": "q",
    "U64": "Q",
}

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


@contextmanager
def workspace_tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = TEST_TMP_ROOT / uuid4().hex
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_fake_safetensors(
    path: str | Path,
    tensors: dict[str, dict[str, Any]],
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    path = Path(path)
    data_chunks: list[bytes] = []
    header: dict[str, Any] = {}
    offset = 0

    for name, spec in tensors.items():
        dtype = str(spec["dtype"])
        shape = list(spec["shape"])
        values = list(spec["values"])
        packed = struct.pack("<" + STRUCT_FORMATS[dtype] * len(values), *values)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(packed)],
        }
        data_chunks.append(packed)
        offset += len(packed)

    if metadata:
        header["__metadata__"] = metadata

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_chunks))
    return path


def write_fake_source_model(source_dir: str | Path) -> Path:
    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen35ForCausalLM"],
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_fake_safetensors(
        source_path / "model-00001-of-00001.safetensors",
        {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 8],
                "values": [float(index) for index in range(16)],
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [8, 8],
                "values": [0.1] * 64,
            },
            "model.layers.0.self_attn.k_proj.weight": {
                "dtype": "F32",
                "shape": [4, 8],
                "values": [0.2] * 32,
            },
            "model.layers.0.self_attn.v_proj.weight": {
                "dtype": "F32",
                "shape": [4, 8],
                "values": [0.3] * 32,
            },
            "model.layers.0.self_attn.o_proj.weight": {
                "dtype": "F32",
                "shape": [8, 8],
                "values": [0.4] * 64,
            },
            "model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [8],
                "values": [1.0] * 8,
            },
            "model.layers.0.post_attention_layernorm.weight": {
                "dtype": "F32",
                "shape": [8],
                "values": [1.0] * 8,
            },
            "model.layers.0.mlp.gate_proj.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "values": [0.01] * 128,
            },
            "model.layers.0.mlp.up_proj.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "values": [0.02] * 128,
            },
            "model.layers.0.mlp.down_proj.weight": {
                "dtype": "F32",
                "shape": [8, 16],
                "values": [0.03] * 128,
            },
            "model.norm.weight": {
                "dtype": "F32",
                "shape": [8],
                "values": [1.0] * 8,
            },
            "lm_head.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "values": [0.5] * 128,
            },
            "visual.patch_embed.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "values": [9.0] * 16,
            },
        },
        metadata={"format": "test"},
    )
    return source_path


def write_fake_nested_text_config_model(source_dir: str | Path) -> Path:
    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "vocab_size": 16,
                        "hidden_size": 8,
                        "intermediate_size": 16,
                        "num_hidden_layers": 4,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 4,
                        "rms_norm_eps": 1e-6,
                        "layer_types": [
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                        ],
                        "rope_parameters": {
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                        },
                        "attn_output_gate": True,
                        "eos_token_id": 15,
                    },
                },
                indent=2,
            ),
        encoding="utf-8",
    )
    write_fake_safetensors(
        source_path / "model-00001-of-00001.safetensors",
        {
            "model.language_model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 8],
                "values": [float(index) for index in range(16)],
            },
            "model.language_model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [8, 8],
                "values": [0.1] * 64,
            },
            "model.language_model.layers.0.self_attn.k_proj.weight": {
                "dtype": "F32",
                "shape": [4, 8],
                "values": [0.2] * 32,
            },
            "model.language_model.layers.0.self_attn.v_proj.weight": {
                "dtype": "F32",
                "shape": [4, 8],
                "values": [0.3] * 32,
            },
            "model.language_model.layers.0.self_attn.o_proj.weight": {
                "dtype": "F32",
                "shape": [8, 8],
                "values": [0.4] * 64,
            },
            "model.language_model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [8],
                "values": [1.0] * 8,
            },
            "model.language_model.layers.0.post_attention_layernorm.weight": {
                "dtype": "F32",
                "shape": [8],
                "values": [1.0] * 8,
            },
            "model.language_model.layers.0.mlp.gate_proj.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "values": [0.01] * 128,
            },
            "model.language_model.layers.0.mlp.up_proj.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "values": [0.02] * 128,
            },
            "model.language_model.layers.0.mlp.down_proj.weight": {
                "dtype": "F32",
                "shape": [8, 16],
                "values": [0.03] * 128,
            },
            "model.language_model.norm.weight": {
                "dtype": "F32",
                "shape": [8],
                "values": [1.0] * 8,
            },
            "lm_head.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "values": [0.5] * 128,
            },
            "model.visual.blocks.0.attn.proj.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "values": [9.0] * 16,
            },
        },
        metadata={"format": "test"},
    )
    return source_path


def write_fake_qwen_full_attention_model(source_dir: str | Path) -> Path:
    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "vocab_size": 4,
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                    "rms_norm_eps": 1e-6,
                    "layer_types": ["full_attention"],
                    "rope_parameters": {
                        "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25,
                    },
                    "attn_output_gate": True,
                    "eos_token_id": 3,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def eye(size: int) -> list[float]:
        return [1.0 if row == col else 0.0 for row in range(size) for col in range(size)]

    def rows(*matrix_rows: list[float]) -> list[float]:
        return [float(value) for row in matrix_rows for value in row]

    write_fake_safetensors(
        source_path / "model-00001-of-00001.safetensors",
        {
            "model.language_model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "values": eye(4),
            },
            "model.language_model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [4],
                "values": [1.0, 1.0, 1.0, 1.0],
            },
            "model.language_model.layers.0.post_attention_layernorm.weight": {
                "dtype": "F32",
                "shape": [4],
                "values": [1.0, 1.0, 1.0, 1.0],
            },
            "model.language_model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [8, 4],
                "values": rows(
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ),
            },
            "model.language_model.layers.0.self_attn.k_proj.weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "values": rows(
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ),
            },
            "model.language_model.layers.0.self_attn.v_proj.weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "values": rows(
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ),
            },
            "model.language_model.layers.0.self_attn.q_norm.weight": {
                "dtype": "F32",
                "shape": [2],
                "values": [1.0, 1.0],
            },
            "model.language_model.layers.0.self_attn.k_norm.weight": {
                "dtype": "F32",
                "shape": [2],
                "values": [1.0, 1.0],
            },
            "model.language_model.layers.0.self_attn.o_proj.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "values": eye(4),
            },
            "model.language_model.layers.0.mlp.gate_proj.weight": {
                "dtype": "F32",
                "shape": [8, 4],
                "values": [0.0] * 32,
            },
            "model.language_model.layers.0.mlp.up_proj.weight": {
                "dtype": "F32",
                "shape": [8, 4],
                "values": [0.0] * 32,
            },
            "model.language_model.layers.0.mlp.down_proj.weight": {
                "dtype": "F32",
                "shape": [4, 8],
                "values": [0.0] * 32,
            },
            "model.language_model.norm.weight": {
                "dtype": "F32",
                "shape": [4],
                "values": [1.0, 1.0, 1.0, 1.0],
            },
            "lm_head.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "values": eye(4),
            },
        },
        metadata={"format": "test"},
    )
    return source_path
