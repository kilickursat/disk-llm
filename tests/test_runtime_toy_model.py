from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np

from disk_llm.manifest import PackedModelManifest, ShardEntry, TensorEntry
from disk_llm.runtime.model import DiskLLMTextModel
from tests.helpers import workspace_tempdir


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class ToyModelTests(unittest.TestCase):
    def test_attention_only_toy_model_generates(self):
        with workspace_tempdir() as tmp_path:
            shard_path = tmp_path / "layers" / "layer_000.bin"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            final_path = tmp_path / "final" / "final.bin"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            embed_path = tmp_path / "embeddings" / "embeddings.bin"
            embed_path.parent.mkdir(parents=True, exist_ok=True)

            tensors = {}

            def add_tensor(path: Path, name: str, array, offset_map: dict[str, int]):
                raw = np.asarray(array, dtype=np.float32).tobytes()
                offset = offset_map.setdefault(str(path), 0)
                with path.open("ab") as handle:
                    handle.write(raw)
                offset_map[str(path)] += len(raw)
                tensors[name] = TensorEntry(
                    name=name,
                    shard=str(path.relative_to(tmp_path)).replace("\\", "/"),
                    offset=offset,
                    nbytes=len(raw),
                    dtype="F32",
                    shape=list(array.shape),
                    group=path.parent.name,
                    source_file="toy",
                    sha256="",
                    numpy_dtype="float32",
                )

            offsets: dict[str, int] = {}
            embed = np.eye(4, dtype=np.float32)
            add_tensor(embed_path, "model.embed_tokens.weight", embed, offsets)
            add_tensor(shard_path, "model.layers.0.input_layernorm.weight", np.ones(4, dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.post_attention_layernorm.weight", np.ones(4, dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.self_attn.q_proj.weight", np.eye(4, dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.self_attn.k_proj.weight", np.eye(4, dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.self_attn.v_proj.weight", np.eye(4, dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.self_attn.o_proj.weight", np.eye(4, dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.mlp.gate_proj.weight", np.zeros((8, 4), dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.mlp.up_proj.weight", np.zeros((8, 4), dtype=np.float32), offsets)
            add_tensor(shard_path, "model.layers.0.mlp.down_proj.weight", np.zeros((4, 8), dtype=np.float32), offsets)
            add_tensor(final_path, "model.norm.weight", np.ones(4, dtype=np.float32), offsets)
            add_tensor(final_path, "lm_head.weight", np.eye(4, dtype=np.float32), offsets)

            manifest = PackedModelManifest(
                format_version=1,
                family="toy",
                variant="tiny",
                text_only=True,
                created_at="2026-01-01T00:00:00Z",
                source_dir=str(tmp_path),
                layout_strategy="layer_prefix_v1",
                config={
                    "vocab_size": 4,
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "rms_norm_eps": 1e-6,
                    "block_kinds": ["attention"],
                },
                tensors=tensors,
                shards={
                    "embeddings/embeddings.bin": ShardEntry(
                        path="embeddings/embeddings.bin",
                        size_bytes=embed_path.stat().st_size,
                        tensor_count=1,
                        sha256="",
                    ),
                    "layers/layer_000.bin": ShardEntry(
                        path="layers/layer_000.bin",
                        size_bytes=shard_path.stat().st_size,
                        tensor_count=9,
                        sha256="",
                    ),
                    "final/final.bin": ShardEntry(
                        path="final/final.bin",
                        size_bytes=final_path.stat().st_size,
                        tensor_count=2,
                        sha256="",
                    ),
                },
            )
            manifest_path = tmp_path / "manifest.json"
            manifest.write(manifest_path)

            model = DiskLLMTextModel.from_manifest(manifest_path)
            generated_ids, telemetry = model.generate_token_ids([0], max_new_tokens=2)
            self.assertEqual(len(generated_ids), 2)
            self.assertIn("tokens_per_second", telemetry)
