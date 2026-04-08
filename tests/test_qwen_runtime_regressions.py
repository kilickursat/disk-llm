from __future__ import annotations

import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
ML_DTYPES_AVAILABLE = importlib.util.find_spec("ml_dtypes") is not None

from disk_llm.converter import convert_model
from disk_llm.inspect import inspect_source_dir
from disk_llm.manifest import PackedModelManifest
from disk_llm.runtime.model import DiskLLMTextModel
from disk_llm.runtime.telemetry import TelemetryRecorder
from tests.helpers import workspace_tempdir, write_fake_qwen_full_attention_model


@unittest.skipUnless(NUMPY_AVAILABLE and ML_DTYPES_AVAILABLE, "numpy or ml-dtypes is not installed")
class QwenRuntimeRegressionTests(unittest.TestCase):
    def test_qwen_full_attention_toy_model_generates_with_gate_and_norms(self):
        with workspace_tempdir() as tmp:
            source_dir = write_fake_qwen_full_attention_model(tmp / "source")
            output_dir = tmp / "packed"

            source_summary = inspect_source_dir(source_dir)
            self.assertEqual(source_summary["num_hidden_layers"], 1)
            self.assertEqual(source_summary["block_kinds"], ["attention"])

            result = convert_model(source_dir, output_dir)
            manifest = PackedModelManifest.from_path(result.manifest_path)
            model = DiskLLMTextModel.from_manifest(result.manifest_path)

            self.assertEqual(model.config.num_hidden_layers, 1)
            self.assertEqual(model.config.attention_head_dim, 2)
            self.assertTrue(model.config.attn_output_gate)
            self.assertFalse(model.config.enable_prefetch)
            self.assertTrue(model.config.use_qwen3_next_norms)
            self.assertEqual(model.config.block_kind(0), "attention")
            self.assertEqual(manifest.layer_ids(), [0])

            generated_ids, telemetry = model.generate_token_ids(
                [0],
                max_new_tokens=2,
                temperature=0.0,
                seed=7,
            )
            self.assertEqual(len(generated_ids), 2)
            self.assertEqual(len(telemetry["layer_times"]), 1)
            self.assertIn("tokens_per_second", telemetry)

    def test_tensor_name_resolution_cache_skips_repeated_manifest_lookups(self):
        with workspace_tempdir() as tmp:
            source_dir = write_fake_qwen_full_attention_model(tmp / "source")
            output_dir = tmp / "packed"

            result = convert_model(source_dir, output_dir)
            model = DiskLLMTextModel.from_manifest(result.manifest_path)

            original_has = model.store.has
            manifest_lookups: list[str] = []

            def counted_has(name: str) -> bool:
                manifest_lookups.append(name)
                return original_has(name)

            model.store.has = counted_has  # type: ignore[method-assign]
            telemetry = TelemetryRecorder(prompt_tokens=1)
            candidates = (
                "model.layers.0.input_layernorm.weight",
                "model.language_model.layers.0.input_layernorm.weight",
            )

            model._get_tensor(candidates, telemetry=telemetry)
            first_lookup_count = len(manifest_lookups)
            self.assertGreater(first_lookup_count, 0)

            model._get_tensor(candidates, telemetry=telemetry)
            self.assertEqual(len(manifest_lookups), first_lookup_count)

            missing_candidates = (
                "missing.tensor.weight",
                "missing.tensor.alias",
            )
            self.assertIsNone(model._maybe_get_tensor(missing_candidates, telemetry=telemetry))
            missing_lookup_count = len(manifest_lookups)

            self.assertIsNone(model._maybe_get_tensor(missing_candidates, telemetry=telemetry))
            self.assertEqual(len(manifest_lookups), missing_lookup_count)

    def test_qwen_hybrid_linear_attention_toy_model(self):
        from tests.helpers import write_fake_qwen_hybrid_model
        with workspace_tempdir() as tmp:
            source_dir = write_fake_qwen_hybrid_model(tmp / "source")
            output_dir = tmp / "packed"

            result = convert_model(source_dir, output_dir)
            model = DiskLLMTextModel.from_manifest(result.manifest_path)

            self.assertEqual(model.config.num_hidden_layers, 2)
            self.assertEqual(model.config.block_kind(0), "linear_attention")
            self.assertEqual(model.config.block_kind(1), "attention")

            generated_ids, telemetry = model.generate_token_ids(
                [0],
                max_new_tokens=2,
                temperature=0.0,
                seed=7,
            )
            self.assertEqual(len(generated_ids), 2)
            self.assertEqual(len(telemetry["layer_times"]), 2)
