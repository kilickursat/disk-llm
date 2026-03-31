from __future__ import annotations

import unittest

from disk_llm.converter import convert_model
from disk_llm.manifest import PackedModelManifest, validate_manifest_files
from tests.helpers import workspace_tempdir, write_fake_source_model


class ConverterTests(unittest.TestCase):
    def test_convert_model_creates_manifest_and_skips_visual_tensors(self):
        with workspace_tempdir() as tmp:
            source_dir = write_fake_source_model(tmp / "source")
            output_dir = tmp / "packed"
            result = convert_model(source_dir, output_dir)

            manifest = PackedModelManifest.from_path(result.manifest_path)
            self.assertIn("model.embed_tokens.weight", manifest.tensors)
            self.assertNotIn("visual.patch_embed.weight", manifest.tensors)
            self.assertIn("visual.patch_embed.weight", manifest.skipped_tensors)
            self.assertIn("layers/layer_000.bin", manifest.shards)
            self.assertTrue((output_dir / "layers" / "layer_000.bin").exists())
            self.assertEqual(manifest.config["block_kinds"][:4], ["delta", "delta", "delta", "attention"])

            errors = validate_manifest_files(manifest, base_dir=output_dir)
            self.assertEqual(errors, [])
