from __future__ import annotations

import hashlib
import struct
import unittest

from disk_llm.safetensors_io import copy_tensor_bytes, read_safetensors_header
from tests.helpers import workspace_tempdir, write_fake_safetensors


class SafetensorsIoTests(unittest.TestCase):
    def test_reads_header_and_copies_tensor_bytes(self):
        with workspace_tempdir() as tmp:
            source_path = write_fake_safetensors(
                tmp / "tiny.safetensors",
                {
                    "weight": {
                        "dtype": "F32",
                        "shape": [2, 2],
                        "values": [1.0, 2.0, 3.0, 4.0],
                    },
                    "bias": {
                        "dtype": "F32",
                        "shape": [2],
                        "values": [5.0, 6.0],
                    },
                },
            )
            parsed = read_safetensors_header(source_path)
            self.assertEqual(set(parsed.tensors), {"weight", "bias"})
            self.assertEqual(parsed.tensors["weight"].shape, (2, 2))
            self.assertEqual(parsed.tensors["weight"].nbytes, 16)

            target_path = tmp / "weight.bin"
            with target_path.open("wb") as handle:
                digest = copy_tensor_bytes(parsed, "weight", handle)

            expected = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            with target_path.open("rb") as handle:
                self.assertEqual(handle.read(), expected)
