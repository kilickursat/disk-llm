from __future__ import annotations

import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np

from disk_llm.runtime.kernels import apply_rope_single, grouped_query_attention_step, rms_norm, sample_from_logits


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class KernelTests(unittest.TestCase):
    def test_rms_norm_matches_manual_formula(self):
        hidden = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)
        actual = rms_norm(hidden, weight, eps=1e-6)
        expected = hidden / np.sqrt(np.mean(hidden * hidden) + 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_apply_rope_preserves_shape(self):
        values = np.arange(8, dtype=np.float32).reshape(2, 4)
        rotated = apply_rope_single(values, position=3, theta=10000.0)
        self.assertEqual(rotated.shape, values.shape)

    def test_grouped_query_attention_step_returns_expected_shape(self):
        query = np.ones((2, 4), dtype=np.float32)
        key_history = np.ones((3, 2, 4), dtype=np.float32)
        value_history = np.ones((3, 2, 4), dtype=np.float32)
        context = grouped_query_attention_step(query, key_history, value_history, scale=0.5)
        self.assertEqual(context.shape, (2, 4))

    def test_sampling_greedy_path(self):
        logits = np.array([0.1, 0.2, 0.9], dtype=np.float32)
        token_id = sample_from_logits(logits, temperature=0.0)
        self.assertEqual(token_id, 2)
