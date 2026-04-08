from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from disk_llm.plotting import generate_plots
from tests.helpers import workspace_tempdir


class _FakeFigure:
    def add_trace(self, *args, **kwargs):
        return None

    def update_layout(self, *args, **kwargs):
        return None

    def add_annotation(self, *args, **kwargs):
        return None

    def write_image(self, path, scale=1.0):
        raise RuntimeError("static export unavailable")

    def write_html(self, path, include_plotlyjs="cdn", full_html=True):
        from pathlib import Path

        Path(path).write_text("<html><body>fallback</body></html>", encoding="utf-8")


class _FakeGo:
    Figure = _FakeFigure

    @staticmethod
    def Bar(**kwargs):
        return kwargs

    @staticmethod
    def Scatter(**kwargs):
        return kwargs


class PlottingTests(unittest.TestCase):
    @patch("disk_llm.plotting.require_plotly", return_value=(_FakeGo, None))
    def test_generate_plots_falls_back_to_html_when_static_export_fails(self, _mock_require_plotly):
        with workspace_tempdir() as tmp:
            results_dir = tmp / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            (results_dir / "benchmark_summary.csv").write_text(
                "backend,backend_label,prompt_tokens,max_new_tokens,tokens_per_second_mean,tokens_per_second_stdev,first_token_seconds_mean,logical_bytes_mapped_mb_mean,rss_mb_peak_mean,io_read_mb_mean\n"
                "disk_llm,Disk-LLM,8,2,0.15,0.0,10.0,64.0,100.0,\n"
                "hf_cpu,HF CPU,8,2,0.12,0.0,6.0,,200.0,\n",
                encoding="utf-8",
            )
            (results_dir / "memory_timeline.csv").write_text(
                "run_id,backend,backend_label,prompt_tokens,max_new_tokens,run_index,sample_index,elapsed_seconds,rss_mb\n"
                "disk_llm|8|2|0,disk_llm,Disk-LLM,8,2,0,0,0.0,100.0\n",
                encoding="utf-8",
            )
            (results_dir / "benchmark_metadata.json").write_text(
                json.dumps({"manifest_path": "/tmp/manifest.json", "runs": 1, "warmup_runs": 0}),
                encoding="utf-8",
            )

            paths = generate_plots(results_dir)

            self.assertEqual(paths["tokens_per_second"].suffix, ".html")
            self.assertEqual(paths["first_token_latency"].suffix, ".html")
            self.assertEqual(paths["logical_mapped_mb"].suffix, ".html")
            self.assertEqual(paths["rss_timeline"].suffix, ".html")
            self.assertTrue(paths["tokens_per_second"].exists())
            self.assertTrue(paths["comparison_summary"].exists())
            self.assertTrue((results_dir / "plots" / "tokens_per_second_export_warning.txt").exists())


if __name__ == "__main__":
    unittest.main()