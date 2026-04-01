from __future__ import annotations

import csv
import json
import unittest

from disk_llm.benchmarking import BenchmarkReport, aggregate_run_rows, build_prompt_cases, write_benchmark_artifacts
from tests.helpers import workspace_tempdir


class BenchmarkingTests(unittest.TestCase):
    def test_build_prompt_cases_cycles_base_tokens(self):
        cases = build_prompt_cases([7, 8, 9], [2, 5])
        self.assertEqual([case.token_ids for case in cases], [[7, 8], [7, 8, 9, 7, 8]])
        self.assertEqual([case.prompt_tokens for case in cases], [2, 5])

    def test_write_benchmark_artifacts_creates_csv_and_metadata(self):
        run_rows = [
            {
                "run_id": "disk_llm|8|16|0",
                "backend": "disk_llm",
                "backend_label": "Disk-LLM",
                "prompt_label": "tokens_0008",
                "prompt_tokens": 8,
                "max_new_tokens": 16,
                "run_index": 0,
                "run_phase": "initial",
                "seed": 0,
                "temperature": 0.0,
                "top_p": 0.95,
                "generated_tokens": 16,
                "elapsed_seconds": 1.2,
                "first_token_seconds": 0.3,
                "tokens_per_second": 13.333333,
                "rss_mb_start": 100.0,
                "rss_mb_peak": 120.0,
                "rss_mb_end": 118.0,
                "rss_delta_mb": 18.0,
                "io_read_mb": 42.0,
                "io_write_mb": 0.5,
                "logical_bytes_mapped_mb": 64.0,
                "tensors_touched": 12,
                "layer_count": 4,
                "notes": "",
            }
        ]
        timeline_rows = [
            {
                "run_id": "disk_llm|8|16|0",
                "backend": "disk_llm",
                "backend_label": "Disk-LLM",
                "prompt_label": "tokens_0008",
                "prompt_tokens": 8,
                "max_new_tokens": 16,
                "run_index": 0,
                "sample_index": 0,
                "elapsed_seconds": 0.0,
                "rss_mb": 100.0,
                "io_read_mb": 0.0,
                "io_write_mb": 0.0,
            }
        ]
        report = BenchmarkReport(
            metadata={"created_at": "2026-04-01T00:00:00Z", "runs": 1},
            run_rows=run_rows,
            timeline_rows=timeline_rows,
            summary_rows=aggregate_run_rows(run_rows),
        )

        with workspace_tempdir() as tmp:
            paths = write_benchmark_artifacts(report, tmp / "bench")
            self.assertTrue(paths["runs"].exists())
            self.assertTrue(paths["timeline"].exists())
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["metadata"].exists())

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["runs"], 1)

            with paths["runs"].open("r", encoding="utf-8", newline="") as handle:
                run_rows_loaded = list(csv.DictReader(handle))
            self.assertEqual(len(run_rows_loaded), 1)
            self.assertEqual(run_rows_loaded[0]["backend"], "disk_llm")

            with paths["summary"].open("r", encoding="utf-8", newline="") as handle:
                summary_rows_loaded = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows_loaded), 1)
            self.assertEqual(summary_rows_loaded[0]["tokens_per_second_mean"], "13.333333")
