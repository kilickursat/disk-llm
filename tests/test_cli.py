from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from disk_llm.cli import main
from tests.helpers import workspace_tempdir, write_fake_source_model


class CliTests(unittest.TestCase):
    def test_help_exit_code(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_inspect_source_json(self):
        with workspace_tempdir() as tmp:
            source_dir = write_fake_source_model(tmp / "source")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["inspect", "--source-dir", str(source_dir), "--json"])
            self.assertEqual(exit_code, 0)
            output = buffer.getvalue()
            self.assertIn('"kind": "source"', output)

    def test_convert_command(self):
        with workspace_tempdir() as tmp:
            source_dir = write_fake_source_model(tmp / "source")
            output_dir = tmp / "packed"
            exit_code = main(["convert", str(source_dir), str(output_dir)])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "manifest.json").exists())
