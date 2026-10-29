import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "jsonl-audit"


class JsonlAuditTests(unittest.TestCase):
    def run_cli(self, *args, input_text=""):
        return subprocess.run([sys.executable, str(CLI), *args], input=input_text,
                              capture_output=True, text=True, check=False)

    def test_valid_rows_report_field_presence_and_type_drift(self):
        result = self.run_cli(input_text='{"id":1,"v":true}\n{"id":2,"v":"x"}\n')
        self.assertEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["schema"]["v"]["types"], {"boolean": 1, "string": 1})
        self.assertTrue(report["consistent_fields"])

    def test_strict_fields_and_unique_findings(self):
        text = '{"id":1,"x":0}\n{"id":1,"y":2}\n'
        result = self.run_cli("--strict-fields", "--unique", "id", input_text=text)
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(report["errors"]), 2)
        self.assertEqual(report["errors"][0]["missing"], ["x"])
        self.assertEqual(report["errors"][0]["extra"], ["y"])
        self.assertIn("duplicate value", report["errors"][1]["error"])

    def test_duplicate_object_keys_rejected(self):
        result = self.run_cli(input_text='{"id":1,"id":2}\n')
        self.assertEqual(result.returncode, 1)
        self.assertIn("duplicate object key", json.loads(result.stdout)["errors"][0]["error"])

    def test_bad_json_and_non_object_have_line_numbers(self):
        result = self.run_cli(input_text='not-json\n[]\n{"ok":true}\n')
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual([item["line"] for item in report["errors"]], [1, 2])
        self.assertEqual(report["rows"], 1)

    def test_missing_file_returns_input_error(self):
        result = self.run_cli(str(ROOT / "no-such-jsonl-fixture"))
        self.assertEqual(result.returncode, 2)
        self.assertIn('"error"', result.stderr)

    def test_blank_lines_ignored(self):
        result = self.run_cli(input_text='\n{"id":1}\n \n')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["rows"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
