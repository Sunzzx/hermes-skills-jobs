import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "trace-timeline"

class TimelineCliTests(unittest.TestCase):
    def run_cli(self, *args, input_text=None):
        return subprocess.run([sys.executable, str(SCRIPT), *args], input=input_text,
                              capture_output=True, text=True, check=False)

    def test_merges_and_sorts_iso_and_json_records(self):
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "api.log"
            second = Path(td) / "worker.jsonl"
            first.write_text("2026-01-02T00:00:02Z ERROR late\n2026-01-02T00:00:01+00:00 INFO early\n", encoding="utf-8")
            second.write_text(json.dumps({"timestamp":"2026-01-02T00:00:01Z", "level":"WARN", "message":"queued"})+"\n", encoding="utf-8")
            result = self.run_cli("--json", str(first), str(second))
            events = json.loads(result.stdout)
            self.assertEqual(result.returncode, 0)
            self.assertEqual([e["message"] for e in events], ["INFO early", "WARN queued", "ERROR late"])
            self.assertEqual(events[0]["timestamp"], "2026-01-02T00:00:01.000000Z")

    def test_syslog_explicit_year_and_min_level(self):
        result = self.run_cli("--year", "2024", "--min-level", "ERROR", "-", input_text="Jan  2 03:04:05 host INFO ready\nJan  2 03:04:06 host ERROR failed\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("2024-01-02T03:04:06", result.stdout)
        self.assertNotIn("ready", result.stdout)

    def test_skips_bad_lines_and_reports_count(self):
        result = self.run_cli("-", input_text="not a timestamp\n\n# comment\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("No timestamped log events", result.stdout)
        self.assertIn("2 non-timestamped", result.stderr)

    def test_invalid_file_is_actionable_error(self):
        result = self.run_cli("/path/that/does/not/exist.log")
        self.assertEqual(result.returncode, 2)
        self.assertIn("trace-timeline:", result.stderr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
