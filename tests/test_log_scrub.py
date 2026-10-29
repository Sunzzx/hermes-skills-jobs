import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "log-scrub"

class LogScrubTests(unittest.TestCase):
    def run_cli(self, *args, input_text=""):
        return subprocess.run([sys.executable, str(CLI), *args], input=input_text,
                              capture_output=True, text=True, check=False)

    def test_scrubs_credentials_and_pii_but_preserves_labels(self):
        text = ("api_key=abcdefghijk password: 'topsecret123'\n"
                "Authorization: Bearer abcdefghijklmno\n"
                "AWS AKIA1234567890ABCDEF\nuser@example.com +1 (415) 555-1212\n")
        result = self.run_cli("-", input_text=text)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("abcdefghijk", result.stdout)
        self.assertNotIn("topsecret123", result.stdout)
        self.assertNotIn("user@example.com", result.stdout)
        self.assertIn("api_key=[REDACTED]", result.stdout)
        self.assertIn("Bearer [REDACTED]", result.stdout)

    def test_check_mode_emits_no_sensitive_text_and_signals_findings(self):
        result = self.run_cli("--check", "-", input_text="password=secret-value\n")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("secret-value", result.stderr)
        self.assertIn("assignment-secret: 1", result.stderr)

    def test_clean_input_check_passes(self):
        result = self.run_cli("--check", "-", input_text="service started successfully\n")
        self.assertEqual(result.returncode, 0)

    def test_private_key_block_removed(self):
        text = "before\n-----BEGIN PRIVATE KEY-----\nsecretmaterial\n-----END PRIVATE KEY-----\nafter\n"
        result = self.run_cli("-", input_text=text)
        self.assertNotIn("secretmaterial", result.stdout)
        self.assertIn("before", result.stdout)
        self.assertIn("after", result.stdout)

    def test_missing_file_returns_two(self):
        result = self.run_cli("/missing/sensitive.log")
        self.assertEqual(result.returncode, 2)
        self.assertIn("log-scrub:", result.stderr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
