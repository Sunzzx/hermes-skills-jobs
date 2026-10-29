import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "artifact-integrity"

class ArtifactIntegrityTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)

    def test_create_verify_then_detect_tamper_and_extra(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); tree = base / "dist"; tree.mkdir()
            (tree / "app.bin").write_bytes(b"release-v1")
            manifest = base / "manifest.json"
            result = self.run_cli("create", tree, "--output", manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(manifest.read_text())["files"]["app.bin"]["size"], 10)
            self.assertEqual(self.run_cli("verify", tree, manifest).returncode, 0)
            (tree / "app.bin").write_bytes(b"release-v2")
            (tree / "unexpected").write_text("x")
            result = self.run_cli("verify", tree, manifest)
            self.assertEqual(result.returncode, 1)
            self.assertIn("CHANGED: app.bin", result.stdout)
            self.assertIn("UNEXPECTED: unexpected", result.stdout)
            (tree / "app.bin").unlink()
            result = self.run_cli("verify", tree, manifest)
            self.assertIn("MISSING: app.bin", result.stdout)

    def test_exclusion_and_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); tree = base / "dist"; tree.mkdir()
            (tree / "app").write_text("ok"); (tree / "debug.map").write_text("ignored")
            a, b = base / "a.json", base / "b.json"
            self.assertEqual(self.run_cli("create", tree, "-o", a, "--exclude", "*.map").returncode, 0)
            self.assertEqual(self.run_cli("create", tree, "-o", b, "--exclude", "*.map").returncode, 0)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(set(json.loads(a.read_text())["files"]), {"app"})
            self.assertEqual(self.run_cli("verify", tree, a, "--exclude", "*.map").returncode, 0)

    def test_rejects_symlink_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); tree = base / "dist"; tree.mkdir()
            target = base / "target"; target.write_text("secret")
            try:
                (tree / "link").symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            result = self.run_cli("create", tree, "-o", base / "manifest.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing symlink", result.stderr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
