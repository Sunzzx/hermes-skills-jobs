import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "release-gate"


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Release Gate Test")
        self.git("config", "user.email", "release-gate@example.invalid")
        (self.repo / "README.md").write_text("fixture\n")
        self.git("add", "README.md")
        self.git("commit", "-m", "initial")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.repo, text=True,
                              capture_output=True, check=check)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(CLI), *args], cwd=self.repo,
                              text=True, capture_output=True, check=False)

    def test_clean_local_repository_without_tag_fails_missing_upstream(self):
        result = self.run_cli()
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing_upstream", {item["code"] for item in report["checks"]})
        self.assertFalse(report["ready"])

    def test_synced_clean_repository_and_unused_semver_tag_pass(self):
        bare = Path(self.temp.name) / "origin.git"
        subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
        self.git("remote", "add", "origin", str(bare))
        self.git("push", "-u", "origin", "main")
        result = self.run_cli("--tag", "v1.2.3")
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["ready"])
        self.assertEqual(report["tag"], "v1.2.3")

    def test_dirty_tree_and_invalid_tag_are_blocking_findings(self):
        (self.repo / "uncommitted.txt").write_text("dirty\n")
        result = self.run_cli("--tag", "release-latest")
        report = json.loads(result.stdout)
        codes = {item["code"] for item in report["checks"]}
        self.assertEqual(result.returncode, 1)
        self.assertIn("dirty_worktree", codes)
        self.assertIn("invalid_tag", codes)

    def test_existing_tag_is_rejected(self):
        self.git("tag", "v1.0.0")
        result = self.run_cli("--tag", "v1.0.0")
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertIn("tag_exists", {item["code"] for item in report["checks"]})

    def test_human_output_is_concise_and_nonzero_when_blocked(self):
        result = self.run_cli("--format", "text")
        self.assertEqual(result.returncode, 1)
        self.assertIn("NOT READY", result.stdout)
        self.assertIn("missing_upstream", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
