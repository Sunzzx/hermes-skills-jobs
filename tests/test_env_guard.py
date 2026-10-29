import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "env-guard"


class EnvGuardTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=cwd or str(ROOT),
            capture_output=True,
            text=True,
            check=False
        )

    def test_matching_env_and_example_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / ".env").write_text("PORT=8080\nDATABASE_URL=postgres://localhost:5432/app\n")
            (td / ".env.example").write_text("PORT=your_port_here\nDATABASE_URL=postgres://user:pass@localhost:5432/db\n")
            
            res = self.run_cli(str(td), "--format", "json")
            self.assertEqual(res.returncode, 0)
            data = json.loads(res.stdout)
            self.assertTrue(data["valid"])
            self.assertEqual(data["missing_in_env"], [])
            self.assertEqual(data["secret_leaks"], [])

    def test_drift_missing_keys_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / ".env").write_text("PORT=8080\n")
            (td / ".env.example").write_text("PORT=8080\nREDIS_URL=redis://localhost:6379\nSECRET_KEY=change_me\n")
            
            res = self.run_cli(str(td), "--format", "json")
            self.assertEqual(res.returncode, 1)
            data = json.loads(res.stdout)
            self.assertFalse(data["valid"])
            self.assertIn("REDIS_URL", data["missing_in_env"])
            self.assertIn("SECRET_KEY", data["missing_in_env"])

    def test_strict_mode_rejects_extra_undocumented_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / ".env").write_text("PORT=8080\nINTERNAL_DEBUG_FLAG=true\n")
            (td / ".env.example").write_text("PORT=8080\n")
            
            # Without --strict should pass
            res1 = self.run_cli(str(td), "--format", "json")
            self.assertEqual(res1.returncode, 0)
            
            # With --strict should fail
            res2 = self.run_cli(str(td), "--strict", "--format", "json")
            self.assertEqual(res2.returncode, 1)
            data = json.loads(res2.stdout)
            self.assertFalse(data["valid"])
            self.assertIn("INTERNAL_DEBUG_FLAG", data["extra_in_env"])

    def test_secret_leak_in_example_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / ".env").write_text("AWS_KEY=AKIAIOSFODNN7EXAMPLE\n")
            # Accidental secret leak in committed example template
            (td / ".env.example").write_text("AWS_KEY=AKIAIOSFODNN7EXAMPLE\n")
            
            res = self.run_cli(str(td), "--format", "json")
            self.assertEqual(res.returncode, 1)
            data = json.loads(res.stdout)
            self.assertFalse(data["valid"])
            self.assertTrue(any(leak["key"] == "AWS_KEY" for leak in data["secret_leaks"]))

    def test_type_and_required_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / ".env").write_text("PORT=not-a-port\nDEBUG=yes\nAPP_URL=https://app.local\nEMAIL=invalid-email\n")
            (td / ".env.example").write_text("PORT=3000\nDEBUG=false\nAPP_URL=https://example.com\nEMAIL=admin@example.com\n")
            
            res = self.run_cli(
                str(td),
                "--types", "PORT:port,DEBUG:bool,APP_URL:url,EMAIL:email",
                "--required", "PORT,MISSING_REQUIRED_KEY",
                "--format", "json"
            )
            self.assertEqual(res.returncode, 1)
            data = json.loads(res.stdout)
            self.assertFalse(data["valid"])
            violation_keys = [v["key"] for v in data["schema_violations"]]
            self.assertIn("PORT", violation_keys)
            self.assertIn("EMAIL", violation_keys)
            self.assertIn("MISSING_REQUIRED_KEY", violation_keys)
            self.assertNotIn("DEBUG", violation_keys) # 'yes' is valid bool
            self.assertNotIn("APP_URL", violation_keys) # valid url

    def test_sync_scaffolding_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / ".env").write_text("PORT=8080\nSECRET_API_TOKEN=super_secret_production_token_12345\n")
            example_file = td / ".env.example"
            example_file.write_text("PORT=3000\n")
            
            # Run --sync
            res = self.run_cli(str(td), "--sync")
            self.assertEqual(res.returncode, 0)
            self.assertIn("Synced 1 missing key", res.stdout)
            
            example_after = example_file.read_text()
            self.assertIn("SECRET_API_TOKEN=your_secret_api_token_here", example_after)
            # Ensure the actual secret was NEVER written to the example!
            self.assertNotIn("super_secret_production_token_12345", example_after)

    def test_missing_env_file_exit_code_2(self):
        res = self.run_cli("/tmp/nonexistent_env_directory_xyz")
        self.assertEqual(res.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
