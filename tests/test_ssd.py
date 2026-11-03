import unittest, subprocess, sys, os
class TestSSD(unittest.TestCase):
    def test_verify_cli(self):
        res = subprocess.run([sys.executable, 'bin/ssd-engine', 'verify'], capture_output=True, text=True, check=True)
        self.assertIn('verification passed', res.stdout)

if __name__ == '__main__':
    unittest.main()
