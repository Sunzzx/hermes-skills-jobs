from importlib.machinery import SourceFileLoader
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "bitnet"

loader = SourceFileLoader("bitnet", str(CLI))
bitnet = types.ModuleType(loader.name)
loader.exec_module(bitnet)


class BitNetTests(unittest.TestCase):
    def test_quantize_weights_ternary_bounds(self):
        weights = [-2.5, -0.8, -0.1, 0.0, 0.2, 0.9, 3.1]
        ternary_w, gamma = bitnet.BitNetQuantizer.quantize_weights(weights)
        self.assertGreater(gamma, 0.0)
        self.assertEqual(len(ternary_w), len(weights))
        for val in ternary_w:
            self.assertIn(val, {-1, 0, 1})

    def test_quantize_activations_int8_bounds(self):
        acts = [-10.0, -3.2, 0.0, 1.5, 9.8]
        quantized, scale = bitnet.BitNetQuantizer.quantize_activations(acts, num_bits=8)
        self.assertGreater(scale, 0.0)
        self.assertEqual(len(quantized), len(acts))
        for val in quantized:
            self.assertGreaterEqual(val, -127)
            self.assertLessEqual(val, 127)

    def test_ternary_gemm_computation(self):
        ternary_w = [1, -1, 0, 1]
        gamma_w = 0.5
        acts = [10, 5, 20, -2]
        scale_a = 2.0
        # acc = 10*1 + 5*(-1) + 20*0 + (-2)*1 = 10 - 5 + 0 - 2 = 3
        # output = (3 * 0.5) / 2.0 = 0.75
        res = bitnet.BitNetQuantizer.ternary_gemm(acts, ternary_w, gamma_w, scale_a)
        self.assertAlmostEqual(res, 0.75, places=5)

    def test_cli_execution_benchmark(self):
        result = subprocess.run(
            [sys.executable, str(CLI), "bench", "256"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Running BitNet b1.58 Benchmark", result.stdout)
        self.assertIn("quantization kernel validated successfully", result.stdout)


if __name__ == "__main__":
    unittest.main()
