import os
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "flash-attn"

# Add bin/ to path so we can import FlashAttentionEngine directly
sys.path.insert(0, str(ROOT / "bin"))
from importlib.machinery import SourceFileLoader
flash_module = SourceFileLoader("flash_attn", str(CLI)).load_module()
FlashAttentionEngine = flash_module.FlashAttentionEngine


class FlashAttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1337)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *map(str, args)],
            capture_output=True,
            text=True,
        )

    def test_non_causal_forward_precision(self):
        for d in [32, 64]:
            for N in [32, 64, 128]:
                Q = self.rng.randn(2, 2, N, d)
                K = self.rng.randn(2, 2, N, d)
                V = self.rng.randn(2, 2, N, d)

                O_ref, _, L_ref = FlashAttentionEngine.naive_attention(Q, K, V, causal=False)
                O_fa, L_fa = FlashAttentionEngine.forward(Q, K, V, causal=False, Br=16, Bc=16)

                max_err_O = np.max(np.abs(O_ref - O_fa))
                max_err_L = np.max(np.abs(L_ref - L_fa))

                self.assertLess(max_err_O, 1e-10, f"Failed on d={d}, N={N} with err={max_err_O}")
                self.assertLess(max_err_L, 1e-10, f"Failed logsumexp on d={d}, N={N}")

    def test_causal_forward_precision(self):
        for N in [48, 64, 96]:
            Q = self.rng.randn(1, 4, N, 32)
            K = self.rng.randn(1, 4, N, 32)
            V = self.rng.randn(1, 4, N, 32)

            O_ref, _, L_ref = FlashAttentionEngine.naive_attention(Q, K, V, causal=True)
            O_fa, L_fa = FlashAttentionEngine.forward(Q, K, V, causal=True, Br=16, Bc=16)

            max_err_O = np.max(np.abs(O_ref - O_fa))
            max_err_L = np.max(np.abs(L_ref - L_fa))

            self.assertLess(max_err_O, 1e-10)
            self.assertLess(max_err_L, 1e-10)

    def test_exact_backward_pass(self):
        B, H, N, d = 1, 2, 64, 32
        Q = self.rng.randn(B, H, N, d)
        K = self.rng.randn(B, H, N, d)
        V = self.rng.randn(B, H, N, d)
        dO = self.rng.randn(B, H, N, d)

        for causal in [False, True]:
            O_ref, P_ref, _ = FlashAttentionEngine.naive_attention(Q, K, V, causal=causal)
            O_fa, L_fa = FlashAttentionEngine.forward(Q, K, V, causal=causal, Br=16, Bc=16)

            dQ_ref, dK_ref, dV_ref = FlashAttentionEngine.naive_backward(
                dO, Q, K, V, O_ref, P_ref, causal=causal
            )
            dQ_fa, dK_fa, dV_fa = FlashAttentionEngine.backward(
                dO, Q, K, V, O_fa, L_fa, causal=causal, Br=16, Bc=16
            )

            err_dQ = np.max(np.abs(dQ_ref - dQ_fa))
            err_dK = np.max(np.abs(dK_ref - dK_fa))
            err_dV = np.max(np.abs(dV_ref - dV_fa))

            self.assertLess(err_dQ, 1e-10, f"dQ mismatch causal={causal}")
            self.assertLess(err_dK, 1e-10, f"dK mismatch causal={causal}")
            self.assertLess(err_dV, 1e-10, f"dV mismatch causal={causal}")

    def test_flash_decoding_split_kv(self):
        B, H, N_kv, d = 2, 4, 1024, 64
        Q = self.rng.randn(B, H, 1, d)
        K = self.rng.randn(B, H, N_kv, d)
        V = self.rng.randn(B, H, N_kv, d)

        O_ref, _, _ = FlashAttentionEngine.naive_attention(Q, K, V, causal=False)

        for splits in [2, 4, 8]:
            O_decode = FlashAttentionEngine.flash_decoding(Q, K, V, num_splits=splits)
            err = np.max(np.abs(O_ref - O_decode))
            self.assertLess(err, 1e-10, f"FlashDecoding mismatch with {splits} splits: {err}")

    def test_grouped_and_multi_query_attention(self):
        B, d, N = 1, 32, 64
        # GQA: 8 query heads, 2 KV heads
        Q = self.rng.randn(B, 8, N, d)
        K = self.rng.randn(B, 2, N, d)
        V = self.rng.randn(B, 2, N, d)

        O_gqa, _ = FlashAttentionEngine.forward(Q, K, V, causal=True, Br=16, Bc=16)
        K_rep = np.repeat(K, 4, axis=1)
        V_rep = np.repeat(V, 4, axis=1)
        O_ref, _, _ = FlashAttentionEngine.naive_attention(Q, K_rep, V_rep, causal=True)

        err_gqa = np.max(np.abs(O_gqa - O_ref))
        self.assertLess(err_gqa, 1e-10)

        # MQA: 4 query heads, 1 KV head
        Q_mqa = self.rng.randn(B, 4, N, d)
        K_mqa = self.rng.randn(B, 1, N, d)
        V_mqa = self.rng.randn(B, 1, N, d)
        O_mqa, _ = FlashAttentionEngine.forward(Q_mqa, K_mqa, V_mqa, causal=False, Br=16, Bc=16)
        K_rep_m = np.repeat(K_mqa, 4, axis=1)
        V_rep_m = np.repeat(V_mqa, 4, axis=1)
        O_ref_m, _, _ = FlashAttentionEngine.naive_attention(Q_mqa, K_rep_m, V_rep_m, causal=False)
        err_mqa = np.max(np.abs(O_mqa - O_ref_m))
        self.assertLess(err_mqa, 1e-10)

    def test_io_profile_scaling(self):
        prof_512 = FlashAttentionEngine.profile_io(512, d=64)
        prof_2048 = FlashAttentionEngine.profile_io(2048, d=64)

        # Check standard peak memory scales quadratically (16x for 4x length)
        self.assertAlmostEqual(
            prof_2048["standard_peak_mem_mb"] / prof_512["standard_peak_mem_mb"],
            16.0,
            places=2,
        )
        # Check flash peak memory stays constant (independent of N)
        self.assertEqual(
            prof_2048["flash_peak_mem_mb"], prof_512["flash_peak_mem_mb"]
        )

    def test_cli_subcommands(self):
        for cmd in ["verify", "demo", "bench", "decode"]:
            res = self.run_cli(cmd)
            self.assertEqual(res.returncode, 0, f"Command '{cmd}' failed: {res.stderr}")
            self.assertIn("FlashAttention-2", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
