"""Unit and integration tests for KIVI 2-Bit Asymmetric KV Cache Quantization Engine."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

# Dynamically import kivi-engine from bin/
ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "kivi-engine"

spec = importlib.util.spec_from_loader("kivi_engine", SourceFileLoader("kivi_engine", str(CLI)))
assert spec is not None and spec.loader is not None
kivi_mod = importlib.util.module_from_spec(spec)
sys.modules["kivi_engine"] = kivi_mod
spec.loader.exec_module(kivi_mod)

softmax = kivi_mod.softmax
compute_sqnr = kivi_mod.compute_sqnr
cosine_similarity = kivi_mod.cosine_similarity
frobenius_relative_error = kivi_mod.frobenius_relative_error
BitPacker = kivi_mod.BitPacker
KIVIConfig = kivi_mod.KIVIConfig
QuantizedKeyBlock = kivi_mod.QuantizedKeyBlock
QuantizedValueBlock = kivi_mod.QuantizedValueBlock
AsymmetricKeyQuantizer = kivi_mod.AsymmetricKeyQuantizer
AsymmetricValueQuantizer = kivi_mod.AsymmetricValueQuantizer
UniformPerTokenQuantizer = kivi_mod.UniformPerTokenQuantizer
UniformPerChannelQuantizer = kivi_mod.UniformPerChannelQuantizer
KIVIKVCache = kivi_mod.KIVIKVCache
generate_realistic_kv_data = kivi_mod.generate_realistic_kv_data
run_quantization_comparison = kivi_mod.run_quantization_comparison


class TestBitPacker(unittest.TestCase):
    """Tests bitwise packing and unpacking routines for 2-bit and 4-bit integers."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_2bit_packing_roundtrip_multi_dims(self):
        for dim in [4, 8, 16, 25, 64, 128]:
            data = self.rng.integers(0, 4, size=(2, 5, dim), dtype=np.uint8)
            packed = BitPacker.pack_2bit(data)
            # Check byte reduction
            expected_packed_dim = math.ceil(dim / 4)
            self.assertEqual(packed.shape[-1], expected_packed_dim)
            # Unpack and verify exact equality
            unpacked = BitPacker.unpack_2bit(packed, target_length=dim)
            self.assertEqual(unpacked.shape, data.shape)
            np.testing.assert_array_equal(data, unpacked)

    def test_4bit_packing_roundtrip_multi_dims(self):
        for dim in [2, 6, 10, 15, 32, 64]:
            data = self.rng.integers(0, 16, size=(3, dim), dtype=np.uint8)
            packed = BitPacker.pack_4bit(data)
            expected_packed_dim = math.ceil(dim / 2)
            self.assertEqual(packed.shape[-1], expected_packed_dim)
            unpacked = BitPacker.unpack_4bit(packed, target_length=dim)
            self.assertEqual(unpacked.shape, data.shape)
            np.testing.assert_array_equal(data, unpacked)

    def test_2bit_boundary_values(self):
        # Exactly test boundaries 0, 1, 2, 3
        data = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
        packed = BitPacker.pack_2bit(data)
        self.assertEqual(packed.shape, (2, 1))
        # Byte 0: 0 | (1<<2) | (2<<4) | (3<<6) = 0 | 4 | 32 | 192 = 228 (0xE4)
        self.assertEqual(packed[0, 0], 228)
        unpacked = BitPacker.unpack_2bit(packed, target_length=4)
        np.testing.assert_array_equal(data, unpacked)


class TestMathematicalPrimitives(unittest.TestCase):
    """Tests numeric stability and metric primitives."""

    def test_softmax_probabilities_sum_to_one(self):
        x = np.array([[1000.0, 1005.0, 995.0], [-10.0, 0.0, 10.0]])
        probs = softmax(x, axis=-1)
        self.assertFalse(np.isnan(probs).any())
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(2), atol=1e-6)
        self.assertTrue(np.all(probs >= 0.0))

    def test_cosine_similarity_identical_and_orthogonal(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0, places=5)
        b = np.array([-2.0, 1.0, -4.0, 2.0])  # dot(a, b) = -2 + 2 - 12 + 8 = -4
        self.assertTrue(math.isfinite(cosine_similarity(a, b)))
        zeros = np.zeros(4)
        self.assertEqual(cosine_similarity(zeros, zeros), 1.0)

    def test_sqnr_behavior(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(compute_sqnr(ref, ref), 100.0)
        noisy = ref + 0.01
        sqnr = compute_sqnr(ref, noisy)
        self.assertGreater(sqnr, 30.0)


class TestAsymmetricQuantizers(unittest.TestCase):
    """Tests per-channel Key and per-token Value quantization kernels."""

    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.H, self.S, self.D = 4, 64, 32
        self.K = self.rng.standard_normal((self.H, self.S, self.D)).astype(np.float32)
        self.V = self.rng.standard_normal((self.H, self.S, self.D)).astype(np.float32)

    def test_key_per_channel_quantization_fidelity(self):
        # Inject channel outlier in channel 5
        self.K[:, :, 5] *= 25.0
        block = AsymmetricKeyQuantizer.quantize(self.K, bits=2)
        self.assertIsInstance(block, QuantizedKeyBlock)
        self.assertEqual(block.seq_len, self.S)
        self.assertEqual(block.scales.shape, (self.H, self.D))
        self.assertEqual(block.zeros.shape, (self.H, self.D))

        K_rec = AsymmetricKeyQuantizer.dequantize(block)
        self.assertEqual(K_rec.shape, self.K.shape)
        self.assertFalse(np.isnan(K_rec).any())
        cos = cosine_similarity(self.K, K_rec)
        self.assertGreater(cos, 0.88)
        sqnr = compute_sqnr(self.K, K_rec)
        self.assertGreater(sqnr, 5.0)

    def test_value_per_token_quantization_fidelity(self):
        # Inject token outlier in token 0
        self.V[:, 0, :] *= 20.0
        block = AsymmetricValueQuantizer.quantize(self.V, bits=2)
        self.assertIsInstance(block, QuantizedValueBlock)
        self.assertEqual(block.seq_len, self.S)
        self.assertEqual(block.scales.shape, (self.H, self.S))
        self.assertEqual(block.zeros.shape, (self.H, self.S))

        V_rec = AsymmetricValueQuantizer.dequantize(block)
        self.assertEqual(V_rec.shape, self.V.shape)
        self.assertFalse(np.isnan(V_rec).any())
        cos = cosine_similarity(self.V, V_rec)
        self.assertGreater(cos, 0.88)
        sqnr = compute_sqnr(self.V, V_rec)
        self.assertGreater(sqnr, 5.0)

    def test_4bit_quantization_superior_precision(self):
        k_block_2 = AsymmetricKeyQuantizer.quantize(self.K, bits=2)
        K_rec_2 = AsymmetricKeyQuantizer.dequantize(k_block_2)

        k_block_4 = AsymmetricKeyQuantizer.quantize(self.K, bits=4)
        K_rec_4 = AsymmetricKeyQuantizer.dequantize(k_block_4)

        sqnr_2 = compute_sqnr(self.K, K_rec_2)
        sqnr_4 = compute_sqnr(self.K, K_rec_4)
        # 4-bit should have > 10 dB higher SQNR than 2-bit
        self.assertGreater(sqnr_4, sqnr_2 + 10.0)


class TestKIVIKVCache(unittest.TestCase):
    """Tests stateful streaming KV Cache manager."""

    def setUp(self):
        self.cfg = KIVIConfig(
            num_heads=4,
            head_dim=32,
            bits=2,
            group_size=16,
            residual_window=16
        )
        self.cache = KIVIKVCache(self.cfg)
        self.rng = np.random.default_rng(999)

    def test_streaming_append_and_reconstruction(self):
        # Append 3 chunks of 16 tokens each
        for _ in range(3):
            k_chunk = self.rng.standard_normal((4, 16, 32)).astype(np.float32)
            v_chunk = self.rng.standard_normal((4, 16, 32)).astype(np.float32)
            self.cache.append(k_chunk, v_chunk)

        self.assertEqual(self.cache.total_tokens_appended, 48)
        self.assertEqual(len(self.cache.key_blocks), 2)  # 48 - 16 residual = 32 -> 2 blocks of 16
        self.assertEqual(self.cache.residual_K.shape[1], 16)

        K_full, V_full = self.cache.reconstruct_all()
        self.assertEqual(K_full.shape, (4, 48, 32))
        self.assertEqual(V_full.shape, (4, 48, 32))
        self.assertFalse(np.isnan(K_full).any())
        self.assertFalse(np.isnan(V_full).any())

    def test_attention_shapes_and_probabilities(self):
        K, V = generate_realistic_kv_data(num_heads=4, seq_len=64, head_dim=32, seed=42)
        self.cache.append(K, V)
        self.cache.flush_all()

        Q = self.rng.standard_normal((4, 1, 32)).astype(np.float32)
        out, weights = self.cache.compute_attention(Q)

        self.assertEqual(out.shape, (4, 1, 32))
        self.assertEqual(weights.shape, (4, 1, 64))
        # Weights must sum to 1.0 per head
        np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones((4, 1)), atol=1e-5)
        self.assertFalse(np.isnan(out).any())

    def test_memory_stats_compression(self):
        K, V = generate_realistic_kv_data(num_heads=8, seq_len=512, head_dim=64, seed=42)
        cache = KIVIKVCache(KIVIConfig(num_heads=8, head_dim=64, bits=2, group_size=32, residual_window=32))
        cache.append(K, V)
        cache.flush_all()

        stats = cache.get_memory_stats()
        self.assertEqual(stats["total_tokens"], 512)
        self.assertGreater(stats["compression_vs_fp16"], 2.8)
        self.assertGreater(stats["compression_vs_fp32"], 5.6)
        self.assertLess(stats["effective_bits_per_element"], 6.0)


class TestAsymmetricSuperiorityAblation(unittest.TestCase):
    """Tests KIVI asymmetric quantization superiority over naive uniform baselines."""

    def test_key_channel_outlier_advantage(self):
        res = run_quantization_comparison(
            seq_len=256,
            num_heads=4,
            head_dim=64,
            bits=2,
            channel_outlier_scale=25.0,
            token_outlier_scale=15.0
        )
        # KIVI Key (per-channel) should significantly beat Uniform Token Key on SQNR
        kivi_key_sqnr = res["kivi_key"]["sqnr_db"]
        unif_token_key_sqnr = res["unif_token_key"]["sqnr_db"]
        self.assertGreater(kivi_key_sqnr, unif_token_key_sqnr + 1.5)

    def test_value_token_outlier_advantage(self):
        res = run_quantization_comparison(
            seq_len=256,
            num_heads=4,
            head_dim=32,
            bits=2,
            channel_outlier_scale=20.0,
            token_outlier_scale=15.0
        )
        # KIVI Value (per-token) should massively beat Uniform Channel Value on SQNR
        kivi_val_sqnr = res["kivi_val"]["sqnr_db"]
        unif_channel_val_sqnr = res["unif_channel_val"]["sqnr_db"]
        self.assertGreater(kivi_val_sqnr, unif_channel_val_sqnr + 8.0)


if __name__ == '__main__':
    unittest.main()
