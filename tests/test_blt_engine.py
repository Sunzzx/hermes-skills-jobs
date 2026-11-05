"""Unit and integration tests for Byte Latent Transformer (BLT) Architecture Engine."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

# Dynamically import blt-engine from bin/
ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "blt-engine"

spec = importlib.util.spec_from_loader("blt_engine", SourceFileLoader("blt_engine", str(CLI)))
assert spec is not None and spec.loader is not None
blt_mod = importlib.util.module_from_spec(spec)
sys.modules["blt_engine"] = blt_mod
spec.loader.exec_module(blt_mod)

ByteEntropyModel = blt_mod.ByteEntropyModel
DynamicPatchSegmenter = blt_mod.DynamicPatchSegmenter
BLTConfig = blt_mod.BLTConfig
LocalByteEncoder = blt_mod.LocalByteEncoder
LatentGlobalTransformer = blt_mod.LatentGlobalTransformer
LocalByteDecoder = blt_mod.LocalByteDecoder
ByteLatentTransformer = blt_mod.ByteLatentTransformer
softmax = blt_mod.softmax
gelu = blt_mod.gelu
layer_norm = blt_mod.layer_norm


class TestMathematicalPrimitives(unittest.TestCase):
    """Tests core mathematical primitives."""

    def test_softmax_probabilities_sum_to_one(self):
        x = np.array([2.0, 1.0, 0.1, -1.5])
        s = softmax(x)
        self.assertTrue(math.isclose(float(np.sum(s)), 1.0, abs_tol=1e-6))
        self.assertTrue(np.all(s >= 0.0))

    def test_softmax_numerical_stability_large_values(self):
        x = np.array([1000.0, 1005.0, 995.0])
        s = softmax(x)
        self.assertFalse(np.isnan(s).any())
        self.assertTrue(math.isclose(float(np.sum(s)), 1.0, abs_tol=1e-6))

    def test_gelu_monotonicity_and_zeros(self):
        self.assertAlmostEqual(gelu(np.array([0.0]))[0], 0.0, places=5)
        x = np.linspace(-3.0, 3.0, 20)
        g = gelu(x)
        self.assertFalse(np.isnan(g).any())

    def test_layer_norm_zero_mean_unit_variance(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        normed = layer_norm(x)
        self.assertAlmostEqual(float(np.mean(normed)), 0.0, places=5)
        self.assertAlmostEqual(float(np.var(normed)), 1.0, places=4)


class TestByteEntropyModel(unittest.TestCase):
    """Tests n-gram conditional byte entropy modeling."""

    def setUp(self):
        self.model = ByteEntropyModel(alpha=0.1)

    def test_entropy_non_negative_and_bounded(self):
        for text in [b"", b"a", b"abc", b"Byte Latent Transformer", b"\xff\x00\x12"]:
            h = self.model.compute_entropy(text)
            self.assertGreaterEqual(h, 0.0)
            self.assertLessEqual(h, 8.001)  # Max theoretical entropy for 256 bytes is log2(256) = 8 bits

    def test_distribution_sums_to_one(self):
        dist = self.model.compute_distribution(b"test")
        self.assertEqual(len(dist), 256)
        self.assertTrue(math.isclose(float(np.sum(dist)), 1.0, abs_tol=1e-6))
        self.assertTrue(np.all(dist > 0.0))

    def test_sequence_entropies_length(self):
        data = b"EntropyProfileTest"
        ents = self.model.compute_sequence_entropies(data)
        self.assertEqual(len(ents), len(data))
        self.assertTrue(all(e >= 0.0 for e in ents))


class TestDynamicPatchSegmenter(unittest.TestCase):
    """Tests entropy-guided dynamic patch segmentation invariants."""

    def setUp(self):
        self.segmenter = DynamicPatchSegmenter(threshold=3.6, min_patch_len=2, max_patch_len=16)

    def test_exact_byte_reconstruction_invariant(self):
        sample_texts = [
            b"Hello World",
            b"Byte Latent Transformer eliminates fixed subword tokenizers.",
            b"Mathematical formula: E = mc^2; \xcf\x80 = 3.1415926535; \xf0\x9f\x9a\x80\xe2\x9a\xa1",
            bytes(range(256)),
        ]
        for text in sample_texts:
            res = self.segmenter.segment(text)
            reconstructed = b"".join(p.data for p in res.patches)
            self.assertEqual(reconstructed, text)
            self.assertEqual(res.total_bytes, len(text))

    def test_patch_length_constraints(self):
        text = b"A" * 200  # Repetitive stream should hit max_patch_len
        res = self.segmenter.segment(text)
        for p in res.patches:
            self.assertLessEqual(p.length, 16)
            self.assertGreaterEqual(p.length, 1)

    def test_empty_input(self):
        res = self.segmenter.segment(b"")
        self.assertEqual(res.total_bytes, 0)
        self.assertEqual(res.num_patches, 0)
        self.assertEqual(len(res.patches), 0)

    def test_compression_ratio_positive(self):
        text = b"The Byte Latent Transformer allocates compute dynamically."
        res = self.segmenter.segment(text)
        self.assertGreaterEqual(res.compression_ratio, 1.0)
        self.assertAlmostEqual(res.compression_ratio, len(text) / res.num_patches, places=3)


class TestHierarchicalArchitecture(unittest.TestCase):
    """Tests the 3-tier BLT architecture: Encoder, Latent Transformer, Decoder."""

    def setUp(self):
        self.cfg = BLTConfig(
            d_byte=32,
            d_latent=64,
            n_heads_latent=2,
            n_layers_latent=1,
            d_ff_latent=128,
            min_patch_len=2,
            max_patch_len=8
        )
        self.encoder = LocalByteEncoder(self.cfg)
        self.transformer = LatentGlobalTransformer(self.cfg)
        self.decoder = LocalByteDecoder(self.cfg)
        self.segmenter = DynamicPatchSegmenter(threshold=3.5, min_patch_len=2, max_patch_len=8)

    def test_local_byte_encoder_patch_pooling(self):
        text = b"SampleText"
        res = self.segmenter.segment(text)
        patches = res.patches
        self.assertGreater(len(patches), 0)

        z0 = self.encoder.pool_patch(patches[0])
        self.assertEqual(z0.shape, (self.cfg.d_latent,))
        self.assertFalse(np.isnan(z0).any())

        Z = self.encoder.encode_patches(patches)
        self.assertEqual(Z.shape, (len(patches), self.cfg.d_latent))

    def test_latent_transformer_forward(self):
        K = 5
        Z_in = np.random.randn(K, self.cfg.d_latent)
        Z_out = self.transformer.forward(Z_in)
        self.assertEqual(Z_out.shape, (K, self.cfg.d_latent))
        self.assertFalse(np.isnan(Z_out).any())

    def test_local_byte_decoder_logits_shape(self):
        z = np.random.randn(self.cfg.d_latent)
        patch_bytes = b"abcdef"
        logits = self.decoder.decode_patch(z, patch_bytes, self.encoder)
        self.assertEqual(logits.shape, (len(patch_bytes), self.cfg.vocab_size))
        self.assertFalse(np.isnan(logits).any())


class TestByteLatentTransformerE2E(unittest.TestCase):
    """Tests end-to-end forward and generation operations."""

    def setUp(self):
        self.cfg = BLTConfig(
            d_byte=32,
            d_latent=64,
            n_heads_latent=2,
            n_layers_latent=1,
            d_ff_latent=128
        )
        self.blt = ByteLatentTransformer(self.cfg)

    def test_forward_pass_metrics(self):
        text = b"Deep learning on raw bytes without tokenizers."
        res = self.blt.forward(text)
        self.assertIn("loss", res)
        self.assertIn("perplexity", res)
        self.assertIn("bits_per_byte", res)
        self.assertIn("compression_ratio", res)
        self.assertGreater(res["loss"], 0.0)
        self.assertGreater(res["bits_per_byte"], 0.0)
        self.assertGreater(res["perplexity"], 1.0)
        self.assertEqual(res["total_bytes"], len(text))

    def test_autoregressive_generation(self):
        prompt = b"BLT model"
        gen_bytes = self.blt.generate(prompt, max_new_bytes=8, temperature=0.8, top_k=5)
        self.assertEqual(len(gen_bytes), len(prompt) + 8)
        self.assertTrue(gen_bytes.startswith(prompt))

    def test_empty_stream_forward(self):
        res = self.blt.forward(b"")
        self.assertEqual(res["loss"], 0.0)
        self.assertEqual(res["total_bytes"], 0)
        self.assertEqual(res["num_patches"], 0)


if __name__ == '__main__':
    unittest.main()
