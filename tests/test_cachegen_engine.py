#!/usr/bin/env python3
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin")))
from importlib.machinery import SourceFileLoader

cachegen_module = SourceFileLoader("cachegen_engine", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "cachegen-engine"))).load_module()

CacheGenQuantizer = cachegen_module.CacheGenQuantizer
CacheGenDeltaEncoder = cachegen_module.CacheGenDeltaEncoder
CacheGenBitstream = cachegen_module.CacheGenBitstream
CacheGenEngine = cachegen_module.CacheGenEngine


class TestCacheGenEngine(unittest.TestCase):
    def test_quantization_roundtrip(self):
        data = [0.12, -0.45, 0.89, 0.05, -0.22, 1.45]
        q, min_v, scale = CacheGenQuantizer.quantize_tensor(data, bits=4)
        self.assertEqual(len(q), len(data))
        self.assertTrue(all(0 <= x <= 15 for x in q))
        rec = CacheGenQuantizer.dequantize_tensor(q, min_v, scale)
        # Check max absolute error within quantization step bound
        step = (max(data) - min(data)) / 15.0
        for orig, recovered in zip(data, rec):
            self.assertAlmostEqual(orig, recovered, delta=step + 1e-4)

    def test_delta_encoding_roundtrip(self):
        tokens = [
            [10, 12, 14],
            [11, 12, 15],
            [13, 10, 15],
            [12, 11, 16],
        ]
        deltas = CacheGenDeltaEncoder.encode_deltas(tokens)
        self.assertEqual(deltas[0], tokens[0])
        self.assertEqual(deltas[1], [1, 0, 1])
        recovered = CacheGenDeltaEncoder.decode_deltas(deltas)
        self.assertEqual(recovered, tokens)

    def test_bitstream_pack_unpack(self):
        deltas = [[2, -1, 0], [0, 1, -2]]
        payload = CacheGenBitstream.pack_payload(deltas, min_val=-1.0, scale=0.1, bits=4, layer_id=3)
        layer_id, dec_deltas, min_v, sc, bits = CacheGenBitstream.unpack_payload(payload)
        self.assertEqual(layer_id, 3)
        self.assertEqual(dec_deltas, deltas)
        self.assertAlmostEqual(min_v, -1.0)
        self.assertAlmostEqual(sc, 0.1)
        self.assertEqual(bits, 4)

    def test_end_to_end_compression(self):
        engine = CacheGenEngine(num_layers=2, hidden_dim=32, quant_bits=4)
        metrics = engine.benchmark(seq_len=128)
        self.assertGreater(metrics["compression_ratio"], 3.0)
        self.assertGreater(metrics["bandwidth_reduction_pct"], 70.0)

if __name__ == "__main__":
    unittest.main()
