'''Comprehensive Unit and Integration Test Suite for HNSW High-Dimensional Vector Search Engine.'''

from importlib.machinery import SourceFileLoader
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'bin' / 'hnsw-search'

loader = SourceFileLoader('hnsw_search', str(CLI))
hnsw_mod = types.ModuleType(loader.name)
loader.exec_module(hnsw_mod)


class HNSWTests(unittest.TestCase):
    def test_distance_metrics(self):
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        w = [1.0, 0.0, 0.0]

        # L2 squared
        l2_sq = hnsw_mod.DistanceMetric.squared_euclidean(u, v)
        self.assertAlmostEqual(l2_sq, 2.0, places=5)
        self.assertAlmostEqual(hnsw_mod.DistanceMetric.squared_euclidean(u, w), 0.0, places=5)

        # Euclidean
        self.assertAlmostEqual(hnsw_mod.DistanceMetric.euclidean(u, v), math.sqrt(2.0), places=5)

        # Cosine distance: orthogonal vectors have cosine distance 1.0
        cos_dist = hnsw_mod.DistanceMetric.cosine(u, v)
        self.assertAlmostEqual(cos_dist, 1.0, places=5)
        # identical vectors have cosine distance 0.0
        self.assertAlmostEqual(hnsw_mod.DistanceMetric.cosine(u, w), 0.0, places=5)

        # Manhattan distance
        self.assertAlmostEqual(hnsw_mod.DistanceMetric.manhattan(u, v), 2.0, places=5)

    def test_scalar_quantizer_sq8(self):
        dim = 16
        vectors = [[float(i + j * 0.1) for i in range(dim)] for j in range(20)]
        sq = hnsw_mod.ScalarQuantizerSQ8(dim=dim)
        sq.fit(vectors)

        sample = vectors[0]
        q_bytes = sq.quantize(sample)
        self.assertEqual(len(q_bytes), dim)
        self.assertIsInstance(q_bytes, bytes)

        deq = sq.dequantize(q_bytes)
        self.assertEqual(len(deq), dim)
        for orig, recon in zip(sample, deq):
            self.assertAlmostEqual(orig, recon, delta=0.1)

        # Asymmetric L2 squared
        adc_dist = sq.asymmetric_l2_squared(sample, q_bytes)
        self.assertLess(adc_dist, 0.5)

    def test_hnsw_index_insert_and_search(self):
        dim = 8
        index = hnsw_mod.HNSWIndex(dim=dim, metric='l2', m=8, ef_construction=32, ef_search=16, seed=42)
        
        # Insert 40 vectors
        vectors = []
        for i in range(40):
            vec = [float((i * 7 + j) % 13) for j in range(dim)]
            vectors.append(vec)
            index.insert(i, vec)

        self.assertEqual(len(index.nodes), 40)
        self.assertGreaterEqual(index.max_level, 0)

        # Search for exact vector inserted at index 5
        target_q = vectors[5]
        results, trace = index.search(target_q, k=3, trace=True)
        self.assertGreaterEqual(len(results), 1)
        # Nearest neighbor should be node 5 with distance approx 0
        best_node, best_dist = results[0]
        self.assertEqual(best_node, 5)
        self.assertAlmostEqual(best_dist, 0.0, places=4)
        self.assertIsNotNone(trace)
        self.assertIn('steps', trace)

    def test_hnsw_recall_against_brute_force(self):
        dim = 16
        vectors, queries = hnsw_mod.HNSWBenchmarkEngine.generate_clustered_dataset(n=100, dim=dim, seed=42)
        
        bf = hnsw_mod.BruteForceKNN(dim=dim, metric='cosine')
        hnsw = hnsw_mod.HNSWIndex(dim=dim, metric='cosine', m=8, ef_construction=50, ef_search=30, seed=42)

        for i, v in enumerate(vectors):
            bf.insert(i, v)
            hnsw.insert(i, v)

        k = 5
        recalls = []
        for q in queries[:10]:
            exact = set(item[0] for item in bf.search(q, k=k))
            ann = set(item[0] for item in hnsw.search(q, k=k)[0])
            recalls.append(len(exact.intersection(ann)) / float(k))

        avg_recall = sum(recalls) / len(recalls)
        self.assertGreaterEqual(avg_recall, 0.80)

    def test_save_and_load_index(self):
        dim = 4
        index = hnsw_mod.HNSWIndex(dim=dim, metric='cosine', m=4, ef_construction=16, seed=42)
        for i in range(10):
            index.insert(i, [float(i), 1.0, 2.0, float(i * 2)])

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            temp_path = tf.name

        try:
            index.save(temp_path)
            loaded = hnsw_mod.HNSWIndex.load(temp_path)
            self.assertEqual(loaded.dim, index.dim)
            self.assertEqual(loaded.metric_name, index.metric_name)
            self.assertEqual(len(loaded.nodes), len(index.nodes))
            self.assertEqual(loaded.entry_point_id, index.entry_point_id)

            # Query both and ensure identical results
            q = [1.0, 1.0, 2.0, 2.0]
            res_orig, _ = index.search(q, k=3)
            res_loaded, _ = loaded.search(q, k=3)
            self.assertEqual([r[0] for r in res_orig], [r[0] for r in res_loaded])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cli_subcommands(self):
        # 1. demo
        res_demo = subprocess.run([sys.executable, str(CLI), 'demo'], capture_output=True, text=True, check=False)
        self.assertEqual(res_demo.returncode, 0, res_demo.stderr)
        self.assertIn('HNSW MULTI-LAYER TRAJECTORY ROUTING DEMO', res_demo.stdout)

        # 2. quantize
        res_q = subprocess.run([sys.executable, str(CLI), 'quantize', '--n', '50', '--dim', '16'], capture_output=True, text=True, check=False)
        self.assertEqual(res_q.returncode, 0, res_q.stderr)
        self.assertIn('SQ8 Scalar Quantization validated successfully', res_q.stdout)

        # 3. bench
        res_b = subprocess.run([sys.executable, str(CLI), 'bench', '--n', '100', '--dim', '8', '--queries', '5'], capture_output=True, text=True, check=False)
        self.assertEqual(res_b.returncode, 0, res_b.stderr)
        self.assertIn('HNSW benchmark completed successfully', res_b.stdout)


if __name__ == '__main__':
    unittest.main()
