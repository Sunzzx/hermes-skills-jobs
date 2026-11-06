"""Unit and integration tests for DeepSeek-V3 / vLLM EPLB Load Balancer."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "eplb-moe-router"

loader = SourceFileLoader("eplb_moe_router", str(BIN_PATH))
spec = importlib.util.spec_from_loader("eplb_moe_router", loader)
assert spec is not None and spec.loader is not None
eplb_mod = importlib.util.module_from_spec(spec)
sys.modules["eplb_moe_router"] = eplb_mod
loader.exec_module(eplb_mod)

EPLBLoadBalancer = eplb_mod.EPLBLoadBalancer
DynamicMoEDispatcher = eplb_mod.DynamicMoEDispatcher


class TestEPLBMoERouter(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_replication_quota_allocation(self):
        num_experts = 32
        num_nodes = 2
        gpus_per_node = 4
        total_slots = 48
        balancer = EPLBLoadBalancer(
            num_experts=num_experts,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            total_slots=total_slots,
        )
        skewed_loads = np.linspace(1.0, 100.0, num_experts)
        factors, replicated = balancer.compute_replication_factors(skewed_loads)

        self.assertEqual(len(factors), num_experts)
        self.assertEqual(np.sum(factors), total_slots)
        self.assertTrue(all(f >= 1 for f in factors))
        self.assertTrue(len(replicated) > 0)
        self.assertGreater(factors[-1], factors[0])

    def test_placement_capacity_and_coverage(self):
        num_experts = 64
        num_nodes = 2
        gpus_per_node = 4
        total_slots = 80
        balancer = EPLBLoadBalancer(
            num_experts=num_experts,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            total_slots=total_slots,
        )
        plan = balancer.balance_and_place()

        for exp_id in range(num_experts):
            self.assertIn(exp_id, plan.expert_to_gpus)
            self.assertGreaterEqual(len(plan.expert_to_gpus[exp_id]), 1)

        for gid, exp_list in plan.gpu_to_experts.items():
            self.assertEqual(len(exp_list), balancer.slots_per_gpu)

        self.assertLess(plan.load_imbalance_ratio, 1.5)

    def test_dynamic_token_dispatch(self):
        num_experts = 64
        balancer = EPLBLoadBalancer(num_experts=num_experts, num_nodes=2, gpus_per_node=4, total_slots=80)
        plan = balancer.balance_and_place()
        dispatcher = DynamicMoEDispatcher(plan)

        num_tokens = 512
        top_k = 4
        result = dispatcher.simulate_routing(num_tokens=num_tokens, top_k=top_k)

        self.assertEqual(result.num_tokens, num_tokens)
        self.assertEqual(result.top_k, top_k)
        self.assertEqual(result.total_dispatched_tokens, num_tokens * top_k)
        self.assertEqual(sum(result.gpu_token_counts.values()), num_tokens * top_k)
        self.assertEqual(result.dropped_tokens, 0)
        self.assertGreaterEqual(result.cross_node_transfers, 0)
        self.assertGreaterEqual(result.intra_node_transfers, 0)
        self.assertLessEqual(result.cross_node_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
