"""Unit tests for specinfer-tree-speculation: Tree-based Speculative Inference and Verification Engine."""

import os
import sys
from importlib.machinery import SourceFileLoader
import pytest

# Load executable script as Python module
bin_path = os.path.join(os.path.dirname(__file__), "..", "bin", "specinfer-tree-speculation")
specinfer_module = SourceFileLoader("specinfer_tree_speculation", bin_path).load_module()

SpeculativeTokenTree = specinfer_module.SpeculativeTokenTree
TreeNode = specinfer_module.TreeNode
build_synthetic_tree = specinfer_module.build_synthetic_tree


def test_tree_construction_and_ancestors():
    tree = SpeculativeTokenTree(root_token_id=101)
    assert len(tree.nodes) == 1
    assert tree.nodes[0].token_id == 101

    # Add children
    c1 = tree.add_token(parent_id=0, token_id=201, prob=0.9)
    c2 = tree.add_token(parent_id=0, token_id=202, prob=0.8)
    c3 = tree.add_token(parent_id=c1, token_id=301, prob=0.85)

    assert len(tree.nodes) == 4
    assert tree.get_ancestor_path(c3) == [0, c1, c3]
    assert tree.get_token_sequence(c3) == [101, 201, 301]
    assert tree.get_ancestor_path(c2) == [0, c2]
    assert tree.get_token_sequence(c2) == [101, 202]


def test_causal_tree_attention_mask_properties():
    tree = build_synthetic_tree(depth=2, branching_factor=2)
    # Root (0) -> (1, 2) -> (1->3,4; 2->5,6) -> total 7 nodes
    nodes, mask = tree.compute_tree_attention_mask()
    assert len(nodes) == 7
    assert len(mask) == 7
    assert len(mask[0]) == 7

    # Node 0 (root) can only attend to itself
    assert mask[0] == [1, 0, 0, 0, 0, 0, 0]

    # Node 3 is child of 1, which is child of 0
    # mask[3] should attend to 0, 1, 3
    idx_map = {nid: i for i, nid in enumerate(nodes)}
    i3 = idx_map[3]
    assert mask[i3][idx_map[0]] == 1
    assert mask[i3][idx_map[1]] == 1
    assert mask[i3][idx_map[3]] == 1
    # But cannot attend to sibling branch 2, 5, 6 or sibling 4
    assert mask[i3][idx_map[2]] == 0
    assert mask[i3][idx_map[4]] == 0
    assert mask[i3][idx_map[5]] == 0
    assert mask[i3][idx_map[6]] == 0


def test_tree_verification_optimal_path():
    tree = SpeculativeTokenTree(root_token_id=10)
    # Branch A: 10 -> 20 -> 30 (high prob)
    a1 = tree.add_token(0, 20, prob=0.95)
    a2 = tree.add_token(a1, 30, prob=0.92)

    # Branch B: 10 -> 40 -> 50 (lower prob)
    b1 = tree.add_token(0, 40, prob=0.40)
    b2 = tree.add_token(b1, 50, prob=0.20)

    # Target eval accepts branch A up to full depth, but rejects branch B at b1
    def target_eval(tokens):
        if tokens == [10, 20, 30]:
            return True, 3
        elif tokens == [10, 20]:
            return True, 2
        elif tokens == [10]:
            return True, 1
        elif 40 in tokens:
            return True, 1  # only root accepted
        return False, 0

    best_tokens, acc_count, best_path = tree.verify_paths(target_eval)
    assert best_tokens == [10, 20, 30]
    assert acc_count == 2  # 2 accepted speculative tokens
    assert best_path == [0, a1, a2]


def test_synthetic_tree_generation():
    tree = build_synthetic_tree(depth=3, branching_factor=2)
    # Depth 0: 1, Depth 1: 2, Depth 2: 4, Depth 3: 8 => Total 15 nodes
    assert len(tree.nodes) == 15
    data = tree.to_dict() if hasattr(tree, "to_dict") else tree.to_json()
    assert "total_nodes" in data
