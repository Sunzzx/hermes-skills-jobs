---
name: specinfer-tree-speculation
description: "Tree-based Speculative Inference and Verification Engine (SpecInfer / ASPLOS 2024) with parallel multi-draft tree expansion and single-pass causal tree attention verification."
category: inference
author: Sunzzx
added_at: 2026-11-14 18:55:00
---

# SpecInfer: Tree-based Speculative Inference Engine

Implementation inspired by **SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification** (Miao et al., ASPLOS 2024, arXiv:2305.01858).

Standard speculative decoding explores only a single candidate draft sequence per step. If the first token diverges or gets rejected, the entire speculative roll-out is wasted. **SpecInfer** overcomes this bottleneck by organizing candidate token predictions into a **speculative token tree** combined with **single-pass causal tree attention verification** on the target LLM.

## Core Innovations

1. **Multi-Candidate Speculative Token Tree**:
   - Explores multiple diverse branch trajectories from small draft model(s) concurrently.
   - Token candidates share common prefix nodes, maximizing candidate density per verification step.

2. **Custom Causal Tree Attention Mask**:
   - Verifies all candidate nodes of the speculative tree in a single forward pass of the target LLM.
   - Attention is permitted from child token $j$ to ancestor token $i$ if and only if $i$ lies on the unique prefix path from root to $j$.
   - Completely eliminates cross-talk across different speculative branches.

3. **Optimal Path Selection & Verification**:
   - Verifies target acceptance criteria across the tree using greedy or stochastic acceptance rules.
   - Retains the longest accepted path, amortizing target model memory bandwidth and prefill overhead.

## CLI Usage

```bash
# Run SpecInfer speculative tree generation and verification benchmark
specinfer-tree-speculation benchmark --branching-factor 3 --depth 4

# Inspect the causal tree attention mask matrix for a multi-branch candidate tree
specinfer-tree-speculation mask --depth 3 --branching-factor 2

# Analyze theoretical speedup and token throughput gains
specinfer-tree-speculation stats --acceptance-rate 0.75 --tree-size 15
```
