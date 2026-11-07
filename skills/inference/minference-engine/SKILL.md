---
name: minference-engine
description: "MInference 1.0 dynamic sparse attention engine: accelerates long-context pre-filling via head-wise A-Shape, Vertical-Slash, and Block-Sparse dynamic routing with up to 10x latency and FLOP reduction."
category: inference
tags: [minference, sparse-attention, long-context, prefill-acceleration, dynamic-sparse, llm-inference]
---

# MInference 1.0: Dynamic Sparse Attention Engine

Implements the breakthrough MInference (Microsoft Research / NeurIPS 2024) technique to break the quadratic $O(N^2)$ computation latency of the prompt pre-filling phase in long-context Large Language Models (32K to 1M+ tokens).

## Core Architecture

1. **Attention Pattern Classification**:
   - Classifies attention heads into three distinct topological classes:
     - **A-Shape**: High energy concentrated on initial prefix tokens (attention sinks) and the local causal sliding diagonal band.
     - **Vertical-Slash**: Salient vertical key columns (retrieval tokens consistently attended across time) combined with local causal bands.
     - **Block-Sparse**: Dispersed associative attention clusters resolved through coarse-to-fine block QK estimation.

2. **Dynamic Pattern Routing**:
   - Automatically identifies and profiles head patterns without model fine-tuning or distillation.
   - Dispatches each head to specialized sparse matrix kernels.

3. **Lossless / Bounded Numerical Fidelity**:
   - Retains >98% cosine similarity to full dense causal attention.
   - Achieves 2.5x to 10x effective FLOP reduction and memory traffic savings during long prompt evaluations.

## CLI Usage

```bash
# Self-test unit verification
minference-engine --test --json

# Benchmark pre-filling against dense causal baseline
minference-engine --benchmark --seq-len 512 --heads 4
```
