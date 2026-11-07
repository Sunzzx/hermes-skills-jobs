---
name: eagle2-engine
description: "EAGLE-2 dynamic lossless speculative sampling engine featuring context-aware dynamic draft tree expansion, 2D tree-structured causal attention verification, and multi-branch rejection sampling."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, SpeculativeDecoding, EAGLE-2, TreeAttention, LosslessSampling, InferenceAcceleration, LLM]
---

# EAGLE-2 Dynamic Lossless Speculative Sampling Engine

## Overview
**EAGLE-2** (Extrapolative Acceleration toward Generalized Large-scale Edge/Cloud Language Models) advances speculative decoding through feature-level auto-regression and confidence-aware dynamic draft trees:
1. **Dynamic Draft Tree Synthesis**: Rather than generating static sequences or fixed branching candidate lists, EAGLE-2 dynamically adjusts draft tree width and depth based on token confidence scores.
2. **2D Tree-Structured Attention Verification**: A single target model forward pass evaluates all tree candidates concurrently using exact ancestor causal attention masking.
3. **Lossless Rejection Sampling**: Implements the Leviathan / Chen rejection sampling theorem across multiple candidate branches, guaranteeing zero distributional shift compared to standard auto-regressive decoding.

## Key Capabilities
- **Confidence-Guided BFS Expansion**: Traverses candidate logits and dynamically branches only high-probability hypotheses, pruning low-confidence paths early.
- **Arbitrary 2D Tree Causal Masking**: Formulates exact adjacency matrices mapping tokens to ancestors, preventing future-leakage across alternative tree hypotheses.
- **Greedy / Temperature Speculative Sampling**: Unbiased target token distribution matching with residual distribution resampling on rollback.
- **Comprehensive Benchmarks & Verification Suite**: Validates output shapes, lack of NaNs, mathematical probability convergence, and speedup factors (typically 2.5x - 4.0x).

## CLI Usage

### Run Tree Speculation Benchmark
```bash
eagle2-engine benchmark --steps 50 --tree-size 16
```

### Formally Verify Mathematical Properties
```bash
eagle2-engine verify --tree-size 12
```

## Python API Integration
```python
import numpy as np
from eagle2_engine import EAGLE2Config, EAGLE2Engine

cfg = EAGLE2Config(max_tree_size=16, num_heads=8, head_dim=64)
engine = EAGLE2Engine(cfg)

# Run simulation with 50 generation steps
stats = engine.benchmark_tree_speculation(num_steps=50, tree_size=16)
print(f"Tokens Per Step: {stats['tokens_per_step']:.2f}")
print(f"Effective Speedup: {stats['speedup_factor']:.2f}x")
```
