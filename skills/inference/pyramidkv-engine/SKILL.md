---
name: pyramidkv-engine
description: Dynamic Layer-Adaptive Hierarchical KV Cache Compression Engine
version: 1.0.0
category: inference
tags:
  - pyramidkv
  - kv-cache
  - layer-adaptive
  - attention-compression
  - llm-inference
  - memory-optimization
---

# PyramidKV: Dynamic Layer-Adaptive Hierarchical KV Cache Compression

Based on *"PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Flow"* (Zhang et al., 2024), `pyramidkv-engine` introduces a layer-adaptive hierarchical KV cache budgeting algorithm that mirrors the pyramidal attention information flow across transformer layers.

## Theoretical Basis

1. **Pyramidal Information Flow**: Empirical analysis of LLMs (LLaMA-3, Mistral, Qwen) reveals that lower layers capture broad contextual interactions across the entire sequence, requiring larger KV cache capacity, while deeper layers exhibit concentrated semantic abstractions and local focus, needing drastically fewer retained KV tokens.
2. **Dynamic Layer Budgeting**: Rather than enforcing uniform KV budgets across all layers ($B_{uniform} = B_{total} / L$), PyramidKV allocates budgets along a parameterized slope $\gamma$ from layer $0$ to layer $L-1$ ($B_0 > B_1 > \dots > B_{L-1}$), preserving critical global contexts where it matters most.
3. **Hybrid Per-Layer Selection**:
   - **Attention Sinks**: Preserves initial prompt tokens ($[0 : N_{sink}]$) across all layers to maintain softmax score normalization and prevent numerical divergence.
   - **Pooled Salient Features**: Applies 1D max pooling across observation window attention scores (or key norm proxies) to retain contiguous semantic chunk representations.
   - **Recent Local Window**: Preserves the rolling local tail tokens ($[-W :]$) for syntactic fluency.
4. **Performance**: Yields 4x–16x KV cache memory compression with negligible perplexity degradation and superior Needle-In-A-Haystack retrieval compared to uniform eviction strategies.

## CLI Usage

```bash
# Run self-verification and check layer budget calculations
pyramidkv-engine

# Run comprehensive compression benchmark across 16 layers
pyramidkv-engine benchmark --seq-len 1024 --layers 16 --total-budget 1024 --pyramid-slope 2.0

# Output structured JSON metrics for CI/CD integration
pyramidkv-engine benchmark --json
```

## Python Library Usage

```python
from pyramidkv_engine import PyramidKVConfig, PyramidKVEngine
import numpy as np

# Initialize 16-layer configuration with total budget of 1024 tokens
config = PyramidKVConfig(
    num_layers=16,
    num_heads=8,
    head_dim=64,
    total_budget=1024,
    pyramid_slope=2.0,
    num_sink_tokens=4,
    recent_window_size=16
)
engine = PyramidKVEngine(config)

print("Layer-specific budgets:", engine.layer_budgets)
# e.g., Layer 0 gets 85 tokens, Layer 15 gets 43 tokens

# Compress all layers simultaneously
compressed_keys, compressed_values, stats = engine.compress_all_layers(
    all_keys, all_values, all_queries
)
print(f"Overall Compression Ratio: {stats['overall_compression_ratio']}x")
print(f"Memory Saved: {stats['memory_saved_pct']}%")
```
