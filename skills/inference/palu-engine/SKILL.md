---
name: palu-engine
description: Palu Low-Rank KV-Cache Compression Engine (ICLR 2025)
category: inference
version: 1.0.0
tags: [llm, inference, kv-cache, palu, low-rank, svd, latent-projection, memory-efficiency]
cli: palu-engine
---

# Palu: KV-Cache Compression with Low-Rank Projection

## Overview

**Palu** (ICLR 2025 / arXiv:2407.21118) provides post-training, low-rank projection compression for Large Language Model (LLM) Key and Value caches. By projecting high-dimensional hidden representations ($d$) into low-rank latent subspaces ($r \ll d$) via offline SVD factorization, Palu achieves **50% to 75% KV cache memory reduction** while eliminating costly memory bandwidth bottlenecks during autoregressive decoding.

### Key Capabilities:
1. **Decoupled Low-Rank Caching**: Compresses and stores KV states directly in low-rank latent representation ($r$), bypassing full-dimensional tensor allocation.
2. **Instant Latent Recomposition**: Projects latent states back or computes low-rank attention factorizations without degrading factual recall.
3. **Post-Training / Zero Fine-Tuning**: Compatible with pre-trained LLM checkpoints out-of-the-box using offline weight/covariance SVD decomposition.
4. **Scalable Decoding Throughput**: Halves memory bandwidth footprint per decoding step, enabling 2x larger batch sizes and extended context windows.

---

## CLI Usage

```bash
# Inspect theoretical memory savings model
palu-engine status --rank-ratio 0.5 --head-dim 128

# Run prefill and token-by-token generation benchmark
palu-engine benchmark --seq-len 1024 --decode-steps 32 --rank-ratio 0.5
```

---

## Python API

```python
from importlib.machinery import SourceFileLoader
import os

bin_path = os.path.expanduser("~/.local/bin/palu-engine")
palu_mod = SourceFileLoader("palu_engine", bin_path).load_module()

config = palu_mod.PaluConfig(
    num_layers=4,
    num_heads=8,
    head_dim=64,
    rank_ratio=0.5  # Compresses head dimension from 64 to 32
)
engine = palu_mod.PaluEngine(config)

# Prefill prompt tokens
stats = engine.prefill(seq_len=512)
print(f"Memory Savings: {stats['memory_savings_pct']}% | Compression Ratio: {stats['compression_ratio']}x")

# Autoregressive decoding step
step_info = engine.decode_step()
print(f"Step Latency: {step_info['step_latency_us']} us")
```
