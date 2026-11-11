---
name: cachegen-engine
description: Fast streaming KV cache compression, delta-encoding, and context migration for distributed LLM serving.
category: inference
type: skill+tool
version: 1.0.0
entry_point: bin/cachegen-engine
---

# CacheGen Engine

Production implementation of **CacheGen** (SIGCOMM / arXiv:2310.07240), an ultra-fast streaming KV cache compression engine tailored for LLM context migration and distributed inference.

## Architecture Overview

- **Adaptive Dynamic Quantization:** Dynamic bit-width token tensor quantization (4-bit default) preserving attention distribution precision.
- **Sequential Delta Residual Encoding:** Exploits temporal and cross-token correlation by computing residual deltas across successive token positions.
- **Compact Compressed Bitstream:** Binary serialization with magic headers, metadata, and zlib/entropy-packed deltas reducing memory footprint by >75%.

## CLI Usage

```bash
# Run synthetic KV cache compression benchmark
bin/cachegen-engine bench --seq-len 512 --layers 4 --dim 64 --bits 4
```

## Python API

```python
from cachegen_engine import CacheGenEngine

engine = CacheGenEngine(num_layers=4, hidden_dim=64, quant_bits=4)
bench_results = engine.benchmark(seq_len=512)
print(bench_results)
```
