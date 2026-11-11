---
name: zipcache-engine
description: "ZipCache: Accurate and Efficient KV Cache Compression via Channel-Wise Discrepancy and Dimension-Specific Non-Uniform Quantization."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
category: inference
tags:
  - kv-cache
  - quantization
  - zipcache
  - llm-inference
---

# ZipCache Engine

ZipCache introduces accurate and efficient KV cache compression by exploiting channel-wise sensitivity discrepancy and dimension-specific non-uniform quantization.

## Key Architecture
1. **Channel Sensitivity Analysis**: Key and Value dimensions exhibit highly non-uniform sensitivity to quantization error. ZipCache dynamically computes channel variance and isolates sensitive channels.
2. **Dimension-Specific Bit Allocation**: Preserves critical outlier channels at 4-bit resolution while compressing non-sensitive channels to 2-bit packing.
3. **BitPacker Engine**: Vectorized bit-level packing and unpacking for high-throughput inference.

## CLI Commands
```bash
# Status check
zipcache-engine status

# Benchmark across context lengths
zipcache-engine bench --seq-len 4096 --gen-tokens 16
```
