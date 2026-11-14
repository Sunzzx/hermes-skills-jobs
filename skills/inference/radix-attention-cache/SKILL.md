---
name: radix-attention-cache
description: "Radix Tree KV Cache Sharing Engine (SGLang / LMSYS) for automatic prefix KV cache reuse across multi-turn chats, few-shot prompts, and parallel speculative sampling."
category: inference
author: Sunzzx
added_at: 2026-11-14 09:48:00
---

# RadixAttention: Automatic KV Cache Sharing via Radix Trees

Implementation inspired by **SGLang: Efficient Execution of Structured Language Model Programs** (Zheng et al., LMSYS Org / UC Berkeley, arXiv:2312.07104).

Unlike traditional static FIFO KV cache managers, **RadixAttention** organizes key-value memory pages as an exact-prefix Radix Tree. During inference:
1. Common prefixes (system prompts, few-shot examples, multi-turn history) are shared across branches with zero recomputation.
2. When new branches fork, nodes automatically split into common prefix sub-edges.
3. When memory reaches hardware capacity, leaf nodes are evicted via Least Recently Used (LRU) policy.

## Architecture
- **Node Split & Coalescing**: Deterministic dynamic edge splitting when prefixes diverge.
- **Page-Aware Allocation**: Quantizes token blocks into GPU-sized memory pages (default: 16 tokens/page).
- **LRU Dynamic Eviction**: Recursive pruning of unreferenced leaves with parent-child edge coalescing.

## CLI Usage

```bash
# Run multi-turn and tree-of-thought branch reuse benchmark
radix-attention-cache benchmark --page-size 16 --max-pages 256

# Calculate theoretical memory footprint and prefill TFLOP savings for LLaMA-style 8B/70B models
radix-attention-cache stats --model-dim 4096 --num-layers 32 --kv-heads 8 --seq-len 4096
```
