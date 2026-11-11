---
name: fastgen-engine
description: FastGen adaptive KV-cache compression via attention profiling (NeurIPS 2024 / arXiv:2310.01801).
category: inference
tags:
  - inference
  - kv-cache
  - attention-profiling
  - fastgen
  - neurips-2024
  - memory-efficiency
---

# FastGen: Adaptive KV Cache Compression via Attention Profiling (NeurIPS 2024)

FastGen is an adaptive KV cache compression architecture that dynamically classifies attention heads into distinct functional archetypes and allocates structural token retention budgets during LLM inference.

## Theoretical Foundations

Traditional KV cache pruning policies (like uniform sliding windows or standard H2O / SnapKV) apply identical eviction heuristics uniformly across all heads. However, attention heads exhibit sharply diverging roles:

1. **Special / Sink Heads**: Focus massive attention weight on initial prompt tokens (BOS, initial delimiter). FastGen preserves dedicated initial sink buffers + compact local history.
2. **Punctuation Heads**: Focus attention on semantic boundaries (periods, commas, line breaks). FastGen maintains indexed delimiter buffers.
3. **Local / Streaming Heads**: Exhibit strict diagonal locality. FastGen allocates bounded FIFO sliding windows.
4. **Retrieval / Context Heads**: Maintain diffuse, content-dependent long-range attention. FastGen applies selective top-K token retention.

## CLI Usage

```bash
# View theoretical budget allocation and savings model
fastgen-engine status --num-layers 32 --num-heads 32 --retrieval-ratio 0.25

# Run prefill and token-by-token generation benchmark
fastgen-engine benchmark --seq-len 1024 --decode-steps 32 --retrieval-ratio 0.25
```

## Architecture

- **Dynamic Classifier**: `FastGenKernel.classify_head()` computes sink mass, local mass, punctuation mass, and Shannon entropy.
- **Archetype Allocation**: Assigns specialized token retention policies (`SPECIAL_SINK`, `LOCAL_STREAMING`, `PUNCTUATION`, `RETRIEVAL_CONTEXT`).
- **Streaming Eviction**: Ensures zero memory spikes during long autoregressive decoding.
