---
name: streaming-llm-engine
description: Infinite context LLM streaming with Attention Sinks
version: 1.0.0
category: inference
tags:
  - streaming-llm
  - attention-sinks
  - kv-cache
  - infinite-context
  - llm-inference
---

# StreamingLLM: Attention Sinks for Infinite Sequence Streaming

Based on *"Efficient Streaming Language Models with Attention Sinks"* (Xiao et al., 2023), `streaming-llm-engine` solves the catastrophic collapse of LLMs when context exceeds the pre-training window during multi-turn or streaming conversations.

## Theoretical Basis

1. **Attention Sink Phenomenon**: Large language models dedicate an outsized portion of their attention mass to the initial tokens (first 1–4 tokens), regardless of semantic relevance, to act as a numerical "sink" for softmax score normalization.
2. **Catastrophic Failure of Sliding Window**: When initial tokens are evicted from a standard rolling KV cache, the softmax denominator explodes or collapses, causing perplexity to diverge to infinity.
3. **Attention Sink Solution**: Retain initial tokens ($[0 : N_{sink}]$) as static anchors in the KV cache, combined with a local rolling window of the latest tokens ($[-W :]$). This maintains stable softmax distribution without retraining or architectural modifications.

## Architecture & Capabilities

- **Fixed Memory Footprint**: Strictly caps KV cache memory at $O(N_{sink} + W)$, enabling unbounded text streaming.
- **Zero-Fine-Tuning**: Operates out-of-the-box on pre-trained causal models (LLaMA, Mistral, Falcon, Gemma, Qwen).
- **Seamless FIFO Eviction**: Auto-manages KV cache displacement at index $N_{sink}$ while holding the anchor sinks invariant.

## CLI Usage

```bash
# Run streaming simulation with 4 sink tokens and a 64-token local window
streaming-llm-engine simulate --stream-length 1024 --sink-tokens 4 --window-size 64

# Output structured JSON metrics
streaming-llm-engine simulate --stream-length 2048 --json
```

## Python Library Usage

```python
from streaming_llm_engine import AttentionSinkKVCache, StreamingLLMConfig

config = StreamingLLMConfig(
    num_sink_tokens=4,
    local_window_size=128,
    head_dim=64
)
cache = AttentionSinkKVCache(config)

for step, (k_token, v_token) in enumerate(token_stream):
    cache.update(k_token, v_token, pos=step)

# Query step
context, sink_fraction = cache.compute_attention(q_token)
stats = cache.stats()
print(f"Tokens seen: {stats['total_tokens_seen']}, Cached: {stats['cached_tokens']}")
```
