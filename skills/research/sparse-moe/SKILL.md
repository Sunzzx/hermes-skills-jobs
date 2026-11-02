---
name: sparse-moe
description: "Sparse Mixture-of-Experts (MoE) Top-K Gating & Capacity-Constrained Routing Engine with load balancing loss, PaLM Z-loss, and DeepSeek-V3 shared expert partitioning."
category: research
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, MoE, MixtureOfExperts, DeepSeekV3, SwiGLU, LoadBalancing, RouterZLoss]
---

# ⚡ Sparse Mixture-of-Experts (MoE) Routing Engine

A mathematically rigorous, production-grade implementation of **Sparse Mixture-of-Experts (MoE)** routing, Top-K gating with Gumbel/Gaussian noise, auxiliary load balancing loss, PaLM/ST-MoE Router Z-loss, DeepSeek-V3 fine-grained expert partitioning, and capacity-constrained dispatch with token overflow mitigation.

---

## 📐 Core Theory & Mathematical Foundations

### 1. Top-K Softmax Gating with Gumbel/Gaussian Noise
Given token hidden state $x \in \mathbb{R}^{d_{model}}$ and gating weights $W_g \in \mathbb{R}^{d_{model} \times E}$:

$$H(x)_i = (x \cdot W_g)_i + \epsilon \cdot \text{Softplus}((x \cdot W_{noise})_i), \quad \epsilon \sim \mathcal{N}(0, 1)$$

The top-$k$ experts are selected:

$$\mathcal{T} = \text{TopK}(H(x), k)$$

The gating distribution is normalized over selected experts:

$$P(x)_i = \begin{cases} \frac{\exp(H(x)_i / T)}{\sum_{j \in \mathcal{T}} \exp(H(x)_j / T)} & \text{if } i \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases}$$

### 2. Auxiliary Load-Balancing Loss (Switch Transformer / Fedus et al.)
To prevent routing collapse where a small subset of experts receives all tokens while others starve:

- **Router Importance**: $m_i = \frac{1}{N} \sum_{x \in X} S(x)_i$, where $S(x)$ is full softmax over all $E$ experts.
- **Dispatch Fraction**: $f_i = \frac{1}{N \cdot k} \sum_{x \in X} \mathbb{I}(i \in \mathcal{T}(x))$.
- **Auxiliary Loss**:

$$\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^E m_i \cdot f_i$$

Under uniform routing, $m_i = f_i = 1/E$, achieving the theoretical minimum loss $\mathcal{L}_{aux}^* = \alpha$.

### 3. Router Z-Loss (ST-MoE / PaLM)
To prevent extreme gating logits that cause numerical instability and FP16/BF16 overflow:

$$\mathcal{L}_z = \frac{\beta}{N} \sum_{b=1}^N \left( \ln \sum_{i=1}^E e^{H(x_b)_i} \right)^2$$

### 4. Expert Capacity Factor & Overflow Residual Bypass
To maintain deterministic compute bounds in distributed execution:

$$C = \left\lceil \frac{N \times k}{E} \times \text{capacity\_factor} \right\rceil$$

When expert $e$ receives $> C$ tokens, overflow tokens bypass the expert via the residual identity stream:

$$y_t = x_t + \sum_{j \in \mathcal{T}(t), \text{count}(j) \le C} P(x_t)_j \cdot \text{Expert}_j(x_t)$$

### 5. DeepSeek-V3 Fine-Grained MoE & Shared Experts
Partitions the layer into $E_{shared}$ shared experts (active for all tokens) plus $E_{routed}$ specialized experts (selected via top-$k$):

$$\text{Output}(x) = x + \frac{1}{|E_{shared}|} \sum_{s \in E_{shared}} \text{Shared}_s(x) + \sum_{j \in \mathcal{T}} P(x)_j \cdot \text{Routed}_j(x)$$

---

## 🛠️ CLI Command Reference (`sparse-moe`)

```bash
# Verify formal mathematical invariants (Top-k sum-to-1, capacity overflow, z-loss, SwiGLU)
sparse-moe verify

# Route a batch of tokens and inspect expert utilization and loss metrics
sparse-moe route --tokens 128 --num-experts 8 --top-k 2 --capacity-factor 1.25

# Benchmark parameter scaling, active parameters, FLOPs, and throughput (tokens/sec)
sparse-moe benchmark --tokens 256 --d-model 64 --d-ffn 128

# Study load balancing dynamics and Gini inequality across auxiliary loss weights
sparse-moe balance --tokens 128 --num-experts 8 --top-k 2

# Run an end-to-end forward pass with SwiGLU activations and DeepSeek-V3 shared experts
sparse-moe simulate --tokens 32 --d-model 32 --num-experts 4 --shared-experts 1

# Export structured routing metadata for telemetry and monitoring
sparse-moe route --tokens 64 --json
```

---

## 💻 Python API Usage

```python
from sparse_moe import MoEConfig, SparseMoELayer, SyntheticDataGenerator

# Configure MoE layer
config = MoEConfig(
    d_model=64,
    d_ffn=128,
    num_experts=8,
    num_shared_experts=1,
    top_k=2,
    capacity_factor=1.25,
    aux_loss_weight=0.01,
    z_loss_weight=0.001,
)

# Initialize Layer
layer = SparseMoELayer(config)

# Generate synthetic token sequence
tokens = SyntheticDataGenerator.generate_clustered_tokens(num_tokens=32, d_model=64)

# Execute forward pass
outputs, routing_res = layer.forward(tokens, training=True)

print(f"Output shape: {len(outputs)} x {len(outputs[0])}")
print(f"Aux Loss: {routing_res.aux_loss:.6f}")
print(f"Z-Loss: {routing_res.z_loss:.6f}")
print(f"Dropped tokens: {routing_res.dropped_tokens}")
```
