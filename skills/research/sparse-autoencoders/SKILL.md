---
name: sparse-autoencoders
description: "Sparse Autoencoders (SAE) & Mechanistic Interpretability Engine: Top-K sparsity, polysemanticity disentanglement, dead latent resampling, causal feature steering, and monosemanticity analysis."
version: 1.0.0
category: research
tags:
  - Frontier-AI
  - Mechanistic-Interpretability
  - Sparse-Autoencoders
  - SAE
  - Superposition
  - Feature-Steering
  - Neural-Activation-Analysis
---

# 🔬 Sparse Autoencoders (SAE) & Mechanistic Interpretability Engine

A production-grade, mathematically rigorous, zero-dependency **Sparse Autoencoder (SAE)** and **Mechanistic Interpretability Engine** built from first principles in Python with NumPy. Disentangles neural network activations trapped in superposition into human-interpretable, monosemantic concept directions using **Top-K sparse dictionary learning**, **dead latent residual resampling**, **Adam optimization**, and **causal feature steering**.

---

## 📐 Mathematical Formulation & Theoretical AI Research Foundations

### 1. The Linear Representation Hypothesis & Polysemantic Superposition
Under the **Linear Representation Hypothesis**, neural representations correspond to directions in activation space $\mathbb{R}^d$. When an artificial or biological neural network must represent $M$ distinct human-interpretable concepts where $M \gg d$ (e.g. $M = 10^5, d = 1024$), the model exploits the almost-orthogonal geometry of high-dimensional Euclidean space (governed by the Johnson-Lindenstrauss lemma) to pack features in **Superposition**:
$$x = \sum_{i=1}^M f_i(x) \cdot v_i + \epsilon$$
where:
- $v_i \in \mathbb{R}^d$ is the ground-truth concept direction ($\|v_i\|_2 = 1$).
- $f_i(x) \ge 0$ is the sparse concept activation (most $f_i(x) = 0$ for any single token).
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ represents cross-feature interference noise.

Because $M > d$, basis vectors along individual neurons are linear combinations of multiple unrelated concepts—yielding **polysemantic neurons**.

---

### 2. Top-K Sparse Dictionary Learning Architecture
Standard SAEs penalize activations via an $\mathcal{L}_1$ sparsity regularizer $\lambda \|f(x)\|_1$, which inevitably induces **shrinkage distortion** (artificially compressing active feature magnitudes) and requires delicate hyperparameter tuning.

The **Top-K Sparse Autoencoder** (Gao et al., 2024; Rajamanoharan et al., DeepMind 2024) solves this by enforcing exact $k$-sparsity directly through the activation function:

#### Encoder:
$$z(x) = (x - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}}, \quad z \in \mathbb{R}^M$$
$$f(x) = \text{TopK}(\text{ReLU}(z(x)), k)$$

#### Decoder:
$$\hat{x} = f(x) W_{\text{dec}} + b_{\text{dec}}$$
where $W_{\text{dec}} \in \mathbb{R}^{M \times d}$ and $b_{\text{dec}} \in \mathbb{R}^d$.

#### Loss Objective:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot d} \sum_{b=1}^B \|x^{(b)} - \hat{x}^{(b)}\|_2^2$$
$$\text{FVU} = \frac{\sum_b \|x^{(b)} - \hat{x}^{(b)}\|_2^2}{\sum_b \|x^{(b)} - \bar{x}\|_2^2}$$

---

### 3. Unit Hypersphere Decoder Normalization
To prevent scale degeneracy where decoder columns shrink $\|W_{\text{dec}}[i, :]\|_2 \to 0$ and encoder activations artificially expand $f_i(x) \to \infty$, every decoder feature vector is strictly normalized after each optimizer step:
$$W_{\text{dec}}[i, :] \longleftarrow \frac{W_{\text{dec}}[i, :]}{\max(\|W_{\text{dec}}[i, :]\|_2, 10^{-12})}$$

---

### 4. Dead Latent Detection & Residual Resampling
In overcomplete dictionaries ($M/d \ge 4$), features initialized far from the data manifold may never fire, receiving zero gradient:
$$\text{FiringRate}_i = \frac{1}{T} \sum_{t=1}^T \mathbf{1}_{f_i(x_t) > 0} < \tau_{\text{dead}} \quad (\tau_{\text{dead}} = 10^{-5})$$

To reclaim dormant capacity, residual resampling computes high-loss error residuals:
$$e = x - \hat{x}$$
Sampling inputs proportional to residual norm $\|e\|_2$, dead feature directions are reinitialized directly along the residual unit direction:
$$W_{\text{dec}}[i_{\text{dead}}, :] \longleftarrow \frac{e}{\|e\|_2}, \quad W_{\text{enc}}[:, i_{\text{dead}}] \longleftarrow \frac{e}{2 \|e\|_2}, \quad b_{\text{enc}}[i_{\text{dead}}] \longleftarrow 0$$

---

### 5. Causal Activation Steering
Given an activation vector $x$, intervenes counterfactually on specific discovered latents by modifying feature coefficients prior to decoding:
$$f_{\text{steered}}(x)_i = f(x)_i + \alpha$$
$$x_{\text{steered}} = f_{\text{steered}}(x) W_{\text{dec}} + b_{\text{dec}}$$
Displacement norm and cosine divergence quantify causal steerability:
$$\Delta x = \|x_{\text{steered}} - x\|_2, \quad \cos(x, x_{\text{steered}}) = \frac{x \cdot x_{\text{steered}}}{\|x\|_2 \|x_{\text{steered}}\|_2}$$

---

## 🛠️ CLI Quickstart & Usage Reference

The CLI is installed globally as `sae-lens`:

### 1. Run Automated Verification Test Matrix
```bash
sae-lens verify
```

### 2. Train Top-K SAE on Superposition Activations
```bash
sae-lens train --d-model 64 --n-dict 256 --k-top 6 --steps 1000 --batch-size 128
```

### 3. Inspect Discovered Monosemantic Features
```bash
sae-lens inspect --limit 10
```

### 4. Demonstrate Causal Activation Steering
```bash
sae-lens steer
```

### 5. Benchmark Forward/Backward Throughput
```bash
sae-lens benchmark
```

### 6. Theoretical Formulations & Reference
```bash
sae-lens info
```
