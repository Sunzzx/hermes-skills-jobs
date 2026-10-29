---
name: flow-matching
description: "Production-grade Flow Matching & Continuous Normalizing Flows (CNF) engine: Optimal Transport displacement interpolants, Conditional Flow Matching (CFM) loss, neural vector field regression with sinusoidal time embeddings, adaptive Dormand-Prince RK45 & Euler numerical ODE integration, Liouville continuous change of variables log-likelihood, trajectory straightness metrics, and high-resolution interactive Unicode Braille phase-flow visualizer."
version: 1.0.0
category: research
tags:
  - FrontierAI
  - FlowMatching
  - ContinuousNormalizingFlows
  - OptimalTransport
  - GenerativeAI
  - ODENumerics
  - VectorFields
  - BrailleVisualizer
---

# 🌊 Flow Matching & Continuous Normalizing Flows (CNF) Engine

A production-grade, mathematically profound, zero-external-framework implementation of **Flow Matching** and **Continuous Normalizing Flows (CNF)** built from first principles in Python with NumPy. Models complex multi-modal data distributions through deterministic ODE probability paths pushed forward from a standard Gaussian prior along an Optimal Transport velocity vector field.

---

## 📐 Mathematical Formulation & Theoretical Foundations

### 1. Continuous Normalizing Flows & The Push-Forward Equation
A Continuous Normalizing Flow (CNF) parameterizes a continuous-time transformation of probability densities via an Ordinary Differential Equation (ODE):
$$\frac{d \mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t), \quad t \in [0, 1]$$

Integrating the vector field $\mathbf{v}_\theta$ from $t = 0$ to $t = 1$ establishes a diffeomorphism $\psi_t: \mathbb{R}^d \to \mathbb{R}^d$ pushing forward a tractable source base prior $p_0(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ to an empirical target distribution $p_1(\mathbf{x}) = q(\mathbf{x}_1)$:
$$\mathbf{x}_1 = \mathbf{x}_0 + \int_0^1 \mathbf{v}_\theta(\mathbf{x}_t, t) \, dt$$

---

### 2. Optimal Transport (OT) Displacement Interpolants
Traditional score-based diffusion models (DDPM, SDEs) rely on curved stochastic trajectories that require hundreds of noise injection steps. 

**Flow Matching** (Lipman et al., 2023; Albergo & Vanden-Eijnden, 2023) specifies deterministic, straight probability trajectories through **Optimal Transport (OT) Displacement Interpolants**:
$$\psi_t(\mathbf{x}_0, \mathbf{x}_1) = (1 - t) \mathbf{x}_0 + t \mathbf{x}_1$$

The conditional velocity field that generates this straight-line geodesic path is constant and time-independent along each sample pair:
$$\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{d}{dt} \psi_t(\mathbf{x}_0, \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0$$

---

### 3. Conditional Flow Matching (CFM) Objective & The Equivalence Theorem
Direct regression on the marginal target vector field $\mathbf{u}_t(\mathbf{x})$ is intractable because the marginal density $p_t(\mathbf{x})$ cannot be computed in closed form. 

The **Conditional Flow Matching Equivalence Theorem** establishes that optimizing the neural vector field $\mathbf{v}_\theta$ against the conditional vector field $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$ yields the exact same gradient with respect to $\theta$ as regressing against the intractable marginal field:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], \mathbf{x}_0 \sim p_0, \mathbf{x}_1 \sim q} \left[ \left\| \mathbf{v}_\theta((1 - t)\mathbf{x}_0 + t\mathbf{x}_1, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|_2^2 \right]$$

---

### 4. Continuous Change of Variables & Exact Log-Likelihood
By Liouville's continuity equation, the instantaneous rate of change of the log-probability density along an ODE trajectory is governed by the divergence of the vector field:
$$\frac{d}{dt} \log p_t(\mathbf{x}_t) = -\text{Tr}\left( \frac{\partial \mathbf{v}_\theta(\mathbf{x}_t, t)}{\partial \mathbf{x}_t} \right) = -\nabla \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$$

Integrating backward from $t = 1$ to $t = 0$ yields the exact continuous log-likelihood without Monte Carlo approximations:
$$\log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \nabla \cdot \mathbf{v}_\theta(\mathbf{x}_t, t) \, dt$$

---

### 5. Geodesic Straightness Metric
To quantify the efficiency of the learned transport paths, we evaluate the trajectory straightness ratio $\mathcal{S}$:
$$\mathcal{S} = \frac{\|\mathbf{x}(1) - \mathbf{x}(0)\|_2}{\sum_{k=0}^{N-1} \|\mathbf{x}(t_{k+1}) - \mathbf{x}(t_k)\|_2} \in (0, 1]$$
Where $\mathcal{S} = 1.0$ represents a perfectly straight Euclidean geodesic (zero path curvature).

---

## 🏛️ Architectural Implementation

1. **Zero External DL Frameworks**: Pure Python 3 + NumPy standard library.
2. **Sinusoidal Time Embedding**: Fourier projection of scalar $t \in [0, 1]$ into a 32-dimensional multi-frequency spectrum $\phi(t) = [\sin(\omega_k t), \cos(\omega_k t)]$.
3. **Deep Residual Vector Field**: Double residual blocks with Swish/SiLU non-linear activations ($\sigma(z) = z \cdot \text{sigmoid}(z)$).
4. **Exact Analytical Backpropagation**: Machine-precision gradient derivations for all network weight tensors and bias vectors.
5. **First-Principles Adam Optimizer**: Bias-corrected first and second moment tracking with decoupled weight decay.
6. **Numerical ODE Suite**:
   - Explicit Euler: 1 function evaluation per step (achieves high accuracy in 10-20 steps due to straight paths).
   - Midpoint (RK2): 2 evaluations per step.
   - Classical Runge-Kutta 4th Order (RK4): 4 evaluations per step.
   - Adaptive Dormand-Prince (RK45): Embedded 5th/4th order adaptive step-size control.
7. **High-Resolution Unicode Braille Visualizer**: Subpixel $2 \times 4$ dot projection (`0x2800`–`0x28FF`) rendering multi-panel flow transport and 2D vector field arrow quiver plots.

---

## 💻 CLI Commands & Usage

```bash
# 1. Run full mathematical & algorithmic certification suite
flow-matching test

# 2. Train continuous vector field on Swiss Roll (or rings, mog8, checkerboard)
flow-matching train --dataset swissroll --steps 500 --batch-size 128 --lr 0.003

# 3. Generate generative samples from Gaussian noise with RK4 solver
flow-matching sample --dataset swissroll --solver rk4 --steps 20 --num-samples 500

# 4. Interactive 4-panel terminal transport progression (Noise -> Flow -> Generated)
flow-matching visualize --dataset swissroll --flow-steps 12

# 5. Display 2D vector field flow directions & speed quiver heatmap
flow-matching quiver --dataset swissroll --t 0.5 --grid-size 18
```
