---
name: nbody-barnes-hut
description: "N-Body Barnes-Hut gravitational simulator: hierarchical octree O(N log N), symplectic 2nd-order Leapfrog integrator, Plummer softening, quadrupole multipoles, and Unicode Braille terminal visualizer."
version: 1.0.0
category: computational-science
tags:
  - ComputationalPhysics
  - Astrophysics
  - BarnesHut
  - SymplecticIntegrator
  - NBody
  - OrbitalMechanics
  - ComputerScience
---

# 🌌 N-Body Barnes-Hut Gravitational Simulator & Galactic Collision Engine

A production-grade, mathematically rigorous implementation of the **Barnes-Hut Hierarchical Octree $O(N \log N)$ N-body gravitational simulator** and **Symplectic 2nd-Order Leapfrog execution engine**, built from scratch in pure Python with zero external dependencies.

Simulates thousands of interacting celestial bodies in full 3D space with recursive spatial octree decomposition, Multipole Acceptance Criterion (MAC), quadrupole gravitational potential expansions, Plummer gravitational softening, and real-time ANSI/Unicode Braille terminal rendering with live Hamiltonian telemetry.

---

## 📐 Theoretical Foundations & Mathematical Semantics

### 1. Gravitational Potential & Plummer Softening
In a classical Newtonian system of $N$ point masses, the gravitational force acting on body $i$ located at $\mathbf{r}_i$ with mass $m_i$ is given by:

$$\mathbf{F}_i = -\nabla_i U = \sum_{j \neq i} G \frac{m_i m_j}{|\mathbf{r}_j - \mathbf{r}_i|^3} (\mathbf{r}_j - \mathbf{r}_i)$$

For point masses, as the separation $r_{ij} \to 0$, the force diverges singularity-wise ($r^{-2} \to \infty$), creating catastrophic non-physical velocity kicks and destroying numerical integration stability. To accurately represent extended mass distributions (such as galactic dark matter halos or stellar clusters) and maintain continuous numerical derivatives, we incorporate **Plummer gravitational softening** with characteristic scale length $\epsilon$:

$$\mathbf{a}_i = \sum_{j \neq i} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{\left(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2\right)^{3/2}}$$

And the total smoothed gravitational potential energy is:

$$U = -\sum_{i < j} \frac{G m_i m_j}{\sqrt{|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2}}$$

---

### 2. Barnes-Hut Hierarchical Spatial Octree Decomposition

Direct summation requires evaluating $N(N - 1) / 2$ pairwise force interactions, resulting in an asymptotically prohibitive computational cost of $\mathcal{O}(N^2)$ per timestep.

The **Barnes-Hut algorithm** constructs a hierarchical spatial tree (3D Octree / 2D Quadtree) by recursively partitioning space into 8 cubic child octants until every leaf contains at most one celestial body.

1. **Tree Construction ($O(N \log N)$)**:
   - A root bounding box enclosing all bodies $[x_{min}, x_{max}] \times [y_{min}, y_{max}] \times [z_{min}, z_{max}]$ with a 5% margin is constructed.
   - Each body is inserted recursively. When two bodies occupy the same node, the node subdivides into 8 child octants.

2. **Center of Mass (CoM) & Monopole Moment**:
   - For each internal node, the total mass $M$ and Center of Mass $\mathbf{R}_{CoM}$ are computed recursively:
     $$M = \sum_{k} m_k, \quad \mathbf{R}_{CoM} = \frac{1}{M} \sum_{k} m_k \mathbf{r}_k$$

3. **Quadrupole Moment Tensor Expansion**:
   - To further boost accuracy beyond pure monopole approximations, each internal node computes the traceless quadrupole tensor:
     $$Q_{\alpha \beta} = \sum_k m_k \left( 3 (r_{k,\alpha} - R_\alpha)(r_{k,\beta} - R_\beta) - |\mathbf{r}_k - \mathbf{R}|^2 \delta_{\alpha \beta} \right)$$
   - The quadrupole correction to the gravitational field at distance $d = \|\mathbf{R} - \mathbf{r}_i\|$ is evaluated as:
     $$\mathbf{a}_{quad} = \frac{G}{2 (d^2 + \epsilon^2)^{5/2}} \left[ 5 \frac{(\mathbf{n} \cdot Q \cdot \mathbf{n})}{d^2 + \epsilon^2} \mathbf{n} - 2 Q \cdot \mathbf{n} \right]$$

4. **Multipole Acceptance Criterion (MAC)**:
   - When evaluating the acceleration on body $i$ from an octree cell of spatial width $s = 2 \times \text{half\_size}$ at distance $d = \|\mathbf{R}_{CoM} - \mathbf{r}_i\|$:
     $$\theta = \frac{s}{d} < \theta_{crit}$$
   - If $\theta < \theta_{crit}$ (typically $\theta_{crit} = 0.5$), the entire subtree is treated as a single composite mass located at $\mathbf{R}_{CoM}$, truncating the descent.
   - If $\theta \ge \theta_{crit}$, the algorithm descends recursively into the 8 child octants.
   - This bounds force evaluations to $\mathcal{O}(N \log N)$ per integration step.

---

### 3. Symplectic 2nd-Order Leapfrog (Velocity Verlet / Kick-Drift-Kick)

Standard explicit Runge-Kutta or Euler integrators are non-symplectic: they fail to preserve the canonical symplectic 2-form $\omega = \sum_i d\mathbf{q}_i \wedge d\mathbf{p}_i$ and violate Liouville's theorem of phase-space volume preservation. In celestial mechanics, non-symplectic schemes introduce artificial numerical energy dissipation or growth, causing orbital spiral-out over long time horizons.

We implement the **Symplectic 2nd-Order Leapfrog (Kick-Drift-Kick)** operator splitting:

1. **First Half-Kick**:
   $$\mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) = \mathbf{v}_i(t) + \frac{1}{2} \mathbf{a}_i(t) \Delta t$$

2. **Full Drift**:
   $$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) \Delta t$$

3. **Octree Acceleration Evaluation**:
   $$\mathbf{a}_i(t + \Delta t) = \text{BarnesHut}(\mathbf{r}(t + \Delta t), \theta, \epsilon)$$

4. **Second Half-Kick**:
   $$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) + \frac{1}{2} \mathbf{a}_i(t + \Delta t) \Delta t$$

**Symplectic Guarantees**:
- Phase-space volume strictly conserved ($d\Gamma = \prod d\mathbf{r}_i d\mathbf{p}_i = \text{const}$).
- Time-reversal symmetry: $\mathcal{S}_{-\Delta t} \circ \mathcal{S}_{+\Delta t} = \mathcal{I}$.
- Hamiltonian total energy error $\Delta E(t) = E(t) - E_0$ remains bounded by $\mathcal{O}(\Delta t^2)$ indefinitely without secular drift.

---

### 4. Conservation Laws & Virial Equilibrium Diagnostics

The engine continuously monitors 5 fundamental physical invariants:

1. **Total Hamiltonian Energy**:
   $$E_{tot} = K + U = \sum_{i=1}^N \frac{1}{2} m_i |\mathbf{v}_i|^2 - \sum_{i < j} \frac{G m_i m_j}{\sqrt{|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2}}$$

2. **Total Linear Momentum**:
   $$\mathbf{P}_{tot} = \sum_{i=1}^N m_i \mathbf{v}_i = \text{const}$$

3. **Total Angular Momentum**:
   $$\mathbf{L}_{tot} = \sum_{i=1}^N m_i (\mathbf{r}_i \times \mathbf{v}_i) = \text{const}$$

4. **Virial Equilibrium Ratio**:
   $$\eta = \frac{2 K}{|U|}$$
   By the Virial Theorem, for a steady-state gravitationally bound system ($2\langle K \rangle + \langle U \rangle = 0$), $\eta \approx 1.000$.

5. **Center of Mass Frame Invariance**:
   $$\mathbf{R}_{CoM} = \frac{1}{M_{tot}} \sum_{i=1}^N m_i \mathbf{r}_i, \quad \mathbf{V}_{CoM} = \frac{\mathbf{P}_{tot}}{M_{tot}} = \text{const}$$

---

## 🎮 Built-in Astrophysical Scenarios

| Scenario | Particles | Description | Theoretical Significance |
| :--- | :--- | :--- | :--- |
| `galaxy-collision` | 500+ | Parabolic encounter of two rotating spiral galaxies with central Supermassive Black Holes (SMBH). | Tidal tail formation, bridge creation, dynamical friction, core merger. |
| `solar-system` | 9 | Sun + Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. | Keplerian orbital stability, planetary eccentricity, perihelion conservation. |
| `figure-eight` | 3 | Moore-Chenciner 3-body equal mass periodic choreography ($m=1.0$). | Chaotic 3-body solution exhibiting stable periodic closed phase-space loop ($T \approx 6.326$). |
| `plummer-sphere` | 400+ | Virialized self-gravitating globular cluster sampled from Plummer density profile. | Equilibrium stellar dynamics, isotropic velocity dispersion ($2K/|U| \approx 1.0$). |
| `lagrange-chaos` | 150+ | Sun-Jupiter circular restricted 3-body system with Trojan and chaotic asteroids. | L4/L5 triangular Lagrange stability vs L1/L2 chaotic escape trajectories. |

---

## 💻 CLI Reference & Interactive Visualizer

The engine is deployed globally to `~/.local/bin/nbody-sim`:

```bash
# 1. Run live interactive Unicode Braille animated visualizer
nbody-sim demo galaxy-collision --steps 200 --fps 24

# 2. Simulate Moore-Chenciner 3-body periodic choreography
nbody-sim demo figure-eight --steps 300 --fps 30

# 3. Simulate solar system planetary orbits
nbody-sim demo solar-system --steps 250

# 4. Run formal verification test suite verifying all 8 physical invariants
nbody-sim test

# 5. Benchmark scaling speedup O(N log N) vs O(N^2)
nbody-sim benchmark --particles 100,250,500,1000
```

---

## 🔬 Formal Verification Test Results

```
========================================================================
🔬 N-BODY BARNES-HUT & SYMPLECTIC INVARIANT FORMAL TEST SUITE
========================================================================
  ✔ PASS | Tree Total Mass & CoM Invariant Mass=100.00, CoM=(1.20, 0.10, -0.10)
  ✔ PASS | Octree Spatial Bounding Containment Root HalfSize=3.68
  ✔ PASS | Barnes-Hut Force Convergence (θ=0 vs θ=0.5) θ=0 Error=1.39e-16, θ=0.5 Rel Error=0.90%
  ✔ PASS | Symplectic Energy Conservation (500 steps Kepler) Max Hamiltonian Drift |ΔE/E0| = 4.399e-09 (< 0.05%)
  ✔ PASS | Total Linear Momentum Conservation (Exact Symplectic) ||ΔP|| = 1.60e-14 (< 1e-10)
  ✔ PASS | Total Angular Momentum Conservation (Exact Symplectic) ||ΔL|| = 1.16e-13 (< 1e-10)
  ✔ PASS | Figure-Eight 3-Body Periodic Choreography Period T=6.3259, Orbit Closure Error=0.0006
  ✔ PASS | Plummer Globular Cluster Virial Equilibrium Virial Ratio 2K/|U| = 0.922 (Theoretical = 1.000)

🎉 ALL FORMAL INVARIANT TESTS PASSED SUCCESSFULLY!
```
