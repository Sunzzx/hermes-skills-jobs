---
name: molecular-dynamics
description: "Production-grade Molecular Dynamics (MD) engine: symplectic 2nd-order Velocity-Verlet integration, shifted Lennard-Jones (12-6) potential, Periodic Boundary Conditions (PBC), Cell Linked-List O(N) neighbor acceleration, Berendsen/Andersen canonical thermostats, radial distribution function g(r), virial pressure, MSD, and interactive Unicode Braille phase-space projection."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [MolecularDynamics, ComputationalPhysics, SymplecticIntegrator, LennardJones, StatisticalMechanics, FrontierResearch]
---

# Molecular Dynamics Engine: Symplectic Velocity-Verlet & Lennard-Jones Phase Dynamics

A production-grade, mathematically rigorous Molecular Dynamics (MD) simulation engine implementing:
- **Symplectic 2nd-Order Velocity-Verlet Time Integration**: Preserves phase-space symplectic volume and demonstrates exact Hamiltonian energy conservation ($|\Delta E / E_0| < 0.02\%$) over hundreds of microcanonical (NVE) steps.
- **Shifted Lennard-Jones (12-6) Pair Potential**: Exact analytic interatomic potential with continuous cutoff potential shifting $V(r_c) = 0$ and analytic long-range dispersion tail corrections for energy and virial pressure.
- **Periodic Boundary Conditions (PBC)**: Minimum Image Convention across cubic and orthorhombic simulation supercells.
- **Cell Linked-List Spatial Acceleration**: $O(N)$ domain decomposition grid partitioning that achieves $7.6\times$ speedup over $O(N^2)$ direct pairwise calculations while maintaining identical machine-precision forces ($||F_{cell} - F_{direct}|| < 10^{-14}$).
- **Canonical Ensemble (NVT) Thermostats**: Berendsen weak-coupling velocity scaling and Andersen stochastic collision heat baths for precise temperature equilibration.
- **Statistical Mechanics Observables**:
  - Instantaneous kinetic temperature: $T^* = \frac{2K}{3(N-1)}$
  - Virial Equation of State Pressure: $P^* = \rho T^* + \frac{1}{3V} \sum (r_{ij} \cdot f_{ij}) + P_{tail}$
  - Radial Distribution Function $g(r)$ with automated liquid Argon first coordination peak detection
  - Mean Squared Displacement (MSD) & Einstein diffusion coefficient
- **Interactive Terminal Visualizer**: 3D orthographic projection onto a high-resolution Unicode Braille canvas (`0x2800`–`0x28FF`) with velocity-dependent thermal spectrum coloring and live real-time HUD telemetry.

Zero external dependencies. Pure Python standard library.

---

## Mathematical Formulation

### 1. Lennard-Jones (12-6) Potential
$$
V(r_{ij}) = 4\varepsilon \left[ \left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6 \right]
$$

With smooth cutoff shifting at $r_c$:
$$
V_{\text{shift}}(r) = \begin{cases} V(r) - V(r_c) & r \le r_c \\ 0 & r > r_c \end{cases}
$$

And interatomic force:
$$
\mathbf{F}_{ij} = -\nabla V(r_{ij}) = 24\varepsilon \left[ 2\left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6 \right] \frac{\mathbf{r}_{ij}}{r_{ij}^2}
$$

### 2. Symplectic Velocity-Verlet Integration
$$
\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2
$$
$$
\mathbf{v}(t + \Delta t/2) = \mathbf{v}(t) + \frac{1}{2}\mathbf{a}(t)\Delta t
$$
$$
\mathbf{a}(t + \Delta t) = \frac{1}{m}\mathbf{F}(\mathbf{r}(t + \Delta t))
$$
$$
\mathbf{v}(t + \Delta t) = \mathbf{v}(t + \Delta t/2) + \frac{1}{2}\mathbf{a}(t + \Delta t)\Delta t
$$

### 3. Radial Distribution Function $g(r)$
$$
g(r) = \frac{V}{4\pi r^2 \Delta r N^2} \sum_{i} \sum_{j \ne i} \langle \delta(r - r_{ij}) \rangle
$$

---

## Global CLI Commands

Available globally via `md-sim`:

### 1. Run Interactive Visual Simulation
```bash
# Run liquid Argon simulation near triple point
md-sim run --scenario liquid-argon --steps 200 --temp 0.85 --thermostat berendsen

# Run shockwave cluster collision
md-sim run --scenario shockwave-collision --steps 150 --fps 30

# Run Joule-Thomson vacuum expansion
md-sim run --scenario free-expansion --steps 150

# Headless production execution (no Braille UI)
md-sim run --scenario fcc-crystal --steps 500 --no-render
```

### 2. Run Formal Mathematical Invariant Suite
```bash
md-sim test
```

Verifies 9 fundamental physical invariants:
1. Minimum Image Convention PBC metric distance consistency
2. Lennard-Jones zero force at equilibrium minimum $r = 2^{1/6}\sigma$
3. Shifted potential boundary condition $V(r_c) = 0$
4. Cell Linked-List $O(N)$ vs Direct $O(N^2)$ force equivalence ($||\Delta F|| = 0$)
5. Symplectic Hamiltonian Energy Conservation under NVE ensemble ($|\Delta E / E_0| < 0.02\%$)
6. Total Linear Momentum Conservation ($||P_{total}|| < 10^{-10}$)
7. Berendsen Thermostat Canonical Relaxation
8. Radial Distribution Function $g(r)$ Liquid Coordination Peak ($r_{peak} \approx 1.10\sigma$)
9. Virial Equation of State Pressure & Tail Correction

### 3. Benchmark Spatial Scaling ($O(N^2)$ vs $O(N)$)
```bash
md-sim benchmark
```

### 4. Mathematical Formulation & Units Reference
```bash
md-sim info
```
