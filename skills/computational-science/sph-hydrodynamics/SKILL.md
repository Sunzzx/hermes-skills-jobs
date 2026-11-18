---
name: sph-hydrodynamics
description: "2D SPH Lagrangian fluid engine (WCSPH, Tait EOS, Monaghan)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [ComputationalPhysics, FluidDynamics, SPH, Lagrangian, CFD, WCSPH, Monaghan]
    related_skills: [lbm-cfd, navier-stokes-2d, nbody-barnes-hut]
---

# SPH 2D Smoothed Particle Hydrodynamics Fluid Engine

Lagrangian weakly-compressible SPH (Gingold & Monaghan 1977; Monaghan 1992/2005) reference: cubic-spline + Wendland C2 kernels, density summation, Tait EOS, Monaghan artificial viscosity, XSPH, leapfrog integration, spatial-hash neighbours, dam-break / hydrostatic / droplet / shear scenarios.

Complements grid LBM (`lbm-cfd`) and Eulerian NS (`navier-stokes-2d`) with a mesh-free particle formulation.

## When to Use

- Mesh-free free-surface flow (dam-break, droplets, sloshing) where grid CFD struggles with interfaces
- Teaching/validating WCSPH kernels, EOS, artificial viscosity, and CFL limits
- Comparing cubic-spline vs Wendland C2 partition-of-unity and stability
- Estimating SPH particle count, neighbour degree, and memory before a production port

Don't use for: 3D production CFD, incompressible projection SPH (IISPH/DFSPH), or continuum solid FEM.

## Theoretical Foundations

**Kernel interpolation.** Field $A(\mathbf{x}) \approx \sum_j A_j V_j W(|\mathbf{x}-\mathbf{x}_j|,h)$ with compact support $\sim 2h$.

**Density summation.** $\rho_i = \sum_j m_j W_{ij}$ (default) or continuity $\mathrm{d}\rho_i/\mathrm{d}t = \sum_j m_j (\mathbf{v}_i-\mathbf{v}_j)\cdot\nabla W_{ij}$.

**Tait EOS (γ=7).** $p = \frac{c_0^2\rho_0}{\gamma}\big[(\rho/\rho_0)^\gamma - 1\big]$ so $p(\rho_0)=0$; choose $c_0 \gtrsim 10\,|\mathbf{v}|_{\max}$ to keep $\delta\rho/\rho\lesssim 1\%$.

**Momentum (symmetric).**
$$\frac{\mathrm{d}\mathbf{v}_i}{\mathrm{d}t} = -\sum_j m_j\Big(\frac{p_i}{\rho_i^2}+\frac{p_j}{\rho_j^2}+\Pi_{ij}\Big)\nabla_i W_{ij} + \mathbf{g}$$
with Monaghan artificial viscosity $\Pi_{ij}$ active only for approaching pairs ($\mathbf{v}_{ij}\cdot\mathbf{x}_{ij}<0$).

**XSPH.** $\mathbf{v}_i \leftarrow \mathbf{v}_i + \varepsilon\sum_j \frac{m_j}{\bar\rho}(\mathbf{v}_j-\mathbf{v}_i)W_{ij}$.

**CFL.** $\Delta t \le \mathrm{CFL}\cdot h / (c_0 + |\mathbf{v}|_{\max})$.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy required)

## How to Run

```bash
terminal(command="sph-hydrodynamics selftest", timeout=60)
terminal(command="sph-hydrodynamics bench --scenario dam_break --steps 40", timeout=60)
terminal(command="sph-hydrodynamics compare --scenario hydrostatic --steps 30", timeout=60)
terminal(command="sph-hydrodynamics run --scenario dam_break --steps 60 --ascii --verbose", timeout=90)
terminal(command="sph-hydrodynamics kernels", timeout=30)
terminal(command="sph-hydrodynamics theory --n 10000 --json", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `sph-hydrodynamics selftest` | CI gate (10 invariant groups) |
| `sph-hydrodynamics bench` | Throughput + mass/energy/density report |
| `sph-hydrodynamics compare` | Cubic spline vs Wendland C2 |
| `sph-hydrodynamics run --ascii` | Scenario + density ASCII field |
| `sph-hydrodynamics kernels` | Partition-of-unity + grad oddness |
| `sph-hydrodynamics theory` | N-particle memory/FLOP ledger |

Scenarios: `dam_break`, `hydrostatic`, `droplet`, `shear`.

## Procedure

1. Pick scenario + kernel; build particle lattice (hex packing) with $m=\rho_0\Delta A$, $h\approx 1.3\Delta x$.
2. Bootstrap density summation and forces; integrate with velocity-Verlet + optional XSPH.
3. Track mass (must be machine-exact under summation density), momentum, KE/PE/UE, $\rho$ moments, CFL.
4. Validate kernels: $\sum_j \Delta A\,W_{ij}\approx 1$, $\nabla W(\mathbf{r})=-\nabla W(-\mathbf{r})$.
5. For free surface, prefer cubic spline; for tensile stability stress-tests, try Wendland C2.

Completion criterion: `selftest` exits 0; `bench` reports `mass_rel_err == 0` and finite `v_max`.

## Pitfalls

- Too-small $c_0$ → large density spikes; too-large $c_0$ → tiny $\Delta t$.
- Neighbour list must rebuild every step (particles advect across hash cells).
- Pair loop uses $j>i$ with Newton reaction — unequal masses need the $m_i/m_j$ symmetric form (implemented).
- Pure-Python pair loops are $O(N\cdot n_{\mathrm{nbr}})$; production needs C++/GPU or vectorized backends.
- Energy drift in WCSPH is expected under artificial viscosity + wall repulsion; mass conservation is the hard invariant.

## Verification

```bash
terminal(command="sph-hydrodynamics selftest", timeout=60)
terminal(command="pytest tests/test_sph_hydrodynamics.py -q", timeout=120)
```

Expect selftest `10/10`, pytest all green, dam-break `mass_rel_err=0`.
