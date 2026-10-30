---
name: navier-stokes-2d
description: "Chorin projection method fluid dynamics engine with advection, diffusion, pressure Poisson iteration, and terminal ASCII vorticity visualization."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [ComputationalPhysics, FluidDynamics, NavierStokes, PartialDifferentialEquations, BrailleVisualization]
---

# 2D Incompressible Navier-Stokes Fluid Dynamics Engine

Production-grade 2D Incompressible Navier-Stokes fluid simulator implemented via Chorin's Projection Method (operator splitting) on a Marker-and-Cell (MAC) staggered grid in pure Python with NumPy.

## Mathematical Formulation

### 1. Incompressible Navier-Stokes Equations

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

Where:
- $\mathbf{u} = (u, v)$ is the velocity vector field.
- $p$ is the scalar kinematic pressure.
- $\rho$ is fluid density ($\rho = 1.0$).
- $\nu$ is kinematic viscosity ($\nu = 1 / \mathrm{Re}$).
- $\mathbf{f}$ is body force.

### 2. Fractional Step / Chorin Splitting

1. **Advection**: Solved via 2nd-order Runge-Kutta Semi-Lagrangian characteristic back-tracing:
   $$\mathbf{u}^* = \mathbf{u}^n(\mathbf{x} - \mathbf{u}_{mid} \Delta t)$$
2. **Viscous Diffusion**: Implicit Helmholtz relaxation:
   $$(I - \nu \Delta t \nabla^2) \mathbf{u}^{**} = \mathbf{u}^*$$
3. **Pressure Poisson Equation**: Divergence of intermediate velocity sources pressure Poisson equation:
   $$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^{**}$$
   Solved via Preconditioned Conjugate Gradient (PCG) with zero-mean solvability conditioning.
4. **Velocity Projection**: Guarantees solenoidality $\nabla \cdot \mathbf{u}^{n+1} = 0$:
   $$\mathbf{u}^{n+1} = \mathbf{u}^{**} - \frac{\Delta t}{\rho} \nabla p$$

### 3. Marker-and-Cell (MAC) Staggered Grid

- Horizontal velocities $u$ are located on vertical cell faces $(i \pm 1/2, j)$.
- Vertical velocities $v$ are located on horizontal cell faces $(i, j \pm 1/2)$.
- Pressure $p$ and vorticity $\omega$ reside at cell centers $(i, j)$.
- Staggered placement prevents odd-even checkerboard pressure-velocity decoupling without requiring artificial numerical dissipation.

## CLI Capabilities

```bash
# 1. Analytical Validation: Taylor-Green vortex decay against exact analytical solution
fluid2d taylor-green --grid 48 --steps 40 --validate

# 2. Lid-Driven Cavity Flow Benchmark (Ghia et al., Re = 100)
fluid2d cavity --re 100 --grid 48 --steps 300 --vis vorticity

# 3. Von Kármán Vortex Shedding past immersed cylinder obstacle
fluid2d vortex --re 150 --nx 100 --ny 40 --steps 300 --vis vorticity

# 4. Numerical Throughput Benchmark (MCUPS: Million Cell Updates Per Second)
fluid2d benchmark --grid 64 --steps 100

# 5. Quick Interactive Terminal Demo
fluid2d demo
```

## Terminal Visualization

Includes a high-resolution Unicode Braille canvas (U+2800..U+28FF) mapping 2x4 subpixels per terminal cell, with ANSI 24-bit RGB palette rendering for vorticity fields, speed magnitudes, and immersed solid obstacle boundaries.
