---
name: navier-stokes-2d
description: "Chorin projection method fluid dynamics engine with advection, diffusion, pressure Poisson iteration, and terminal ASCII vorticity visualization."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [ComputationalPhysics, FluidDynamics, NavierStokes, PartialDifferentialEquations]
---

# 2D Incompressible Navier-Stokes Fluid Simulator

## Overview
Numerical solution to incompressible Navier-Stokes equations via operator splitting (Chorin Projection):
$$rac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot 
abla)\mathbf{u} = -rac{1}{ho}
abla p + 
u 
abla^2 \mathbf{u} + \mathbf{f}$$

## CLI Usage
```bash
fluid2d    # Run simulation and render ASCII vorticity field
```
