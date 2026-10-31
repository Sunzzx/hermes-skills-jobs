---
name: kerr-geodesic
description: "Production-grade Kerr metric general relativistic geodesic raytracer & black hole shadow engine: Boyer-Lindquist geometry, Carter constant Mino-time Hamiltonian integration, exact Bardeen analytical critical curves, Novikov-Thorne thin accretion disk with relativistic Doppler boosting and gravitational redshift, and subpixel Unicode Braille terminal visualizer."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
category: computational-science
tags:
  - Astrophysics
  - GeneralRelativity
  - KerrMetric
  - BlackHoleShadow
  - RelativisticRaytracing
  - ComputationalPhysics
  - GeodesicIntegrator
  - AccretionDisk
---

# 🌌 Kerr Relativistic Geodesic Raytracer & Black Hole Shadow Engine

A production-grade, mathematically rigorous General Relativistic Ray Tracing (GRRT) and null geodesic simulation engine for spinning Kerr black holes from first principles in pure Python + NumPy. Implements exact Boyer-Lindquist spacetime geometry, Carter constant first-integrals in Mino affine parameter, Bardeen analytical critical curve benchmarking, Novikov-Thorne Keplerian thin accretion disk physics, relativistic Doppler boosting, gravitational redshift, Liouville's theorem invariant beaming ($I_\text{obs} = g^4 I_\text{emit}$), and subpixel Unicode Braille terminal rendering with 24-bit ANSI TrueColor astrophysical heatmaps.

Zero external heavy C++ wrappers, raymarching approximations, or opaque CFD toolkits. Immediate standalone CLI and importable Python API.

---

## 📐 Theoretical Foundations & Mathematical Formulation

### 1. Spacetime Metric in Boyer-Lindquist Coordinates
In natural units where $G = c = M = 1$, the Kerr metric describing a stationary, axisymmetric vacuum spacetime with angular momentum $J = aM$ is given by:
$$
ds^2 = -\left(1 - \frac{2r}{\Sigma}\right) dt^2 - \frac{4ar\sin^2\theta}{\Sigma} dt d\phi + \frac{\Sigma}{\Delta} dr^2 + \Sigma d\theta^2 + \left(r^2 + a^2 + \frac{2a^2 r \sin^2\theta}{\Sigma}\right) \sin^2\theta d\phi^2
$$
where the metric auxiliary functions are:
$$
\Sigma(r, \theta) = r^2 + a^2 \cos^2\theta, \qquad \Delta(r) = r^2 - 2r + a^2
$$

### 2. Spacetime Horizons & Ergosphere Geometry
- **Event Horizons**: Roots of $\Delta(r) = 0$:
  $$
  r_\pm = 1 \pm \sqrt{1 - a^2}
  $$
  where $r_+$ is the outer event horizon (one-way causal membrane) and $r_-$ is the inner Cauchy horizon.
- **Ergosphere Radius**: The static limit surface where $g_{tt} = 0$:
  $$
  r_E(\theta) = 1 + \sqrt{1 - a^2 \cos^2\theta}
  $$
  In the ergosphere region ($r_+ < r < r_E$), the frame-dragging effect is so intense that no physical observer can remain static relative to infinity ($dt/d\tau > 0$ forces $d\phi/d\tau > 0$).
- **Innermost Stable Circular Orbit (ISCO)** (Bardeen, Press, Teukolsky 1972):
  $$
  Z_1 = 1 + (1 - a^2)^{1/3} \left[(1 + a)^{1/3} + (1 - a)^{1/3}\right]
  $$
  $$
  Z_2 = \sqrt{3a^2 + Z_1^2}
  $$
  $$
  r_\text{ISCO} = 3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)}
  $$
  where the negative sign corresponds to prograde (co-rotating) equatorial orbits ($r_\text{ISCO} = 6M$ at $a=0$, down to $1M$ at $a=1$) and the positive sign to retrograde orbits ($r_\text{ISCO} = 9M$ at $a=1$).

---

### 3. Null Geodesic Equations & Carter Constants
For photons ($m=0$), the geodesic equations possess four constants of motion:
1. Energy at infinity: $E = -p_t = 1$
2. Axial angular momentum: $L_z = p_\phi$
3. Massless condition: $g_{\mu\nu} p^\mu p^\nu = 0$
4. Carter Constant: $\mathcal{Q} = p_\theta^2 + \cos^2\theta \left[a^2(m^2 - E^2) + \frac{L_z^2}{\sin^2\theta}\right]$

Defining normalized impact parameters $\xi = L_z / E$ and $\eta = \mathcal{Q} / E^2$, and introducing the Mino affine parameter $\lambda$ defined by $d\lambda = d\tau / \Sigma$:
$$
\left(\frac{dr}{d\lambda}\right)^2 = R(r) = P(r)^2 - \Delta(r)\left[\eta + (\xi - a)^2\right]
$$
$$
\left(\frac{d\theta}{d\lambda}\right)^2 = \Theta(\theta) = \eta + a^2 \cos^2\theta - \xi^2 \cot^2\theta
$$
$$
\frac{d\phi}{d\lambda} = \frac{a P(r)}{\Delta(r)} - a + \frac{\xi}{\sin^2\theta}
$$
where $P(r) = r^2 + a^2 - a\xi$.

### 4. Carter Quartic Radial Polynomial
Expanding the radial potential $R(r)$ yields a depressed monic quartic with no cubic term:
$$
R(r) = r^4 + A_2 r^2 + A_1 r + A_0
$$
where:
$$
A_2 = a^2 - \xi^2 - \eta
$$
$$
A_1 = 2M \left[ \eta + (\xi - a)^2 \right]
$$
$$
A_0 = -a^2 \eta
$$

**Theorem (Shadow Classification)**:
Let $\mathcal{R} = \{r \in \mathbb{R} : R(r) = 0 \land r > r_+\}$ be the set of real roots outside the event horizon.
- If $\mathcal{R} = \emptyset$: the radial velocity never vanishes outside $r_+$. The ray falls monotonically across the horizon $\implies$ **Inside Black Hole Shadow**.
- If $\mathcal{R} \ne \emptyset$: the ray encounters a periapsis turning point at $r_\text{turn} = \max(\mathcal{R})$ and scatters back to spatial infinity $\implies$ **Outside Shadow**.

---

### 5. Bardeen Analytical Critical Curve
Unstable spherical photon orbits occur where $R(r) = 0$ and $R'(r) = 0$ simultaneously. Solving for impact parameters as functions of orbit radius $r \in [r_\text{ph, pro}, r_\text{ph, retro}]$:
$$
\xi_c(r) = \frac{r^2 - a^2 - r \Delta}{a(r - 1)}
$$
$$
\eta_c(r) = \frac{r^3 \left[4\Delta - r(r - 1)^2\right]}{a^2(r - 1)^2}
$$
Projecting onto the celestial coordinates $(\alpha, \beta)$ of an observer at inclination $\theta_0$:
$$
\alpha_c(r) = -\frac{\xi_c(r)}{\sin\theta_0}
$$
$$
\beta_c(r) = \pm \sqrt{\eta_c(r) + a^2 \cos^2\theta_0 - \xi_c^2 \cot^2\theta_0}
$$
For $a=0$ (Schwarzschild), this reduces to a circle of radius $R = \sqrt{27} M \approx 5.196 M$. For $a \to 1$, frame-dragging displaces the shadow centroid by $\delta \approx 2M$ and flattens the prograde edge.

---

### 6. Relativistic Accretion Disk & Beaming
A geometrically thin, optically thick Novikov-Thorne accretion disk in the equatorial plane $\theta = \pi/2$ spans from $r_\text{in} = r_\text{ISCO}$ to $r_\text{out}$.
- Keplerian angular velocity: $\Omega_K = \frac{1}{r^{3/2} + a}$
- Four-velocity component: $u^t = \frac{1}{\sqrt{1 - 3/r + 2a/r^{3/2}}}$
- Relativistic Doppler and gravitational redshift factor:
  $$
  g = \frac{\nu_\text{obs}}{\nu_\text{emit}} = \frac{1}{u^t(1 - \Omega_K \xi)}
  $$
- Observed Intensity (Liouville's Theorem invariant):
  $$
  I_\text{obs} = g^4 I_\text{emit}(r)
  $$
  where $I_\text{emit}(r) = \frac{1}{r^3} \left(1 - \sqrt{\frac{r_\text{ISCO}}{r}}\right)$ is the Novikov-Thorne thermal profile.

---

## 💻 CLI Reference

```bash
# Verify all 7 relativistic invariants (horizons, ISCO, Bardeen curve, Doppler beaming)
kerr-sim verify

# Render high-spin Kerr shadow (a=0.9, theta=85 deg) with Inferno TrueColor palette
kerr-sim shadow -a 0.9 -i 85 --fov 24 --palette inferno

# Render glowing relativistic accretion disk with Doppler boosting and redshift
kerr-sim disk -a 0.9 -i 85 --fov 28 --palette blackbody

# Fast terminal ASCII snapshot
kerr-sim ascii -a 0.9

# High-throughput raytracing benchmark (5,000 rays)
kerr-sim bench -n 5000
```

---

## 🐍 Python API Usage

```python
import math
import numpy as np
from pathlib import Path

# Load kerr-sim engine
from kerr_sim import KerrSpacetime, KerrVisualizer

sp = KerrSpacetime(a=0.9, M=1.0)
print(f"Event Horizon r+: {sp.r_plus:.4f} M")
print(f"ISCO Radius:      {sp.r_isco:.4f} M")
print(f"Photon Orbits:    [{sp.r_ph_pro:.4f}, {sp.r_ph_retro:.4f}] M")

# Trace a single ray from celestial screen (alpha, beta)
theta0 = math.radians(85.0)
ray = sp.trace_ray(alpha=-8.0, beta=1.0, theta0=theta0, r0=35.0)

if ray.hit_disk:
    print(f"Accretion Disk Hit at r = {ray.disk_r:.2f} M")
    print(f"Relativistic Doppler Factor g = {ray.doppler_g:.3f}")
    print(f"Observed Beamed Intensity I_obs = {ray.intensity:.6f}")

# Analytical Bardeen critical curve
alp_c, bet_c = sp.analytical_shadow(theta0, num_points=200)
print(f"Shadow bounds: [{alp_c.min():.2f}, {alp_c.max():.2f}] M")
```

---

## 🔬 Test Suite Verification

```bash
pytest tests/test_kerr_geodesic.py -v
```

Validates:
1. Exact Schwarzschild horizon and ISCO limits ($a=0 \implies r_+=2M, r_\text{isco}=6M$).
2. Sub-extremal Kerr horizon and prograde photon sphere contraction ($a=0.998$).
3. Ergosphere contact at poles and equatorial extension to $2M$.
4. Analytical Bardeen critical curve bounds and gravitomagnetic frame-dragging centroid displacement.
5. Hamiltonian energy and Carter invariant conservation along numerical null geodesics ($< 10^{-6}$ drift).
6. Relativistic Doppler blueshift ($g > 1$) on the approaching side vs redshift ($g < 1$) on the receding side.
7. Full CLI invocation across all subcommands (`verify`, `shadow`, `disk`, `ascii`, `bench`).
