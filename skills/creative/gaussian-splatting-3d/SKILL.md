---
name: gaussian-splatting-3d
description: "3D Gaussian Splatting & Real-Time Radiance Field Rasterizer: pure Python+NumPy 3DGS, SO(3) quaternion rotation, Zwicker 2D projection, volumetric alpha compositing, and Truecolor ANSI terminal visualization."
version: 1.0.0
category: creative
tags:
  - Generative-Aesthetics
  - 3D-Gaussian-Splatting
  - Radiance-Fields
  - Computer-Vision
  - Volumetric-Rendering
  - Truecolor-ANSI
---

# 🎨 3D Gaussian Splatting & Real-Time Radiance Field Rasterization Engine

A mathematically rigorous, zero-dependency 3D Gaussian Splatting (3DGS) rasterization engine implemented from scratch in pure Python + NumPy. Synthesizes explicit radiance fields, computes projective 2D covariance ellipses, performs front-to-back volumetric alpha compositing, optimizes Gaussian parameters via differentiable photometric gradients, and renders interactive 24-bit Truecolor ANSI displays and standard 3DGS PLY point clouds.

---

## 📐 Mathematical Formulation

### 1. Parametric 3D Gaussian Primitive
A 3D Gaussian is parameterized by its mean position $\boldsymbol{\mu} \in \mathbb{R}^3$, positive semi-definite covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$, opacity $\alpha \in [0, 1]$, and view-dependent color $C(\mathbf{d})$:
$$G(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

To ensure that $\boldsymbol{\Sigma}$ remains positive semi-definite and symmetric throughout transformations, it is decomposed into an anisotropic scaling matrix $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$ and an orthogonal rotation matrix $\mathbf{R} \in \mathrm{SO}(3)$:
$$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

---

### 2. Quaternionic $SO(3)$ Rotation Manifold
Rotation is parameterized by a normalized unit quaternion $\mathbf{q} = [w, x, y, z]^T$ with $\|\mathbf{q}\|_2 = 1$. The corresponding orthogonal rotation matrix $\mathbf{R}(\mathbf{q}) \in \mathrm{SO}(3)$ is given by:
$$\mathbf{R} = \begin{pmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\n2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\n2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{pmatrix}$$

Properties verified by formal invariants:
$$\det(\mathbf{R}) = +1.0, \quad \mathbf{R}^T \mathbf{R} = \mathbf{I}_3$$

---

### 3. Projective Camera Transformation & 2D Screen Covariance
Given world-to-camera extrinsic matrix $\mathbf{W} = [\mathbf{R}_{\text{cam}} \mid \mathbf{t}_{\text{cam}}]$, the Gaussian center in camera space is:
$$\mathbf{t} = \mathbf{R}_{\text{cam}} \boldsymbol{\mu} + \mathbf{t}_{\text{cam}} = [t_x, t_y, t_z]^T$$

Under perspective projection with focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$:
$$x' = \frac{f_x t_x}{t_z} + c_x, \quad y' = \frac{f_y t_y}{t_z} + c_y$$

The affine approximation of the perspective transformation is represented by the Jacobian $\mathbf{J} \in \mathbb{R}^{2 \times 3}$ evaluated at $\mathbf{t}$:
$$\mathbf{J} = \begin{pmatrix}
\frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\n0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2}
\end{pmatrix}$$

The 2D covariance in screen coordinates is computed via Zwicker's EWA (Elliptical Weighted Average) formulation with an anti-aliasing low-pass regularization filter:
$$\boldsymbol{\Sigma}'_{2D} = \mathbf{J} \mathbf{R}_{\text{cam}} \boldsymbol{\Sigma} \mathbf{R}_{\text{cam}}^T \mathbf{J}^T + 0.3 \mathbf{I}_{2 \times 2}$$

Screen bounding radius is derived from the maximum eigenvalue $\lambda_{\max}$ of $\boldsymbol{\Sigma}'_{2D}$ with a $3\sigma$ confidence bound ($99.7\%$ energy):
$$r_{\text{bound}} = \left\lceil 3 \sqrt{\lambda_{\max}} \right\rceil$$

---

### 4. Spherical Harmonics View-Dependent Radiance
Radiance emission is modeled using spherical harmonics expansions up to degree $L=1$ over viewing direction unit vector $\mathbf{d} = [d_x, d_y, d_z]^T$:
$$C(\mathbf{d}) = \sum_{l=0}^L \sum_{m=-l}^l c_{lm} Y_{lm}(\mathbf{d})$$

Basis functions:
$$Y_0^0 = \frac{1}{2}\sqrt{\frac{1}{\pi}} \approx 0.282095$$
$$Y_1^{-1} = \frac{1}{2}\sqrt{\frac{3}{\pi}} d_y, \quad Y_1^0 = \frac{1}{2}\sqrt{\frac{3}{\pi}} d_z, \quad Y_1^1 = \frac{1}{2}\sqrt{\frac{3}{\pi}} d_x$$

---

### 5. Volumetric Front-to-Back Alpha Compositing
All visible 3D Gaussians are sorted front-to-back according to camera depth $t_{z,(1)} \le t_{z,(2)} \le \dots \le t_{z,(M)}$. For each pixel $(u, v)$ with displacement $\mathbf{\Delta} = [u - x', v - y']^T$:

Mahalanobis distance squared:
$$m^2 = \mathbf{\Delta}^T (\boldsymbol{\Sigma}'_{2D})^{-1} \mathbf{\Delta} = \frac{c \Delta_x^2 - 2 b \Delta_x \Delta_y + a \Delta_y^2}{\det(\boldsymbol{\Sigma}'_{2D})}$$

Pixel alpha contribution:
$$\alpha_p = \alpha_i \exp\left(-\frac{1}{2} m^2\right)$$

Accumulated color and transmittance update:
$$C_{\text{pixel}} = \sum_{i=1}^M c_i \alpha_{p,i} T_i, \quad T_i = \prod_{j=1}^{i-1} (1 - \alpha_{p,j})$$

Early termination occurs when optical transmittance falls below threshold $T_i < 10^{-4}$.

---

## 🖥️ CLI Quickstart

The `gaussian-splat` CLI is installed globally at `/Users/ayu/.local/bin/gaussian-splat`.

### 1. Run Verification Test Suite
```bash
gaussian-splat test
```

### 2. Render Procedural 3D Scenes
```bash
# Render 2-arm spiral galaxy in ANSI Truecolor
gaussian-splat render --scene galaxy --res 80x40

# Render Trefoil Torus Knot
gaussian-splat render --scene torus --res 80x40

# Render Utah Teapot point cloud
gaussian-splat render --scene teapot --res 80x40

# Render Planetary Atmosphere
gaussian-splat render --scene planetary --res 80x40
```

### 3. Depth Map & Alpha Field Visualization
```bash
# Render Turbo colormap depth buffer
gaussian-splat render --scene galaxy --mode depth

# Render accumulated alpha optical thickness
gaussian-splat render --scene galaxy --mode alpha
```

### 4. Interactive Turntable Orbit Animation
```bash
gaussian-splat orbit --scene galaxy --frames 16 --res 70x34
```

### 5. Differentiable Fitting Optimization
```bash
gaussian-splat fit --steps 15 --lr 0.1
```

### 6. High-Throughput Performance Benchmark
```bash
gaussian-splat benchmark --n-gaussians 2000 --res 80x40
```

### 7. Export Standard 3DGS PLY Format
```bash
gaussian-splat render --scene galaxy --ply /tmp/galaxy.ply
```
