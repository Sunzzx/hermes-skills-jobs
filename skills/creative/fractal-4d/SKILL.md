---
name: fractal-4d
description: "Hyperdimensional 4D Fractal Raymarcher & Quaternion Julia Engine: pure Python SDF raymarching, SO(4) isoclinic rotations, Blinn-Phong lighting, ambient occlusion, and Truecolor ANSI terminal visualization."
version: 1.0.0
category: creative
tags:
  - Generative-Aesthetics
  - 4D-Fractals
  - Quaternions
  - Raymarching
  - Computer-Graphics
  - Truecolor-ANSI
---

# 🌌 Hyperdimensional 4D Fractal Raymarcher & Quaternion Julia Engine

A mathematically rigorous, zero-dependency 4D raymarching engine written in pure Python. Renders 4D Quaternion Julia sets, 4D Mandelbulbs, and Clifford Tori via Signed Distance Fields (SDFs), Hart distance estimation, $SO(4)$ rotations, Blinn-Phong illumination, ambient occlusion, and soft shadows directly to 24-bit Truecolor terminal displays and lossless PNG files.

---

## 📐 Mathematical Formulation

### 1. Quaternionic Algebra $\mathbb{H}$
A quaternion $q \in \mathbb{H}$ is defined over the 4-dimensional real vector space with basis $\{1, i, j, k\}$:
$$q = w + x i + y j + z k = (w, \mathbf{v}), \quad \mathbf{v} = (x, y, z) \in \mathbb{R}^3$$

Fundamental non-commutative multiplication laws:
$$i^2 = j^2 = k^2 = i j k = -1$$
$$i j = k, \quad j k = i, \quad k i = j, \quad j i = -k, \quad k j = -i, \quad i k = -j$$

For two quaternions $q_1 = (w_1, \mathbf{v}_1)$ and $q_2 = (w_2, \mathbf{v}_2)$:
$$q_1 q_2 = (w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2, \; w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2)$$

Squaring a quaternion $q^2$ reduces efficiently to:
$$q^2 = (w^2 - \|\mathbf{v}\|^2, \; 2 w \mathbf{v})$$

---

### 2. Quaternionic Julia Set Dynamics & Hart Distance Estimation
A 4D Julia set is the boundary of the filled Julia set $K_c$ for the dynamical system:
$$q_{n+1} = q_n^2 + c, \quad c \in \mathbb{H}$$

To render the 3D hypersurface intersection formed by taking a slice $w = w_0$ through $\mathbb{R}^4$, we employ **Hart's Distance Estimator** (Hart, Sandin, & Kauffman, 1989). Differentiating the recurrence relation with respect to the initial point $q_0$:
$$z'_0 = 1, \quad z'_{n+1} = 2 z_n z'_n$$
$$\frac{d}{d q_0} (q_n) = \prod_{i=0}^{n-1} 2 q_i$$

Upon escaping the escape radius $R = 2.0$ at iteration $N$ with norm $|q_N| > R$, the lower bound to the fractal boundary is given by:
$$d(q_0) \approx \frac{1}{2} \frac{|q_N| \ln |q_N|}{|q'_N|}$$

This provides a conservative Lipschitz-bounded Signed Distance Field (SDF) suitable for sphere tracing.

---

### 3. $SO(4)$ 6-Plane Hyper-Rotations
Rotations in 4-dimensional Euclidean space $\mathbb{R}^4$ require specifying rotations across pairs of coordinate axes. The Lie group $SO(4)$ has dimension $\frac{4 \times 3}{2} = 6$. Any 4D rotation is parameterized by six planar rotations:
- $R_{xy}(\theta)$ in the $xy$-plane
- $R_{xz}(\theta)$ in the $xz$-plane
- $R_{xw}(\theta)$ in the $xw$-plane (rotating 3D space into the 4th dimension)
- $R_{yz}(\theta)$ in the $yz$-plane
- $R_{yw}(\theta)$ in the $yw$-plane
- $R_{zw}(\theta)$ in the $zw$-plane

Isoclinic double rotations preserve the Euclidean norm $\|v\|_2 = \sqrt{x^2 + y^2 + z^2 + w^2}$ while revealing non-trivial 3D cross-sections of 4D manifolds.

---

### 4. Physically-Based Shading & Numerical Normals

#### Numerical Gradient for Surface Normals
The surface normal $\mathbf{N}$ at hit point $\mathbf{p} \in \mathbb{R}^3$ is calculated using the numerical forward gradient of the distance field $d(\mathbf{p})$ with finite difference step $\epsilon = 0.002$:
$$\mathbf{N}(\mathbf{p}) = \frac{\nabla d(\mathbf{p})}{\|\nabla d(\mathbf{p})\|}, \quad \nabla d(\mathbf{p}) = \begin{pmatrix} d(\mathbf{p} + \epsilon \hat{\mathbf{x}}) - d(\mathbf{p}) \\ d(\mathbf{p} + \epsilon \hat{\mathbf{y}}) - d(\mathbf{p}) \\ d(\mathbf{p} + \epsilon \hat{\mathbf{z}}) - d(\mathbf{p}) \end{pmatrix}$$

#### Ambient Occlusion (AO)
Surface self-occlusion is approximated along the surface normal $\mathbf{N}$ across $K$ step samples:
$$\text{AO}(\mathbf{p}) = 1.0 - \alpha \sum_{i=1}^K 2^{-i} (i \delta - d(\mathbf{p} + i \delta \mathbf{N}))$$

#### Soft Penumbra Shadows
Raymarching secondary shadow rays toward key light direction $\mathbf{L}$ with hardness coefficient $k = 16.0$:
$$S(\mathbf{p}, \mathbf{L}) = \min_{t \in [t_{\min}, t_{\max}]} \left( \frac{k \cdot d(\mathbf{p} + t \mathbf{L})}{t} \right)$$

---

## 🖥️ CLI Quickstart

The `fractal-4d` CLI is installed globally at `/Users/ayu/.local/bin/fractal-4d`.

### 1. Run Verification Test Suite
```bash
fractal-4d test
```

### 2. View Preconfigured Gallery
```bash
fractal-4d gallery
```
Available targets: `julia-classic`, `julia-rabbit`, `julia-sanmarco`, `julia-dendrite`, `mandelbulb`, `clifford-torus`.

### 3. Render 4D Cross-Section to Terminal (Truecolor)
```bash
fractal-4d render --target julia-classic --res 80x48 --palette cosmic
```

### 4. High-Resolution PNG Export
```bash
fractal-4d render --target julia-rabbit --res 640x480 --palette neon --out /tmp/rabbit.png
```

### 5. 4D Hyperplane Tomography Slicing
Generate a series of parallel 3D cross-sections as $w$ steps across $[-0.6, +0.6]$:
```bash
fractal-4d slice --target julia-sanmarco --slices 5 --w-min -0.5 --w-max 0.5
```

### 6. Interactive Real-Time 4D Rotation Animation
```bash
fractal-4d anim --target julia-classic --frames 40 --palette cosmic
```

### 7. Performance Benchmark
```bash
fractal-4d benchmark
```
