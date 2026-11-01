---
name: granular-synth
description: "Granular Synthesis & Dynamic Gabor Acoustic Texture DSP Engine: Gabor quantum acoustic atomization, Catmull-Rom Hermite sub-sample interpolation, Poisson and quasi-synchronous stochastic scheduling, independent time-scale modification (TSM), infinite freeze, constant-power stereo panning, Braille scatter matrix, and 16-bit 44.1kHz WAV export."
version: 1.0.0
category: creative
tags:
  - Audio-DSP
  - GranularSynthesis
  - GaborAcoustics
  - SoundDesign
  - GenerativeAesthetics
  - TimeStretching
  - QuantumAcoustics
---

# 🌊 Granular Synthesis & Dynamic Gabor Acoustic Texture DSP Engine

A production-grade, mathematically rigorous implementation of Curtis Roads & Dennis Gabor's granular synthesis quantum acoustics in pure Python with NumPy. Zero external deep learning, C-extension, or audio toolchain dependencies required.

Implements Dennis Gabor elementary time-frequency acoustic atoms, 5 canonical window envelope geometries (Hann, Gaussian, Blackman-Harris, Tukey, Bartlett), 4-point 3rd-order Catmull-Rom Hermite sub-sample polynomial interpolation, asynchronous Poisson process and quasi-synchronous stochastic grain emission schedulers, continuous pitch transposition with microtonal detune spray and harmonic chord quantization, independent Time-Scale Modification (TSM) with infinite acoustic freeze, spatial constant-power stereo panning ($L^2 + R^2 \equiv 1.0$), real-time Unicode Braille time-frequency scatter matrix visualizer, and 16-bit 44.1 kHz stereo RIFF WAV export.

---

## 📐 Mathematical Formulation & Acoustic Physics

### 1. Dennis Gabor Quantum Acoustics & Time-Frequency Atomization
In 1946, Nobel laureate Dennis Gabor established the quantum theory of acoustic communication: any continuous sound signal $s(t)$ can be represented as an expansion of elementary acoustic quanta or "grains" centered at time coordinate $\tau$ and frequency $\omega$:

$$s(t) = \sum_{i} a_i \cdot w_i(t - \tau_i) \cdot \cos(\omega_i (t - \tau_i) + \phi_i)$$

where each grain is an elementary information quantum bounded by the Heisenberg-Gabor acoustic uncertainty principle:
$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$

For granular sound design, grain durations typically range between $10\text{ ms}$ and $150\text{ ms}$ ($440$ to $6,600$ samples at $f_s = 44,100\text{ Hz}$). Below $10\text{ ms}$, grains dissolve into microscopic pitch clicks; above $200\text{ ms}$, individual grains are perceived as isolated musical phrases rather than a continuous fused acoustic cloud.

---

### 2. Window Envelope Geometries & Partition of Unity
To prevent abrupt boundary discontinuities from generating severe high-frequency spectral splatter (Nyquist edge clicks), each grain is multiplied by a smooth window envelope $w[n]$ defined over $N$ samples ($n \in [0, N-1]$):

1. **Periodic DFT-Even Hann Window** (Optimal for continuous overlap-add partition of unity):
   $$w_{\text{Hann}}[n] = \frac{1}{2} \left(1 - \cos\left(\frac{2\pi n}{N}\right)\right)$$
   Satisfies exact **Partition of Unity** under $50\%$ overlap ($R = N/2$):
   $$\sum_{k} w_{\text{Hann}}[n - k R] \equiv 1.0 \pm 10^{-15}$$

2. **Truncated Gaussian Window** (Minimizes time-frequency Heisenberg uncertainty product):
   $$w_{\text{Gauss}}[n] = \exp\left(-\frac{1}{2} \left(\frac{n - (N-1)/2}{\sigma (N-1)/2}\right)^2\right), \quad \sigma = 0.35$$

3. **4-Term Blackman-Harris Window** (Maximum sidelobe rejection: $-92\text{ dB}$ attenuation):
   $$w_{\text{BH}}[n] = a_0 - a_1 \cos\left(\frac{2\pi n}{N-1}\right) + a_2 \cos\left(\frac{4\pi n}{N-1}\right) - a_3 \cos\left(\frac{6\pi n}{N-1}\right)$$
   where $a_0 = 0.35875, a_1 = 0.48829, a_2 = 0.14128, a_3 = 0.01168$.

4. **Tukey Window** (Tapered cosine with flat sustained central plateau, $\alpha = 0.5$):
   Preserves original audio RMS dynamics while smoothly feathering grain edges.

---

### 3. Catmull-Rom Hermite Sub-Sample Resampling
To pitch-shift grains by arbitrary frequency transposition ratios $\rho = 2^{\Delta c / 1200}$ without altering grain duration or inducing aliasing distortion, the engine performs continuous fractional lookup in the audio source buffer.

Linear interpolation introduces unacceptable high-frequency attenuation ($|H(e^{j\omega})| < 1$). Sinc interpolation is computationally prohibitive for thousands of concurrent grains. We implement a **4-point, 3rd-order Catmull-Rom Hermite polynomial**:

For a floating-point sample index $x$ decomposed into integer floor $n = \lfloor x \rfloor$ and fractional offset $\alpha = x - n \in [0, 1)$:

$$c_0 = y[n]$$
$$c_1 = \frac{1}{2}(y[n+1] - y[n-1])$$
$$c_2 = y[n-1] - \frac{5}{2} y[n] + 2 y[n+1] - \frac{1}{2} y[n+2]$$
$$c_3 = \frac{1}{2}(y[n+2] - y[n-1]) + \frac{3}{2}(y[n] - y[n+1])$$
$$y(n + \alpha) = ((c_3 \alpha + c_2)\alpha + c_1)\alpha + c_0$$

**Guarantees**:
- Exact interpolation at integer knots: $y(n + 0) = y[n]$
- Continuous first derivatives ($C^1$ smoothness): eliminates slope discontinuities and phase tearing
- Flat passband response up to $0.4 f_s$ with maximum reconstruction error $< 5 \times 10^{-4}$ on bandlimited sinusoidal waves.

---

### 4. Stochastic Poisson vs. Quasi-Synchronous Scheduling

1. **Asynchronous Poisson Point Process**:
   Inter-grain arrival intervals $\Delta t$ follow an Exponential distribution with parameter $\lambda$ (grain density in grains/sec):
   $$p(\Delta t) = \lambda e^{-\lambda \Delta t}, \quad \mathbb{E}[\Delta t] = \frac{1}{\lambda}, \quad \text{Var}(\Delta t) = \frac{1}{\lambda^2}$$
   Produces organic, rain-like, asynchronous acoustic textures without comb-filtering periodic resonance peaks.

2. **Quasi-Synchronous Scheduling**:
   Grains are emitted at regular nominal periods $T = 1 / \lambda$ perturbed by controlled stochastic jitter $\epsilon \in [0, 1]$:
   $$\Delta t = \frac{1}{\lambda} + \epsilon \cdot \mathcal{U}\left(-\frac{0.5}{\lambda}, \frac{0.5}{\lambda}\right)$$
   At high densities ($\lambda > 30\text{ Hz}$), quasi-synchronous grains fuse into a perceived pitch formant fundamental $f_{\text{grain}} = \lambda$.

---

### 5. Time-Scale Modification (TSM) & Infinite Time Freeze
Traversing the source audio buffer with an independent scrub velocity $v_{\text{scrub}}$ decouples playback speed from acoustic pitch:

$$P_{\text{scrub}}(t) = (v_{\text{scrub}} \cdot t \cdot f_s) \pmod{N_{\text{buffer}}}$$

- **Normal Speed** ($v_{\text{scrub}} = 1.0$): Original playback rate.
- **Time Dilation / Stretch** ($v_{\text{scrub}} = 0.25$): $4\times$ temporal expansion; timbre and pitch fundamental are fully preserved without phase-vocoder smearing or transient transient degradation.
- **Infinite Acoustic Freeze** ($v_{\text{scrub}} = 0.0$): Scrub head holds position; grains continuously sample the localized temporal window with position spray $\sigma_P$. Sustains an infinite, morphing spectral drone with steady RMS ($< 1\text{ dB}$ variation) and zero looping seams.
- **Reverse Granular Flow** ($v_{\text{scrub}} = -0.5$): Traverses backwards in slow motion.

---

### 6. Constant-Power Spatial Panning Law
To create expansive 3D acoustic width without center-channel volume drop, each grain is assigned an azimuth $\theta_{\text{az}} \in [-1.0, 1.0]$ mapped to radiant angle $\phi = (\theta_{\text{az}} + 1) \frac{\pi}{4} \in [0, \frac{\pi}{2}]$:

$$g_L = \cos(\phi), \quad g_R = \sin(\phi)$$

Satisfying the **Constant-Power Law** everywhere:
$$g_L^2 + g_R^2 = \cos^2(\phi) + \sin^2(\phi) \equiv 1.0$$

---

## 💻 CLI Commands & Usage

The engine is packaged as an executable CLI tool: `grain-synth` (symlinked into `~/.local/bin/grain-synth`).

### 1. Synthesize Ambient Evolving Soundscape Demo
```bash
grain-synth demo --duration 4.0 --output granular_soundscape.wav
```

### 2. Infinite Acoustic Time Freeze on Cello Drone
```bash
grain-synth freeze --duration 5.0 --output cello_freeze.wav
```

### 3. Generate Dense 120-Grain/sec Poisson Cloud
```bash
grain-synth cloud --density 120.0 --duration 3.0 --output grain_cloud.wav
```

### 4. 4x Time-Stretch Buffer with Zero Pitch Shift
```bash
grain-synth stretch --ratio 4.0 --output stretched.wav
```

### 5. Run High-Throughput Microbenchmark
```bash
grain-synth bench
```

### 6. Formally Verify Mathematical Invariants
```bash
grain-synth verify
```

---

## 🐍 Python API & Programmatic Usage

```python
import numpy as np
from grain_synth import AudioSourceFactory, GranularEngine, StereoWavWriter, BrailleGrainMatrix

# 1. Generate or load an audio source buffer
buffer = AudioSourceFactory.generate_cello_drone(duration=4.0, f0=110.0)

# 2. Configure Granular Engine
engine = GranularEngine(
    buffer=buffer,
    sample_rate=44100,
    grain_duration_ms=65.0,
    duration_spray=0.25,
    density_hz=70.0,
    scheduling="poisson",
    scrub_speed=0.25,                  # 4x Time Dilation
    position_spray_ms=20.0,
    pitch_shift_cents=0.0,
    pitch_spray_cents=15.0,
    chord_intervals=[0.0, 700.0, 1200.0], # Octave + Fifth Shimmer
    pan_spray=0.75,
    window_type="blackman_harris",
)

# 3. Synthesize stereo audio
out_l, out_r, manifest = engine.synthesize(output_duration_sec=3.0, seed=42)

# 4. Export 16-bit 44.1 kHz PCM WAV
StereoWavWriter.write("output.wav", out_l, out_r, fs=44100)

# 5. Render Unicode Braille scatter matrix
print(BrailleGrainMatrix.render_scatter(manifest, total_duration=3.0))
```

---

## 🔬 Formal Invariant Test Suite

Every build verifies 7 core physical invariants:
- **Invariant 1**: Constant-power stereo panning power deviation $< 10^{-6}$
- **Invariant 2**: Periodic Hann 50% overlap-add partition of unity error $< 10^{-4}$
- **Invariant 3**: Catmull-Rom Hermite interpolation sinusoidal error $< 5 \times 10^{-3}$
- **Invariant 4**: Dennis Gabor Heisenberg uncertainty bound $\Delta t \cdot \Delta f \ge 1 / (4\pi)$
- **Invariant 5**: Poisson arrival distribution mean and std error $< 10\%$
- **Invariant 6**: Time dilation conservation (exact sample count match)
- **Invariant 7**: Infinite time freeze stability (RMS variation $< 2.5\text{ dB}$)
