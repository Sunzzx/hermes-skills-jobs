---
name: polyphonic-fm
description: "Polyphonic 6-Operator Algorithmic FM Synthesizer & Phase Modulation DSP Engine: all 32 Yamaha DX7 routing topologies, 4-Rate/4-Level exponential envelopes, feedback anti-aliasing loops, 16-voice dynamic polyphony with priority voice stealing, Braille oscilloscope, and 16-bit 44.1kHz stereo WAV export."
version: 1.0.0
category: creative
tags:
  - Audio-DSP
  - FM-Synthesis
  - Phase-Modulation
  - Sound-Design
  - Generative-Aesthetics
  - Polyphony
  - Yamaha-DX7
---

# 🎹 Polyphonic 6-Operator Algorithmic FM Synthesizer & Phase Modulation DSP Engine

A production-grade, mathematically rigorous implementation of Yamaha DX7-style 6-operator algorithmic Frequency Modulation (Phase Modulation) synthesis in pure Python with NumPy. Zero external deep learning or audio synthesis toolkits required.

Implements all 32 classical DX7 topological modulation algorithms, 4-rate / 4-level exponential envelope generators with acoustic time scaling, 2-sample anti-aliasing moving-average feedback loops, 16-voice dynamic polyphony with priority voice stealing, soft-knee hyperbolic tangent saturation limiting, real-time Unicode Braille subpixel oscilloscope, 32-band DFT spectral power visualizer, and 16-bit 44.1 kHz stereo RIFF WAV export.

---

## 📐 Mathematical Formulation & DSP Theory

### 1. Phase Modulation vs. Frequency Modulation
True frequency modulation varies the instantaneous frequency $\omega(t)$ of a carrier sine wave:
$$\omega(t) = \omega_c + 2\pi k_f m(t)$$
$$y(t) = A(t) \sin\left(\int_0^t \omega(\tau) d\tau\right) = A(t) \sin\left(\omega_c t + 2\pi k_f \int_0^t m(\tau) d\tau\right)$$

In digital implementations, integrating $m(\tau)$ introduces pitch drift whenever the modulator contains DC bias or asymmetric envelope transients. To guarantee absolute pitch stability, John Chowning and Yamaha engineered **Phase Modulation (PM)**:
$$y[n] = A[n] \sin\left(\theta_c[n] + I[n] \cdot m[n]\right)$$
where instantaneous phase accumulates linearly without drift:
$$\theta_c[n] = \left(\theta_c[n-1] + \frac{2\pi (f_c \cdot r + \Delta f)}{f_s}\right) \pmod{2\pi}$$
with $r$ the operator harmonic ratio, $\Delta f$ the fine detune, and $I[n]$ the envelope-modulated modulation index.

---

### 2. Jacobi-Anger Expansion & Bessel Harmonic Sidebands
When a sinusoidal carrier of frequency $\omega_c$ is phase-modulated by a sinusoidal modulator of frequency $\omega_m$ with modulation index $\beta$:
$$y(t) = \sin(\omega_c t + \beta \sin(\omega_m t))$$

By the Jacobi-Anger identity, this expands into an infinite series of harmonic sidebands weighted by Bessel functions of the first kind $J_k(\beta)$:
$$\sin(\omega_c t + \beta \sin(\omega_m t)) = \sum_{k=-\infty}^{+\infty} J_k(\beta) \sin((\omega_c + k \omega_m)t)$$
where:
$$J_k(\beta) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m! \Gamma(m + k + 1)} \left(\frac{\beta}{2}\right)^{2m + k}$$

- **Rational Frequency Ratios** ($f_c / f_m = N_1 / N_2$): Sidebands occur at integer multiples of $f_0 / N_2$, generating rich harmonic acoustic timbres (pianos, brass, flutes, acoustic basses).
- **Irrational Frequency Ratios** ($f_c / f_m \notin \mathbb{Q}$, e.g. $\sqrt{2}, \pi, 2.756$): Generates dense inharmonic partials simulating metallic bells, struck rods, and cymbals.

---

### 3. Self-Feedback Modulation Loop & Anti-Aliasing Averaging
In FM synthesis, an operator can modulate its own phase to transform a pure sine wave into complex spectra (sawtooth, square, or noise):
$$y[n] = \sin\left(\theta[n] + \beta_{fb} \cdot \bar{y}[n]\right)$$

A naive feedback loop $y[n] = \sin(\theta[n] + \beta_{fb} y[n-1])$ exhibits numerical instability and chaotic aliasing at high frequencies. We implement the Yamaha 2-sample moving average filter:
$$\bar{y}[n] = \frac{y[n-1] + y[n-2]}{2}$$
In the $z$-domain, this filter has transfer function:
$$H_{aa}(z) = \frac{1 + z^{-1}}{2}$$
with a transmission zero at $z = -1$ ($\omega = \pi f_s$, the Nyquist frequency). This zero suppresses high-frequency feedback amplification, maintaining stable limit cycles for $\beta_{fb} \le \pi / 2 \approx 1.57$.

---

### 4. The 32 Yamaha DX7 Topological Routing Algorithms
6 operators are arranged into modular directed acyclic graphs (DAGs). Each algorithm defines:
1. **Carriers**: Operators whose outputs sum directly into the stereo output bus.
2. **Modulators**: Operators whose outputs modulate the phase of downstream operators.
3. **Feedback Loop**: Dedicated operator self-feedback with configurable depth.

```
Algorithm 1: [6(FB)->5->4->3] + [2->1]  -> Carriers: [1, 3]
Algorithm 5: [6(FB)->5] + [4->3] + [2->1] -> Carriers: [1, 3, 5] (Classic Rhodes)
Algorithm 16: [6(FB)->5->4] + 3 + 2 + 1  -> Carriers: [1, 2, 3, 4] (Slap Bass)
Algorithm 32: All 6 parallel carriers     -> Pure additive organ bank
```

Topological sorting ensures every modulator is evaluated strictly before its target operator in discrete time.

---

### 5. 4-Rate / 4-Level Exponential Envelope Generators (EG)
Each operator possesses an autonomous 4-stage rate/level envelope generator:
- **Attack**: Transitions from Level 4 towards Level 1 at Rate 1.
- **Decay 1**: Transitions from Level 1 towards Level 2 at Rate 2.
- **Decay 2**: Transitions from Level 2 towards Level 3 at Rate 3.
- **Sustain**: Holds at Level 3 while key is pressed.
- **Release**: Transitions from current level towards Level 4 at Rate 4 upon key off.

Rates ($R \in [0, 99]$) map exponentially to physical decay times from $\tau_{max} = 45.0$ s down to $\tau_{min} = 1.5$ ms:
$$\tau(R) = \tau_{min} \cdot \left(\frac{\tau_{max}}{\tau_{min}}\right)^{\frac{99 - R}{99}}, \quad \alpha = 1.0 - \exp\left(-\frac{2.5}{f_s \cdot \tau(R)}\right)$$
Levels ($L \in [0, 99]$) map to normalized linear amplitudes in $[0.0, 1.0]$.

---

### 6. Dynamic 16-Voice Polyphony & Voice Stealing
- **Pool**: 16 concurrent voice instances (totaling up to 96 simultaneous FM operators).
- **State Lifecycle**: `FREE` $\to$ `ACTIVE` $\to$ `RELEASED` $\to$ `FREE`.
- **Voice Stealing Protocol**:
  1. If a voice is `FREE`, allocate immediately.
  2. If all voices are busy, select the earliest allocated voice in `RELEASED` state.
  3. If all voices are `ACTIVE`, steal the oldest active voice with dynamic cross-fade to prevent audible pops.

---

### 7. Output Signal Chain & Master Limiter
1. **Voice Stereo Spread**: Panning $p \in [0.3, 0.7]$ across voices creates wide stereo separation:
   $$L = S \cdot \cos(p \cdot \pi / 2), \quad R = S \cdot \sin(p \cdot \pi / 2)$$
2. **DC Blocking Filter**: Eliminates DC offsets accumulated from asymmetric phase modulation:
   $$y[n] = x[n] - x[n-1] + 0.995 \cdot y[n-1]$$
3. **Soft-Knee Saturation Limiter**: Prevents digital clipping on multi-voice chord clusters:
   $$y_{out} = \tanh(0.8 \cdot y_{in})$$

---

## 🚀 CLI Commands & Usage

```bash
# 1. Synthesize and preview a preset note with Braille oscilloscope and spectrum
fm-synth play dx7-rhodes --note C4 --duration 2.5

# 2. Play a polyphonic 4-note chord
fm-synth chord dx7-rhodes --notes C4,E4,G4,B4 --duration 3.0

# 3. Export high-quality 44.1kHz 16-bit Stereo WAV
fm-synth render dx7-rhodes -o rhodes_c4.wav --duration 4.0

# 4. Inspect any of the 32 Yamaha DX7 modulation algorithms
fm-synth algo 5

# 5. List all production presets
fm-synth presets

# 6. Benchmark DSP sample throughput (Million Samples/sec)
fm-synth bench

# 7. Run formal mathematical invariant & DSP validation test suite
fm-synth test

# 8. Display theoretical formulations and system info
fm-synth info
```

---

## 🎹 Built-In Production Preset Library

| Preset | Algorithm | Character & Musical Role | Focus Features |
| :--- | :--- | :--- | :--- |
| `dx7-rhodes` | Alg 5 (3 Pairs) | Classic 1980s Electric Piano | 14x tine modulator, warm decay body, subtle chorus |
| `tubular-bells` | Alg 1 (Stack 4) | Inharmonic church chime & tubular bell | Bessel partial ratios (1.0, 2.76, 5.40, 8.93, 13.34) |
| `fm-slap-bass` | Alg 16 (4 Carriers) | Punchy metallic slap bass | Percussive envelope attack, tight feedback sub |
| `synth-brass` | Alg 5 (3 Pairs) | Warm polyphonic analog-style brass | Detuned operator unison (+/-0.5 Hz), filter swell |
| `glass-pad` | Alg 5 (3 Pairs) | Shimmering celestial ambient pad | Slow evolving envelopes, stereo beating |
| `cyberpunk-drone`| Alg 1 (Stack 4) | Heavy industrial distorted drone | Overdriven feedback loop (beta=1.2), sub-harmonics |
| `marimba` | Alg 5 (3 Pairs) | Tuned acoustic wooden mallet | Euler-Bernoulli 1:4:10 modal partial alignment |
| `laser-fx` | Alg 1 (Stack 4) | Modular sci-fi pitch-swept discharge | Ultra-fast multi-rate envelope frequency sweep |

---

## 🧪 Verification & Invariant Testing

Run the comprehensive unit test suite with pytest:
```bash
pytest ~/Developer/hermes-skills-jobs/tests/test_polyphonic_fm.py -v
```

Validates:
- Exact MIDI tuning against equal temperament standard ($A_4 = 440.0$ Hz, $C_4 = 261.625$ Hz).
- 4-Rate / 4-Level state machine trajectory and exponential convergence.
- Bessel sideband expansion generating $>3$ harmonic partials.
- Anti-aliasing feedback loop stability under high feedback depths ($\beta_{fb} = 1.5$).
- DAG structural validity and loop-free topological ordering for all 32 algorithms.
- Dynamic 16-voice allocation, release state reclamation, and voice stealing.
- Soft-knee hyperbolic tangent saturation bound ($|y| \le 1.0$).
- Lossless RIFF 16-bit 44.1 kHz stereo WAV header byte integrity.
