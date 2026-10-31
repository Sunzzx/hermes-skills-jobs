---
name: binaural-spatial-audio
description: "Production-grade 3D binaural spatial audio & Ambisonics DSP engine: Woodworth spherical head diffraction ITD with cubic Lagrange fractional delay, Duda-Brown ILD head shadow biquad filtering, pinna elevation spectral notch cues, Ambisonics B-format spherical harmonics encoding, Image Source Method room acoustics, FDN reverberation, Braille polar soundfield radar, and 16-bit 44.1kHz stereo WAV export."
version: 1.0.0
category: creative
tags:
  - Audio-DSP
  - Spatial-Audio
  - Binaural
  - HRTF
  - Ambisonics
  - Acoustics
  - Generative-Aesthetics
---

# 🎧 Binaural 3D Spatial Audio & Ambisonics DSP Engine

A production-grade, mathematically rigorous implementation of 3D binaural spatial audio and Higher-Order Ambisonics (HOA) digital signal processing in pure Python and NumPy. Zero external deep learning, C-extension, or audio toolchain dependencies required.

Implements the complete psychoacoustic spatialization pipeline: Woodworth-Schlosberg spherical head acoustic diffraction with sub-sample cubic Lagrange fractional delay lines, Duda-Brown continuous-time spherical head shadow filters discretized via the Bilinear Transform, elevation pinna notch and concha resonance filtering, ISO 9613-1 atmospheric air damping, Allen & Berkley Image Source Method (ISM) shoebox early reflections, 4-channel unitary Householder Feedback Delay Network (FDN) late reverberation, real-time Unicode Braille polar soundfield radar, Lissajous stereo phase goniometer, and 16-bit 44.1 kHz stereo RIFF WAV export.

---

## 📐 Mathematical Formulation & Acoustic Physics

### 1. Interaural Time Difference (ITD) & Woodworth Diffraction
For an acoustic wave arriving at azimuth angle $	heta \in [-\pi, \pi]$ relative to the listener median plane, the acoustic path difference between the ipsilateral and contralateral ears wraps around the spherical skull. Under the Woodworth-Schlosberg spherical diffraction model:
$$\Delta t(	heta) = rac{a}{c} (\sin|	heta| + |	heta|)$$
where:
- $a pprox 0.0875	ext{ m}$ is the average adult human head radius ($8.75	ext{ cm}$)
- $c = 343.2	ext{ m/s}$ is the acoustic speed of sound in dry air at $20^\circ	ext{C}$

At maximum azimuth $	heta = \pm 90^\circ$ (lateral arrival):
$$	ext{ITD}_{\max} = rac{0.0875}{343.2} \left(1 + rac{\pi}{2}ight) pprox 655.4\,\mu	ext{s}$$
At $f_s = 44,100	ext{ Hz}$, this corresponds to $\sim 28.9$ samples of acoustic delay.

---

### 2. Cubic Lagrange Fractional Delay Interpolation
When sound sources or listener head move dynamically, path delays vary continuously across non-integer sample boundaries. Integer rounding causes severe pitch-quantization clicks and phase tearing. Linear interpolation introduces high-frequency damping ($|H(e^{j\omega})| < 1$).

To maintain sub-sample precision and smooth Doppler frequency shifts, we utilize a 4-point **Cubic Lagrange fractional delay polynomial**:
Given delay $D = \lfloor D floor + d$ where $d \in [0, 1)$:
$$y[n - D] = \sum_{k=-1}^2 L_k(d) \cdot x[n - \lfloor D floor - k]$$
where the Lagrange basis polynomials are:
$$L_{-1}(d) = -rac{d(d-1)(d-2)}{6}, \quad L_0(d) = rac{(d+1)(d-1)(d-2)}{2}$$
$$L_1(d) = -rac{(d+1)d(d-2)}{2}, \quad L_2(d) = rac{(d+1)d(d-1)}{6}$$
Guaranteeing $\sum_{k=-1}^2 L_k(d) \equiv 1.0$ for all $d$, preserving unity DC gain and continuous phase.

---

### 3. Interaural Level Difference (ILD) & Duda-Brown Head Shadow Filter
The human skull acts as an acoustic lowpass obstacle, creating a frequency-dependent acoustic shadow on the contralateral ear. We implement the Brown & Duda (1998) continuous-time spherical head shadow model:
$$H(s, 	heta_e) = rac{1 + s rac{lpha(	heta_e) 	au_0}{2}}{1 + s rac{	au_0}{2}}$$
where:
- $	au_0 = rac{2a}{c} pprox 0.51	ext{ ms}$
- $	heta_e$ is the incidence angle relative to the ear acoustic axis
- $lpha(	heta_e) = 0.15 + 1.05 \left(rac{1 + \cos	heta_e}{2}ight)^{1.8} \in [0.15, 1.20]$

Discretized via the Bilinear Transform ($s \leftarrow rac{2}{T_s} rac{1 - z^{-1}}{1 + z^{-1}}$):
$$H(z) = rac{b_0 + b_1 z^{-1}}{1 + a_1 z^{-1}}$$
$$b_0 = rac{2 + lpha 	au_0 f_s}{2 + 	au_0 f_s}, \quad b_1 = rac{lpha 	au_0 f_s - 2}{2 + 	au_0 f_s}, \quad a_1 = rac{	au_0 f_s - 2}{2 + 	au_0 f_s}$$
Yielding up to $-18	ext{ dB}$ attenuation for contralateral high frequencies ($>3	ext{ kHz}$) and $+2	ext{ dB}$ ipsilateral boost.

---

### 4. Elevation Pinna Notch & Concha Resonance
Vertical elevation localization and front-back disambiguation are provided by spectral coloration from the outer ear (pinna/concha folds):
- **Concha Resonance**: A parametric 2nd-order bell peaking filter centered at $4.5	ext{ kHz}$ ($Q = 2.2$, Gain $+3.5	ext{ dB}$).
- **Pinna Elevation Notch**: Destructive acoustic interference in the pinna cymba creates a spectral notch whose center frequency sweeps monotonically with elevation angle $\phi \in [-\pi/2, +\pi/2]$:
$$f_{	ext{notch}}(\phi) = 8000 + 2200 \cdot \left(rac{\phi}{\pi/2}ight)	ext{ Hz} \in [5.8	ext{ kHz}, 10.2	ext{ kHz}]$$
- **Rear Shadow**: High-frequency shelf attenuation above $3.5	ext{ kHz}$ when sound arrives from behind ($y < 0$) due to ear flap shielding.

---

### 5. Distance Attenuation & ISO 9613-1 Air Absorption
- **Inverse Distance Law**: Direct acoustic pressure scales with $1/r$ with near-field safety clamp $r_{\min} = 0.25	ext{ m}$:
$$g(r) = rac{r_0}{\max(r, r_{\min})}$$
- **Atmospheric Air Absorption**: Molecular relaxation of $O_2$ and $N_2$ absorbs high frequencies over distance. Modeled via a 1st-order dynamic lowpass filter with distance-dependent cutoff:
$$f_c(r) = rac{20000}{1 + 0.05 \cdot r}	ext{ Hz}$$

---

### 6. First-Order Ambisonics (HOA B-Format) Encoding
Encodes a 3D spherical soundfield into 4 orthogonal spherical harmonic components:
$$W = rac{S}{\sqrt{2}} \quad (	ext{0th order omnidirectional pressure})$$
$$X = S \cos(	heta) \cos(\phi) \quad (	ext{1st order figure-8 front-back})$$
$$Y = S \sin(	heta) \cos(\phi) \quad (	ext{1st order figure-8 left-right})$$
$$Z = S \sin(\phi) \quad (	ext{1st order figure-8 up-down})$$

B-format audio can be decoded to an 8-virtual-loudspeaker cube array:
$$s_k = rac{W}{\sqrt{8}} + X \cdot k_{x,k} + Y \cdot k_{y,k} + Z \cdot k_{z,k}$$
and binauralized by filtering each virtual loudspeaker through its corresponding static HRTF.

---

### 7. Shoebox Image Source Method (ISM) & Unitary FDN Reverb
- **Early Reflections**: Allen & Berkley Image Source Method in a rectangular shoebox room $[L_x, L_y, L_z]$ calculates up to 26 1st and 2nd-order wall reflection mirror sources with reflection coefficient $R = \sqrt{1 - lpha_{	ext{wall}}}$.
- **Late Reverberation**: 4-channel Feedback Delay Network (FDN) with prime coprime delay lengths $[1117, 1361, 1601, 1987]$ samples and a 4x4 unitary Householder scattering matrix:
$$\mathbf{A} = \mathbf{I} - rac{2}{N}\mathbf{1}\mathbf{1}^T = rac{1}{2} egin{bmatrix} -1 & 1 & 1 & 1 \ 1 & -1 & 1 & 1 \ 1 & 1 & -1 & 1 \ 1 & 1 & 1 & -1 \end{bmatrix}$$
Guarantees zero energy explosion ($\det(\mathbf{A}) = -1$), infinite impulse response stability, and decorrelated stereo output.

---

## 💻 Global CLI Usage (`spatial-audio`)

### Subcommands Overview
```bash
# 1. Render a 360-degree circular orbit trajectory around listener
spatial-audio orbit --radius 2.0 --speed 0.5 --duration 3.0 --output orbit.wav

# 2. Render high-speed supersonic/subsonic flyby with Doppler shift
spatial-audio flyby --speed 35.0 --miss-distance 1.5 --duration 3.0 --output flyby.wav

# 3. Render sound inside a reflective shoebox room with ISM early reflections & FDN
spatial-audio room --duration 3.0 --output room.wav

# 4. View real-time 3D Polar Soundfield Radar in Unicode Braille
spatial-audio radar --trajectory orbit
spatial-audio radar --trajectory flyby

# 5. View Stereo Phase Goniometer & Vector Scope (Mid-Side spatial width)
spatial-audio goniometer

# 6. Run comprehensive performance benchmark
spatial-audio bench

# 7. Run self-test validation suite
spatial-audio test

# 8. Print theoretical formulations and filter coefficients
spatial-audio info
```

---

## 🧪 Verification & Unit Test Suite

Run the comprehensive pytest suite:
```bash
python3 -m pytest tests/test_spatial_audio.py -v
```

Validates:
- Exact Woodworth-Schlosberg bounds ($600\,\mu	ext{s} \le 	ext{ITD}_{90^\circ} \le 750\,\mu	ext{s}$)
- Cubic Lagrange fractional delay interpolation error $< 10^{-3}$
- Contralateral head shadow attenuation asymmetry ($>12	ext{ dB}$)
- Pinna elevation notch frequency sweeping with $\phi$
- Ambisonics B-format energy preservation $2W^2 + X^2 + Y^2 + Z^2 \equiv 2.0$
- 4-Channel Unitary Householder FDN Reverb energy decay stability
- Stereo panning RMS separation ($R_{	ext{RMS}} > 1.5 L_{	ext{RMS}}$ for right-side source)
- Bit-exact RIFF 16-bit 44.1 kHz stereo WAV file header compliance
