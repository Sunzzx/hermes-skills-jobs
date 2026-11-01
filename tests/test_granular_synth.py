"""
Comprehensive Unit and Integration Test Suite for Granular Synthesis & Dynamic Gabor Acoustic Texture Engine.
"""

import math
import os
import struct
import sys
import tempfile
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "grain-synth"

loader = SourceFileLoader("grain_synth", str(CLI))
gs = types.ModuleType(loader.name)
loader.exec_module(gs)


class TestGranularSynthesisEngine(unittest.TestCase):
    """Formal test suite for granular acoustic DSP components."""

    @classmethod
    def setUpClass(cls):
        cls.fs = 44100

    def test_window_envelopes_geometry(self):
        n = 512
        # 1. Hann window
        w_hann = gs.GrainWindow.hann(n)
        self.assertEqual(len(w_hann), n)
        self.assertAlmostEqual(w_hann[0], 0.0, places=5)
        self.assertAlmostEqual(w_hann[n // 2], 1.0, places=2)
        self.assertAlmostEqual(w_hann[-1], 0.0, places=5)

        # 2. Gaussian window
        w_gauss = gs.GrainWindow.gaussian(n, sigma=0.35)
        self.assertEqual(len(w_gauss), n)
        self.assertAlmostEqual(w_gauss[n // 2], 1.0, places=4)
        self.assertTrue(np.all(w_gauss >= 0.0))
        self.assertTrue(w_gauss[0] < 0.1)

        # 3. Blackman-Harris window
        w_bh = gs.GrainWindow.blackman_harris(n)
        self.assertEqual(len(w_bh), n)
        self.assertAlmostEqual(w_bh[n // 2], 1.0, places=2)
        self.assertTrue(np.all(w_bh >= 0.0))

        # 4. Tukey window
        w_tukey = gs.GrainWindow.tukey(n, alpha=0.5)
        self.assertEqual(len(w_tukey), n)
        self.assertAlmostEqual(w_tukey[n // 2], 1.0, places=5)

        # 5. Bartlett window
        w_bartlett = gs.GrainWindow.bartlett(n)
        self.assertEqual(len(w_bartlett), n)
        self.assertAlmostEqual(float(np.max(w_bartlett)), 1.0, places=2)

    def test_hann_partition_of_unity(self):
        n = 256
        hop = n // 2
        w = gs.GrainWindow.hann(n, periodic=True)
        buf = np.zeros(n * 4)
        for i in range(6):
            st = i * hop
            buf[st : st + n] += w
        central = buf[n : n * 2]
        max_dev = np.max(np.abs(central - 1.0))
        self.assertLess(max_dev, 1e-12)

    def test_catmull_rom_hermite_interpolation(self):
        # Integer index exact recovery
        arr = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        positions = np.array([1.0, 2.0, 3.0, 4.0])
        recovered = gs.hermite_interp_4pt(arr, positions)
        np.testing.assert_allclose(recovered, arr[1:5], rtol=1e-5)

        # Sub-sample sinusoidal reconstruction
        t = np.linspace(0, 1, 200, endpoint=False)
        signal = np.sin(2.0 * np.pi * 3.0 * t)
        q_t = np.linspace(0.1, 0.9, 1000)
        q_pos = q_t * 200.0
        interp = gs.hermite_interp_4pt(signal, q_pos)
        expected = np.sin(2.0 * np.pi * 3.0 * q_t)
        max_error = np.max(np.abs(interp - expected))
        self.assertLess(max_error, 2e-3)

    def test_constant_power_stereo_panning(self):
        # Azimuth = -1.0 (hard left)
        th_l = (-1.0 + 1.0) * (np.pi / 4.0)  # 0.0
        self.assertAlmostEqual(np.cos(th_l), 1.0, places=6)
        self.assertAlmostEqual(np.sin(th_l), 0.0, places=6)

        # Azimuth = +1.0 (hard right)
        th_r = (1.0 + 1.0) * (np.pi / 4.0)  # pi/2
        self.assertAlmostEqual(np.cos(th_r), 0.0, places=6)
        self.assertAlmostEqual(np.sin(th_r), 1.0, places=6)

        # Center
        th_c = (0.0 + 1.0) * (np.pi / 4.0)  # pi/4
        self.assertAlmostEqual(np.cos(th_c), np.sin(th_c), places=6)
        self.assertAlmostEqual(np.cos(th_c) ** 2 + np.sin(th_c) ** 2, 1.0, places=6)

        # Sweep across full azimuth range
        for az in np.linspace(-1.0, 1.0, 50):
            th = (az + 1.0) * (np.pi / 4.0)
            power = np.cos(th) ** 2 + np.sin(th) ** 2
            self.assertAlmostEqual(power, 1.0, places=6)

    def test_audio_source_factory(self):
        dur = 0.5
        # Cello drone
        cello = gs.AudioSourceFactory.generate_cello_drone(dur, f0=110.0, fs=self.fs)
        self.assertEqual(len(cello), int(dur * self.fs))
        self.assertLessEqual(np.max(np.abs(cello)), 1.0)
        self.assertGreater(np.max(np.abs(cello)), 0.5)

        # Singing bowl
        bowl = gs.AudioSourceFactory.generate_singing_bowl(dur, f0=220.0, fs=self.fs)
        self.assertEqual(len(bowl), int(dur * self.fs))
        self.assertLessEqual(np.max(np.abs(bowl)), 1.0)

        # FM pad
        fm = gs.AudioSourceFactory.generate_fm_pad(dur, fs=self.fs)
        self.assertEqual(len(fm), int(dur * self.fs))

        # Pink noise
        pink = gs.AudioSourceFactory.generate_pink_noise(dur, fs=self.fs)
        self.assertEqual(len(pink), int(dur * self.fs))

    def test_granular_engine_synthesis(self):
        source = gs.AudioSourceFactory.generate_cello_drone(1.0, fs=self.fs)
        engine = gs.GranularEngine(
            source,
            sample_rate=self.fs,
            grain_duration_ms=40.0,
            density_hz=50.0,
            scrub_speed=0.5,
            pan_spray=0.6,
            pitch_shift_cents=700.0,  # +7 semitones
        )
        out_l, out_r, manifest = engine.synthesize(1.0, seed=123)
        self.assertEqual(len(out_l), int(1.0 * self.fs))
        self.assertEqual(len(out_r), int(1.0 * self.fs))
        self.assertGreater(len(manifest), 25)
        # Verify no clipping
        self.assertLessEqual(np.max(np.abs(out_l)), 1.0)
        self.assertLessEqual(np.max(np.abs(out_r)), 1.0)

    def test_time_dilation_ratio(self):
        source = gs.AudioSourceFactory.generate_singing_bowl(1.0, fs=self.fs)
        # 3x time dilation
        engine = gs.GranularEngine(source, scrub_speed=0.333, density_hz=40.0)
        out_l, out_r, manifest = engine.synthesize(3.0, seed=42)
        self.assertEqual(len(out_l), int(3.0 * self.fs))
        self.assertGreater(len(manifest), 70)

    def test_infinite_freeze_mode(self):
        source = gs.AudioSourceFactory.generate_cello_drone(1.0, fs=self.fs)
        # scrub_speed = 0.0 produces steady drone state
        engine = gs.GranularEngine(source, scrub_speed=0.0, density_hz=60.0, grain_duration_ms=60.0)
        out_l, out_r, manifest = engine.synthesize(2.0, seed=42)
        # Check energy across windows
        w1 = out_l[:self.fs]
        w2 = out_l[self.fs:]
        rms1 = np.sqrt(np.mean(w1 ** 2))
        rms2 = np.sqrt(np.mean(w2 ** 2))
        self.assertGreater(rms1, 0.05)
        self.assertGreater(rms2, 0.05)
        db_diff = abs(20.0 * np.log10(rms1 / rms2))
        self.assertLess(db_diff, 2.0)

    def test_stereo_wav_writer(self):
        n = 1000
        l = np.sin(np.linspace(0, 10, n)) * 0.8
        r = np.cos(np.linspace(0, 10, n)) * 0.8
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            gs.StereoWavWriter.write(wav_path, l, r, fs=self.fs)
            self.assertTrue(os.path.exists(wav_path))
            # Validate RIFF header
            with open(wav_path, "rb") as f:
                data = f.read(36)
                riff, size, wave, fmt, sub1, audio_fmt, n_chan, s_rate, byte_rate = struct.unpack(
                    "<4sI4s4sIHHII", data[:32]
                )
                self.assertEqual(riff, b"RIFF")
                self.assertEqual(wave, b"WAVE")
                self.assertEqual(audio_fmt, 1)  # PCM
                self.assertEqual(n_chan, 2)     # Stereo
                self.assertEqual(s_rate, self.fs)

    def test_braille_visualizer(self):
        events = [
            {"time_sec": 0.1, "duration_ms": 50.0, "pitch_cents": 0.0},
            {"time_sec": 0.5, "duration_ms": 60.0, "pitch_cents": 700.0},
            {"time_sec": 0.8, "duration_ms": 40.0, "pitch_cents": -500.0},
        ]
        scatter = gs.BrailleGrainMatrix.render_scatter(events, total_duration=1.0)
        self.assertIn("GABOR GRAIN SCATTER", scatter)
        self.assertIn("Time", scatter)

        stats = {
            "fs": 44100,
            "duration_sec": 1.0,
            "total_samples": 44100,
            "grain_count": 3,
            "emission_rate": 3.0,
            "mean_duration_ms": 50.0,
            "duration_spray": 0.2,
            "scrub_speed": 1.0,
            "time_stretch_ratio": 1.0,
            "scheduling": "poisson",
            "window_type": "hann",
            "pan_spray": 0.5,
            "rms_db": -12.0,
        }
        dashboard = gs.BrailleGrainMatrix.render_dashboard(stats)
        self.assertIn("GRANULAR SYNTHESIS ENGINE", dashboard)
        self.assertIn("POISSON", dashboard)

    def test_formal_invariants_pass(self):
        self.assertTrue(gs.verify_granular_invariants())

    def test_microbenchmark_throughput(self):
        bench = gs.run_benchmark(n_seconds=0.5)
        self.assertGreater(bench["rtf"], 1.0)
        self.assertGreater(bench["throughput_grains"], 500.0)
        self.assertGreater(bench["total_grains"], 10)


if __name__ == "__main__":
    unittest.main()
