import math
import os
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "spatial-audio"

loader = SourceFileLoader("spatial_audio", str(CLI))
sa = types.ModuleType(loader.name)
loader.exec_module(sa)


class TestBinauralSpatialAudio(unittest.TestCase):
    def test_vector_spherical_coordinate_transform(self):
        # Front (+y)
        v_front = sa.Vector3D(0.0, 2.0, 0.0)
        r, az, el = v_front.to_spherical()
        self.assertAlmostEqual(r, 2.0, places=5)
        self.assertAlmostEqual(az, 0.0, places=5)
        self.assertAlmostEqual(el, 0.0, places=5)

        # Right (+x)
        v_right = sa.Vector3D(3.0, 0.0, 0.0)
        r, az, el = v_right.to_spherical()
        self.assertAlmostEqual(r, 3.0, places=5)
        self.assertAlmostEqual(az, math.pi * 0.5, places=5)

        # Up (+z)
        v_up = sa.Vector3D(0.0, 0.0, 4.0)
        r, az, el = v_up.to_spherical()
        self.assertAlmostEqual(r, 4.0, places=5)
        self.assertAlmostEqual(el, math.pi * 0.5, places=5)

        # Invert spherical to cartesian roundtrip
        v_recon = sa.Vector3D.from_spherical(r, az, el)
        self.assertAlmostEqual(v_recon.x, v_up.x, places=5)
        self.assertAlmostEqual(v_recon.y, v_up.y, places=5)
        self.assertAlmostEqual(v_recon.z, v_up.z, places=5)

    def test_woodworth_itd_acoustics(self):
        # 0 azimuth (straight ahead) -> equal delay
        tau_l0, tau_r0 = sa.calculate_woodworth_itd(0.0)
        self.assertEqual(tau_l0, 0.0)
        self.assertEqual(tau_r0, 0.0)

        # 90 degrees right: right ear has zero diff delay, left ear delayed by (a/c)*(1 + pi/2)
        tau_l90, tau_r90 = sa.calculate_woodworth_itd(math.pi * 0.5)
        self.assertEqual(tau_r90, 0.0)
        expected_itd = (sa.HEAD_RADIUS / sa.SPEED_OF_SOUND) * (1.0 + math.pi * 0.5)
        self.assertAlmostEqual(tau_l90, expected_itd, places=6)
        self.assertGreater(tau_l90 * 1e6, 600.0)  # >600 us
        self.assertLess(tau_l90 * 1e6, 750.0)     # <750 us

        # Symmetry check: -90 degrees left
        tau_l_neg, tau_r_neg = sa.calculate_woodworth_itd(-math.pi * 0.5)
        self.assertEqual(tau_l_neg, 0.0)
        self.assertAlmostEqual(tau_r_neg, expected_itd, places=6)

    def test_fractional_delay_lagrange_interpolation(self):
        fd = sa.FractionalDelayLine(max_delay_samples=200)
        # Feed linear ramp
        for i in range(100):
            fd.write(float(i))

        # Exact integer delay of 10
        # Current write is at index 100, last written value was 99
        val_int10 = fd.read(10.0)
        self.assertAlmostEqual(val_int10, 89.0, places=3)

        # Half-sample delay of 10.5 should interpolate linearly to 88.5 for a straight ramp
        val_frac10 = fd.read(10.5)
        self.assertAlmostEqual(val_frac10, 88.5, places=2)

    def test_head_shadow_ild_filter(self):
        hs = sa.HeadShadowFilter()
        # Ipsilateral direct angle (0 rad)
        hs.update_coefficients(0.0)
        # Step response
        y_ipsi = [hs.process(1.0) for _ in range(50)]

        # Contralateral shadowed angle (pi rad)
        hs_contra = sa.HeadShadowFilter()
        hs_contra.update_coefficients(math.pi)
        y_contra = [hs_contra.process(1.0) for _ in range(50)]

        # High-frequency transient response must be noticeably smaller for contralateral ear
        self.assertLess(y_contra[0], y_ipsi[0])
        self.assertFalse(np.any(np.isnan(y_contra)))
        self.assertFalse(np.any(np.isinf(y_contra)))

    def test_pinna_elevation_notch(self):
        pinna = sa.PinnaElevationFilter()
        # Below horizon (-45 deg)
        out_low = [pinna.process(math.sin(i * 0.2), -math.pi * 0.25, False) for i in range(200)]
        # Above horizon (+45 deg)
        out_high = [pinna.process(math.sin(i * 0.2), math.pi * 0.25, False) for i in range(200)]
        self.assertFalse(np.any(np.isnan(out_low)))
        self.assertFalse(np.any(np.isnan(out_high)))

    def test_ambisonics_bformat_energy_and_decoding(self):
        # Test various directions
        for az_deg in (0, 45, 90, 180, 270):
            for el_deg in (-30, 0, 45):
                az = math.radians(az_deg)
                el = math.radians(el_deg)
                w, x, y, z = sa.AmbisonicsBFormat.encode(1.0, az, el)
                # Theoretical B-format energy
                energy = w * w * 2.0 + (x * x + y * y + z * z)
                self.assertAlmostEqual(energy, 2.0, places=4)

                # Decode to virtual cube (8 speakers)
                speakers = sa.AmbisonicsBFormat.decode_to_virtual_cube(w, x, y, z)
                self.assertEqual(len(speakers), 8)
                self.assertFalse(np.any(np.isnan(speakers)))

    def test_feedback_delay_network_stability(self):
        fdn = sa.FeedbackDelayNetworkReverb(sample_rate=44100, t60=0.15)
        # Unit impulse
        outputs_l = []
        outputs_r = []
        for i in range(15000):
            in_val = 1.0 if i == 0 else 0.0
            l, r = fdn.process(in_val, in_val)
            outputs_l.append(l)
            outputs_r.append(r)

        # Energy when echoes arrive (samples 1200..3000) vs long tail (samples 12000..15000)
        active_energy = sum(x * x for x in outputs_l[1200:3000])
        tail_energy = sum(x * x for x in outputs_l[12000:15000])
        self.assertGreater(active_energy, 1e-4)
        self.assertLess(tail_energy, active_energy * 0.01)
        self.assertLess(max(abs(x) for x in outputs_l), 1.5)

    def test_master_spatializer_stereo_panning(self):
        spat = sa.BinauralSpatializer(enable_room=False)
        duration = 0.1
        sig = np.sin(2.0 * np.pi * 440.0 * np.arange(int(duration * sa.SAMPLE_RATE)) / sa.SAMPLE_RATE)

        # Source strictly on the RIGHT (+x = 2.0m)
        left_r, right_r = spat.render_trajectory(sig, lambda t, d: sa.Vector3D(2.0, 0.0, 0.0), duration)
        rms_left = np.sqrt(np.mean(left_r ** 2))
        rms_right = np.sqrt(np.mean(right_r ** 2))
        self.assertGreater(rms_right, rms_left * 1.5, "Right channel must dominate when sound is on the right")

        # Clamping limiter check: no audio values exceed [-1.0, 1.0]
        self.assertTrue(np.all(np.abs(left_r) <= 1.0))
        self.assertTrue(np.all(np.abs(right_r) <= 1.0))

    def test_riff_wav_export(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            l = np.array([0.5, -0.5, 0.2, -0.2], dtype=np.float32)
            r = np.array([0.4, -0.4, 0.1, -0.1], dtype=np.float32)
            sa.export_wav(tmp_path, l, r)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "rb") as f:
                hdr = f.read(44)
            self.assertEqual(hdr[:4], b"RIFF")
            self.assertEqual(hdr[8:12], b"WAVE")
            self.assertEqual(hdr[12:16], b"fmt ")
            audio_fmt, channels, srate, brate, balign, bits = struct.unpack("<HHIIHH", hdr[20:36])
            self.assertEqual(audio_fmt, 1)      # PCM
            self.assertEqual(channels, 2)       # Stereo
            self.assertEqual(srate, 44100)      # 44.1 kHz
            self.assertEqual(bits, 16)          # 16-bit
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_cli_execution(self):
        # Test 'test' subcommand
        r_test = subprocess.run([sys.executable, str(CLI), "test"], capture_output=True, text=True)
        self.assertEqual(r_test.returncode, 0, r_test.stderr)
        self.assertIn("7/7 tests passed", r_test.stdout)

        # Test 'info' subcommand
        r_info = subprocess.run([sys.executable, str(CLI), "info"], capture_output=True, text=True)
        self.assertEqual(r_info.returncode, 0, r_info.stderr)
        self.assertIn("Woodworth-Schlosberg", r_info.stdout)

        # Test 'radar' subcommand
        r_radar = subprocess.run([sys.executable, str(CLI), "radar", "--trajectory", "orbit"], capture_output=True, text=True)
        self.assertEqual(r_radar.returncode, 0, r_radar.stderr)
        self.assertIn("3D Polar Soundfield Radar", r_radar.stdout)

        # Test 'bench' subcommand
        r_bench = subprocess.run([sys.executable, str(CLI), "bench"], capture_output=True, text=True)
        self.assertEqual(r_bench.returncode, 0, r_bench.stderr)
        self.assertIn("Benchmark finished successfully", r_bench.stdout)


if __name__ == "__main__":
    unittest.main()
