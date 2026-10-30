import os
import struct
import subprocess
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "fm-synth"

loader = SourceFileLoader("fm_synth", str(CLI))
fm = types.ModuleType(loader.name)
loader.exec_module(fm)


class PolyphonicFMTests(unittest.TestCase):
    def test_midi_pitch_tuning(self):
        # Standard pitch A4 (MIDI 69) must equal 440 Hz
        f_a4 = fm.midi_to_freq(69)
        self.assertAlmostEqual(f_a4, 440.0, places=5)
        # C4 (MIDI 60)
        f_c4 = fm.midi_to_freq(60)
        self.assertAlmostEqual(f_c4, 261.625565, places=3)
        # Note string parsing
        self.assertEqual(fm.note_to_midi("A4"), 69)
        self.assertEqual(fm.note_to_midi("C4"), 60)
        self.assertEqual(fm.note_to_midi("F#3"), 54)

    def test_dx7_envelope_transitions(self):
        eg = fm.DX7Envelope(rates=(99, 99, 99, 99), levels=(99, 80, 50, 0))
        self.assertEqual(eg.state, fm.EGState.IDLE)
        eg.key_on(velocity=100)
        self.assertEqual(eg.state, fm.EGState.ATTACK)
        self.assertFalse(eg.is_finished)

        # Progress through attack
        levels = [eg.tick() for _ in range(300)]
        self.assertGreater(max(levels), 0.9)

        # Key off into release
        eg.key_off()
        self.assertEqual(eg.state, fm.EGState.RELEASE)
        rel_levels = [eg.tick() for _ in range(500)]
        self.assertLess(rel_levels[-1], 0.05)
        self.assertTrue(eg.is_finished)

    def test_operator_feedback_stability(self):
        # Self-feedback loop with 2-sample anti-aliasing average must not diverge
        config = fm.OperatorConfig(ratio=1.0, feedback=1.5, output_level=99)
        op = fm.Operator(1, config)
        op.set_frequency(440.0, velocity=127)
        samples = [op.tick() for _ in range(4000)]
        self.assertFalse(np.any(np.isnan(samples)) or np.any(np.isinf(samples)))
        self.assertLess(max(abs(s) for s in samples), 2.0)

    def test_all_32_dx7_algorithms_dag_integrity(self):
        self.assertEqual(len(fm.DX7_ALGORITHMS), 32)
        for algo_id in range(1, 33):
            algo = fm.DX7_ALGORITHMS[algo_id]
            self.assertIn("carriers", algo)
            self.assertIn("routes", algo)
            self.assertIn("feedback_op", algo)
            self.assertTrue(1 <= algo["feedback_op"] <= 6)
            for c in algo["carriers"]:
                self.assertTrue(1 <= c <= 6)
            # Verify topological sort does not cycle
            order = fm.get_topological_eval_order(algo_id)
            self.assertEqual(len(order), 6)
            self.assertEqual(set(order), {1, 2, 3, 4, 5, 6})

    def test_polyphony_and_voice_stealing(self):
        poly = fm.PolyphonicFMSynth(num_voices=4)
        # Allocate 4 voices
        v1 = poly.note_on(60, 100)
        v2 = poly.note_on(64, 100)
        v3 = poly.note_on(67, 100)
        v4 = poly.note_on(71, 100)
        active_voices = [v for v in poly.voices if v.state == fm.VoiceState.ACTIVE]
        self.assertEqual(len(active_voices), 4)

        # 5th note on must steal the oldest voice without crashing
        v5 = poly.note_on(76, 100)
        self.assertIsNotNone(v5)
        active_after_steal = [v for v in poly.voices if v.state == fm.VoiceState.ACTIVE]
        self.assertEqual(len(active_after_steal), 4)

    def test_master_limiter_bounds(self):
        poly = fm.PolyphonicFMSynth(num_voices=16)
        poly.load_preset(fm.PRESETS["cyberpunk-drone"])
        # Massive 12-note cluster strike at maximum velocity
        for note in range(36, 48):
            poly.note_on(note, 127)
        lb, rb = poly.render_buffer(0.05)
        self.assertTrue(np.all(np.abs(lb) <= 1.0))
        self.assertTrue(np.all(np.abs(rb) <= 1.0))
        self.assertFalse(np.any(np.isnan(lb)))
        self.assertFalse(np.any(np.isnan(rb)))

    def test_riff_wav_export_conformance(self):
        synth = fm.PolyphonicFMSynth(2)
        synth.note_on(60, 100)
        left, right = synth.render_buffer(0.1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fm.export_wav(tmp_path, left, right)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "rb") as f:
                hdr = f.read(44)
            self.assertEqual(hdr[:4], b"RIFF")
            self.assertEqual(hdr[8:12], b"WAVE")
            self.assertEqual(hdr[12:16], b"fmt ")
            audio_format = struct.unpack("<H", hdr[20:22])[0]
            channels = struct.unpack("<H", hdr[22:24])[0]
            samplerate = struct.unpack("<I", hdr[24:28])[0]
            bits = struct.unpack("<H", hdr[34:36])[0]
            self.assertEqual(audio_format, 1)  # PCM
            self.assertEqual(channels, 2)      # Stereo
            self.assertEqual(samplerate, 44100)
            self.assertEqual(bits, 16)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_cli_subcommands(self):
        # Test bench
        r_bench = subprocess.run([sys.executable, str(CLI), "bench"], capture_output=True, text=True)
        self.assertEqual(r_bench.returncode, 0, r_bench.stderr)
        self.assertIn("Benchmarking Polyphonic FM DSP Engine", r_bench.stdout)

        # Test test command
        r_test = subprocess.run([sys.executable, str(CLI), "test"], capture_output=True, text=True)
        self.assertEqual(r_test.returncode, 0, r_test.stderr)
        self.assertIn("Summary: 8/8 tests passed", r_test.stdout)

        # Test algo visualizer
        r_algo = subprocess.run([sys.executable, str(CLI), "algo", "5"], capture_output=True, text=True)
        self.assertEqual(r_algo.returncode, 0, r_algo.stderr)
        self.assertIn("Yamaha DX7 Algorithm 5", r_algo.stdout)

        # Test presets list
        r_presets = subprocess.run([sys.executable, str(CLI), "presets"], capture_output=True, text=True)
        self.assertEqual(r_presets.returncode, 0, r_presets.stderr)
        self.assertIn("dx7-rhodes", r_presets.stdout)


if __name__ == "__main__":
    unittest.main()
