"""Comprehensive Unit and Integration Test Suite for Quantum Circuit Simulator & Multi-Qubit Entanglement Engine."""

from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "quantum-sim"

loader = SourceFileLoader("quantum_sim", str(BIN_PATH))
qsim_mod = types.ModuleType(loader.name)
loader.exec_module(qsim_mod)

QuantumCircuit = qsim_mod.QuantumCircuit
run_verification_suite = qsim_mod.run_verification_suite
run_quantum_teleportation = qsim_mod.run_quantum_teleportation
run_grover_search = qsim_mod.run_grover_search


class TestQuantumGatePrimitives(unittest.TestCase):
    def test_hadamard_inversion(self):
        """H^2 = I: Applying Hadamard twice restores the initial state."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        self.assertAlmostEqual(qc.probabilities()[0], 1.0, places=7)
        self.assertAlmostEqual(qc.probabilities()[1], 0.0, places=7)

    def test_pauli_involutions(self):
        """X^2 = I, Y^2 = I, Z^2 = I: Pauli gates are self-inverse unitaries."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)
        self.assertAlmostEqual(qc.probabilities()[0], 1.0, places=7)

        qc.y(0)
        qc.y(0)
        self.assertAlmostEqual(qc.probabilities()[0], 1.0, places=7)

        qc.z(0)
        qc.z(0)
        self.assertAlmostEqual(qc.probabilities()[0], 1.0, places=7)

    def test_phase_and_t_gates(self):
        """T^2 = S, S^2 = Z, T^4 = Z: Phase hierarchy holds."""
        qc1 = QuantumCircuit(1)
        qc1.x(0)  # prepare |1>
        qc1.s(0)
        qc1.s(0)

        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.z(0)

        self.assertAlmostEqual(qc1.state[1].real, qc2.state[1].real, places=7)
        self.assertAlmostEqual(qc1.state[1].imag, qc2.state[1].imag, places=7)

    def test_born_rule_normalization(self):
        """Statevector probabilities must sum to 1.0 under arbitrary unitary rotations."""
        qc = QuantumCircuit(3)
        qc.rx(0, 0.785)
        qc.ry(1, 1.234)
        qc.rz(2, 2.345)
        qc.cx(0, 1)
        qc.cx(1, 2)
        total_p = sum(qc.probabilities())
        self.assertAlmostEqual(total_p, 1.0, places=9)


class TestMultiQubitEntanglement(unittest.TestCase):
    def test_bell_state_generation_and_entropy(self):
        """Bell state |Phi+> = (|00> + |11>)/sqrt(2) has S = 1.0 ebit and purity = 0.5."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        probs = qc.probabilities()
        self.assertAlmostEqual(probs[0], 0.5, places=7)
        self.assertAlmostEqual(probs[1], 0.0, places=7)
        self.assertAlmostEqual(probs[2], 0.0, places=7)
        self.assertAlmostEqual(probs[3], 0.5, places=7)

        entropy, purity, _ = qc.entanglement_entropy_and_purity(0)
        self.assertAlmostEqual(entropy, 1.0, places=6)
        self.assertAlmostEqual(purity, 0.5, places=6)

    def test_separable_product_state_entropy(self):
        """Separable product states have zero entanglement entropy and unit purity."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        entropy, purity, _ = qc.entanglement_entropy_and_purity(0)
        self.assertAlmostEqual(entropy, 0.0, places=6)
        self.assertAlmostEqual(purity, 1.0, places=6)

    def test_ghz_multipartite_entanglement(self):
        """5-qubit GHZ state (|00000> + |11111>)/sqrt(2) has S(q0) = 1.0 ebit."""
        qc = QuantumCircuit(5)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)
        entropy, purity, _ = qc.entanglement_entropy_and_purity(0)
        self.assertAlmostEqual(entropy, 1.0, places=6)
        self.assertAlmostEqual(purity, 0.5, places=6)


class TestUniversalLogicAndControlledGates(unittest.TestCase):
    def test_toffoli_ccx_gate(self):
        """CCX flips target if and only if both control qubits are |1>."""
        for c0 in [0, 1]:
            for c1 in [0, 1]:
                qc = QuantumCircuit(3)
                if c0: qc.x(0)
                if c1: qc.x(1)
                qc.ccx(0, 1, 2)
                expected_target = 1 if (c0 and c1) else 0
                expected_idx = c0 | (c1 << 1) | (expected_target << 2)
                self.assertAlmostEqual(qc.probabilities()[expected_idx], 1.0, places=7)

    def test_cswap_fredkin_gate(self):
        """CSWAP swaps targets if and only if control qubit is |1>."""
        qc = QuantumCircuit(3)
        qc.x(0)  # control = 1 (bit 0 = 1)
        qc.x(1)  # target1 = 1, target2 = 0 (bit 1 = 1, bit 2 = 0 -> idx = 1 + 2 = 3)
        qc.cswap(0, 1, 2)  # swaps target1 and target2 -> bit 1 = 0, bit 2 = 1 -> idx = 1 + 4 = 5
        self.assertAlmostEqual(qc.probabilities()[5], 1.0, places=7)

        qc_off = QuantumCircuit(3)
        qc_off.x(1)  # control = 0, target1 = 1, target2 = 0 -> idx = 2
        qc_off.cswap(0, 1, 2)  # no swap -> still idx = 2
        self.assertAlmostEqual(qc_off.probabilities()[2], 1.0, places=7)


class TestLandmarkQuantumAlgorithms(unittest.TestCase):
    def test_quantum_teleportation_fidelity(self):
        """Quantum teleportation transfers an arbitrary unknown state with 100% fidelity."""
        random.seed(1337)
        for _ in range(5):
            alpha = complex(random.gauss(0, 1), random.gauss(0, 1))
            beta = complex(random.gauss(0, 1), random.gauss(0, 1))
            res = run_quantum_teleportation(alpha, beta)
            self.assertAlmostEqual(res["fidelity"], 1.0, places=6)

    def test_grover_search_amplification(self):
        """Grover's search on 3 qubits amplifies marked state to > 90% probability."""
        res = run_grover_search(num_qubits=3, target_index=5)
        self.assertGreater(res["final_probability"], 0.90)
        self.assertEqual(res["target_binary"], "101")
        self.assertTrue(res["success"])

    def test_bloch_vector_computations(self):
        """Bloch vectors match theoretical coordinates for basis states."""
        qc = QuantumCircuit(1)
        # |0> has Bloch vector (0, 0, 1)
        bx, by, bz = qc.bloch_vector(0)
        self.assertAlmostEqual(bx, 0.0, places=6)
        self.assertAlmostEqual(by, 0.0, places=6)
        self.assertAlmostEqual(bz, 1.0, places=6)

        # |1> has Bloch vector (0, 0, -1)
        qc.x(0)
        bx, by, bz = qc.bloch_vector(0)
        self.assertAlmostEqual(bx, 0.0, places=6)
        self.assertAlmostEqual(by, 0.0, places=6)
        self.assertAlmostEqual(bz, -1.0, places=6)

        # |+> has Bloch vector (1, 0, 0)
        qc.reset()
        qc.h(0)
        bx, by, bz = qc.bloch_vector(0)
        self.assertAlmostEqual(bx, 1.0, places=6)
        self.assertAlmostEqual(by, 0.0, places=6)
        self.assertAlmostEqual(bz, 0.0, places=6)


class TestCLIExecution(unittest.TestCase):
    def test_cli_test_suite(self):
        res = subprocess.run([str(BIN_PATH), "test"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("12 Passed", res.stdout)
        self.assertIn("0 Failed", res.stdout)

    def test_cli_entropy_bell(self):
        res = subprocess.run([str(BIN_PATH), "entropy", "--state", "bell"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("1.000000 ebits", res.stdout)

    def test_cli_demo_teleport(self):
        res = subprocess.run([str(BIN_PATH), "demo", "teleport"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("100% Exact Transfer", res.stdout)


if __name__ == '__main__':
    unittest.main()
