"""
Comprehensive Unit and Integration Test Suite for zk-SNARK & R1CS-to-QAP Engine.
"""

from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "zksnark"

loader = SourceFileLoader("zksnark", str(CLI))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

FieldElement = mod.FieldElement
Polynomial = mod.Polynomial
Constraint = mod.Constraint
R1CS = mod.R1CS
CircuitBuilder = mod.CircuitBuilder
QAP = mod.QAP
r1cs_to_qap = mod.r1cs_to_qap
compute_qap_polynomials = mod.compute_qap_polynomials
Groth16System = mod.Groth16System
Proof = mod.Proof
build_polynomial_circuit = mod.build_polynomial_circuit
build_range_proof_circuit = mod.build_range_proof_circuit
build_mimc_hash_circuit = mod.build_mimc_hash_circuit
PRIME = mod.PRIME
F_ZERO = mod.F_ZERO
F_ONE = mod.F_ONE


class TestFieldElement(unittest.TestCase):
    def test_basic_arithmetic(self):
        a = FieldElement(40)
        b = FieldElement(2)
        self.assertEqual(a + b, 42)
        self.assertEqual(a - b, 38)
        self.assertEqual(a * b, 80)
        self.assertEqual(a / b, 20)
        self.assertEqual(-a, PRIME - 40)

    def test_multiplicative_inverse(self):
        a = FieldElement(123456789)
        inv_a = a.inv()
        self.assertEqual(a * inv_a, 1)

    def test_division_by_zero(self):
        a = FieldElement(10)
        with self.assertRaises(ZeroDivisionError):
            _ = a / 0
        with self.assertRaises(ZeroDivisionError):
            _ = FieldElement(0).inv()

    def test_modular_wrap(self):
        a = FieldElement(PRIME + 5)
        self.assertEqual(a.v, 5)
        b = FieldElement(-3)
        self.assertEqual(b.v, PRIME - 3)


class TestPolynomial(unittest.TestCase):
    def test_deg_and_evaluation(self):
        # P(x) = 2 + 3*x + 4*x^2
        p = Polynomial([2, 3, 4])
        self.assertEqual(p.deg(), 2)
        # P(2) = 2 + 6 + 16 = 24
        self.assertEqual(p.eval(2), 24)

    def test_addition_and_multiplication(self):
        # P(x) = x + 1, Q(x) = x - 1 -> P*Q = x^2 - 1
        p = Polynomial([1, 1])
        q = Polynomial([-1, 1])
        prod = p * q
        self.assertEqual(prod.deg(), 2)
        self.assertEqual(prod, Polynomial([-1, 0, 1]))

    def test_euclidean_division(self):
        # (x^3 - 1) / (x - 1) = x^2 + x + 1, R = 0
        p = Polynomial([-1, 0, 0, 1])
        d = Polynomial([-1, 1])
        q, r = p.divmod(d)
        self.assertEqual(q, Polynomial([1, 1, 1]))
        self.assertTrue(r.is_zero())

    def test_lagrange_interpolation(self):
        # Points: (1, 5), (2, 9), (3, 15) -> y = x^2 + x + 3
        pts = [
            (FieldElement(1), FieldElement(5)),
            (FieldElement(2), FieldElement(9)),
            (FieldElement(3), FieldElement(15)),
        ]
        poly = Polynomial.interpolate(pts)
        self.assertEqual(poly, Polynomial([3, 1, 1]))
        for x, y in pts:
            self.assertEqual(poly.eval(x), y)


class TestR1CSAndQAP(unittest.TestCase):
    def test_circuit_builder_mul_and_add(self):
        builder = CircuitBuilder()
        a = builder.allocate_var("a", is_public=False)
        b = builder.allocate_var("b", is_public=False)
        c = builder.allocate_var("c", is_public=True)
        builder.enforce_mul(a, b, c)

        witness_valid = [FieldElement(1), FieldElement(3), FieldElement(4), FieldElement(12)]
        sat, err = builder.r1cs.is_satisfied(witness_valid)
        self.assertTrue(sat)
        self.assertIsNone(err)

        witness_invalid = [FieldElement(1), FieldElement(3), FieldElement(4), FieldElement(15)]
        sat_inv, err_inv = builder.r1cs.is_satisfied(witness_invalid)
        self.assertFalse(sat_inv)
        self.assertEqual(err_inv, 0)

    def test_qap_reduction_vanishing_quotient(self):
        r1cs, witness, y_val = build_polynomial_circuit(4)
        # x=4 -> y = 4^3 + 4 + 5 = 73
        self.assertEqual(y_val, 73)
        self.assertTrue(r1cs.is_satisfied(witness)[0])

        qap = r1cs_to_qap(r1cs)
        self.assertEqual(len(qap.roots), len(r1cs.constraints))

        A_p, B_p, C_p, H_p, R_p = compute_qap_polynomials(qap, witness)
        self.assertTrue(R_p.is_zero())
        # Check that (A*B - C) == H * T
        self.assertEqual((A_p * B_p) - C_p, H_p * qap.target_poly)


class TestGroth16Protocol(unittest.TestCase):
    def setUp(self):
        self.r1cs, self.witness, self.y = build_polynomial_circuit(5)
        self.qap = r1cs_to_qap(self.r1cs)
        self.pk, self.vk = Groth16System.setup(self.qap, num_public=self.r1cs.num_public)

    def test_valid_proof_verification(self):
        proof = Groth16System.prove(self.pk, self.qap, self.r1cs.num_public, self.witness)
        public_inputs = [self.witness[i] for i in range(self.r1cs.num_public + 1)]
        self.assertTrue(Groth16System.verify(self.vk, public_inputs, proof))

    def test_zero_knowledge_blinding(self):
        # Two different random proofs for same witness must yield distinct proof elements
        proof1 = Groth16System.prove(self.pk, self.qap, self.r1cs.num_public, self.witness)
        proof2 = Groth16System.prove(self.pk, self.qap, self.r1cs.num_public, self.witness)
        self.assertNotEqual(proof1.pi_a, proof2.pi_a)
        self.assertNotEqual(proof1.pi_b, proof2.pi_b)
        self.assertNotEqual(proof1.pi_c, proof2.pi_c)
        # Yet both verify
        pub = [self.witness[0], self.witness[1]]
        self.assertTrue(Groth16System.verify(self.vk, pub, proof1))
        self.assertTrue(Groth16System.verify(self.vk, pub, proof2))

    def test_tamper_resistance_soundness(self):
        proof = Groth16System.prove(self.pk, self.qap, self.r1cs.num_public, self.witness)
        pub = [self.witness[0], self.witness[1]]

        # Tamper pi_A
        bad_a = Proof(proof.pi_a + 1, proof.pi_b, proof.pi_c)
        self.assertFalse(Groth16System.verify(self.vk, pub, bad_a))

        # Tamper pi_B
        bad_b = Proof(proof.pi_a, proof.pi_b + 1, proof.pi_c)
        self.assertFalse(Groth16System.verify(self.vk, pub, bad_b))

        # Tamper pi_C
        bad_c = Proof(proof.pi_a, proof.pi_b, proof.pi_c + 1)
        self.assertFalse(Groth16System.verify(self.vk, pub, bad_c))

        # Tamper public input
        bad_pub = [self.witness[0], self.witness[1] + 1]
        self.assertFalse(Groth16System.verify(self.vk, bad_pub, proof))


class TestPrebuiltCircuits(unittest.TestCase):
    def test_mimc_hash_circuit(self):
        secret = 777
        r1cs, witness, digest = build_mimc_hash_circuit(secret=secret, rounds=3)
        self.assertTrue(r1cs.is_satisfied(witness)[0])

        qap = r1cs_to_qap(r1cs)
        pk, vk = Groth16System.setup(qap, num_public=r1cs.num_public)
        proof = Groth16System.prove(pk, qap, r1cs.num_public, witness)
        pub = [witness[0], digest]
        self.assertTrue(Groth16System.verify(vk, pub, proof))

    def test_range_proof_circuit(self):
        # Proves 42 is in [0, 2^8 - 1]
        r1cs, witness = build_range_proof_circuit(val=42, num_bits=8)
        self.assertTrue(r1cs.is_satisfied(witness)[0])
        qap = r1cs_to_qap(r1cs)
        pk, vk = Groth16System.setup(qap, num_public=r1cs.num_public)
        proof = Groth16System.prove(pk, qap, r1cs.num_public, witness)
        pub = [witness[0]]
        self.assertTrue(Groth16System.verify(vk, pub, proof))


class TestCLIIntegration(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_info(self):
        res = self.run_cli("info")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Rank-1 Constraint System", res.stdout)
        self.assertIn("Groth16", res.stdout)

    def test_cli_compile(self):
        res = self.run_cli("compile", "--circuit", "poly", "--input", "3")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Matrix A Sparsity Pattern", res.stdout)
        self.assertIn("x=3 -> y=35", res.stdout)

    def test_cli_qap(self):
        res = self.run_cli("qap", "--circuit", "poly", "--input", "3")
        self.assertEqual(res.returncode, 0)
        self.assertIn("ZERO remainder", res.stdout)

    def test_cli_mimc(self):
        res = self.run_cli("mimc", "--secret", "42", "--rounds", "3")
        self.assertEqual(res.returncode, 0)
        self.assertIn("VALID PROOF!", res.stdout)

    def test_cli_range(self):
        res = self.run_cli("range", "--value", "1337", "--bits", "16")
        self.assertEqual(res.returncode, 0)
        self.assertIn("VALID!", res.stdout)

    def test_cli_benchmark(self):
        res = self.run_cli("benchmark")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Benchmark completed successfully", res.stdout)


if __name__ == '__main__':
    unittest.main()
