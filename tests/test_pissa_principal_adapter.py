"""Unit tests for PiSSA: Principal Singular values and Singular vectors Adaptation."""

import math
import os
from importlib.machinery import SourceFileLoader
import pytest

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "pissa-principal-adapter",
)
mod = SourceFileLoader("pissa_principal_adapter", MOD_PATH).load_module()

_matmul = mod._matmul
_transpose = mod._transpose
_frobenius = mod._frobenius
_rand_gauss = mod._rand_gauss
_sub = mod._sub
_add = mod._add
thin_svd_subspace = mod.thin_svd_subspace
decompose_pissa = mod.decompose_pissa
effective_weight = mod.effective_weight
PiSSATrainer = mod.PiSSATrainer
peft_parameter_comparison = mod.peft_parameter_comparison
run_convergence_benchmark = mod.run_convergence_benchmark


def test_matrix_operations():
    A = [[1.0, 2.0], [3.0, 4.0]]
    I = [[1.0, 0.0], [0.0, 1.0]]
    assert _matmul(A, I) == A
    AT = _transpose(A)
    assert AT == [[1.0, 3.0], [2.0, 4.0]]
    assert round(_frobenius(I), 4) == round(math.sqrt(2.0), 4)


def test_svd_orthonormality():
    W = _rand_gauss(20, 16, seed=123)
    rank = 4
    U, S, V = thin_svd_subspace(W, rank=rank, n_iter=20, seed=42)
    assert len(S) == rank
    # Check singular values are non-negative and descending
    for i in range(len(S) - 1):
        assert S[i] >= S[i + 1] >= 0.0

    # U columns must be approximately orthonormal: Uᵀ U ≈ I_r
    UT = _transpose(U)
    UTU = _matmul(UT, U)
    for i in range(rank):
        for j in range(rank):
            expected = 1.0 if i == j else 0.0
            assert abs(UTU[i][j] - expected) < 1e-3


def test_pissa_decomposition_and_reconstruction():
    W = _rand_gauss(24, 16, seed=777)
    rank = 6
    decomp = decompose_pissa(W, rank=rank, n_iter=20, seed=42)

    assert decomp.rank == rank
    assert 0.0 < decomp.energy_explained <= 1.0
    assert len(decomp.A) == 24 and len(decomp.A[0]) == rank
    assert len(decomp.B) == rank and len(decomp.B[0]) == 16

    # Verify exact reconstruction: W == W_res + A * B
    W_eff = effective_weight(decomp)
    err = _frobenius(_sub(W, W_eff))
    assert err < 1e-10


def test_pissa_trainer_loss_decrease():
    m, n, r = 16, 16, 4
    W_base = _rand_gauss(m, n, seed=1)
    W_target = _rand_gauss(m, n, seed=2)
    X = _rand_gauss(8, n, seed=3)
    Y_target = _matmul(X, _transpose(W_target))

    trainer = PiSSATrainer(W_base, rank=r, mode="pissa", lr=0.05, seed=10)
    losses = [trainer.step(X, Y_target) for _ in range(15)]
    assert losses[-1] < losses[0]


def test_pissa_faster_than_lora():
    bench = run_convergence_benchmark(steps=25)
    assert bench["pissa_final_loss"] < bench["lora_final_loss"]
    assert bench["pissa_loss_reduction_pct"] > bench["lora_loss_reduction_pct"]


def test_parameter_scaling_stats():
    stats = peft_parameter_comparison(model_dim=2048, hidden_dim=8192, num_layers=16, rank=8)
    assert stats["trainable_pissa_params"] < stats["full_model_params"]
    assert stats["compression_ratio_x"] > 50
    assert stats["inference_latency_overhead_ms"] == 0.0
