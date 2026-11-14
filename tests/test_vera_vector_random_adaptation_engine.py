"""Unit tests for vera-vector-random-adaptation-engine (ICLR 2024, arXiv:2310.11454)."""

import math
import os
from importlib.machinery import SourceFileLoader
import pytest

MOD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "bin", "vera-vector-random-adaptation-engine")
)
vera_mod = SourceFileLoader("vera_engine", MOD_PATH).load_module()


def test_matrix_primitives():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[2.0, 0.0], [1.0, 2.0]]
    C = vera_mod._matmul(A, B)
    assert C == [[4.0, 4.0], [10.0, 8.0]]

    v = [2.0, 3.0]
    Mv = vera_mod._mat_vec(A, v)
    assert Mv == [8.0, 18.0]

    vM = vera_mod._vec_mat(v, A)
    assert vM == [11.0, 16.0]


def test_shared_random_matrices_determinism():
    gen1 = vera_mod.SharedRandomMatrices(d_max_out=64, d_max_in=64, rank=8, seed=42)
    gen2 = vera_mod.SharedRandomMatrices(d_max_out=64, d_max_in=64, rank=8, seed=42)
    assert gen1.B_hat == gen2.B_hat
    assert gen1.A_hat == gen2.A_hat


def test_shared_random_matrices_slicing():
    gen = vera_mod.SharedRandomMatrices(d_max_out=128, d_max_in=128, rank=16, seed=123)
    B_sub, A_sub = gen.slice_for_layer(d_out=32, d_in=48)
    assert len(B_sub) == 32
    assert len(B_sub[0]) == 16
    assert len(A_sub) == 16
    assert len(A_sub[0]) == 48

    # Submatrix matches prefix of full shared matrix
    for i in range(32):
        assert B_sub[i] == gen.B_hat[i][:16]
    for r in range(16):
        assert A_sub[r] == gen.A_hat[r][:48]


def test_vera_layer_initial_delta_zero():
    # When d is initialized to zero, effective Delta W must be strictly zero matrix
    d_out, d_in, rank = 16, 20, 4
    W0 = vera_mod._rand_gauss(d_out, d_in, seed=10, scale=0.1)
    shared = vera_mod.SharedRandomMatrices(d_out, d_in, rank, seed=10)
    B_sub, A_sub = shared.slice_for_layer(d_out, d_in)

    layer = vera_mod.VeRALayer(
        name="test_init",
        W0=W0,
        B_hat=B_sub,
        A_hat=A_sub,
        rank=rank,
        alpha=1.0,
        init_d_val=0.0,
        init_b_val=1.0,
    )

    delta = layer.effective_delta_w()
    norm_delta = vera_mod._norm_frob(delta)
    assert norm_delta == 0.0

    # Forward output must match base W0 @ x exactly
    x = [1.0] * d_in
    y_vera = layer.forward(x)
    y_base = vera_mod._mat_vec(W0, x)
    assert y_vera == y_base


def test_vera_layer_forward_matches_effective_weight():
    # Verify layer.forward(x) == layer.weight() @ x with non-zero d
    d_out, d_in, rank = 12, 16, 4
    W0 = vera_mod._rand_gauss(d_out, d_in, seed=20, scale=0.1)
    shared = vera_mod.SharedRandomMatrices(d_out, d_in, rank, seed=20)
    B_sub, A_sub = shared.slice_for_layer(d_out, d_in)

    layer = vera_mod.VeRALayer(
        name="test_fwd",
        W0=W0,
        B_hat=B_sub,
        A_hat=A_sub,
        rank=rank,
        alpha=2.0,
        init_d_val=0.5,
        init_b_val=0.8,
    )

    x = [0.5 * (i % 3) for i in range(d_in)]
    y_fwd = layer.forward(x)

    W_eff = layer.weight()
    y_mat = vera_mod._mat_vec(W_eff, x)

    for y1, y2 in zip(y_fwd, y_mat):
        assert math.isclose(y1, y2, rel_tol=1e-5, abs_tol=1e-7)


def test_backward_gradient_accuracy_finite_diff():
    # Validate analytical gradients vs numerical finite-difference gradients
    d_out, d_in, rank = 6, 8, 3
    W0 = vera_mod._rand_gauss(d_out, d_in, seed=30, scale=0.1)
    shared = vera_mod.SharedRandomMatrices(d_out, d_in, rank, seed=30)
    B_sub, A_sub = shared.slice_for_layer(d_out, d_in)

    layer = vera_mod.VeRALayer(
        name="test_grad",
        W0=W0,
        B_hat=B_sub,
        A_hat=A_sub,
        rank=rank,
        alpha=1.0,
        init_d_val=0.2,
        init_b_val=0.5,
    )

    x_batch = [
        [0.1 * i for i in range(d_in)],
        [-0.05 * i for i in range(d_in)],
    ]
    target_batch = [
        [0.2 * j for j in range(d_out)],
        [0.1 * j for j in range(d_out)],
    ]

    def eval_loss(curr_d, curr_b):
        old_d, old_b = layer.d[:], layer.b[:]
        layer.d = curr_d[:]
        layer.b = curr_b[:]
        loss = 0.0
        for xk, ytk in zip(x_batch, target_batch):
            yp = layer.forward(xk)
            for ypi, yti in zip(yp, ytk):
                diff = ypi - yti
                loss += 0.5 * diff * diff
        layer.d, layer.b = old_d, old_b
        return loss / len(x_batch)

    # Analytical
    grad_y_batch = []
    for xk, ytk in zip(x_batch, target_batch):
        yp = layer.forward(xk)
        grad_y_batch.append([yp[i] - ytk[i] for i in range(d_out)])

    grad_d_ana, grad_b_ana = layer.backward_gradients(x_batch, grad_y_batch)

    eps = 1e-6
    # Numerical for d
    for i in range(d_out):
        d_plus = layer.d[:]
        d_minus = layer.d[:]
        d_plus[i] += eps
        d_minus[i] -= eps
        num_g = (eval_loss(d_plus, layer.b) - eval_loss(d_minus, layer.b)) / (2.0 * eps)
        assert math.isclose(grad_d_ana[i], num_g, rel_tol=1e-4, abs_tol=1e-5)

    # Numerical for b
    for r in range(rank):
        b_plus = layer.b[:]
        b_minus = layer.b[:]
        b_plus[r] += eps
        b_minus[r] -= eps
        num_g = (eval_loss(layer.d, b_plus) - eval_loss(layer.d, b_minus)) / (2.0 * eps)
        assert math.isclose(grad_b_ana[r], num_g, rel_tol=1e-4, abs_tol=1e-5)


def test_parameter_compression_audit_llama7b():
    ledger = vera_mod.compute_architecture_peft_ledger("llama-7b", rank=16)
    assert ledger["layers_count"] == 32
    assert ledger["full_model_params"] > 6_000_000_000
    assert ledger["lora_trainable_params"] == 39_976_960
    assert ledger["vera_trainable_params"] == 1_363_456
    # VeRA achieves ~29.3x compression compared to LoRA on LLaMA-7B
    assert ledger["compression_vs_lora"] > 25.0
    assert ledger["vera_checkpoint_mib"] < 3.0  # Under 3 MiB


def test_multilayer_network_simulation_convergence():
    dims = [(32, 32), (32, 32), (24, 32)]
    net = vera_mod.MultiLayerVeRANetwork(dims, rank=4, seed=42)
    assert len(net.layers) == 3
    assert net.total_trainable_parameters() == (32 + 4) + (32 + 4) + (24 + 4)
    assert net.total_lora_parameters() == (32 * 4 + 4 * 32) * 2 + (24 * 4 + 4 * 32)
    assert net.parameter_compression_ratio_vs_lora() > 6.0


def test_cli_status_and_compare_dispatch(capsys):
    import argparse
    vera_mod.cmd_status(argparse.Namespace(rank=16))
    captured = capsys.readouterr()
    assert "vera-vector-random-adaptation-engine" in captured.out

    vera_mod.cmd_compare(argparse.Namespace(arch="llama-3-8b", rank=16, json=True))
    captured_json = capsys.readouterr()
    assert "LLaMA-3-8B" in captured_json.out
