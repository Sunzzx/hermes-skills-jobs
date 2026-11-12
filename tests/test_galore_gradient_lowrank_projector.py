"""Unit tests for galore-gradient-lowrank-projector (ICML 2024, arXiv:2403.03507)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "galore-gradient-lowrank-projector",
)
mod = SourceFileLoader("galore_gradient_lowrank_projector", MOD_PATH).load_module()

_matmul = mod._matmul
_transpose = mod._transpose
_frobenius = mod._frobenius
_mse_mat = mod._mse_mat
_cosine_mat = mod._cosine_mat
_eye = mod._eye
_zeros = mod._zeros
_rand_gauss = mod._rand_gauss
_rand_ortho_cols = mod._rand_ortho_cols
_outer = mod._outer
_axpy_inplace = mod._axpy_inplace
_add = mod._add
subspace_iteration_svd = mod.subspace_iteration_svd
stable_rank = mod.stable_rank
nuclear_proxy = mod.nuclear_proxy
GaLoreConfig = mod.GaLoreConfig
AdamConfig = mod.AdamConfig
refresh_projector = mod.refresh_projector
project = mod.project
project_back = mod.project_back
choose_side = mod.choose_side
GaLoreAdamW = mod.GaLoreAdamW
FullAdamW = mod.FullAdamW
LoRAAdamW = mod.LoRAAdamW
LowRankGradSystem = mod.LowRankGradSystem
LinearRegressionTask = mod.LinearRegressionTask
GaLoreEngine = mod.GaLoreEngine
memory_theory = mod.memory_theory
theory_report = mod.theory_report


def test_matmul_identity():
    A = _rand_gauss(5, 5, seed=1)
    I = _eye(5)
    B = _matmul(A, I)
    assert _mse_mat(A, B) < 1e-18


def test_transpose_involution():
    A = _rand_gauss(4, 7, seed=2)
    assert _mse_mat(A, _transpose(_transpose(A))) < 1e-18


def test_choose_side_std():
    assert choose_side(32, 64, "std") == "left"
    assert choose_side(64, 32, "std") == "right"
    assert choose_side(32, 64, "reverse") == "right"
    assert choose_side(10, 10, "left") == "left"
    assert choose_side(10, 10, "right") == "right"


def test_svd_recovers_planted_rank():
    m, n, r = 20, 24, 3
    U = _rand_ortho_cols(m, r, seed=3)
    V = _rand_ortho_cols(n, r, seed=4)
    G = _zeros(m, n)
    for k in range(r):
        uk = [U[i][k] for i in range(m)]
        vk = [V[j][k] for j in range(n)]
        _axpy_inplace(G, _outer(uk, vk), 2.0 + k)
    Uh, S, Vh = subspace_iteration_svd(G, rank=r, n_iter=15, seed=0)
    # reconstruction
    recon = _zeros(m, n)
    for k in range(r):
        uk = [Uh[i][k] for i in range(m)]
        vk = [Vh[j][k] for j in range(n)]
        _axpy_inplace(recon, _outer(uk, vk), S[k])
    assert _cosine_mat(G, recon) > 0.99
    assert S[0] >= S[-1]


def test_stable_rank_low_for_rank1():
    u = [1.0, 0.0, 0.0, 0.0]
    v = [1.0, 2.0, 3.0]
    G = _outer(u, v)
    sr = stable_rank(G)
    assert sr < 1.5


def test_project_roundtrip_on_core():
    G = _rand_gauss(16, 20, seed=5)
    st = refresh_projector(G, GaLoreConfig(rank=4, svd_iters=10, seed=5))
    R = project(G, st)
    R2 = project(project_back(R, st), st)
    assert _mse_mat(R, R2) < 1e-10


def test_rank_r_matrix_high_cosine_recon():
    m, n, r = 18, 22, 3
    U = _rand_ortho_cols(m, r, seed=6)
    V = _rand_ortho_cols(n, r, seed=7)
    G = _zeros(m, n)
    for k in range(r):
        uk = [U[i][k] for i in range(m)]
        vk = [V[j][k] for j in range(n)]
        _axpy_inplace(G, _outer(uk, vk), 1.5)
    st = refresh_projector(G, GaLoreConfig(rank=r, svd_iters=12, seed=6))
    Ghat = project_back(project(G, st), st)
    assert _cosine_mat(G, Ghat) > 0.98


def test_lowrank_grad_system_collapses():
    sys = LowRankGradSystem.create(m=16, n=20, seed=8, eta=0.12)
    sr0 = stable_rank(sys.gradient())
    for _ in range(50):
        sys.step_gd()
    sr1 = stable_rank(sys.gradient())
    assert sr1 < sr0 * 0.85 or sr1 < 4.5


def test_galore_decreases_loss():
    eng = GaLoreEngine()
    out = eng.train(m=14, n=18, rank=4, steps=35, seed=9, method="galore", lr=0.08, gap=10)
    assert out["init_loss"] is not None
    assert out["final_loss"] < out["init_loss"] * 0.9


def test_full_adam_decreases_loss():
    eng = GaLoreEngine()
    out = eng.train(m=12, n=16, rank=3, steps=30, seed=10, method="full", lr=0.08)
    assert out["final_loss"] < out["init_loss"] * 0.9


def test_lora_decreases_loss():
    eng = GaLoreEngine()
    out = eng.train(m=12, n=16, rank=4, steps=40, seed=11, method="lora", lr=0.05)
    assert out["final_loss"] < out["init_loss"] * 0.95


def test_memory_theory_saving():
    mem = memory_theory(4096, 4096, 128)
    assert mem["galore_vs_full_saving_pct"] > 50.0
    assert mem["galore_optimizer_bytes"] < mem["full_adam_moments_bytes"]


def test_galore_optimizer_bytes_less_than_full():
    opt = GaLoreAdamW(GaLoreConfig(rank=8), AdamConfig())
    mb = opt.memory_bytes(64, 64)
    assert mb["total_optimizer"] < mb["full_adam_moments"]


def test_engine_status_ok():
    eng = GaLoreEngine()
    s = eng.status(m=20, n=24, rank=4, seed=0)
    assert s["ok"] is True
    assert s["init_loss"] > 0


def test_engine_verify_battery():
    eng = GaLoreEngine()
    out = eng.verify(trials=2, seed=0)
    assert out["n_pass"] == out["n_total"]
    assert out["ok"] is True


def test_engine_project_once():
    eng = GaLoreEngine()
    out = eng.project_once(m=20, n=24, rank=3, seed=1)
    assert out["reconstruction_cosine"] > 0.9


def test_engine_ablate_runs():
    eng = GaLoreEngine()
    out = eng.ablate(m=12, n=16, rank=3, steps=20, seed=2)
    assert out["ok"]
    assert len(out["ranking"]) == 3
    assert out["winner"] in ("full", "galore", "lora")


def test_theory_report_positive_saving():
    t = theory_report(rank=128, bytes_per=2)
    assert t["galore_saving_vs_full_pct"] > 40.0
    assert t["model_galore_gib"] < t["model_full_adam_gib"]


def test_nuclear_proxy_keys():
    G = _rand_gauss(12, 16, seed=3)
    p = nuclear_proxy(G, rank=4)
    assert "stable_rank" in p
    assert 0.0 <= p["top_r_energy_frac"] <= 1.0 + 1e-9


def test_linear_task_grad_finite():
    task = LinearRegressionTask.create(m=10, n=12, batch=8, seed=4)
    W = _rand_gauss(10, 12, seed=5, scale=0.1)
    loss, G = task.loss_grad(W)
    assert math.isfinite(loss)
    assert _frobenius(G) > 0
