"""Unit tests for dora-weight-decomposed-adapter (ICML 2024, arXiv:2402.09353)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "dora-weight-decomposed-adapter",
)
mod = SourceFileLoader("dora_weight_decomposed_adapter", MOD_PATH).load_module()

_matmul = mod._matmul
_transpose = mod._transpose
_frobenius = mod._frobenius
_mse_mat = mod._mse_mat
_cosine_mat = mod._cosine_mat
_eye = mod._eye
_zeros = mod._zeros
_rand_gauss = mod._rand_gauss
_add = mod._add
_scale = mod._scale
column_norms = mod.column_norms
normalize_columns = mod.normalize_columns
recompose = mod.recompose
decompose_weight = mod.decompose_weight
effective_weight_lora = mod.effective_weight_lora
effective_weight_dora = mod.effective_weight_dora
split_grad_mag_dir = mod.split_grad_mag_dir
peft_param_ledger = mod.peft_param_ledger
theory_report = mod.theory_report
DoRAConfig = mod.DoRAConfig
AdamConfig = mod.AdamConfig
DoRATrainer = mod.DoRATrainer
LoRATrainer = mod.LoRATrainer
FullFTTrainer = mod.FullFTTrainer
LinearRegressionTask = mod.LinearRegressionTask
DoRAEngine = mod.DoRAEngine


def test_matmul_identity():
    A = _rand_gauss(5, 5, seed=1)
    I = _eye(5)
    B = _matmul(A, I)
    assert _mse_mat(A, B) < 1e-18


def test_transpose_involution():
    A = _rand_gauss(4, 7, seed=2)
    assert _mse_mat(A, _transpose(_transpose(A))) < 1e-18


def test_column_norms_unit_after_normalize():
    W = _rand_gauss(10, 6, seed=3, scale=2.0)
    V, mag = normalize_columns(W)
    norms = column_norms(V)
    for n in norms:
        assert abs(n - 1.0) < 1e-9
    assert all(m > 0 for m in mag)


def test_decompose_recompose_identity():
    W = _rand_gauss(12, 9, seed=4, scale=1.5)
    dec = decompose_weight(W)
    W2 = recompose(dec["magnitude"], dec["direction"])
    assert _mse_mat(W, W2) < 1e-18
    assert dec["recon_cosine"] > 0.999999


def test_lora_init_zero_delta():
    W0 = _rand_gauss(8, 10, seed=5, scale=0.5)
    cfg = DoRAConfig(rank=3, seed=5)
    tr = LoRATrainer(W0, cfg, AdamConfig(lr=0.01))
    assert _mse_mat(tr.weight(), W0) < 1e-18


def test_dora_init_matches_base():
    W0 = _rand_gauss(8, 10, seed=6, scale=0.7)
    cfg = DoRAConfig(rank=3, seed=6)
    tr = DoRATrainer(W0, cfg, AdamConfig(lr=0.01))
    # B=0 so V=W0; m=||W0||_c → exact recovery
    assert _mse_mat(tr.weight(), W0) < 1e-10


def test_dora_params_equal_lora_plus_n():
    led = peft_param_ledger(64, 48, 8)
    assert led["dora_params"] == led["lora_params"] + 48
    assert led["dora_params"] < led["full_ft_params"]
    # small matrices have higher PEFT ratio; LLaMA-scale is <<5%
    assert led["lora_vs_full_pct"] < 50.0
    big = peft_param_ledger(4096, 4096, 16)
    assert big["lora_vs_full_pct"] < 1.0
    assert big["dora_vs_full_pct"] < 1.0


def test_grad_split_sums_to_one():
    W = _rand_gauss(15, 11, seed=7)
    G = _rand_gauss(15, 11, seed=8)
    sp = split_grad_mag_dir(W, G)
    assert abs(sp["magnitude_frac"] + sp["direction_frac"] - 1.0) < 1e-9
    assert sp["magnitude_frac"] >= 0.0
    assert sp["direction_frac"] >= 0.0


def test_dora_train_decreases_loss():
    eng = DoRAEngine()
    out = eng.train(m=12, n=16, rank=4, steps=35, seed=9, method="dora", lr=0.08)
    assert out["final_loss"] < out["init_loss"] * 0.95


def test_lora_train_decreases_loss():
    eng = DoRAEngine()
    out = eng.train(m=12, n=16, rank=4, steps=35, seed=10, method="lora", lr=0.08)
    assert out["final_loss"] < out["init_loss"] * 0.95


def test_full_train_decreases_loss():
    eng = DoRAEngine()
    out = eng.train(m=12, n=16, rank=4, steps=30, seed=11, method="full", lr=0.08)
    assert out["final_loss"] < out["init_loss"] * 0.9


def test_linear_task_grad_finite():
    task = LinearRegressionTask.create(m=10, n=12, batch=8, seed=12)
    W = _rand_gauss(10, 12, seed=13, scale=0.1)
    loss, G = task.loss_grad(W)
    assert math.isfinite(loss)
    assert _frobenius(G) > 0.0


def test_engine_status_ok():
    eng = DoRAEngine()
    s = eng.status(m=20, n=24, rank=4, seed=0)
    assert s["ok"] is True
    assert s["recon_cosine"] > 0.999
    assert s["trainable_dora"] == s["trainable_lora"] + 24


def test_engine_verify_battery():
    eng = DoRAEngine()
    out = eng.verify(trials=2, seed=0)
    assert out["n_pass"] == out["n_total"]
    assert out["ok"] is True


def test_engine_ablate_runs():
    eng = DoRAEngine()
    out = eng.ablate(m=12, n=16, rank=3, steps=20, seed=2)
    assert out["ok"]
    assert len(out["ranking"]) == 3
    assert out["winner"] in ("full", "galore", "lora", "dora")


def test_theory_report_peft_saving():
    t = theory_report(rank=16, bytes_per=2)
    assert t["dora_vs_full_pct"] < 5.0
    assert t["dora_adapter_mib"] < t["full_ft_gib"] * 1024
    assert t["dora_params"] > t["lora_params"]


def test_effective_dora_equals_recompose_path():
    W0 = _rand_gauss(6, 8, seed=20, scale=0.4)
    cfg = DoRAConfig(rank=2, seed=20)
    tr = DoRATrainer(W0, cfg, AdamConfig(lr=0.01))
    # nudge A a bit
    tr.A[0][0] += 0.1
    W = tr.weight()
    V = effective_weight_lora(tr.W0, tr.B, tr.A, alpha=cfg.alpha)
    W2 = recompose(tr.magnitude, V)
    assert _mse_mat(W, W2) < 1e-18


def test_step_once_finite():
    eng = DoRAEngine()
    out = eng.step_once(m=10, n=12, rank=2, seed=21, method="dora")
    assert math.isfinite(out["loss_after"])
    assert out["n_trainable"] == 10 * 2 + 2 * 12 + 12


def test_benchmark_rows():
    eng = DoRAEngine()
    out = eng.benchmark(dims=[10, 12], ranks=[2, 4], steps=8, seed=0)
    assert out["ok"]
    assert out["n"] >= 2


def test_pack_keys():
    eng = DoRAEngine()
    p = eng.pack(m=16, n=20, rank=4, seed=0)
    assert p["engine"] == "dora-weight-decomposed-adapter"
    assert "theory" in p
    assert "status" in p
