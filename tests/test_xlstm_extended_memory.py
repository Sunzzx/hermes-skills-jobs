"""Unit tests for xlstm-extended-memory (xLSTM sLSTM/mLSTM engine)."""

import os
import math
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/xlstm-extended-memory")
)
eng = SourceFileLoader("xlstm_extended_memory", bin_path).load_module()


def test_sigmoid_tanh_exp():
    assert abs(eng.sigmoid(0.0) - 0.5) < 1e-9
    assert abs(eng.tanh(0.0)) < 1e-12
    assert eng.exp_gate(0.0) == 1.0
    assert eng.exp_gate(100.0) < 1e10  # capped


def test_matmul_identity():
    I = eng._eye(4)
    v = [1.0, 2.0, 3.0, 4.0]
    assert eng._matvec(I, v) == v
    II = eng._matmul(I, I)
    assert eng._frobenius(eng._sub_mat(II, I)) < 1e-12


def test_vanilla_lstm_forward():
    import random

    rng = random.Random(0)
    w = eng.LSTMWeights.init(5, 6, rng)
    cell = eng.VanillaLSTM(w)
    xs = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(4)]
    hs, st = cell.forward(xs)
    assert len(hs) == 4 and len(hs[0]) == 6
    assert all(math.isfinite(x) for h in hs for x in h)
    assert all(math.isfinite(x) for x in st.c)


def test_slstm_normalizer_and_mixing():
    import random

    rng = random.Random(1)
    w = eng.SLSTMWeights.init(4, 8, n_heads=2, rng=rng)
    cell = eng.SLSTM(w)
    xs = [[rng.gauss(0, 0.5) for _ in range(4)] for _ in range(5)]
    hs, st = cell.forward(xs)
    assert len(hs) == 5
    assert all(math.isfinite(x) for x in st.c + st.n + st.m)

    w2 = eng.SLSTMWeights.init(4, 8, n_heads=2, rng=random.Random(1))
    w2.R = eng._eye(8)
    cell_id = eng.SLSTM(w2)
    h_mix, _ = cell.forward(xs)
    h_id, _ = cell_id.forward(xs)
    diff = sum(eng._norm(eng._sub(a, b)) for a, b in zip(h_mix, h_id))
    assert diff > 1e-6


def test_mlstm_covariance_grows():
    import random

    rng = random.Random(2)
    w = eng.MLSTMWeights.init(6, 6, 6, rng)
    cell = eng.MLSTM(w)
    st = cell.initial_state()
    assert eng._frobenius(st.C) == 0.0
    for _ in range(5):
        x = [rng.gauss(0, 1) for _ in range(6)]
        _, st = cell.step(x, st)
    assert eng._frobenius(st.C) > 0.0
    assert all(math.isfinite(v) for row in st.C for v in row)


def test_xlstm_stack_logits():
    stack = eng.XLSTMStack(
        d_model=8,
        blocks=[eng.BlockSpec("slstm", 8, n_heads=2), eng.BlockSpec("mlstm", 8)],
        n_classes=4,
        seed=3,
    )
    xs = [[0.1 * i for i in range(8)] for _ in range(3)]
    hs, logits = stack.forward(xs)
    assert len(logits) == 4
    assert all(math.isfinite(x) for x in logits)
    assert stack.n_params() > 0


def test_tasks():
    import random

    r = random.Random(0)
    seq, t = eng.task_copy(r, length=5, vocab=8)
    assert len(seq) == 10 and 1 <= t < 8
    seq, t = eng.task_parity(r, length=9)
    assert t in (0, 1) and len(seq) == 9
    seq, t = eng.task_associative_recall(r, n_pairs=3, vocab=10)
    assert len(seq) == 7
    seq, t = eng.task_majority(r, 11)
    assert t in (0, 1)


def test_softmax_ce():
    p = eng.softmax([0.0, 0.0, 0.0])
    assert abs(sum(p) - 1.0) < 1e-9
    assert eng.cross_entropy([5.0, 0.0], 0) < 0.1


def test_gate_revision_demo():
    d = eng.gate_revision_demo()
    assert len(d["slstm_trajectory"]) == 4
    assert len(d["lstm_trajectory"]) == 4


def test_selftest_gate():
    result = eng.run_selftest(verbose=False)
    assert result["ok"] is True
    assert result["passed"] >= 18


def test_param_counts_ordered():
    import random

    rng = random.Random(0)
    d = 16
    lp = eng.LSTMWeights.init(d, d, rng).n_params()
    sp = eng.SLSTMWeights.init(d, d, 4, random.Random(1)).n_params()
    mp = eng.MLSTMWeights.init(d, d, d, random.Random(2)).n_params()
    assert sp > lp > 0
    assert mp > 0


def test_rms_norm_unit():
    v = eng.rms_norm([2.0, 0.0, 0.0, 0.0])
    mean_sq = sum(x * x for x in v) / len(v)
    assert abs(mean_sq - 1.0) < 1e-6


def test_stabilizer_no_nan():
    import random

    w = eng.SLSTMWeights.init(2, 2, 1, random.Random(0))
    for row in w.W_x:
        for j in range(len(row)):
            row[j] *= 40.0
    cell = eng.SLSTM(w)
    st = cell.initial_state()
    for _ in range(4):
        _, st = cell.step([8.0, -8.0], st)
    assert all(math.isfinite(x) for x in st.c + st.n + st.m)
