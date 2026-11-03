"""Health check: every stream, learner and plasticity path for a few hundred steps,
plus a stop/resume test that must reproduce the uninterrupted run exactly."""
from __future__ import annotations
import shutil
import traceback
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from . import RUNS_DIR, CACHE_DIR
from .config import load_config
from .loop import Runner
from .metrics import read_metrics

ROOT = RUNS_DIR / "_selftest"


def _flat(model):
    return np.concatenate([np.array(v).ravel() for _, v in tree_flatten(model.parameters())])


def _scr_cfg(**over):
    base = {"run": {"device": "cpu", "log_every": 50, "batch_size": 1, "seed": 3},
            "stream": {"kind": "scr", "flip_every": 100},
            "learner": {"kind": "mlp", "hidden": [5], "act": "tanh"},
            "optim": {"kind": "sgd", "lr": 0.01},
            "plasticity": {"kind": "cbp", "replacement_rate": 0.05, "maturity": 10},
            "sleep": {"wake_steps": 150, "replay": False}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v}
    return load_config(overrides=base)


def t_scr_cbp():
    r = Runner(_scr_cfg(), ROOT / "scr_cbp", fresh=True, quiet=True)
    r.run(max_steps=300)
    rows = read_metrics(ROOT / "scr_cbp" / "metrics.csv")
    assert r.plast.total_reinits > 0, "continual backprop never re-initialised a unit"
    assert sum(1 for x in rows if x["phase"] == "sleep") == 2, "expected 2 sleeps"
    assert all(np.isfinite(x["loss"]) for x in rows if x["phase"] == "wake"), "non-finite loss"
    return f"300 steps, {r.plast.total_reinits} re-inits, 2 sleeps, retention loss {rows[-1]['ret_loss']:.3f}"


def t_resume_exact():
    cfg = _scr_cfg(plasticity={"kind": "cbp", "replacement_rate": 0.02, "maturity": 10}, sleep={"wake_steps": 70})
    a = Runner(cfg, ROOT / "resume_a", fresh=True, quiet=True)
    a.run(max_steps=220)
    b = Runner(cfg, ROOT / "resume_b", fresh=True, quiet=True)
    b.run(max_steps=110)
    b2 = Runner(cfg, ROOT / "resume_b", fresh=False, quiet=True)
    assert b2.state.step == 110, f"resumed at {b2.state.step}, expected 110"
    b2.run(max_steps=220)
    d = float(np.abs(_flat(a.model) - _flat(b2.model)).max())
    assert d < 1e-6, f"resumed run diverged from the uninterrupted run: max |dw| = {d:.3g}"
    assert a.stream.state()["task"] == b2.stream.state()["task"]
    assert a.plast.total_reinits == b2.plast.total_reinits, "re-init counts differ after resume"
    return f"stop at 110 / resume to 220 reproduces the straight run (max |dw| = {d:.1e}, {a.plast.total_reinits} re-inits)"


def t_l2init_shrink():
    r = Runner(_scr_cfg(plasticity={"kind": "l2init", "l2_lambda": 1e-3}), ROOT / "l2", fresh=True, quiet=True)
    r.run(max_steps=120)
    r2 = Runner(_scr_cfg(plasticity={"kind": "shrink_perturb", "every": 25}), ROOT / "sp", fresh=True, quiet=True)
    r2.run(max_steps=120)
    return "l2init and shrink_perturb run"


def t_text_sleep():
    tdir = ROOT / "textsrc"
    tdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    words = ["awake", "stream", "learn", "sleep", "weight", "unit", "the", "and", "of", "never", "reset", "twenty", "watts"]
    text = " ".join(rng.choice(words, size=40000)) + "\n"
    (tdir / "sample.txt").write_text(text)
    cfg = load_config(overrides={
        "run": {"device": "gpu", "log_every": 20, "batch_size": 1, "seed": 1},
        "stream": {"kind": "text", "path": str(tdir), "seq_len": 16, "retention_bytes": 2048},
        "learner": {"kind": "chargru", "embed": 8, "hidden": 32},
        "optim": {"kind": "adam", "lr": 0.003},
        "plasticity": {"kind": "cbp", "replacement_rate": 0.02, "maturity": 5},
        "sleep": {"wake_steps": 40, "replay": True, "memory_size": 32, "replay_steps": 10, "slow_ema": 0.1, "consolidate": 0.5}})
    r = Runner(cfg, ROOT / "text", fresh=True, quiet=True)
    r.run(max_steps=90)
    rows = read_metrics(ROOT / "text" / "metrics.csv")
    sleeps = [x for x in rows if x["phase"] == "sleep"]
    assert len(sleeps) == 2 and all(np.isfinite(s["ret_loss"]) for s in sleeps)
    assert sleeps[-1]["replayed"] == 10, "replay did not run"
    assert r.plast.total_reinits > 0, "cbp on the GRU never re-initialised a unit"
    first, last = rows[0]["metric"], [x for x in rows if x["phase"] == "wake"][-1]["metric"]
    return f"byte GRU on {r.device}: {first:.2f} -> {last:.2f} bits/byte in 90 steps, 2 sleeps with replay, {r.plast.total_reinits} GRU units re-inited"


def t_pmnist():
    if not (CACHE_DIR / "mnist" / "mnist.npz").exists():
        return "skipped (no MNIST cached; run `awake data mnist`)"
    cfg = load_config(overrides={
        "run": {"device": "cpu", "log_every": 20, "batch_size": 8, "seed": 2},
        "stream": {"kind": "pmnist", "samples_per_task": 400, "retention_samples": 200},
        "learner": {"kind": "mlp", "hidden": [64, 64], "act": "relu"},
        "optim": {"kind": "sgd", "lr": 0.05},
        "plasticity": {"kind": "cbp", "replacement_rate": 0.01, "maturity": 10},
        "sleep": {"wake_steps": 60}})
    r = Runner(cfg, ROOT / "pmnist", fresh=True, quiet=True)
    r.run(max_steps=120)
    rows = read_metrics(ROOT / "pmnist" / "metrics.csv")
    last = [x for x in rows if x["phase"] == "wake"][-1]
    return f"120 steps, task {last['task']}, online acc {last['metric']:.2f}, retention acc {rows[-1]['ret_metric']:.2f}"


TESTS = [t_scr_cbp, t_resume_exact, t_l2init_shrink, t_text_sleep, t_pmnist]


def run_selftest() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    failed = 0
    print(f"awake selftest · mlx {mx.__version__ if hasattr(mx, '__version__') else ''} · metal {mx.metal.is_available()}")
    for t in TESTS:
        try:
            msg = t()
            print(f"  PASS {t.__name__:18s} {msg}")
        except Exception:
            failed += 1
            print(f"  FAIL {t.__name__}")
            traceback.print_exc()
    print("all good" if not failed else f"{failed} failed")
    shutil.rmtree(ROOT, ignore_errors=True)
    return 1 if failed else 0
