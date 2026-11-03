"""TOML config with defaults. Unknown keys inside a section pass straight through to
the stream/learner constructors, so new knobs need no plumbing here."""
from __future__ import annotations
import copy
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "run": {
        "name": None,
        "device": "cpu",
        "seed": 0,
        "batch_size": 1,
        "max_steps": 0,
        "max_steps_per_sec": 0.0,
        "log_every": 1000,
        "ckpt_every": 0,
        "compile": True,
    },
    "stream": {"kind": "scr"},
    "learner": {"kind": "mlp", "hidden": [5], "act": "tanh"},
    "optim": {"kind": "sgd", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0},
    "plasticity": {
        "kind": "none",
        "decay": 0.99,
        "replacement_rate": 1e-4,
        "maturity": 100,
        "adaptation": False,
        "l2_lambda": 1e-3,
        "shrink": 0.999,
        "perturb": 1e-4,
        "every": 1000,
    },
    "sleep": {
        "wake_steps": 50000,
        "replay": False,
        "memory_size": 1024,
        "replay_steps": 100,
        "slow_ema": 0.0,
        "consolidate": 0.0,
    },
}


class Section(dict):
    """A dict with attribute access. Nested dicts are Sections too."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(d):
    if isinstance(d, dict):
        return Section({k: _wrap(v) for k, v in d.items()})
    return d


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Section:
    data: dict = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    cfg = _merge(DEFAULTS, data)
    if overrides:
        cfg = _merge(cfg, overrides)
    return _wrap(cfg)
