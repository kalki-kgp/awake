from __future__ import annotations
import numpy as np
from .mlp import MLP
from .chargru import CharGRU


def build_learner(cfg_learner, stream):
    kind = cfg_learner["kind"]
    if kind == "mlp":
        return MLP(int(np.prod(stream.x_shape)), list(cfg_learner.get("hidden", [5])),
                   stream.out_dim, act=cfg_learner.get("act", "tanh"))
    if kind == "chargru":
        return CharGRU(vocab=stream.out_dim, embed=int(cfg_learner.get("embed", 64)),
                       hidden=int(cfg_learner.get("hidden", 256)))
    raise ValueError(f"unknown learner kind {kind!r}")
