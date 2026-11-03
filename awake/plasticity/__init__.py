from __future__ import annotations
from .cbp import CBP
from .regularizers import L2Init, ShrinkPerturb


def build_plasticity(cfg_p, model, rng):
    kind = cfg_p.get("kind", "none")
    if kind in (None, "none", ""):
        return None
    kw = {k: v for k, v in cfg_p.items() if k != "kind"}
    if kind == "cbp":
        return CBP(model, rng, **kw)
    if kind == "l2init":
        return L2Init(model, **kw)
    if kind == "shrink_perturb":
        return ShrinkPerturb(model, rng, **kw)
    raise ValueError(f"unknown plasticity kind {kind!r}")
