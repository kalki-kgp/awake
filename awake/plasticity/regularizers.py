"""Two cheap plasticity baselines to compare continual backprop against."""
from __future__ import annotations
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from ..util import module_set_param


class L2Init:
    """Regenerative regularisation (Kumar et al., 2023): pull weights toward their
    initial values, loss += lambda * ||theta - theta_0||^2."""
    kind = "l2init"

    def __init__(self, model, l2_lambda=1e-3, **_ignored):
        self.lam = float(l2_lambda)
        self.theta0 = tree_map(lambda p: mx.array(np.array(p)), model.parameters())

    def penalty(self, model):
        total = mx.zeros(())
        for (_, p), (_, p0) in zip(tree_flatten(model.parameters()), tree_flatten(self.theta0)):
            total = total + mx.sum((p - p0) ** 2)
        return self.lam * total

    def arrays_to_eval(self):
        return []

    def arrays(self):
        return {f"theta0.{k}": v for k, v in tree_flatten(self.theta0)}

    def meta(self):
        return {}

    def load(self, arrays, meta):
        self.theta0 = tree_unflatten([(k[len("theta0."):], v) for k, v in arrays.items() if k.startswith("theta0.")])

    def describe(self):
        return f"l2init(lambda={self.lam:g})"


class ShrinkPerturb:
    """Ash & Adams (2020): every `every` steps, theta <- shrink * theta + perturb * noise."""
    kind = "shrink_perturb"

    def __init__(self, model, rng, shrink=0.999, perturb=1e-4, every=1000, **_ignored):
        self.model, self.rng = model, rng
        self.shrink, self.perturb, self.every = float(shrink), float(perturb), int(every)

    def apply(self):
        for path, p in tree_flatten(self.model.parameters()):
            noise = mx.array(self.rng.normal(size=p.shape).astype(np.float32))
            module_set_param(self.model, path, p * self.shrink + self.perturb * noise)

    def arrays_to_eval(self):
        return []

    def arrays(self):
        return {}

    def meta(self):
        return {}

    def load(self, arrays, meta):
        pass

    def describe(self):
        return f"shrink_perturb(shrink={self.shrink}, perturb={self.perturb:g}, every={self.every})"
