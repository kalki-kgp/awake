"""Slowly-Changing Regression, after Dohare et al. (Nature, 2024).

Inputs are n_bits binary values. The first n_flip bits are held fixed for `flip_every`
examples, then one of them flips (a new "task"). The remaining bits are random every
example. The target is a fixed random network of linear-threshold units, so the
function the learner must track changes slightly at every flip and never repeats
exactly. A learner far smaller than the target (5 hidden units vs 100) has to keep
re-adapting forever. Plain backprop slowly loses the ability to do so.
"""
from __future__ import annotations
import numpy as np
from .base import Stream
from ..util import rng_state, set_rng_state


class SCRStream(Stream):
    name = "scr"
    loss = "mse"
    sequential = False

    def __init__(self, seed=0, n_bits=20, n_flip=15, flip_every=10_000, target_units=100,
                 beta=0.7, retention_samples=512, y_scale=1.0):
        assert 0 < n_flip < n_bits
        self.seed = int(seed)
        self.n_bits, self.n_flip, self.flip_every = int(n_bits), int(n_flip), int(flip_every)
        self.x_shape, self.y_shape, self.out_dim = (self.n_bits,), (1,), 1
        self.rng = np.random.default_rng(self.seed)
        t = np.random.default_rng(self.seed + 1_000_003)          # the target network never changes
        pm = np.array([-1.0, 1.0], dtype=np.float32)
        self.W = t.choice(pm, size=(int(target_units), self.n_bits + 1))
        self.theta = ((self.n_bits + 1) * float(beta) - (self.W < 0).sum(1)).astype(np.float32)
        self.v = t.choice(pm, size=int(target_units))
        self.y_scale = float(y_scale)
        self.flip_bits = self.rng.integers(0, 2, self.n_flip).astype(np.float32)
        self.task0_bits = self.flip_bits.copy()
        self.step = 0
        self._task = 0
        self.retention_samples = int(retention_samples)
        self._ret = None

    def target(self, x: np.ndarray) -> np.ndarray:
        xb = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
        act = (xb @ self.W.T > self.theta).astype(np.float32)
        return (act @ self.v)[:, None] * self.y_scale

    def next(self):
        if self.step > 0 and self.step % self.flip_every == 0:
            i = int(self.rng.integers(0, self.n_flip))
            self.flip_bits[i] = 1.0 - self.flip_bits[i]
            self._task += 1
        var = self.rng.integers(0, 2, self.n_bits - self.n_flip).astype(np.float32)
        x = np.concatenate([self.flip_bits, var])
        y = self.target(x[None])[0]
        self.step += 1
        return x, y, False

    def retention(self):
        if self._ret is None:
            n_var = self.n_bits - self.n_flip
            if n_var <= 12:                       # enumerate every pattern of the varying bits
                grid = np.array(np.meshgrid(*([[0, 1]] * n_var), indexing="ij")).reshape(n_var, -1).T
                var = grid.astype(np.float32)
            else:
                r = np.random.default_rng(self.seed + 2_000_003)
                var = r.integers(0, 2, (self.retention_samples, n_var)).astype(np.float32)
            x = np.concatenate([np.tile(self.task0_bits, (len(var), 1)), var], axis=1)
            self._ret = (x, self.target(x))
        return self._ret

    @property
    def task(self) -> int:
        return self._task

    def state(self) -> dict:
        return {"step": self.step, "task": self._task, "flip_bits": self.flip_bits.tolist(),
                "task0_bits": self.task0_bits.tolist(), "rng": rng_state(self.rng)}

    def load_state(self, d: dict) -> None:
        self.step = int(d["step"]); self._task = int(d["task"])
        self.flip_bits = np.array(d["flip_bits"], dtype=np.float32)
        self.task0_bits = np.array(d["task0_bits"], dtype=np.float32)
        set_rng_state(self.rng, d["rng"])
        self._ret = None

    def describe(self) -> str:
        return f"scr(n_bits={self.n_bits}, n_flip={self.n_flip}, flip_every={self.flip_every})"
