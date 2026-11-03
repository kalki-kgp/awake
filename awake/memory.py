"""Reservoir-sampled memory of past examples, used only during sleep (replay is a
choice, not a default: the Alberta camp forbids it, the sleep papers rely on it)."""
from __future__ import annotations
from pathlib import Path
import numpy as np


class ReservoirMemory:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.cap = int(capacity)
        self.rng = rng
        self.xs = None
        self.ys = None
        self.seen = 0

    @property
    def count(self) -> int:
        return min(self.seen, self.cap)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.xs is None:
            self.xs = np.zeros((self.cap, *np.shape(x)), dtype=np.asarray(x).dtype)
            self.ys = np.zeros((self.cap, *np.shape(y)), dtype=np.asarray(y).dtype)
        if self.seen < self.cap:
            j = self.seen
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j >= self.cap:
                self.seen += 1
                return
        self.xs[j] = x
        self.ys[j] = y
        self.seen += 1

    def sample(self, b: int):
        idx = self.rng.integers(0, self.count, size=b)
        return self.xs[idx], self.ys[idx]

    def save(self, path: Path) -> None:
        if self.xs is None:
            return
        np.savez(path, xs=self.xs, ys=self.ys, seen=np.array(self.seen))

    def load(self, path: Path) -> None:
        if not Path(path).exists():
            return
        d = np.load(path)
        self.xs, self.ys, self.seen = d["xs"], d["ys"], int(d["seen"])
