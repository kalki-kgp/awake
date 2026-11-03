"""A Stream hands out one example at a time, forever, and can describe its own
position so a run can be checkpointed and resumed mid-stream."""
from __future__ import annotations
import numpy as np


class Stream:
    name = "base"
    loss = "mse"            # "mse" or "ce"
    sequential = False      # True if a hidden state should be carried between consecutive examples
    x_shape: tuple = ()
    y_shape: tuple = ()
    out_dim: int = 1
    x_dtype = np.float32
    y_dtype = np.float32

    def next(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (x, y, reset). reset=True means: clear any carried hidden state first."""
        raise NotImplementedError

    def retention(self) -> tuple[np.ndarray, np.ndarray]:
        """A fixed evaluation set drawn from the very start of the stream ('week one')."""
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError

    def load_state(self, d: dict) -> None:
        raise NotImplementedError

    @property
    def task(self) -> int:
        return 0

    def describe(self) -> str:
        return self.name
