"""Continual backprop (Dohare et al., 2024): the generate-and-test half of it.

Every step, each hidden unit's *utility* is updated: a running average of how much
it contributes to the next layer (mean-corrected activation magnitude times the
size of its outgoing weights). Mature units with the lowest utility are re-initialised
at a small rate: fresh random incoming weights, zero outgoing weights, optimizer
moments cleared. This keeps a supply of fresh, trainable units so the network never
fills up with dead or saturated ones. It is the one algorithm from the Alberta camp
you can point at any network without changing its loss.
"""
from __future__ import annotations
import numpy as np
import mlx.core as mx
from ..util import tree_get, tree_set, module_set_param, zero_columns, set_rows


class CBP:
    kind = "cbp"

    def __init__(self, model, rng: np.random.Generator, decay=0.99, replacement_rate=1e-4,
                 maturity=100, adaptation=False, **_ignored):
        self.model = model
        self.rng = rng
        self.decay = float(decay)
        self.rate = float(replacement_rate)
        self.maturity = int(maturity)
        self.adaptation = bool(adaptation)
        self.specs = model.unit_specs()
        self.util = [mx.zeros((s["n"],)) for s in self.specs]
        self.mean = [mx.zeros((s["n"],)) for s in self.specs]
        self.age = [np.zeros(s["n"], dtype=np.int64) for s in self.specs]
        self.accum = [0.0 for _ in self.specs]
        self.total_reinits = 0
        # index matrices (n, count): which entries along the unit axis belong to unit i
        self._out_idx = [[(p, a, self._index_matrix(spec, s["n"])) for (p, a, spec) in s["out_params"]] for s in self.specs]
        self._in_idx = [[(p, a, self._index_matrix(spec, s["n"])) for (p, a, spec) in s["in_params"]] for s in self.specs]

    @staticmethod
    def _index_matrix(idx_spec, n: int) -> mx.array:
        units = np.arange(n)
        if idx_spec is None:
            return mx.array(units[:, None])
        period, count = idx_spec
        return mx.array(np.stack([units + k * period for k in range(count)], axis=1))

    @staticmethod
    def _expand(idx_spec, units: np.ndarray) -> np.ndarray:
        if idx_spec is None:
            return units
        period, count = idx_spec
        return np.concatenate([units + k * period for k in range(count)])

    # ---- every step (lazy MLX ops, evaluated together with the training step) ----
    def observe(self, hiddens: list[mx.array]) -> None:
        params = self.model.parameters()
        d = self.decay
        for j, spec in enumerate(self.specs):
            n = spec["n"]
            h = hiddens[spec["act"]].reshape(-1, n)
            self.mean[j] = d * self.mean[j] + (1 - d) * h.mean(axis=0)
            corr = mx.array((1.0 - d ** (self.age[j] + 1)).astype(np.float32))
            f = self.mean[j] / corr
            out_mag = mx.zeros((n,))
            for path, axis, idx in self._out_idx[j]:
                W = tree_get(params, path)
                mag = mx.abs(W).sum(axis=0) if axis == 1 else mx.abs(W).sum(axis=1)
                out_mag = out_mag + mag[idx].sum(axis=1)
            contrib = mx.abs(h - f).mean(axis=0) * out_mag
            if self.adaptation:
                in_mag = mx.zeros((n,))
                for path, axis, idx in self._in_idx[j]:
                    W = tree_get(params, path)
                    if W.ndim == 2:
                        in_mag = in_mag + mx.abs(W).sum(axis=1)[idx].sum(axis=1)
                contrib = contrib / (in_mag + 1e-8)
            self.util[j] = d * self.util[j] + (1 - d) * contrib

    def arrays_to_eval(self) -> list[mx.array]:
        return [*self.util, *self.mean]

    # ---- occasionally: pick the least useful mature units and replace them ----
    def maybe_replace(self, opt) -> int:
        total = 0
        for j, spec in enumerate(self.specs):
            self.age[j] += 1
            eligible = self.age[j] > self.maturity
            n_elig = int(eligible.sum())
            if n_elig == 0:
                continue
            self.accum[j] += self.rate * n_elig
            k = int(self.accum[j])
            if k == 0:
                continue
            self.accum[j] -= k
            k = min(k, n_elig)
            corr = (1.0 - self.decay ** self.age[j]).astype(np.float32)
            u = np.array(self.util[j]) / corr
            u[~eligible] = np.inf
            units = np.argpartition(u, k - 1)[:k]
            self._reinit(j, np.sort(units), opt)
            total += k
        self.total_reinits += total
        return total

    def _reinit(self, j: int, units: np.ndarray, opt) -> None:
        spec = self.specs[j]
        s = spec["init_scale"]
        params = self.model.parameters()
        for path, axis, idx_spec in spec["in_params"]:
            rows = self._expand(idx_spec, units)
            W = tree_get(params, path)
            if W.ndim == 2:
                new = self.rng.uniform(-s, s, size=(len(rows), W.shape[1])).astype(np.float32)
                set_rows(W, rows, mx.array(new))
            else:
                set_rows(W, rows, 0.0)
            self._zero_opt(opt, path, 0, rows)
        for path, axis, idx_spec in spec["out_params"]:
            cols = self._expand(idx_spec, units)
            W = tree_get(self.model.parameters(), path)
            module_set_param(self.model, path, zero_columns(W, cols))
            self._zero_opt(opt, path, 1, cols)
        idx = mx.array(units)
        self.util[j][idx] = 0.0
        self.mean[j][idx] = 0.0
        self.age[j][units] = 0

    @staticmethod
    def _zero_opt(opt, path: str, axis: int, idx: np.ndarray) -> None:
        try:
            st = tree_get(opt.state, path)
        except (KeyError, IndexError, TypeError):
            return
        if not isinstance(st, dict):
            return
        for key, arr in list(st.items()):
            if not isinstance(arr, mx.array) or arr.ndim == 0:
                continue
            if axis == 0:
                set_rows(arr, idx, 0.0)
            elif arr.ndim == 2:
                tree_set(opt.state, f"{path}.{key}", zero_columns(arr, idx))

    # ---- checkpointing ----
    def arrays(self) -> dict[str, mx.array]:
        out = {}
        for j in range(len(self.specs)):
            out[f"util.{j}"] = self.util[j]
            out[f"mean.{j}"] = self.mean[j]
            out[f"age.{j}"] = mx.array(self.age[j])
        return out

    def meta(self) -> dict:
        return {"accum": self.accum, "total_reinits": self.total_reinits}

    def load(self, arrays: dict[str, mx.array], meta: dict | None) -> None:
        for j in range(len(self.specs)):
            self.util[j] = arrays[f"util.{j}"]
            self.mean[j] = arrays[f"mean.{j}"]
            self.age[j] = np.array(arrays[f"age.{j}"]).astype(np.int64)
        if meta:
            self.accum = [float(a) for a in meta.get("accum", self.accum)]
            self.total_reinits = int(meta.get("total_reinits", 0))

    def describe(self) -> str:
        return f"cbp(rate={self.rate:g}, maturity={self.maturity}, decay={self.decay})"
