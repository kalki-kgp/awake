"""A plain MLP that also reports its hidden activations (needed for continual backprop)
and describes which parameter slices belong to each hidden unit."""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn

ACTS = {"tanh": mx.tanh, "relu": nn.relu, "gelu": nn.gelu, "sigmoid": mx.sigmoid, "silu": nn.silu}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, act: str = "tanh"):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.act_name = act
        self.hidden_sizes = list(hidden)
        self.hidden = hidden[-1] if hidden else out_dim

    def forward(self, x, h0=None):
        act = ACTS[self.act_name]
        h, hs = x, []
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < last:
                h = act(h)
                hs.append(h)
        return h, mx.zeros((1,)), hs

    def unit_specs(self):
        """One spec per hidden layer: which activation index it is, which parameter
        slices feed each unit (rows) and which carry its output (columns)."""
        specs = []
        for j, n in enumerate(self.hidden_sizes):
            fan_in = self.layers[j].weight.shape[1]
            specs.append({
                "n": n,
                "act": j,
                "in_params": [(f"layers.{j}.weight", 0, None), (f"layers.{j}.bias", 0, None)],
                "out_params": [(f"layers.{j + 1}.weight", 1, None)],
                "init_scale": 1.0 / math.sqrt(fan_in),
            })
        return specs
