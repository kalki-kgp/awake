"""Small helpers for reaching into parameter trees (dicts and lists of arrays)."""
from __future__ import annotations
import numpy as np
import mlx.core as mx
import mlx.nn as nn

LN2 = float(np.log(2.0))


def tree_get(tree, path: str):
    node = tree
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def tree_set(tree, path: str, value) -> None:
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def module_set_param(model: nn.Module, path: str, value: mx.array) -> None:
    """Assign a new array to a parameter given its dotted path (handles lists of modules)."""
    parts = path.split(".")
    node = model
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else getattr(node, part)
    setattr(node, parts[-1], value)


def zero_columns(W: mx.array, cols: np.ndarray) -> mx.array:
    """Return W with the given columns (axis 1) set to zero."""
    mask = np.ones(W.shape[1], dtype=np.float32)
    mask[cols] = 0.0
    return W * mx.array(mask)[None, :]


def set_rows(W: mx.array, rows: np.ndarray, value) -> None:
    """In-place: W[rows] = value (works for 1-D and 2-D arrays)."""
    W[mx.array(rows)] = value


def rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def set_rng_state(rng: np.random.Generator, st: dict) -> None:
    rng.bit_generator.state = st
