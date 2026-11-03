"""Everything needed to stop and continue as if nothing happened: weights, optimizer
moments, plasticity statistics, slow weights, stream position, RNG, replay memory."""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from .state import LoopState
from .util import rng_state, set_rng_state

LATEST = "latest.safetensors"
TMP = "latest.tmp.safetensors"
META = "latest.json"
MEM = "memory.npz"


def exists(run_dir: Path) -> bool:
    return (run_dir / LATEST).exists() and (run_dir / META).exists()


def save(r) -> None:
    run_dir = r.run_dir
    r.state.wall_s = r.wall()
    arrays: dict[str, mx.array] = {}
    for k, v in tree_flatten(r.model.parameters()):
        arrays[f"model.{k}"] = v
    for k, v in tree_flatten(r.opt.state):
        arrays[f"opt.{k}"] = v
    if r.plast is not None:
        for k, v in r.plast.arrays().items():
            arrays[f"plast.{k}"] = v
    if r.slow is not None:
        for k, v in tree_flatten(r.slow):
            arrays[f"slow.{k}"] = v
    mx.eval(arrays)
    mx.save_safetensors(str(run_dir / TMP), arrays)
    os.replace(run_dir / TMP, run_dir / LATEST)
    meta = {
        "version": 1,
        "step": r.state.step,
        "saved_at": time.time(),
        "loop": asdict(r.state),
        "stream": r.stream.state(),
        "plast": r.plast.meta() if r.plast is not None else None,
        "rng": rng_state(r.rng),
        "device": r.device,
    }
    tmp = run_dir / (META + ".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, run_dir / META)
    if r.memory is not None:
        r.memory.save(run_dir / MEM)
    r.state.last_ckpt_step = r.state.step


def load(r) -> None:
    run_dir = r.run_dir
    arrays = mx.load(str(run_dir / LATEST))
    meta = json.loads((run_dir / META).read_text())
    r.model.update(tree_unflatten([(k[len("model."):], v) for k, v in arrays.items() if k.startswith("model.")]))
    opt_items = [(k[len("opt."):], v) for k, v in arrays.items() if k.startswith("opt.")]
    if opt_items:
        r.opt.state = tree_unflatten(opt_items)
        r.opt.learning_rate = float(r.cfg.optim.lr)
    if r.plast is not None:
        r.plast.load({k[len("plast."):]: v for k, v in arrays.items() if k.startswith("plast.")}, meta.get("plast"))
    slow_items = [(k[len("slow."):], v) for k, v in arrays.items() if k.startswith("slow.")]
    if slow_items and r.slow is not None:
        r.slow = tree_unflatten(slow_items)
    r.state = LoopState(**meta["loop"])
    r.stream.load_state(meta["stream"])
    set_rng_state(r.rng, meta["rng"])
    if r.memory is not None:
        r.memory.load(run_dir / MEM)
    mx.eval(r.model.parameters(), r.opt.state)
