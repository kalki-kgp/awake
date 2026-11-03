"""Measure training throughput on this Mac: CPU vs Metal GPU, compiled vs not, for the
two model shapes this repo uses. Tiny nets are kernel-launch bound and win on CPU;
the byte GRU wins on the GPU. Run it again on the M5 to re-decide the defaults."""
from __future__ import annotations
import time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from .learners.mlp import MLP
from .learners.chargru import CharGRU


def _bench_mlp(device, compiled, n=1500):
    mx.set_default_device(device)
    model, opt = MLP(20, [5], 1, "tanh"), optim.SGD(learning_rate=0.01)
    x, y = mx.random.uniform(shape=(1, 20)), mx.array([[1.0]])

    def loss_fn(m, x, y):
        out, _, hs = m.forward(x)
        return mx.mean((out - y) ** 2), hs
    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def step(x, y):
        (l, hs), g = vg(model, x, y)
        opt.update(model, g)
        return l
    f = mx.compile(step, inputs=state, outputs=state) if compiled else step
    for _ in range(50):
        mx.eval(f(x, y), state)
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f(x, y), state)
    return n / (time.perf_counter() - t)


def _bench_gru(device, compiled, hidden=256, seq=64, n=120):
    mx.set_default_device(device)
    model, opt = CharGRU(256, 64, hidden), optim.Adam(learning_rate=1e-3)
    idx = mx.random.randint(0, 256, (seq,)); tgt = mx.random.randint(0, 256, (seq,)); h0 = mx.zeros((hidden,))

    def loss_fn(m, idx, tgt, h0):
        logits, hl, _ = m.forward(idx, h0)
        return nn.losses.cross_entropy(logits, tgt, reduction="mean"), hl
    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def step(idx, tgt, h0):
        (l, hl), g = vg(model, idx, tgt, h0)
        opt.update(model, g)
        return l, hl
    f = mx.compile(step, inputs=state, outputs=state) if compiled else step
    for _ in range(8):
        l, h0 = f(idx, tgt, h0); mx.eval(l, h0, state)
    t = time.perf_counter()
    for _ in range(n):
        l, h0 = f(idx, tgt, h0); mx.eval(l, h0, state)
    return n / (time.perf_counter() - t)


def run_bench() -> None:
    devs = [("cpu", mx.cpu)] + ([("gpu", mx.gpu)] if mx.metal.is_available() else [])
    print(f"{'model':34s} {'device':7s} {'compiled':9s} {'steps/s':>10s}")
    best = {}
    for label, fn in [("MLP 20-5-1, batch 1 (SCR)", _bench_mlp), ("byte GRU h256, seq 64, batch 1", _bench_gru)]:
        for dname, dev in devs:
            for compiled in (False, True):
                sps = fn(dev, compiled)
                print(f"{label:34s} {dname:7s} {str(compiled):9s} {sps:10,.0f}")
                if compiled:
                    best[(label, dname)] = sps
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    if len(devs) == 2:
        print()
        for label in ["MLP 20-5-1, batch 1 (SCR)", "byte GRU h256, seq 64, batch 1"]:
            c, g = best[(label, "cpu")], best[(label, "gpu")]
            print(f"{label}: {'CPU' if c > g else 'GPU'} wins by {max(c, g) / max(min(c, g), 1e-9):.1f}x -> set [run] device accordingly")
