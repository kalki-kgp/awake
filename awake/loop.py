"""The always-on loop.

wake:  take the next example from the stream, update the weights, score the unit
       utilities (continual backprop), maybe re-initialise a dead unit, log.
sleep: replay a little from reservoir memory (optional), fold fast weights into slow
       weights (optional), re-score the retention set from the start of the stream,
       checkpoint, rebuild the report. Then wake again. Forever.
"""
from __future__ import annotations
import json
import math
import signal
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from .streams import build_stream
from .learners import build_learner
from .plasticity import build_plasticity, CBP, L2Init, ShrinkPerturb
from .memory import ReservoirMemory
from .metrics import MetricsLog
from .state import LoopState
from .util import LN2
from . import checkpoint as ckpt
from .report import write_report


def build_optimizer(c):
    kind = c.get("kind", "sgd")
    lr = float(c.get("lr", 0.01))
    if kind == "sgd":
        return optim.SGD(learning_rate=lr, momentum=float(c.get("momentum", 0.0)),
                         weight_decay=float(c.get("weight_decay", 0.0)))
    if kind == "adam":
        return optim.Adam(learning_rate=lr)
    if kind == "adamw":
        return optim.AdamW(learning_rate=lr, weight_decay=float(c.get("weight_decay", 0.01)))
    if kind == "rmsprop":
        return optim.RMSprop(learning_rate=lr)
    raise ValueError(f"unknown optimizer {kind!r}")


class Runner:
    def __init__(self, cfg, run_dir: Path, fresh: bool = False, quiet: bool = False):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        if fresh:
            for f in (ckpt.LATEST, ckpt.META, ckpt.MEM, "metrics.csv", "report.html"):
                (self.run_dir / f).unlink(missing_ok=True)

        want_gpu = str(cfg.run.device).lower() == "gpu"
        mx.set_default_device(mx.gpu if (want_gpu and mx.metal.is_available()) else mx.cpu)
        self.device = "gpu" if mx.default_device() == mx.gpu else "cpu"
        seed = int(cfg.run.seed)
        mx.random.seed(seed)

        self.B = int(cfg.run.batch_size)
        self.stream = build_stream(cfg.stream, seed)
        self.sequential = self.stream.sequential
        self.loss_kind = self.stream.loss
        if self.sequential and self.B != 1:
            raise ValueError("sequential streams need batch_size = 1 (the hidden state is carried between chunks)")
        self.metric_name = "acc" if (self.loss_kind == "ce" and not self.sequential) else ("bpb" if self.sequential else "")

        self.model = build_learner(cfg.learner, self.stream)
        self.opt = build_optimizer(cfg.optim)
        self.opt.init(self.model.trainable_parameters())
        self.rng = np.random.default_rng(seed + 7)
        self.plast = build_plasticity(cfg.plasticity, self.model, self.rng)
        self.memory = ReservoirMemory(int(cfg.sleep.memory_size), self.rng) if bool(cfg.sleep.replay) else None
        self.slow = None
        if float(cfg.sleep.slow_ema) > 0:
            self.slow = tree_map(lambda p: mx.array(np.array(p)), self.model.parameters())

        self.state = LoopState()
        self.h = None
        self._dummy_h = mx.zeros((1,))
        self._stop = False
        self.exhausted = False
        self._build_step()
        mx.eval(self.model.parameters(), self.opt.state)

        resumed = False
        if not fresh and ckpt.exists(self.run_dir):
            ckpt.load(self)
            resumed = True
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        self.metrics = MetricsLog(self.run_dir / "metrics.csv")
        self._wall_offset = self.state.wall_s
        self._t_start = time.perf_counter()
        self._reset_window()
        self.log(f"{'resumed' if resumed else 'fresh'} | step {self.state.step:,} | {self.device} | "
                 f"{self.stream.describe()} | {self.model.__class__.__name__} | "
                 f"{self.plast.describe() if self.plast else 'no plasticity'}"
                 f"{' | replay' if self.memory is not None else ''}")

    # ------------------------------------------------------------------ setup
    def _build_step(self) -> None:
        model, opt = self.model, self.opt
        seq, loss_kind = self.sequential, self.loss_kind
        l2 = self.plast if isinstance(self.plast, L2Init) else None

        def loss_fn(model, x, y, h0):
            out, h_last, hs = model.forward(x, h0 if seq else None)
            if loss_kind == "mse":
                loss = mx.mean((out - y) ** 2)
            else:
                loss = nn.losses.cross_entropy(out, y, reduction="mean")
            if l2 is not None:
                loss = loss + l2.penalty(model)
            return loss, (out, h_last, hs)

        vg = nn.value_and_grad(model, loss_fn)

        def step(x, y, h0):
            (loss, (out, h_last, hs)), grads = vg(model, x, y, h0)
            opt.update(model, grads)
            return (loss, out, h_last, *hs)

        if bool(self.cfg.run.get("compile", True)):
            state = [model.state, opt.state]
            self._step = mx.compile(step, inputs=state, outputs=state)
        else:
            self._step = step

    def log(self, msg: str) -> None:
        print(f"[awake {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def wall(self) -> float:
        return self._wall_offset + (time.perf_counter() - self._t_start)

    def _zero_h(self) -> mx.array:
        return mx.zeros((self.model.hidden,))

    # ------------------------------------------------------------------ wake
    def _next_batch(self):
        xs, ys, reset = [], [], False
        for _ in range(self.B):
            x, y, r = self.stream.next()
            xs.append(x)
            ys.append(y)
            reset = reset or r
        return (np.stack(xs).astype(self.stream.x_dtype, copy=False),
                np.stack(ys).astype(self.stream.y_dtype, copy=False), reset)

    def step_once(self) -> None:
        try:
            xs, ys, reset = self._next_batch()
        except StopIteration:
            self._stop, self.exhausted = True, True
            return
        if self.sequential:
            if reset or self.h is None:
                self.h = self._zero_h()
            x, y, h0 = mx.array(xs[0]), mx.array(ys[0]), self.h
        else:
            x, y, h0 = mx.array(xs), mx.array(ys), self._dummy_h

        loss, out, h_last, *hs = self._step(x, y, h0)
        if self.sequential:
            self.h = h_last
        cbp = self.plast if isinstance(self.plast, CBP) else None
        if cbp is not None:
            cbp.observe(hs)
        acc = mx.mean(mx.argmax(out, axis=-1) == y) if self.metric_name == "acc" else None

        evals = [loss, self.model.parameters(), self.opt.state]
        if cbp is not None:
            evals += cbp.arrays_to_eval()
        if self.sequential:
            evals.append(self.h)
        if acc is not None:
            evals.append(acc)
        mx.eval(*evals)

        l = float(loss.item())
        if acc is not None:
            metric = float(acc.item())
        elif self.sequential:
            metric = l / LN2
        else:
            metric = float("nan")
        reinits = cbp.maybe_replace(self.opt) if cbp is not None else 0
        if isinstance(self.plast, ShrinkPerturb) and (self.state.step + 1) % self.plast.every == 0:
            self.plast.apply()
            mx.eval(self.model.parameters())
        if self.memory is not None:
            for i in range(self.B):
                self.memory.add(xs[i], ys[i])

        self.state.step += 1
        self._w_loss += l
        self._w_n += 1
        self._w_reinits += reinits
        if metric == metric:
            self._w_metric += metric
            self._w_mn += 1
        if not math.isfinite(l):
            self.log(f"non-finite loss at step {self.state.step:,}; stopping so the checkpoint stays sane")
            self._stop = True
        if self.state.step % int(self.cfg.run.log_every) == 0:
            self._flush_log()

    def _reset_window(self) -> None:
        self._w_loss = 0.0
        self._w_n = 0
        self._w_metric = 0.0
        self._w_mn = 0
        self._w_reinits = 0
        self._w_t = time.perf_counter()

    def _flush_log(self) -> None:
        if self._w_n == 0:
            return
        now = time.perf_counter()
        sps = self._w_n / max(now - self._w_t, 1e-9)
        loss = self._w_loss / self._w_n
        metric = self._w_metric / self._w_mn if self._w_mn else float("nan")
        self.metrics.log(step=self.state.step, wall_s=self.wall(), phase="wake", loss=loss, metric=metric,
                         task=self.stream.task, reinits=self._w_reinits, sps=sps)
        if not self.quiet:
            extra = f" | {self.metric_name} {metric:.4f}" if metric == metric else ""
            extra += f" | reinit {self._w_reinits}" if self._w_reinits else ""
            self.log(f"step {self.state.step:>11,} | task {self.stream.task:>5} | loss {loss:.4f}{extra} | {sps:,.0f} steps/s")
        self._reset_window()

    # ------------------------------------------------------------------ sleep
    def sleep(self) -> None:
        self._flush_log()
        t0 = time.perf_counter()
        replayed = 0
        cbp = self.plast if isinstance(self.plast, CBP) else None
        if self.memory is not None and self.memory.count >= self.B:
            for _ in range(int(self.cfg.sleep.replay_steps)):
                xs, ys = self.memory.sample(self.B)
                if self.sequential:
                    x, y, h0 = mx.array(xs[0]), mx.array(ys[0]), self._zero_h()
                else:
                    x, y, h0 = mx.array(xs), mx.array(ys), self._dummy_h
                loss, out, h_last, *hs = self._step(x, y, h0)
                if cbp is not None:
                    cbp.observe(hs)
                mx.eval(loss, self.model.parameters(), self.opt.state, *(cbp.arrays_to_eval() if cbp else []))
                if cbp is not None:
                    cbp.maybe_replace(self.opt)
                replayed += 1
            if self.sequential:
                self.h = None
        if self.slow is not None:
            tau = float(self.cfg.sleep.slow_ema)
            c = float(self.cfg.sleep.consolidate)
            params = self.model.parameters()
            self.slow = tree_map(lambda s, p: (1 - tau) * s + tau * p, self.slow, params)
            if c > 0:
                self.model.update(tree_map(lambda p, s: (1 - c) * p + c * s, params, self.slow))
            mx.eval(self.slow, self.model.parameters())
        ret = self.evaluate_retention()
        self.state.sleeps += 1
        dt = time.perf_counter() - t0
        self.metrics.log(step=self.state.step, wall_s=self.wall(), phase="sleep", task=self.stream.task,
                         ret_loss=ret["loss"], ret_metric=ret["metric"], replayed=replayed, sps=0.0)
        extra = f" | retention {self.metric_name} {ret['metric']:.4f}" if ret["metric"] == ret["metric"] else ""
        extra += f" | replayed {replayed}" if replayed else ""
        self.log(f"sleep #{self.state.sleeps} | step {self.state.step:,} | retention loss {ret['loss']:.4f}{extra} | {dt:.1f}s")
        self.save()
        write_report(self.run_dir)
        self._w_t = time.perf_counter()

    def evaluate_retention(self) -> dict:
        X, Y = self.stream.retention()
        m = self.model
        if self.sequential:
            h = self._zero_h()
            losses = []
            for i in range(len(X)):
                out, h, _ = m.forward(mx.array(X[i]), h)
                losses.append(nn.losses.cross_entropy(out, mx.array(Y[i]), reduction="mean"))
                if i % 32 == 31:
                    mx.eval(h, *losses)
            loss = float(mx.mean(mx.stack(losses)).item())
            return {"loss": loss, "metric": loss / LN2}
        losses, corrects = [], []
        for i in range(0, len(X), 512):
            x, y = mx.array(X[i:i + 512]), mx.array(Y[i:i + 512])
            out, _, _ = m.forward(x, None)
            if self.loss_kind == "mse":
                losses.append(mx.mean((out - y) ** 2, axis=-1))
            else:
                losses.append(nn.losses.cross_entropy(out, y, reduction="none"))
                corrects.append(mx.argmax(out, axis=-1) == y)
        loss = float(mx.mean(mx.concatenate(losses)).item())
        metric = float(mx.mean(mx.concatenate(corrects).astype(mx.float32)).item()) if corrects else float("nan")
        return {"loss": loss, "metric": metric}

    # ------------------------------------------------------------------ run
    def save(self) -> None:
        ckpt.save(self)

    def _install_signals(self) -> None:
        def handler(signum, frame):
            self.log(f"signal {signum}: finishing this step, then checkpointing")
            self._stop = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, handler)
            except ValueError:
                pass

    def run(self, max_steps: int | None = None) -> None:
        self._install_signals()
        cfg = self.cfg
        rate = float(cfg.run.max_steps_per_sec)
        wake_steps = int(cfg.sleep.wake_steps)
        ckpt_every = int(cfg.run.get("ckpt_every", 0))
        target = int(max_steps) if max_steps else (int(cfg.run.max_steps) or None)
        if target and self.state.step >= target:
            self.log(f"already at step {self.state.step:,} >= {target:,}; nothing to do (use --fresh to restart)")
            return
        self._w_t = time.perf_counter()
        next_t = time.perf_counter()
        try:
            while not self._stop:
                self.step_once()
                if self._stop:
                    break
                self.state.wake_pos += 1
                if wake_steps > 0 and self.state.wake_pos >= wake_steps:
                    self.sleep()
                    self.state.wake_pos = 0
                elif ckpt_every > 0 and self.state.step % ckpt_every == 0:
                    self.save()
                if target and self.state.step >= target:
                    break
                if rate > 0:
                    next_t += 1.0 / rate
                    now = time.perf_counter()
                    if next_t > now:
                        time.sleep(next_t - now)
                    elif now - next_t > 1.0:
                        next_t = now
        except KeyboardInterrupt:
            self._stop = True
        self._flush_log()
        self.save()
        write_report(self.run_dir)
        self.log(f"{'stream exhausted' if self.exhausted else 'stopped'} at step {self.state.step:,} | "
                 f"wall {self.wall() / 3600:.2f} h | checkpoint + report written to {self.run_dir}")
