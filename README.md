# awake

An always-on learner for Apple Silicon. One stream, one example at a time, weights
updated on every step, a sleep phase that consolidates and measures what was kept,
and checkpoints that let it run for a month without ever being reset.

Built on [MLX](https://github.com/ml-explore/mlx): training runs on the M-series GPU
through Metal. No CUDA anywhere.

## The thesis this repo tests

A model you have to invoke is not the thing. The thing learns from every step it
takes, keeps state between steps, and is still improving and still remembering its
first week after thirty days of never being restarted. Nobody has a clean public
demo of that. This is the base for building one.

Of the five pieces a frozen model behind an API is missing, this repo attacks three:

| piece  | what it means                                 | here                                                                 |
|--------|-----------------------------------------------|----------------------------------------------------------------------|
| learns | weights change from experience, no train/deploy split | batch size 1, an update per example, continual backprop to keep units alive |
| state  | something persists between moments           | recurrent hidden state carried across chunks; slow weights that survive sleep |
| stream | there is always input                        | three never-ending streams, resumable mid-stream                     |
| drive  | it wants things when nobody asks             | not yet (see roadmap)                                                |
| watts  | 24/7 has to be cheap                         | tiny models that run at thousands of steps/s on laptop CPU cores; a launchd daemon at nice 10 |

The number to own: **online loss and retention after N days of continuous
operation, no resets.** Every run writes that curve to `runs/<name>/report.html`.

## Why MLX, and why not write your own

MLX already is "something that runs on my M1 for training": arrays, autograd, lazy
evaluation, unified memory, Metal kernels, optimizers. The alternatives on a Mac are
PyTorch's MPS backend (works, patchier op coverage, slower for small ops), JAX-Metal
(experimental, training is flaky), and tinygrad (Metal backend, small community).
Writing a framework from scratch would cost months the M5 does not need. When you hit
an op MLX lacks, for example event-driven sparse updates for a spiking learner,
`mx.fast.metal_kernel` lets you write one Metal kernel inline and keep everything else.

Measured on this M1 Air (16 GB), one training step = forward + backward + update:

| model                                  | CPU, compiled | GPU, compiled | use |
|----------------------------------------|--------------:|--------------:|-----|
| MLP 20-5-1, batch 1 (SCR)              | 10,646 steps/s | 2,619 steps/s | CPU |
| byte GRU, hidden 256, seq 64, batch 1  | 41 steps/s    | 101 steps/s   | GPU |

(`awake bench`, 16 Sep 2026, nothing else running.) Tiny nets are kernel-launch
bound, so the CPU cores win by 4x. The GRU wins on Metal by 2.5x. `device` is
therefore a per-config setting. Run `awake bench` on the M5 to re-decide.

## Install

```sh
cd ~/Developer/awake
uv venv && uv pip install -e .
source .venv/bin/activate
awake selftest          # every path, plus a stop/resume test that must match bit-for-bit
awake bench             # CPU vs GPU on this machine
awake data shakespeare  # 1 MB text stream
awake data mnist        # for permuted MNIST
```

`scripts/quickstart.sh` does all of that.

## Commands

```
awake run CONFIG [--name N] [--steps K] [--fresh] [--device cpu|gpu] [--rate R] [--wake W]
awake status [NAME]          # step, wall-clock, last loss, last retention, daemon state
awake report NAME [--open]   # rebuild runs/NAME/report.html
awake compare A B [C...] [--open]   # overlay runs on shared charts -> runs/compare-A-vs-B.html
awake daemon install CONFIG [--name N] [--rate 200]   # run forever under launchd, survives reboot
awake daemon uninstall|status|logs --name N
awake bench | selftest | data shakespeare|mnist
```

`awake run` resumes from `runs/<name>/latest.*` if it exists. Ctrl-C or `launchctl`
stop sends a signal; the loop finishes the step, checkpoints, then exits. `--fresh`
throws the checkpoint and metrics away.

## Streams, learners, plasticity

**Streams** (`awake/streams/`) hand out one example at a time forever and can report
their own position, so a run resumes mid-stream:

- `scr`: Slowly-Changing Regression (Dohare et al., *Nature* 2024). Twenty input bits,
  fifteen held fixed for 10,000 examples then one flips. A fixed random target network
  of 100 threshold units. A 5-unit learner has to keep re-adapting forever.
- `pmnist`: online permuted MNIST. Every task is a new pixel permutation, every image
  is seen once. Retention is measured on task 0's permutation.
- `text`: any text file or folder as a byte stream, 64 bytes per chunk, hidden state
  carried across chunks. Retention is the first 64 KB of the stream.

**Learners** (`awake/learners/`) expose their hidden activations and describe which
weight slices belong to each hidden unit, which is what continual backprop needs:

- `mlp`: plain MLP (tanh / relu / gelu / silu).
- `chargru`: byte-level GRU language model, truncated backprop through one chunk.

**Plasticity** (`awake/plasticity/`):

- `cbp`: continual backprop. Every unit carries a running utility (mean-corrected
  activation magnitude times outgoing weight size). Mature units with the lowest
  utility are re-initialised at a small rate: fresh incoming weights, zero outgoing
  weights, optimizer moments cleared. Works on the MLP and on the GRU's gates.
- `l2init`: pull weights toward their initial values (regenerative regularisation).
- `shrink_perturb`: shrink and add noise every N steps.

**Sleep** (`[sleep]` in the config): every `wake_steps`, optionally replay a small
reservoir-sampled memory of past examples, optionally fold fast weights into a slow
EMA copy (and pull the fast weights back toward it), then re-score the retention set,
checkpoint, and rebuild the report. Replay is off by default because the Alberta camp
forbids it and on in `text_sleep.toml` because the sleep papers rely on it. That
difference is the first ablation worth running.

## First experiments

```sh
awake run configs/scr_bp.toml  --steps 3000000     # 300 tasks, ~10 min on the M1
awake run configs/scr_cbp.toml --steps 3000000
awake report scr_bp --open; awake report scr_cbp --open
```
Look at the online-loss chart. Plain backprop should creep upward over hundreds of
tasks (loss of plasticity); continual backprop should stay flat.

```sh
awake run configs/pmnist_bp.toml  --steps 125000    # 200 tasks
awake run configs/pmnist_cbp.toml --steps 125000
```
Watch online accuracy per task and the retention accuracy on task 0 at each sleep.

```sh
awake run configs/text_bp.toml    --steps 20000     # ~1 pass over tiny shakespeare
awake run configs/text_sleep.toml --steps 20000
```
Compare retention bits-per-byte at each sleep: does replay during sleep keep the
start of the stream cheaper to predict?

## First results on this M1 (16 Sep 2026)

The metrics, configs and HTML reports behind these numbers are in [`results/`](results/)
(download a `compare-*.html` file and open it in a browser for the charts).

Online loss per 30-task decile on the slowly-changing regression, 3M steps each
(`awake compare scr_bp scr_cbp`):

| decile      | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
|-------------|------|------|------|------|------|------|------|------|------|------|
| plain backprop      | 0.53 | 1.01 | 1.21 | 1.05 | 1.09 | 1.33 | 1.28 | 1.26 | 1.25 | 1.44 |
| continual backprop  | 0.39 | 0.51 | 0.38 | 0.45 | 0.39 | 0.53 | 0.51 | 0.44 | 0.51 | 0.51 |

Plain backprop loses the ability to track the target: its error nearly triples over
300 tasks. Continual backprop, re-initialising about one unit per 2,000 steps, stays
flat. That is the *Nature* 2024 result, reproduced in a quarter of an hour on a
laptop. The SCR retention number is expected to be large and noisy in both runs: a
5-unit tracker is not supposed to hold the task-0 regime, so for this stream the
online loss is the metric that matters.

Permuted MNIST, 200 tasks: online accuracy rises to 0.85 (backprop) and 0.84 (CBP);
backprop peaks in decile 7 and drifts down, CBP is still rising at the end. Two
hundred tasks is too few to separate them clearly; the paper's decline takes
thousands. Retention on task 0 sits near chance (0.10 to 0.26) for both, which is
catastrophic forgetting on display; `replay = true` is the obvious next ablation.

Text, 20k chunks of tiny shakespeare: online bits-per-byte 3.41 -> 2.76 (no sleep)
vs 3.41 -> 2.61 (sleep with replay), and retention on the first 64 KB 3.10 vs 2.70
at the last sleep. Replay during sleep buys about 0.4 bits per byte of retention.
Caveat: with `on_end = "loop"` the last 2,600 chunks re-read the start of the file,
which inflates the final retention score equally for both runs; use `on_end = "stop"`
or a bigger corpus for a clean number.

## The month-long run

```sh
awake daemon install configs/scr_cbp.toml --name month_scr --rate 200
awake daemon install configs/text_sleep.toml --name month_text --rate 50
awake status
```
At 200 steps/s that is 17 million updates a day and a few hundred KB of metrics a
day; the report rebuilds at every sleep. `awake daemon uninstall --name month_scr`
stops it and keeps the checkpoint. Nothing is installed until you run those lines.

## Layout

```
awake/
  cli.py          awake run / status / report / daemon / bench / data / selftest
  loop.py         the wake/sleep Runner: stream -> step -> plasticity -> sleep -> checkpoint
  streams/        scr.py, pmnist.py, text.py
  learners/       mlp.py, chargru.py
  plasticity/     cbp.py, regularizers.py
  memory.py       reservoir memory for sleep replay
  checkpoint.py   weights + optimizer + plasticity stats + stream position + RNG, atomic
  metrics.py      metrics.csv, one row per log window and per sleep
  report.py       self-contained HTML report with inline SVG charts
  daemon.py       launchd plist install / uninstall / status / logs
  bench.py        CPU vs GPU throughput
  selftest.py     every path + exact resume test
configs/          scr_bp, scr_cbp, pmnist_bp, pmnist_cbp, text_bp, text_sleep
runs/<name>/      config.json, metrics.csv, latest.safetensors, latest.json, memory.npz, report.html
```

## Roadmap, in order

1. **Per-weight step sizes** (IDBD / Autostep): the other half of the Alberta recipe
   for continual learning; `cbp` only covers generate-and-test.
2. **A test-time-training layer** for the text learner (Sun et al. 2024, TTT-E2E
   2025): a fast-weights module that keeps learning on the context it reads.
3. **A streaming RL agent** on a real environment (Craftax, NetHack): the *drive* and
   *stream* pieces, one update per step, no replay, following Elsayed et al. 2024.
4. **An event-driven spiking learner** with custom Metal kernels: the *watts* piece.
5. **The month**: on the M5, `scr_cbp` and `text_sleep` as daemons for 30 days. The
   report is the result.

## Honest notes

- The SCR target is built from Dohare et al.'s recipe as read from the paper; check it
  against their reference code before publishing numbers.
- `cbp` uses the contribution utility; set `adaptation = true` for the variant that
  also divides by incoming weight magnitude.
- Exact stop/resume is verified on CPU. On Metal, float reductions can differ in the
  last bits between runs; resumed runs stay statistically identical, not bitwise.
- Sequential streams use batch size 1 because the hidden state is carried between
  consecutive chunks. That is the point, not a limitation to engineer away.
