from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from . import RUNS_DIR, REPO_ROOT, __version__


def _run_dir(name: str) -> Path:
    return RUNS_DIR / name


def cmd_run(a) -> int:
    from .config import load_config
    from .loop import Runner
    over = {"run": {}, "sleep": {}}
    if a.device:
        over["run"]["device"] = a.device
    if a.rate is not None:
        over["run"]["max_steps_per_sec"] = a.rate
    if a.wake is not None:
        over["sleep"]["wake_steps"] = a.wake
    if a.no_compile:
        over["run"]["compile"] = False
    cfg = load_config(a.config, over)
    name = a.name or Path(a.config).stem
    runner = Runner(cfg, _run_dir(name), fresh=a.fresh, quiet=a.quiet)
    runner.run(max_steps=a.steps)
    print(f"report: {_run_dir(name) / 'report.html'}")
    return 0


def cmd_status(a) -> int:
    from .metrics import read_metrics
    from .daemon import status as dstatus
    names = [a.name] if a.name else sorted(p.name for p in RUNS_DIR.glob("*") if p.is_dir() and not p.name.startswith("_"))
    if not names:
        print(f"no runs under {RUNS_DIR}")
        return 0
    for name in names:
        d = _run_dir(name)
        meta = json.loads((d / "latest.json").read_text()) if (d / "latest.json").exists() else {}
        rows = read_metrics(d / "metrics.csv")
        wake = [r for r in rows if r["phase"] == "wake"]
        sleep = [r for r in rows if r["phase"] == "sleep"]
        loop = meta.get("loop", {})
        age = f"{(time.time() - meta['saved_at']) / 60:.0f} min ago" if meta else "never"
        print(f"{name}")
        print(f"  step {loop.get('step', 0):,} · wall {loop.get('wall_s', 0) / 3600:.2f} h · sleeps {loop.get('sleeps', 0)} · checkpoint {age} · daemon: {dstatus(name)}")
        if wake:
            w = wake[-1]
            m = f" · metric {w['metric']:.4f}" if w["metric"] == w["metric"] else ""
            print(f"  last window: loss {w['loss']:.4f}{m} · task {w['task']} · {w['sps']:,.0f} steps/s")
        if sleep:
            s = sleep[-1]
            m = f" · metric {s['ret_metric']:.4f}" if s["ret_metric"] == s["ret_metric"] else ""
            print(f"  last sleep:  retention loss {s['ret_loss']:.4f}{m} at step {s['step']:,}")
    return 0


def cmd_report(a) -> int:
    from .report import write_report
    out = write_report(_run_dir(a.name))
    print(out)
    if a.open:
        subprocess.run(["open", str(out)])
    return 0


def cmd_compare(a) -> int:
    from .report import write_compare
    out = write_compare([_run_dir(n) for n in a.names], RUNS_DIR / ("compare-" + "-vs-".join(a.names) + ".html"))
    print(out)
    if a.open:
        subprocess.run(["open", str(out)])
    return 0


def cmd_bench(a) -> int:
    from .bench import run_bench
    run_bench()
    return 0


def cmd_data(a) -> int:
    from .data import fetch_shakespeare, fetch_mnist
    {"shakespeare": fetch_shakespeare, "mnist": fetch_mnist}[a.what]()
    return 0


def cmd_daemon(a) -> int:
    from . import daemon
    if a.action == "install":
        daemon.install(a.config, a.name or Path(a.config).stem, a.rate)
    elif a.action == "uninstall":
        daemon.uninstall(a.name)
    elif a.action == "status":
        print(daemon.status(a.name))
    elif a.action == "logs":
        daemon.logs(a.name)
    return 0


def cmd_selftest(a) -> int:
    from .selftest import run_selftest
    return run_selftest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="awake", description=f"awake {__version__}: an always-on learner for Apple Silicon (runs under {RUNS_DIR})")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="learn from a stream (resumes from the run's checkpoint if one exists)")
    r.add_argument("config")
    r.add_argument("--name", help="run name (default: config file stem)")
    r.add_argument("--steps", type=int, help="stop at this total step count (default: forever)")
    r.add_argument("--fresh", action="store_true", help="discard any existing checkpoint and metrics for this run")
    r.add_argument("--device", choices=["cpu", "gpu"])
    r.add_argument("--rate", type=float, help="throttle to at most this many steps per second")
    r.add_argument("--wake", type=int, help="override [sleep] wake_steps")
    r.add_argument("--no-compile", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="summarise one run or all runs")
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_status)

    rp = sub.add_parser("report", help="rebuild runs/<name>/report.html")
    rp.add_argument("name")
    rp.add_argument("--open", action="store_true", help="open it in the browser")
    rp.set_defaults(fn=cmd_report)

    cp = sub.add_parser("compare", help="overlay several runs on shared charts (same stream, different methods)")
    cp.add_argument("names", nargs="+")
    cp.add_argument("--open", action="store_true")
    cp.set_defaults(fn=cmd_compare)

    sub.add_parser("bench", help="CPU vs GPU training throughput on this machine").set_defaults(fn=cmd_bench)

    d = sub.add_parser("data", help="download a dataset into ~/.cache/awake")
    d.add_argument("what", choices=["shakespeare", "mnist"])
    d.set_defaults(fn=cmd_data)

    dm = sub.add_parser("daemon", help="run a config forever under launchd")
    dm.add_argument("action", choices=["install", "uninstall", "status", "logs"])
    dm.add_argument("config", nargs="?", help="(install) config path")
    dm.add_argument("--name", help="run name (default: config stem)")
    dm.add_argument("--rate", type=float, default=200.0, help="(install) max steps per second, default 200")
    dm.set_defaults(fn=cmd_daemon)

    sub.add_parser("selftest", help="exercise every path for a few hundred steps").set_defaults(fn=cmd_selftest)

    a = p.parse_args(argv)
    if a.cmd == "daemon":
        if a.action == "install" and not a.config:
            p.error("daemon install needs a config path")
        if a.action != "install" and not a.name:
            p.error(f"daemon {a.action} needs --name")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
