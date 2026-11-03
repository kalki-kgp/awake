"""Run a config forever under launchd, restarted on crash or reboot, at low priority."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from . import REPO_ROOT, RUNS_DIR

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key><array>
    <string>{python}</string><string>-m</string><string>awake.cli</string>
    <string>run</string><string>{config}</string><string>--name</string><string>{name}</string><string>--rate</string><string>{rate}</string>
  </array>
  <key>WorkingDirectory</key><string>{repo}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>Nice</key><integer>10</integer>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
</dict></plist>
"""


def _label(name: str) -> str:
    return f"com.awake.{name}"


def _plist_path(name: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_label(name)}.plist"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(config: str, name: str, rate: float) -> None:
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    plist = _plist_path(name)
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(PLIST.format(label=_label(name), python=sys.executable, config=config_path, name=name,
                                  rate=rate, repo=REPO_ROOT, log=run_dir / "daemon.log"))
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{_label(name)}"], capture_output=True)
    r = subprocess.run(["launchctl", "bootstrap", _domain(), str(plist)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"launchctl bootstrap failed: {r.stderr.strip()}")
    print(f"installed {plist}\nrunning {config_path} as '{name}' at <= {rate:g} steps/s, nice 10\n"
          f"log: {run_dir / 'daemon.log'}\nstop with: awake daemon uninstall {name}")


def uninstall(name: str) -> None:
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{_label(name)}"], capture_output=True)
    plist = _plist_path(name)
    if plist.exists():
        plist.unlink()
        print(f"stopped and removed {plist} (the checkpoint in runs/{name} is kept)")
    else:
        print(f"no daemon installed for '{name}'")


def status(name: str) -> str:
    r = subprocess.run(["launchctl", "print", f"{_domain()}/{_label(name)}"], capture_output=True, text=True)
    if r.returncode != 0:
        return "not installed" if not _plist_path(name).exists() else "installed but not loaded"
    state, pid = "?", "?"
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("state = "):
            state = s.split("=", 1)[1].strip()
        elif s.startswith("pid = "):
            pid = s.split("=", 1)[1].strip()
    return f"{state} (pid {pid})"


def logs(name: str, n: int = 40) -> None:
    log = RUNS_DIR / name / "daemon.log"
    if not log.exists():
        print(f"no log at {log}")
        return
    lines = log.read_text(errors="replace").splitlines()
    print("\n".join(lines[-n:]))
