"""Append-only CSV of what the learner did, one row per log window and one per sleep."""
from __future__ import annotations
import csv
import math
from pathlib import Path

COLUMNS = ["step", "wall_s", "phase", "loss", "metric", "task", "ret_loss", "ret_metric", "reinits", "replayed", "sps"]
FLOATS = {"wall_s", "loss", "metric", "ret_loss", "ret_metric", "sps"}
INTS = {"step", "task", "reinits", "replayed"}


class MetricsLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNS)
        if new:
            self.w.writeheader()
            self.f.flush()

    def log(self, **kw) -> None:
        row = {}
        for c in COLUMNS:
            v = kw.get(c, "")
            if isinstance(v, float):
                v = "" if not math.isfinite(v) else f"{v:.6g}"
            row[c] = v
        self.w.writerow(row)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def read_metrics(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = {}
            for k, v in row.items():
                if k in FLOATS:
                    r[k] = float(v) if v not in ("", None) else float("nan")
                elif k in INTS:
                    r[k] = int(float(v)) if v not in ("", None) else 0
                else:
                    r[k] = v
            out.append(r)
    return out
