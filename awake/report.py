"""A self-contained HTML report (inline SVG, no libraries) rebuilt at every sleep."""
from __future__ import annotations
import html
import json
import math
import time
from pathlib import Path
from .metrics import read_metrics

COLORS = ["#D9731A", "#2A7C72", "#4B4F9B", "#8A8FAD"]
METRIC_NAMES = {"pmnist": "online accuracy", "text": "bits per byte", "scr": ""}


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if raw <= step:
            break
    t = math.floor(lo / step) * step
    ticks = []
    while t <= hi + step * 1e-6:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:g}k"
    return f"{v:.4g}"


def _downsample(xs, ys, max_pts=700):
    n = len(xs)
    if n <= max_pts:
        return list(xs), list(ys)
    k = math.ceil(n / max_pts)
    ox, oy = [], []
    for i in range(0, n, k):
        seg = [v for v in ys[i:i + k] if v == v]
        ox.append(xs[min(i + k - 1, n - 1)])
        oy.append(sum(seg) / len(seg) if seg else float("nan"))
    return ox, oy


def svg_chart(series, title, y_label="", logy=False, width=920, height=250) -> str:
    pl, pr, pt, pb = 64, 16, 30, 36
    W, H = width - pl - pr, height - pt - pb
    clean = []
    for s in series:
        pts = [(x, y) for x, y in zip(s["xs"], s["ys"]) if y == y and (not logy or y > 0)]
        if pts:
            xs, ys = _downsample([p[0] for p in pts], [p[1] for p in pts])
            if logy:
                ys = [math.log10(y) if y == y and y > 0 else float("nan") for y in ys]
            clean.append({"label": s["label"], "xs": xs, "ys": ys})
    head = f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" role="img" aria-label="{html.escape(title)}">'
    if not clean:
        return head + f'<text x="{pl}" y="18" class="t">{html.escape(title)}</text><text x="{pl}" y="{pt + H / 2}" class="l">no data yet</text></svg>'
    xmin = min(min(s["xs"]) for s in clean); xmax = max(max(s["xs"]) for s in clean)
    ymin = min(min(v for v in s["ys"] if v == v) for s in clean); ymax = max(max(v for v in s["ys"] if v == v) for s in clean)
    if xmax <= xmin: xmax = xmin + 1
    if ymax <= ymin: ymax = ymin + (abs(ymin) * 0.1 or 1.0)
    pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad
    sx = lambda x: pl + (x - xmin) / (xmax - xmin) * W
    sy = lambda y: pt + H - (y - ymin) / (ymax - ymin) * H
    out = [head, f'<text x="{pl}" y="18" class="t">{html.escape(title)}</text>']
    for t in _nice_ticks(ymin, ymax, 4):
        if ymin <= t <= ymax:
            y = sy(t)
            label = _fmt(10 ** t) if logy else _fmt(t)
            out.append(f'<line x1="{pl}" x2="{pl + W}" y1="{y:.1f}" y2="{y:.1f}" class="g"/>')
            out.append(f'<text x="{pl - 8}" y="{y + 4:.1f}" class="l" text-anchor="end">{label}</text>')
    for t in _nice_ticks(xmin, xmax, 6):
        if xmin <= t <= xmax:
            x = sx(t)
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pt + H}" y2="{pt + H + 4}" class="a"/>')
            out.append(f'<text x="{x:.1f}" y="{pt + H + 18}" class="l" text-anchor="middle">{_fmt(t)}</text>')
    out.append(f'<line x1="{pl}" x2="{pl + W}" y1="{pt + H}" y2="{pt + H}" class="a"/>')
    if y_label:
        out.append(f'<text x="{pl}" y="{pt - 4}" class="l">{html.escape(y_label)}</text>')
    for i, s in enumerate(clean):
        c = COLORS[i % len(COLORS)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["xs"], s["ys"]) if y == y)
        if len(s["xs"]) == 1:
            out.append(f'<circle cx="{sx(s["xs"][0]):.1f}" cy="{sy(s["ys"][0]):.1f}" r="3" fill="{c}"/>')
        else:
            out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>')
        lx = pl + W - 150 * (len(clean) - i)
        out.append(f'<rect x="{lx}" y="8" width="10" height="10" fill="{c}"/><text x="{lx + 14}" y="18" class="l">{html.escape(s["label"])}</text>')
    out.append("</svg>")
    return "".join(out)


def _tile(label: str, value: str) -> str:
    return f'<div class="tile"><div class="k">{html.escape(label)}</div><div class="v">{html.escape(value)}</div></div>'


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    rows = read_metrics(run_dir / "metrics.csv")
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    meta = json.loads((run_dir / "latest.json").read_text()) if (run_dir / "latest.json").exists() else {}
    wake = [r for r in rows if r["phase"] == "wake"]
    sleep = [r for r in rows if r["phase"] == "sleep"]
    kind = cfg.get("stream", {}).get("kind", "")
    mname = METRIC_NAMES.get(kind, "metric")
    loop = meta.get("loop", {})
    step = loop.get("step", wake[-1]["step"] if wake else 0)
    wall_h = loop.get("wall_s", 0.0) / 3600
    last = wake[-1] if wake else None
    tiles = [
        _tile("steps", f"{step:,}"),
        _tile("wall-clock", f"{wall_h:.2f} h"),
        _tile("steps / s (last window)", f"{last['sps']:,.0f}" if last else "-"),
        _tile("task / epoch", f"{last['task']}" if last else "-"),
        _tile("online loss (last window)", f"{last['loss']:.4f}" if last else "-"),
    ]
    if last and mname and last["metric"] == last["metric"]:
        tiles.append(_tile(mname + " (last window)", f"{last['metric']:.4f}"))
    if sleep:
        tiles.append(_tile("retention loss (last sleep)", f"{sleep[-1]['ret_loss']:.4f}"))
        if mname and sleep[-1]["ret_metric"] == sleep[-1]["ret_metric"]:
            tiles.append(_tile("retention " + mname, f"{sleep[-1]['ret_metric']:.4f}"))
    tiles.append(_tile("sleeps", str(loop.get("sleeps", len(sleep)))))
    total_reinit = sum(r["reinits"] for r in wake)
    if total_reinit:
        tiles.append(_tile("units re-initialised", f"{total_reinit:,}"))
    tiles.append(_tile("device", meta.get("device", cfg.get("run", {}).get("device", "?"))))
    tiles.append(_tile("plasticity", cfg.get("plasticity", {}).get("kind", "none")))

    charts = [svg_chart([{"label": "online loss", "xs": [r["step"] for r in wake], "ys": [r["loss"] for r in wake]}],
                        "Online loss, averaged per log window (prequential: scored before the update)")]
    if mname and any(r["metric"] == r["metric"] for r in wake):
        charts.append(svg_chart([{"label": mname, "xs": [r["step"] for r in wake], "ys": [r["metric"] for r in wake]}],
                                f"Online {mname}"))
    if sleep:
        charts.append(svg_chart([{"label": "retention loss", "xs": [r["step"] for r in sleep], "ys": [r["ret_loss"] for r in sleep]}],
                                "Retention: loss on the fixed set from the start of the stream, measured at each sleep"))
        if mname and any(r["ret_metric"] == r["ret_metric"] for r in sleep):
            charts.append(svg_chart([{"label": "retention " + mname, "xs": [r["step"] for r in sleep], "ys": [r["ret_metric"] for r in sleep]}],
                                    f"Retention {mname}"))
    if total_reinit:
        charts.append(svg_chart([{"label": "re-inits / window", "xs": [r["step"] for r in wake], "ys": [float(r["reinits"]) for r in wake]}],
                                "Hidden units re-initialised by continual backprop, per log window"))
    charts.append(svg_chart([{"label": "steps/s", "xs": [r["step"] for r in wake], "ys": [r["sps"] for r in wake]}],
                            "Throughput (steps per second, includes any throttling)"))

    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>awake · {html.escape(run_dir.name)}</title>
<style>
:root{{--bg:#F1F2F6;--fg:#1C1E33;--fg2:#4B4F6B;--fg3:#787C98;--line:#D3D6E3;--surface:#fff;--accent:#A5520A}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151828;--fg:#E9EAF3;--fg2:#B3B7CF;--fg3:#8A8FAD;--line:#2F3350;--surface:#1C1F33;--accent:#F4A95E}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:var(--fg3);font-family:ui-monospace,Menlo,monospace;font-size:12px;margin-bottom:22px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:26px}}
.tile{{background:var(--surface);padding:12px 14px}} .k{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--fg3);font-family:ui-monospace,Menlo,monospace}}
.v{{font-size:20px;font-variant-numeric:tabular-nums;margin-top:4px}}
.chart{{background:var(--surface);border:1px solid var(--line);padding:6px 8px 2px;margin-bottom:18px}}
svg .t{{font-size:13px;fill:var(--fg);font-weight:600}} svg .l{{font-size:11px;fill:var(--fg3);font-family:ui-monospace,Menlo,monospace}}
svg .g{{stroke:var(--line);stroke-width:1}} svg .a{{stroke:var(--fg3);stroke-width:1}}
pre{{background:var(--surface);border:1px solid var(--line);padding:12px;font-size:12px;overflow:auto;color:var(--fg2)}}
</style></head><body><div class="wrap">
<h1>awake · {html.escape(run_dir.name)}</h1>
<div class="sub">rebuilt {time.strftime('%Y-%m-%d %H:%M:%S')} · {html.escape(str(run_dir))}</div>
<div class="tiles">{''.join(tiles)}</div>
{''.join(f'<div class="chart">{c}</div>' for c in charts)}
<pre>{html.escape(json.dumps(cfg, indent=1))}</pre>
</div></body></html>"""
    out = run_dir / "report.html"
    out.write_text(page)
    return out


def write_compare(run_dirs: list[Path], out: Path) -> Path:
    """Overlay several runs (same stream, different methods) on shared charts."""
    loss_s, metric_s, ret_s, retm_s = [], [], [], []
    mname = "metric"
    for d in run_dirs:
        d = Path(d)
        rows = read_metrics(d / "metrics.csv")
        wake = [r for r in rows if r["phase"] == "wake"]
        sleep = [r for r in rows if r["phase"] == "sleep"]
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        mname = METRIC_NAMES.get(cfg.get("stream", {}).get("kind", ""), "metric") or mname
        loss_s.append({"label": d.name, "xs": [r["step"] for r in wake], "ys": [r["loss"] for r in wake]})
        metric_s.append({"label": d.name, "xs": [r["step"] for r in wake], "ys": [r["metric"] for r in wake]})
        ret_s.append({"label": d.name, "xs": [r["step"] for r in sleep], "ys": [r["ret_loss"] for r in sleep]})
        retm_s.append({"label": d.name, "xs": [r["step"] for r in sleep], "ys": [r["ret_metric"] for r in sleep]})
    charts = [svg_chart(loss_s, "Online loss per log window (prequential)")]
    if any(y == y for s in metric_s for y in s["ys"]):
        charts.append(svg_chart(metric_s, f"Online {mname}"))
    charts.append(svg_chart(ret_s, "Retention loss on the start-of-stream set, at each sleep"))
    if any(y == y for s in retm_s for y in s["ys"]):
        charts.append(svg_chart(retm_s, f"Retention {mname}"))
    names = " vs ".join(Path(d).name for d in run_dirs)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>awake · {html.escape(names)}</title>
<style>
:root{{--bg:#F1F2F6;--fg:#1C1E33;--fg3:#787C98;--line:#D3D6E3;--surface:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151828;--fg:#E9EAF3;--fg3:#8A8FAD;--line:#2F3350;--surface:#1C1F33}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:28px 20px 60px}} h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:var(--fg3);font-family:ui-monospace,Menlo,monospace;font-size:12px;margin-bottom:22px}}
.chart{{background:var(--surface);border:1px solid var(--line);padding:6px 8px 2px;margin-bottom:18px}}
svg .t{{font-size:13px;fill:var(--fg);font-weight:600}} svg .l{{font-size:11px;fill:var(--fg3);font-family:ui-monospace,Menlo,monospace}}
svg .g{{stroke:var(--line);stroke-width:1}} svg .a{{stroke:var(--fg3);stroke-width:1}}
</style></head><body><div class="wrap"><h1>awake · {html.escape(names)}</h1>
<div class="sub">rebuilt {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
{''.join(f'<div class="chart">{c}</div>' for c in charts)}
</div></body></html>"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
