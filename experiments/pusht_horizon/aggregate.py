"""Aggregate the extended sweeps into figures + a summary:
  (1) full open-loop horizon curve  (T1o + T1oL, seed 42)   -> fig1o_full
  (2) scaled- vs fixed-CEM-budget    (T1oS vs T1o/T1oL)       -> fig5_scaled_vs_fixed
  (3) multi-seed mean±std             (T0/T1o/T2, seeds 42..)  -> fig6_multiseed + table

Scans a root for sweep dirs, reading each dir's meta.json (test, seed) + metrics.jsonl.
Usage: python -m experiments.pusht_horizon.aggregate --root <dir> --out <figdir>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scan(root: Path) -> list[dict]:
    """Return [{test, seed, rows}] for every sweep dir with a meta.json + metrics.jsonl."""
    out = []
    for meta_f in sorted(root.glob("*/meta.json")):
        d = meta_f.parent
        mj = json.loads(meta_f.read_text())
        mf = d / "metrics.jsonl"
        if not mf.exists():
            continue
        rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "success_rate" in r]
        out.append({"test": mj.get("test"), "seed": mj.get("seed"), "dir": d, "rows": rows})
    return out


def _errbars(rows):
    rows = sorted(rows, key=lambda r: r["horizon"])
    x = [r["horizon"] for r in rows]
    y = [r["success_rate"] for r in rows]
    lo = [r["success_rate"] - r["ci95_low"] for r in rows]
    hi = [r["ci95_high"] - r["success_rate"] for r in rows]
    return x, y, [lo, hi]


def fig_full_horizon(sweeps, out: Path):
    rows = []
    for s in sweeps:
        if s["test"] in ("T1o", "T1oL") and s["seed"] == 42:
            rows += s["rows"]
    by_h = {r["horizon"]: r for r in rows}          # dedup by horizon
    rows = list(by_h.values())
    if not rows:
        return
    x, y, err = _errbars(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, lw=1.6)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(h) for h in x])
    ax.set_xlabel("planning horizon H (action-blocks, open-loop k=H)")
    ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.set_title("Open-loop horizon curve (inverted-U, peak H*=5)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def fig_scaled_vs_fixed(sweeps, out: Path):
    fixed = {r["horizon"]: r for s in sweeps if s["test"] in ("T1o", "T1oL") and s["seed"] == 42 for r in s["rows"]}
    scaled = {r["horizon"]: r for s in sweeps if s["test"] == "T1oS" for r in s["rows"]}
    hs = sorted(set(scaled) & set(fixed))
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(hs, [fixed[h]["success_rate"] for h in hs], marker="o", lw=1.6,
            label="fixed CEM budget (300)")
    ax.plot(hs, [scaled[h]["success_rate"] for h in hs], marker="s", lw=1.6,
            label="scaled CEM budget (∝H)")
    ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("planning horizon H (action-blocks)"); ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Does more search rescue large H?")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def multiseed(sweeps, test: str, key: str):
    """Group condition success across seeds -> {param: [success_rate,...]}."""
    acc = defaultdict(list)
    for s in sweeps:
        if s["test"] != test:
            continue
        for r in s["rows"]:
            acc[r[key]].append(r["success_rate"])
    return dict(sorted(acc.items()))


def fig_multiseed(sweeps, out: Path):
    specs = [("T0", "horizon", "T0 (H5,k5)"), ("T1o", "horizon", "T1o (open-loop H)"),
             ("T2", "receding_horizon", "T2 (replan k, H5)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    table = {}
    for ax, (test, key, title) in zip(axes, specs):
        g = multiseed(sweeps, test, key)
        xs = list(g); means = [np.mean(v) for v in g.values()]
        stds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in g.values()]
        ax.errorbar(range(len(xs)), means, yerr=stds, marker="o", capsize=3, lw=1.6)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(title); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
        ax.set_xlabel(key); ax.set_ylabel("success (%)")
        table[test] = {str(x): {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1) if len(v) > 1 else 0),
                                "n_seeds": len(v)} for x, v in g.items()}
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas2/lewm/experiments/pusht")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sweeps = scan(Path(args.root))
    print(f"scanned {len(sweeps)} sweeps:",
          {(s['test'], s['seed']) for s in sweeps})
    fig_full_horizon(sweeps, out / "fig1o_full_horizon.png")
    fig_scaled_vs_fixed(sweeps, out / "fig5_scaled_vs_fixed.png")
    table = fig_multiseed(sweeps, out / "fig6_multiseed.png")
    (out / "multiseed_table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
