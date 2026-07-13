"""Build Fig 1-4 from the T1/T2/T3/T4 run outputs. Each flag is optional; only figures with
data are produced. Units: H and k in action-blocks (env steps = x5). Usage:

    python -m experiments.pusht_horizon.figures --out <dir> \
        --t1 <t1 sweep> --t2 <t2 sweep> --t3-curve <run> --t3-corr <run> --t4 <run> --hstar 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rows(sweep: Path) -> list[dict]:
    return [json.loads(l) for l in (Path(sweep) / "metrics.jsonl").read_text().splitlines() if l.strip()]


def fig_success_vs_x(sweep: Path, key: str, xlabel: str, title: str, out: Path):
    rows = [r for r in _rows(sweep) if "success_rate" in r]
    rows.sort(key=lambda r: r[key])
    x = [r[key] for r in rows]
    y = [r["success_rate"] for r in rows]
    lo = [r["success_rate"] - r["ci95_low"] for r in rows]
    hi = [r["ci95_high"] - r["success_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3, lw=1.6)
    ax.set_xlabel(xlabel); ax.set_ylabel("success rate (%)"); ax.set_title(title)
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_drift_curve(run: Path, hstar: int, context: int, out: Path):
    d = np.load(Path(run) / "drift_curve.npz")
    s = json.loads((Path(run) / "summary.json").read_text())
    steps = np.array(s["steps"]); mean = np.array(s["drift_mean"])
    lo = np.array(s["drift_ci95_low"]); hi = np.array(s["drift_ci95_high"])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(steps, mean, marker="o", lw=1.6, label="latent drift (rms-norm)")
    ax.fill_between(steps, lo, hi, alpha=0.2)
    ax.axvline(context, ls="--", c="gray", label=f"C={context}")
    ax.axvline(hstar, ls=":", c="crimson", label=f"H*={hstar}")
    ax.set_xlabel("open-loop horizon (action-blocks)"); ax.set_ylabel("latent drift")
    ax.set_title("T3: open-loop rollout drift"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_drift_corr(run: Path, out: Path):
    d = np.load(Path(run) / "drift_corr.npz")
    s = json.loads((Path(run) / "summary.json").read_text())
    drift, succ = d["drift_at_h"], d["success"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    jit = (np.random.default_rng(0).random(succ.shape) - 0.5) * 0.06
    ax.scatter(drift, succ + jit, alpha=0.5, s=18)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fail", "success"])
    ax.set_xlabel("drift @ H*"); ax.set_title(
        f"T3: drift@H* vs success (r={s.get('pearson_drift_vs_success'):.2f}, n={s.get('n_used')})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def fig_markov(run: Path, out: Path):
    s = json.loads((Path(run) / "summary.json").read_text())
    o, pr = s["onestep"], s["probe"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].bar(["C=1 (min)", "C=3 (full)"],
                [o["err_min_C1_mean"], o["err_full_C3_mean"]],
                yerr=[o["err_min_C1_std"], o["err_full_C3_std"]], capsize=4,
                color=["#bbb", "#4577b0"])
    axes[0].set_ylabel("one-step latent error")
    axes[0].set_title(f"T4: context gain = {o['rel_reduction_pct']:.1f}%")
    axes[1].bar(["N=1", "N=3"], [pr["N1"]["pos_rmse_px"], pr["N3"]["pos_rmse_px"]],
                color=["#bbb", "#4577b0"])
    axes[1].set_ylabel("block-position RMSE (px)")
    axes[1].set_title(f"pose probe (r²: {pr['N1']['r2']:.2f} vs {pr['N3']['r2']:.2f})")
    for a in axes:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t1"); ap.add_argument("--t2")
    ap.add_argument("--t3-curve"); ap.add_argument("--t3-corr"); ap.add_argument("--t4")
    ap.add_argument("--hstar", type=int, default=5); ap.add_argument("--context", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.t1:
        fig_success_vs_x(Path(args.t1), "horizon", "planning horizon H (action-blocks)",
                         "T1: success vs planning horizon", out / "fig1_success_vs_H.png")
    if args.t2:
        fig_success_vs_x(Path(args.t2), "receding_horizon", "replan interval k (action-blocks)",
                         "T2: success vs replan interval", out / "fig2_success_vs_k.png")
    if args.t3_curve:
        fig_drift_curve(Path(args.t3_curve), args.hstar, args.context, out / "fig3_drift_curve.png")
    if args.t3_corr:
        fig_drift_corr(Path(args.t3_corr), out / "fig3b_drift_vs_success.png")
    if args.t4:
        fig_markov(Path(args.t4), out / "fig4_markov.png")


if __name__ == "__main__":
    main()
