"""Post-hoc analysis of the baseline-validation rollouts (run_baseline_validation.py).

From each seed's per-step state log (n, T, 7) we compute, WITHOUT re-running planning:
  - E0: success@(20px, 20deg) at budget 50, env-native (4-D agent+block) AND T-only (block+angle),
        per seed + pooled, with Wilson 95% CI.
  - E1: T-only success vs tolerance at fixed budget 50 (tighten the tolerance -> harder).
  - E2: T-only success over a (tolerance x budget) grid (does more budget recover precision?).
  - Efficiency: action-blocks-to-success (first-crossing step / 5) mean & variance, per seed per
        tolerance, over the successful episodes.

Success is "exists an env step t <= B with err(t) < thr" (monotone in thr and B). Index j of the
log is the state AFTER (j+1) env steps, so budget B keeps indices [0, B-1] and blocks-to-success
= (j_first + 1) / action_block.

Usage:
    python -m experiments.pusht_horizon.analyze_baseline_validation --run-dir outputs/pusht/runs/e0-...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ACTION_BLOCK = 5
TOL_LEVELS = [20, 15, 10, 7, 5, 3]           # coupled: pos in px, angle in deg
BUDGETS = [50, 75, 100, 150, 200]            # env steps
ENV_NATIVE_ANG_DEG = math.degrees(math.pi / 9)   # 20 deg


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100 * (center - half), 100 * (center + half))


def per_step_errors(states, goals):
    """states (n,T,7), goals (n,7) -> dict of (n,T) error arrays."""
    g = goals[:, None, :]                                   # (n,1,7)
    env_pos = np.linalg.norm(states[:, :, :4] - g[:, :, :4], axis=2)       # 4-D agent+block px
    block_pos = np.linalg.norm(states[:, :, 2:4] - g[:, :, 2:4], axis=2)   # block xy px
    da = np.abs(states[:, :, 4] - g[:, :, 4]) % (2 * np.pi)
    ang_deg = np.degrees(np.minimum(da, 2 * np.pi - da))                   # block angle deg
    return {"env_pos": env_pos, "block_pos": block_pos, "ang_deg": ang_deg}


def first_cross(mask_ok):
    """mask_ok (n,T) bool -> (succeeded (n,), first_idx (n,) int or -1)."""
    succeeded = mask_ok.any(axis=1)
    first = np.where(succeeded, mask_ok.argmax(axis=1), -1)
    return succeeded, first


def success_vec(err, pos_key, thr_pos, thr_ang, budget):
    ok = (err[pos_key][:, :budget] < thr_pos) & (err["ang_deg"][:, :budget] < thr_ang)
    return first_cross(ok)


def load_seeds(run_dir: Path):
    seeds = {}
    for sd in sorted(run_dir.glob("seed*")):
        z = np.load(sd / "rollout.npz")
        seeds[int(sd.name[4:])] = {"states": z["states"], "goals": z["goals"]}
    if not seeds:
        raise SystemExit(f"no seed*/rollout.npz under {run_dir}")
    return seeds


def pooled_sr(per_seed_bool):
    """list of bool arrays -> (percent, (lo,hi) wilson, k, n)."""
    allv = np.concatenate(per_seed_bool)
    k, n = int(allv.sum()), int(allv.size)
    return 100 * k / n, wilson_ci(k, n), k, n


def analyze(run_dir: Path):
    seeds = load_seeds(run_dir)
    errs = {s: per_step_errors(d["states"], d["goals"]) for s, d in seeds.items()}
    T = next(iter(seeds.values()))["states"].shape[1]
    budgets = [b for b in BUDGETS if b <= T]

    out = {"run_dir": str(run_dir), "seeds": sorted(seeds), "T": T, "budgets": budgets}

    # ---- E0: baseline @ (20px, 20deg), budget 50, env-native + T-only ----
    def sr_block(metric, thr_pos, thr_ang, budget):
        per = {s: success_vec(errs[s], metric, thr_pos, thr_ang, budget)[0] for s in seeds}
        per_pct = {s: 100 * float(v.mean()) for s, v in per.items()}
        pooled = pooled_sr(list(per.values()))
        return {"per_seed_pct": per_pct,
                "per_seed_ci": {s: wilson_ci(int(per[s].sum()), int(per[s].size)) for s in seeds},
                "pooled_pct": pooled[0], "pooled_ci": pooled[1], "pooled_k": pooled[2],
                "pooled_n": pooled[3]}

    b0 = 50 if 50 <= T else T
    out["E0"] = {
        "budget": b0,
        "env_native_20px_20deg": sr_block("env_pos", 20, ENV_NATIVE_ANG_DEG, b0),
        "t_only_20px_20deg": sr_block("block_pos", 20, 20, b0),
    }

    # ---- E1: T-only SR vs tolerance @ budget 50 ----
    out["E1_tighten_budget50"] = {
        f"{L}px_{L}deg": sr_block("block_pos", L, L, b0) for L in TOL_LEVELS
    }

    # ---- E2: T-only SR over (tolerance x budget) ----
    grid = {}
    for L in TOL_LEVELS:
        grid[f"{L}px_{L}deg"] = {b: sr_block("block_pos", L, L, b)["pooled_pct"] for b in budgets}
    out["E2_tolerance_x_budget_pooled_pct"] = grid

    # ---- Efficiency: blocks-to-success (T-only), per seed per tolerance, budget = max ----
    Bmax = budgets[-1]
    eff = {}
    for L in TOL_LEVELS:
        per_seed = {}
        for s in seeds:
            ok, first = success_vec(errs[s], "block_pos", L, L, Bmax)
            blocks = (first[ok] + 1) / ACTION_BLOCK
            per_seed[s] = {
                "n_success": int(ok.sum()),
                "blocks_mean": float(blocks.mean()) if ok.any() else None,
                "blocks_var": float(blocks.var()) if ok.any() else None,
                "blocks_std": float(blocks.std()) if ok.any() else None,
            }
        # aggregate across seeds (mean of per-seed means)
        means = [v["blocks_mean"] for v in per_seed.values() if v["blocks_mean"] is not None]
        eff[f"{L}px_{L}deg"] = {
            "per_seed": per_seed,
            "across_seed_mean_of_means": float(np.mean(means)) if means else None,
            "across_seed_std_of_means": float(np.std(means)) if means else None,
            "budget_env_steps": Bmax,
        }
    out["efficiency_blocks_to_success_Tonly"] = eff
    return out, seeds, errs


def make_figures(out, seeds, errs, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    budgets = out["budgets"]

    # Fig A: E1 — SR vs tolerance @ budget 50 (env-native ref + T-only)
    Ls = TOL_LEVELS
    tonly = [out["E1_tighten_budget50"][f"{L}px_{L}deg"]["pooled_pct"] for L in Ls]
    cis = [out["E1_tighten_budget50"][f"{L}px_{L}deg"]["pooled_ci"] for L in Ls]
    lo = [t - c[0] for t, c in zip(tonly, cis)]; hi = [c[1] - t for t, c in zip(tonly, cis)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(Ls, tonly, yerr=[lo, hi], marker="o", capsize=3, label="T-only (block pose)")
    ax.axhline(out["E0"]["env_native_20px_20deg"]["pooled_pct"], ls="--", c="gray",
               label="env-native @20px (paper metric)")
    ax.invert_xaxis()
    ax.set_xlabel("success tolerance (px & deg)  — tighter →"); ax.set_ylabel("success rate (%)")
    ax.set_title(f"E1: precision vs success (budget=50, N pooled={out['E0']['t_only_20px_20deg']['pooled_n']})")
    ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "val_fig1_precision_budget50.png", dpi=130); plt.close(fig)

    # Fig B: E2 — SR vs budget, one line per tolerance
    fig, ax = plt.subplots(figsize=(6, 4))
    for L in Ls:
        ys = [out["E2_tolerance_x_budget_pooled_pct"][f"{L}px_{L}deg"][b] for b in budgets]
        ax.plot(budgets, ys, marker="o", label=f"{L}px/{L}°")
    ax.set_xlabel("execution budget (env steps)"); ax.set_ylabel("success rate (%)")
    ax.set_title("E2: does more budget recover precision?")
    ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(title="tolerance", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "val_fig2_budget_sweep.png", dpi=130); plt.close(fig)

    # Fig C: blocks-to-success vs tolerance (mean ± std across seeds)
    eff = out["efficiency_blocks_to_success_Tonly"]
    ms = [eff[f"{L}px_{L}deg"]["across_seed_mean_of_means"] for L in Ls]
    ss = [eff[f"{L}px_{L}deg"]["across_seed_std_of_means"] or 0 for L in Ls]
    xs = [L for L, m in zip(Ls, ms) if m is not None]
    ys = [m for m in ms if m is not None]; es = [s for m, s in zip(ms, ss) if m is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=es, marker="s", capsize=3, color="C3")
    ax.invert_xaxis()
    ax.set_xlabel("success tolerance (px & deg) — tighter →")
    ax.set_ylabel("action-blocks to success (mean ± std over seeds)")
    ax.set_title("Efficiency: blocks needed to seat the T"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "val_fig3_blocks_to_success.png", dpi=130); plt.close(fig)
    return ["val_fig1_precision_budget50.png", "val_fig2_budget_sweep.png",
            "val_fig3_blocks_to_success.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fig-dir", default="outputs/pusht/figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out, seeds, errs = analyze(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    e0 = out["E0"]
    print(f"== E0 baseline (budget {e0['budget']}) ==")
    print(f"  env-native @20px/20deg : pooled {e0['env_native_20px_20deg']['pooled_pct']:.1f}% "
          f"CI[{e0['env_native_20px_20deg']['pooled_ci'][0]:.1f},{e0['env_native_20px_20deg']['pooled_ci'][1]:.1f}] "
          f"n={e0['env_native_20px_20deg']['pooled_n']}  per-seed={ {s: round(v,1) for s,v in e0['env_native_20px_20deg']['per_seed_pct'].items()} }")
    print(f"  T-only    @20px/20deg : pooled {e0['t_only_20px_20deg']['pooled_pct']:.1f}%")
    print("== E1 tighten tolerance (T-only, budget 50) ==")
    for L in TOL_LEVELS:
        d = out["E1_tighten_budget50"][f"{L}px_{L}deg"]
        print(f"  {L:>2}px/{L:>2}deg : {d['pooled_pct']:5.1f}%  CI[{d['pooled_ci'][0]:.1f},{d['pooled_ci'][1]:.1f}]")
    print("== E2 tolerance x budget (T-only, pooled %) ==")
    hdr = "tol\\budget " + " ".join(f"{b:>6}" for b in out["budgets"])
    print("  " + hdr)
    for L in TOL_LEVELS:
        row = out["E2_tolerance_x_budget_pooled_pct"][f"{L}px_{L}deg"]
        print(f"  {L:>2}px/{L:>2}d  " + " ".join(f"{row[b]:6.1f}" for b in out["budgets"]))
    print("== efficiency: blocks-to-success (T-only, mean±std across seeds) ==")
    for L in TOL_LEVELS:
        e = out["efficiency_blocks_to_success_Tonly"][f"{L}px_{L}deg"]
        m = e["across_seed_mean_of_means"]
        print(f"  {L:>2}px/{L:>2}d : " + (f"{m:.2f} ± {e['across_seed_std_of_means']:.2f} blocks"
                                          if m is not None else "no successes"))

    if not args.no_figures:
        figs = make_figures(out, seeds, errs, Path(args.fig_dir))
        print("figures:", figs)
    print(f"\nsummary -> {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
