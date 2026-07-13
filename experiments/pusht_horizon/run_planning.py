"""Planning-sweep driver (T0/T1/T2). Reuses eval.py end-to-end via Hydra overrides — no
planning logic duplicated. Each condition gets its own uniquely-named run dir on the NAS,
runs paired episodes (fixed seed), and its results.json is aggregated with 95% CIs.

CPU-only by default (per plan). Usage:
    python -m experiments.pusht_horizon.run_planning --test T0 --n 50 --seed 42
    python -m experiments.pusht_horizon.run_planning --test T1 --n 50
    python -m experiments.pusht_horizon.run_planning --test T2 --hstar 5 --n 50
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Default output location is the repo outputs/ folder (per EXPERIMENT_RULES.md);
# large videos are still written next to results but can be pruned to the NAS.
EXPERIMENTS_ROOT = REPO / "outputs" / "pusht" / "runs"
INDEX = REPO / "outputs" / "pusht" / "INDEX.md"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100 * (center - half), 100 * (center + half))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def conditions(test: str, hstar: int) -> list[dict]:
    """Return a list of (label, plan_config overrides) for the test."""
    if test == "T0":
        return [{"label": "H5-k5", "horizon": 5, "receding_horizon": 5}]
    if test == "T1":  # horizon sweep, frequent replanning (k=1) isolates horizon
        return [
            {"label": f"H{h}-k1", "horizon": h, "receding_horizon": 1}
            for h in (1, 3, 5, 10)
        ]
    if test == "T1o":  # horizon sweep in the OPEN-LOOP regime (k=H) LeWM actually uses
        return [
            {"label": f"H{h}-k{h}", "horizon": h, "receding_horizon": h}
            for h in (1, 3, 5, 10)
        ]
    if test == "T1oL":  # large-H open-loop tail (task fixed: budget 50, goal 25)
        return [
            {"label": f"H{h}-k{h}", "horizon": h, "receding_horizon": h}
            for h in (20, 40)
        ]
    if test == "T1oS":  # scaled-CEM-budget open-loop: num_samples grows ~ H
        return [
            {"label": f"H{h}-k{h}-s{300 * h // 5}", "horizon": h, "receding_horizon": h,
             "num_samples": 300 * h // 5}
            for h in (5, 10, 20, 40)
        ]
    if test == "T2":  # replan-interval sweep at H*
        ks = sorted({1, 2, 3, hstar})
        return [
            {"label": f"H{hstar}-k{k}", "horizon": hstar, "receding_horizon": k}
            for k in ks
        ]
    raise ValueError(f"unknown test {test!r} (expected T0/T1/T2)")


def run_condition(cond: dict, *, n: int, seed: int, device: str, threads: int,
                  run_dir: Path, gpu: str = "0") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        "--config-name=pusht.yaml",
        "policy=pusht/lewm",
        f"eval.num_eval={n}",
        f"seed={seed}",
        f"solver.device={device}",
        f"+model_device={device}",
        f"plan_config.horizon={cond['horizon']}",
        f"plan_config.receding_horizon={cond['receding_horizon']}",
        *([f"solver.num_samples={cond['num_samples']}"] if "num_samples" in cond else []),
        "+allow_long_horizon=true",
        f"+output_dir={run_dir}",
        f"hydra.run.dir={run_dir}/hydra",
        "hydra.output_subdir=null",
    ]
    env_prefix = {
        "STABLEWM_HOME": "/mnt/nas2/lewm",
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else str(gpu),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    }
    import os
    env = {**os.environ, **env_prefix}
    cmd = [str(REPO / ".venv/bin/python"), "eval.py", *overrides]
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    with (run_dir / "stdout.log").open("w") as log:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    res_file = run_dir / "results.json"
    if proc.returncode != 0 or not res_file.exists():
        return {"label": cond["label"], "error": f"exit={proc.returncode}",
                "run_dir": str(run_dir)}
    res = json.loads(res_file.read_text())
    succ = res["episode_successes"]
    k = int(sum(bool(x) for x in succ))
    lo, hi = wilson_ci(k, len(succ))
    return {
        "label": cond["label"], "horizon": cond["horizon"],
        "receding_horizon": cond["receding_horizon"], "n": len(succ),
        "success_rate": res["success_rate"], "ci95_low": lo, "ci95_high": hi,
        "eval_time_s": res.get("evaluation_time_s"), "run_dir": str(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True,
                    choices=["T0", "T1", "T1o", "T1oL", "T1oS", "T2"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hstar", type=int, default=5, help="H* for T2 (from T1 peak)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES index when --device cuda")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = EXPERIMENTS_ROOT / f"{args.test.lower()}-n{args.n}-{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "meta.json").write_text(json.dumps({
        "test": args.test, "n": args.n, "seed": args.seed, "hstar": args.hstar,
        "device": args.device, "threads": args.threads, "git_sha": git_sha(),
        "stamp": stamp, "cmd": " ".join(sys.argv),
    }, indent=2))

    rows = []
    for cond in conditions(args.test, args.hstar):
        run_dir = sweep_dir / cond["label"]
        print(f"[{args.test}] {cond['label']} -> {run_dir}", flush=True)
        row = run_condition(cond, n=args.n, seed=args.seed, device=args.device,
                            threads=args.threads, run_dir=run_dir, gpu=args.gpu)
        rows.append(row)
        print("  ", {k: row[k] for k in ("success_rate", "ci95_low", "ci95_high",
                                          "eval_time_s") if k in row}, flush=True)

    with (sweep_dir / "metrics.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("a") as f:
        ok = [r for r in rows if "success_rate" in r]
        best = max((r["success_rate"] for r in ok), default=float("nan"))
        f.write(f"- {sweep_dir.name} — {args.test} N={args.n} seed={args.seed}: "
                f"{len(ok)}/{len(rows)} conditions ok, best={best:.1f}% — {sweep_dir}\n")

    print(f"\nDONE {args.test}: wrote {sweep_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
