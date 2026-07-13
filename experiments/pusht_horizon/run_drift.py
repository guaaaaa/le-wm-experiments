"""T3 — open-loop rollout drift. Encodes C real context frames, rolls the predictor open-loop
along the real action sequence, and measures per-step latent drift vs the encoder of the real
future frames. Optionally correlates drift@H* with planning success (--pair-with a sweep).

Reuses the verified primitives (data/rollout/metrics). CPU-only. Usage:
    python -m experiments.pusht_horizon.run_drift --n 100 --horizon 20
    python -m experiments.pusht_horizon.run_drift --pair-with <t1 sweep dir>/H5-k1 --hstar 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

FRAMESKIP = 5
OUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "pusht" / "runs"


def _load_results(pair_with: str) -> dict:
    p = Path(pair_with)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text())


def _blocks(dataset, ep_id: int) -> int:
    return int(math.ceil(int(dataset.lengths[ep_id]) / FRAMESKIP))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=20, help="drift horizon in action-blocks")
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--pair-with", default=None,
                    help="planning sweep dir/results.json for drift@H* vs success")
    ap.add_argument("--hstar", type=int, default=5)
    args = ap.parse_args()

    os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    torch.set_num_threads(args.threads)
    import stable_worldmodel as swm
    from experiments.pusht_horizon import data as D, metrics as M, rollout as R

    model = swm.wm.utils.load_pretrained("pusht/lewm").to("cpu").eval()
    model.requires_grad_(False)
    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5", num_steps=4, frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action", "state"], keys_to_cache=["action", "state"],
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "corr" if args.pair_with else "curve"
    run_dir = OUT_ROOT / f"t3-drift-{mode}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    context, horizon = args.context, args.horizon

    if args.pair_with:
        # --- correlation mode: drift@H* on the exact planning episodes ---
        rj = _load_results(args.pair_with)
        episodes = list(rj["episodes"])
        starts_blk = [int(s) // FRAMESKIP for s in rj["start_steps"]]
        successes = [bool(x) for x in rj["episode_successes"]]
        horizon = args.hstar
        need = context + horizon
        drift_at_h, succ_used, used = [], [], 0
        for ep, sb, sc in zip(episodes, starts_blk, successes):
            if sb + need > _blocks(dataset, ep):
                continue  # rollout window doesn't fit from the planning start
            ch = D.load_episode_frames_actions(dataset, ep, start=sb, length=need)
            out = R.open_loop_rollout(model, ch["frames_u8"], ch["actions_2d"],
                                      context=context, horizon=horizon)
            d = M.latent_drift(out["pred_latents"], out["true_latents"], normalize="rms")
            drift_at_h.append(float(d[-1]))
            succ_used.append(1.0 if sc else 0.0)
            used += 1
        r = M.pearson(np.asarray(drift_at_h), np.asarray(succ_used)) if used > 2 else float("nan")
        summary = {
            "mode": "corr", "hstar": args.hstar, "context": context,
            "n_used": used, "n_total": len(episodes),
            "pearson_drift_vs_success": r,
            "mean_drift_success": float(np.mean([d for d, s in zip(drift_at_h, succ_used) if s == 1])) if any(succ_used) else None,
            "mean_drift_failure": float(np.mean([d for d, s in zip(drift_at_h, succ_used) if s == 0])) if any(s == 0 for s in succ_used) else None,
            "pair_with": args.pair_with,
        }
        np.savez(run_dir / "drift_corr.npz", drift_at_h=np.asarray(drift_at_h),
                 success=np.asarray(succ_used), episodes=np.asarray(episodes[:used]))
    else:
        # --- main curve: per-step drift over N long episodes, start=0 ---
        need = context + horizon
        eps = D.select_episodes(dataset, args.n, args.seed,
                                min_length=(need + 1) * FRAMESKIP).tolist()
        per_ep = []
        for ep in eps:
            ch = D.load_episode_frames_actions(dataset, ep, start=0, length=need)
            out = R.open_loop_rollout(model, ch["frames_u8"], ch["actions_2d"],
                                      context=context, horizon=horizon)
            per_ep.append(M.latent_drift(out["pred_latents"], out["true_latents"],
                                         normalize="rms"))
        arr = np.stack(per_ep)  # (N, H)
        mean = arr.mean(0)
        sem = arr.std(0, ddof=1) / math.sqrt(arr.shape[0])
        summary = {
            "mode": "curve", "n": len(eps), "context": context, "horizon": horizon,
            "steps": list(range(1, horizon + 1)),
            "drift_mean": mean.tolist(),
            "drift_ci95_low": (mean - 1.96 * sem).tolist(),
            "drift_ci95_high": (mean + 1.96 * sem).tolist(),
        }
        np.savez(run_dir / "drift_curve.npz", per_episode=arr, mean=mean, sem=sem,
                 episodes=np.asarray(eps))

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nDONE T3 ({mode}): {run_dir}")


if __name__ == "__main__":
    main()
