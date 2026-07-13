"""T4 — Is Push-T Markov-in-window? Two readouts, both on dataset trajectories:
(1) one-step latent prediction error for the SAME target frame given minimal (C=1) vs full
    (C=3) real context — if extra context doesn't help, memory can't help here.
(2) linear pose probe (latent -> block x,y,sin/cos angle) fit on N=1 vs N=3 stacked latents.

Reuses the verified primitives (data/metrics/probe). CPU-only. Usage:
    python -m experiments.pusht_horizon.run_markov --n 100 --n-probe 60
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


def _blocks(dataset, ep_id: int) -> int:
    return int(math.ceil(int(dataset.lengths[ep_id]) / FRAMESKIP))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="episodes for one-step error")
    ap.add_argument("--n-probe", type=int, default=60, help="episodes for pose probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    torch.set_num_threads(args.threads)
    import stable_worldmodel as swm
    from experiments.pusht_horizon import data as D, metrics as M, probe as P

    model = swm.wm.utils.load_pretrained("pusht/lewm").to("cpu").eval()
    model.requires_grad_(False)
    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5", num_steps=4, frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action", "state"], keys_to_cache=["action", "state"],
    )
    tf = D.image_transform()

    def encode_frames(frames_u8):
        """(L,224,224,3) uint8 -> (L, Dlat) float32 latents."""
        px = tf(frames_u8).unsqueeze(0)  # (1, L, 3, 224, 224)
        with torch.no_grad():
            emb = model.encode({"pixels": px})["emb"][0]  # (L, D)
        return emb.cpu().numpy().astype(np.float32)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / f"t4-markov-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- (1) one-step error: minimal (C=1) vs full (C=3), SAME target frame ----
    eps = D.select_episodes(dataset, args.n, args.seed, min_length=12 * FRAMESKIP).tolist()
    err_full, err_min = [], []
    for ep in eps:
        L = _blocks(dataset, ep)
        for p in (3, 6, 9):                       # predict block p from a 4-block window
            if p + 1 > L:
                continue
            ch = D.load_episode_frames_actions(dataset, ep, start=p - 3, length=4)
            f, a = ch["frames_u8"], ch["actions_2d"]
            # full: frames[0:4], C=3 -> predicts frames[3] (=block p)
            err_full.append(M.one_step_error(model, f, a, context=3))
            # minimal: frames[2:4], C=1 -> predicts frames[3] (=block p)
            err_min.append(M.one_step_error(model, f[2:4], a[2 * FRAMESKIP:4 * FRAMESKIP],
                                            context=1))
    err_full, err_min = np.asarray(err_full), np.asarray(err_min)
    onestep = {
        "n_samples": int(err_full.size),
        "err_full_C3_mean": float(err_full.mean()), "err_full_C3_std": float(err_full.std()),
        "err_min_C1_mean": float(err_min.mean()), "err_min_C1_std": float(err_min.std()),
        "paired_mean_reduction_full_minus_min": float((err_min - err_full).mean()),
        "rel_reduction_pct": float(100 * (err_min - err_full).mean() / err_min.mean()),
    }

    # ---- (2) pose probe: N=1 vs N=3 stacked latents -> block (x, y, sin, cos) ----
    peps = D.select_episodes(dataset, args.n_probe, args.seed + 1,
                             min_length=12 * FRAMESKIP).tolist()
    X1, X3, Y = [], [], []
    for ep in peps:
        L = _blocks(dataset, ep)
        ch = D.load_episode_frames_actions(dataset, ep, start=0, length=L)
        lat = encode_frames(ch["frames_u8"])           # (L, D)
        st = ch["state"]                               # (L, 7)
        bx, by, ang = st[:, 2], st[:, 3], st[:, 4]
        tgt = np.stack([bx, by, np.sin(ang), np.cos(ang)], axis=1)  # (L, 4)
        for t in range(2, L):
            X1.append(lat[t])
            X3.append(np.concatenate([lat[t - 2], lat[t - 1], lat[t]]))
            Y.append(tgt[t])
    X1, X3, Y = np.asarray(X1), np.asarray(X3), np.asarray(Y)

    def fit_eval(X):
        # 80/20 split for honest probe error
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
        tr, te = idx[:cut], idx[cut:]
        pr = P.fit_linear_probe(X[tr], Y[tr])
        e = P.probe_error(pr, X[te], Y[te])
        pos_rmse = float(np.sqrt((e["per_dim_rmse"][:2] ** 2).mean()))
        return {"rmse": e["rmse"], "r2": e["r2"], "pos_rmse_px": pos_rmse,
                "per_dim_rmse": [float(x) for x in e["per_dim_rmse"]]}

    probe_res = {"n_samples": int(len(X1)), "N1": fit_eval(X1), "N3": fit_eval(X3)}

    summary = {"onestep": onestep, "probe": probe_res}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(run_dir / "onestep.npz", err_full=err_full, err_min=err_min)
    print(json.dumps(summary, indent=2))
    print(f"\nDONE T4: {run_dir}")


if __name__ == "__main__":
    main()
