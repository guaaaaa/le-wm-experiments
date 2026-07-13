"""Success-trajectory videos at three precision tiers: 20, 10, and 3 px/deg (T-only metric).

Candidates are picked from the saved seed-42 validation rollout (states only — no pixels were
recorded), binned by the TIGHTEST tolerance they reach within budget 50:
    tier 20 = ok@20 & ~ok@10  (loose success: T near goal but > 10 px off)
    tier 10 = ok@10 & ~ok@3   (good but not exact)
    tier  3 = ok@3            (precise seating)
Because planning is not bit-reproducible, candidates are RE-RUN (batched, with rendering) and
re-binned from the fresh rollout; only episodes still in-bin are kept. Each clip is trimmed at the
first tolerance-crossing frame (the moment success fires — afterwards the flat endpoint cost lets
the agent wander) + a 1 s hold, with a live banner: step, block pos err (px), angle err (deg).
Also writes a side-by-side compare of one clip per tier.

Usage:
    python -m experiments.pusht_horizon.make_precision_videos --gpu 1 \
        --run-dir outputs/pusht/runs/e0-valN100-20260701 --out outputs/pusht/videos/precision
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")

import cv2
import numpy as np

from experiments.pusht_horizon.run_baseline_validation import (
    CALLABLES, img_transform, make_policy)

TIERS = [20, 10, 3]          # requested tolerance tiers (px & deg)
NEXT_TIGHTER = {20: 10, 10: 3, 3: None}   # in-bin = ok@tier & ~ok@next
FPS = 8
HOLD = 8                     # frames held after the crossing (~1 s)


def block_errs(states, goals):
    """states (n,T,7), goals (n,7) -> pos err (n,T) px, angle err (n,T) deg."""
    pos = np.linalg.norm(states[:, :, 2:4] - goals[:, None, 2:4], axis=2)
    da = np.abs(states[:, :, 4] - goals[:, None, 4]) % (2 * np.pi)
    ang = np.degrees(np.minimum(da, 2 * np.pi - da))
    return pos, ang


def ok_any(pos, ang, L, budget):
    return ((pos[:, :budget] < L) & (ang[:, :budget] < L)).any(axis=1)


def write_h264(frames, path, fps=FPS):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def banner(frame, lines, ok):
    """Stack a white banner (one row per text line) above an RGB frame."""
    bar = np.full((16 * len(lines) + 8, frame.shape[1], 3), 255, np.uint8)
    col = (30, 140, 30) if ok else (60, 60, 60)   # RGB: green once inside tolerance
    for r, text in enumerate(lines):
        cv2.putText(bar, text, (5, 14 + 16 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1,
                    cv2.LINE_AA)
    return np.vstack([bar, frame])


def rollout_with_frames(world, model, dataset, episodes, start_steps, *, goal_offset, seed,
                        budget, device, process, tf, horizon=5, k=5):
    """Batched open-loop rollout with termination disabled, logging states AND rendered frames."""
    from stable_worldmodel.world.world import _extract_init_goal, _apply_callables
    n = len(episodes)
    assert n == world.num_envs
    policy = make_policy(model, horizon, k, device, seed, process, tf)
    world.set_policy(policy)
    init_state, goal_state, _ = _extract_init_goal(dataset, episodes, start_steps, goal_offset)
    world.reset(seed=init_state.get("seed"))
    merged = {**init_state, **goal_state}
    for i in range(n):
        _apply_callables(world.envs.envs[i].unwrapped, CALLABLES,
                         {kk: vv[i] for kk, vv in merged.items()})
    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for key, v in src.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(v[:, None, ...],
                                                   shape_prefix + v.shape[1:]).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    for i in range(n):
        env_i = world.envs.envs[i].unwrapped
        env_i.eval_state = lambda gs, cs: (False, float(np.linalg.norm(gs - cs)))
        # The renderer's green T is env.goal_pose = the env's DEFAULT goal (env.py:577), which
        # _set_goal_state does NOT move (it only sets the metric's goal_state). Point it at the
        # true eval goal (block x, y, angle) so the video shows the actual target.
        g7 = np.asarray(goal_state["goal_state"][i], dtype=float)
        env_i.goal_pose = np.concatenate([g7[2:4], [g7[4]]])

    states_log, frames_log = [], []

    def on_step(w):
        w.infos.update(deepcopy(goal_snapshot))
        states_log.append(np.asarray(w.infos["state"])[:, 0, :].copy())
        arr = np.asarray(w.infos["pixels"])
        frames_log.append((arr[:, -1] if arr.ndim == 5 else arr).astype(np.uint8).copy())

    world._run(max_steps=budget, mode="wait", on_step=on_step)
    states = np.stack(states_log, axis=1)                     # (n, budget, 7)
    frames = np.stack(frames_log, axis=1)                     # (n, budget, H, W, 3)
    return states, frames, np.asarray(goal_state["goal_state"])


def make_clip(frames_i, pos_i, ang_i, L):
    """Trim at the first crossing of tolerance L, annotate every frame, add a hold."""
    mask = (pos_i < L) & (ang_i < L)
    t_star = int(mask.argmax())
    clip = []
    for t in range(t_star + 1):
        lines = [f"tolerance {L}px/{L}deg   step {t + 1}",
                 f"T err: {pos_i[t]:5.1f}px  {ang_i[t]:5.1f}deg"
                 + ("   SUCCESS" if mask[t] else "")]
        clip.append(banner(frames_i[t], lines, bool(mask[t])))
    clip += [clip[-1]] * HOLD
    return clip, t_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/pusht/runs/e0-valN100-20260701")
    ap.add_argument("--source-seed", type=int, default=42)
    ap.add_argument("--per-tier", type=int, default=8, help="candidates re-run per tier")
    ap.add_argument("--keep", type=int, default=2, help="videos kept per tier")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--min-push", type=float, default=30.0,
                    help="min INITIAL block->goal distance (px), so the clip shows a real push")
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="outputs/pusht/videos/precision")
    args = ap.parse_args()
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import stable_worldmodel as swm
    from sklearn import preprocessing

    # ---- pick candidates per tier from the saved validation rollout ----
    z = np.load(Path(args.run_dir) / f"seed{args.source_seed}" / "rollout.npz")
    pos0, ang0 = block_errs(z["states"], z["goals"])
    ok0 = {L: ok_any(pos0, ang0, L, args.budget) for L in (20, 10, 3)}
    pushed = pos0[:, 0] > args.min_push        # exclude episodes that START near the goal
    bins0 = {20: ok0[20] & ~ok0[10] & pushed, 10: ok0[10] & ~ok0[3] & pushed,
             3: ok0[3] & pushed}
    rows, row_tier = [], []
    for L in TIERS:
        cand = np.nonzero(bins0[L])[0][:args.per_tier]
        rows.extend(int(r) for r in cand)
        row_tier.extend([L] * len(cand))
        print(f"tier {L:>2}px: {bins0[L].sum()} candidates in source run, re-running {len(cand)}")
    episodes = [int(e) for e in z["episodes"][rows]]
    start_steps = [int(s) for s in z["start_steps"][rows]]

    # ---- model / data / world (mirrors run_baseline_validation.main) ----
    dataset = swm.data.HDF5Dataset("pusht_expert_train",
                                   keys_to_cache=["action", "proprio", "state"],
                                   cache_dir=swm.data.utils.get_cache_dir())
    process = {}
    for col in ("action", "proprio", "state"):
        sc = preprocessing.StandardScaler()
        cd = dataset.get_col_data(col); cd = cd[~np.isnan(cd).any(axis=1)]
        sc.fit(cd); process[col] = sc
        if col != "action":
            process[f"goal_{col}"] = sc
    model = swm.wm.utils.load_pretrained("pusht/lewm").to(args.device).eval()
    model.requires_grad_(False); model.interpolate_pos_encoding = True
    tf = {"pixels": img_transform(), "goal": img_transform()}
    world = swm.World(env_name="swm/PushT-v1", num_envs=len(rows),
                     max_episode_steps=2 * args.budget, image_shape=(224, 224))

    states, frames, goals = rollout_with_frames(
        world, model, dataset, episodes, start_steps, goal_offset=args.goal_offset,
        seed=args.source_seed, budget=args.budget, device=args.device, process=process, tf=tf)
    pos, ang = block_errs(states, goals)

    # ---- re-bin from the fresh rollout (planning is not bit-reproducible), keep in-bin ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta, tier_first_clip = [], {}
    for L in TIERS:
        tight = NEXT_TIGHTER[L]
        kept = 0
        for i in range(len(rows)):
            if row_tier[i] != L or kept >= args.keep:
                continue
            ok_L = bool(((pos[i, :] < L) & (ang[i, :] < L)).any())
            ok_tight = tight is not None and bool(
                ((pos[i, :] < tight) & (ang[i, :] < tight)).any())
            if not ok_L or ok_tight:
                print(f"  tier {L}: ep {episodes[i]} drifted out of bin on re-run "
                      f"(ok@{L}={ok_L}, ok@{tight}={ok_tight}), skipped")
                continue
            clip, t_star = make_clip(frames[i], pos[i], ang[i], L)
            name = out / f"success_{L}px_{L}deg_ep{episodes[i]}.mp4"
            write_h264(clip, name)
            best = int(np.argmin(pos[i]))
            rec = {"tier_px_deg": L, "episode": episodes[i], "start_step": start_steps[i],
                   "crossing_step": t_star + 1,
                   "err_at_crossing_px": round(float(pos[i, t_star]), 2),
                   "err_at_crossing_deg": round(float(ang[i, t_star]), 2),
                   "best_pos_err_px": round(float(pos[i, best]), 2),
                   "video": name.name}
            meta.append(rec)
            tier_first_clip.setdefault(L, clip)
            kept += 1
            print(f"  tier {L}: ep {episodes[i]} crossed at step {t_star + 1} "
                  f"({pos[i, t_star]:.1f}px, {ang[i, t_star]:.1f}deg) -> {name.name}")

    # ---- side-by-side compare: one clip per tier ----
    if len(tier_first_clip) == len(TIERS):
        T = max(len(c) for c in tier_first_clip.values())
        cols = []
        for L in TIERS:
            c = tier_first_clip[L]
            cols.append(c + [c[-1]] * (T - len(c)))
        grid = [np.hstack([cols[j][t] for j in range(len(cols))]) for t in range(T)]
        write_h264(grid, out / "precision_compare_20_10_3.mp4")
        print(f"  compare -> precision_compare_20_10_3.mp4")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE: {len(meta)} clips in {out}")


if __name__ == "__main__":
    main()
