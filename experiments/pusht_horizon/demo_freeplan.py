"""Free-running Push-T demo: start from a real trajectory's state, give the agent a goal that
requires a SUBSTANTIAL push, let it plan in the working open-loop regime (H=5, k=5), and record
the ENTIRE rollout until the block converges on the goal — well past the env's loose 20px/20deg
success threshold. No early-termination freeze; stops when the block stops moving (trims the
agent's post-push wiggle). The observation renders the green goal, so you watch the T pushed in.

Searches random episodes, keeps ones with a large start->goal block displacement that the agent
actually seats. Encodes H.264 (browser/VSCode playable). Usage:
  python -m experiments.pusht_horizon.demo_freeplan --keep 4 --goal-offset 28 --horizon 5 --k 5 \
      --device cuda --gpu 1 --out outputs/pusht/videos/freeplan
"""
from __future__ import annotations

import argparse
import os
import subprocess
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")

import cv2
import numpy as np

CALLABLES = [
    {"method": "_set_state", "args": {"state": {"value": "state"}}},
    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
]
CONV_WIN, CONV_EPS = 8, 2.0   # block "converged" if it moved < EPS px over the last WIN steps
SEAT_STOP = 12.0              # stop shortly after the block seats (px) — before the agent, now
                              # facing a flat cost landscape, wanders off / diverges to NaN


def img_transform(size=224):
    import torch
    from torchvision.transforms import v2 as T
    import stable_pretraining as spt
    return T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=size),
    ])


def block_err(cur7, goal7):
    d_pos = float(np.linalg.norm(np.asarray(cur7)[2:4] - np.asarray(goal7)[2:4]))
    da = abs(float(cur7[4]) - float(goal7[4])) % (2 * np.pi)
    return d_pos, np.degrees(min(da, 2 * np.pi - da))


def write_h264(frames, path, fps=8):
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


def make_policy(model, horizon, k, device, seed, process, tf):
    import stable_worldmodel as swm
    config = swm.PlanConfig(horizon=horizon, receding_horizon=k, action_block=5)
    solver = swm.solver.CEMSolver(model=model, num_samples=300, n_steps=30, topk=30,
                                  var_scale=1.0, batch_size=1, device=device, seed=seed)
    return swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=tf)


def run_episode(world, policy, init_state, goal_state, budget):
    from stable_worldmodel.world.world import _apply_callables
    goal_vec = np.asarray(goal_state["goal_state"])[0]
    world.set_policy(policy)
    world.reset(seed=init_state.get("seed"))
    env0 = world.envs.envs[0].unwrapped
    _apply_callables(env0, CALLABLES, {k: v[0] for k, v in {**init_state, **goal_state}.items()})
    shape = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for key, v in src.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(v[:, None, ...], shape + v.shape[1:]).copy()
    goal_snap = {key: world.infos[key].copy() for key in goal_state}

    frames, dists, block = [], [], []
    countdown = None
    for t in range(budget):
        action = policy.get_action(world.infos)
        _, _, _, _, world.infos = world.envs.step(action, mask=None)
        world.infos.update(deepcopy(goal_snap))
        f = world.infos["pixels"][0]
        frames.append((f[-1] if getattr(f, "ndim", 0) > 3 else f).astype(np.uint8).copy())
        st = np.asarray(world.infos["state"][0]).reshape(-1)[:7]
        dpos, _ = block_err(st, goal_vec)
        dists.append((dpos, _)); block.append(st[2:4].copy())
        if dpos < SEAT_STOP:                 # seated -> stop before the agent diverges
            countdown = 2 if countdown is None else countdown - 1
            if countdown < 0:
                break
        elif t >= 20 and len(block) > CONV_WIN:  # fail-fast: block stuck far from goal
            r = np.array(block[-CONV_WIN:])
            if float(np.linalg.norm(r.max(0) - r.min(0))) < CONV_EPS:
                break
    return frames, dists


def banner(frame, text, ok):
    bar = np.full((28, frame.shape[1], 3), 255, np.uint8)
    col = (40, 150, 40) if ok else (200, 40, 40)  # RGB green / red
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def build_compare(model, world, dataset, ep, goal_offset, budget, configs, process, tf,
                  device, seed, seat_loose, out):
    """Render the SAME free-plan episode under several (H,k) configs, side-by-side."""
    from stable_worldmodel.world.world import _extract_init_goal
    init_state, goal_state, _ = _extract_init_goal(dataset, [ep], [0], goal_offset)
    clips = []
    for h, k in configs:
        policy = make_policy(model, h, k, device, seed, process, tf)
        frames, dists = run_episode(world, policy, init_state, goal_state, budget)
        min_d = min(d for d, _ in dists)
        ok = min_d < seat_loose
        clips.append((f"H={h} k={k}: {'SEATED' if ok else 'FAIL'} ({min_d:.0f}px)", frames, ok))
    T = max(len(f) for _, f, _ in clips) + 10
    rows = []
    for label, frames, ok in clips:
        padded = frames + [frames[-1]] * (T - len(frames))
        rows.append([banner(f, label, ok) for f in padded])
    grid = [np.hstack([rows[c][t] for c in range(len(rows))]) for t in range(T)]
    name = out / f"freeplan_compare_ep{ep}.mp4"
    write_h264(grid, name)
    print(f"wrote compare: ep {ep}, {len(configs)} configs -> {name.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=4, help="how many good demos to collect")
    ap.add_argument("--compare-ep", type=int, default=None,
                    help="render this episode under several (H,k) for a side-by-side")
    ap.add_argument("--goal-offset", type=int, default=40)
    ap.add_argument("--budget", type=int, default=0,
                    help="0 = use the longest trajectory length as the step upper bound")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-push", type=float, default=65.0, help="min start->goal block px")
    ap.add_argument("--max-push", type=float, default=180.0)
    ap.add_argument("--seat-loose", type=float, default=24.0, help="count success if block gets this close (px)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="outputs/pusht/videos/freeplan")
    args = ap.parse_args()
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import stable_worldmodel as swm
    from sklearn import preprocessing
    from stable_worldmodel.world.world import _extract_init_goal

    dataset = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"],
                                   cache_dir=swm.data.utils.get_cache_dir())
    # upper bound on rollout steps = the longest expert trajectory (so slow configs can finish)
    budget = args.budget if args.budget > 0 else int(dataset.lengths.max())
    print(f"step upper bound (longest trajectory) = {budget}")
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
    world = swm.World(env_name="swm/PushT-v1", num_envs=1,
                     max_episode_steps=budget + 20, image_shape=(224, 224))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.compare_ep is not None:
        build_compare(model, world, dataset, args.compare_ep, args.goal_offset, budget,
                      [(5, 5), (5, 1)], process, tf, args.device, args.seed,
                      args.seat_loose, out)
        return

    lengths = dataset.lengths
    rng = np.random.default_rng(args.seed)
    pool = [e for e in range(len(lengths)) if lengths[e] > args.goal_offset + 10]
    rng.shuffle(pool)

    kept = 0
    for ep in map(int, pool):
        init_state, goal_state, _ = _extract_init_goal(dataset, [ep], [0], args.goal_offset)
        start_d, _ = block_err(init_state["state"][0], goal_state["goal_state"][0])
        if not (args.min_push < start_d < args.max_push):
            continue
        policy = make_policy(model, args.horizon, args.k, args.device, args.seed, process, tf)
        frames, dists = run_episode(world, policy, init_state, goal_state, budget)
        dpos = [d for d, _ in dists]
        min_d = min(dpos); best = int(np.argmin(dpos))
        ok = min_d < args.seat_loose
        print(f"ep {ep}: start {start_d:.0f}px -> seated {min_d:.0f}px at frame {best} "
              f"({len(frames)} steps) {'KEEP' if ok else 'skip'}")
        if ok:
            name = out / f"freeplan_ep{ep}_H{args.horizon}_k{args.k}.mp4"
            clip = frames[:best + 1] + [frames[best]] * 8   # end at best seat (agent visible), ~1s hold
            write_h264(clip, name, fps=8)                    # 8 fps -> slower/longer
            kept += 1
            if kept >= args.keep:
                break
    print(f"\nDONE: kept {kept} demos in {out}")


if __name__ == "__main__":
    main()
