"""Baseline validation + precision/budget characterization for LeWM Push-T.

For each seed we sample N=100 expert trajectories (goal = state `goal_offset` env steps ahead) and
run ONE open-loop MPC rollout (H=5, k=5, the paper regime) out to `max_budget` env steps, with the
env's success/termination DISABLED (monkeypatched `eval_state`) so every env keeps stepping the full
budget. We log the full 7-dim state at every step. From that single rollout, `analyze_*` computes,
post-hoc, the whole (success-tolerance x budget) grid, because success is monotone in both:
    success@(thr, B) := exists step t <= B with block_err(t) < thr.

Faithfulness: env-native@20px within budget-50 computed this way equals the real terminating
protocol (planning up to the success step is identical whether or not the env freezes afterward).

Config is held at the paper regime: horizon 5, receding_horizon 5 (open-loop), action_block 5,
CEM 300 samples x 30 iters x 30 elites, var 1. Seeds resample per seed (own trajectories + own CEM
RNG). Usage (one process per GPU):
    python -m experiments.pusht_horizon.run_baseline_validation \
        --seeds 42,43,44 --num-eval 100 --max-budget 200 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")

import numpy as np

OUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "pusht" / "runs"

# the eval callables (config/eval/pusht.yaml): set the initial state and the goal state
CALLABLES = [
    {"method": "_set_state", "args": {"state": {"value": "state"}}},
    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
]


def img_transform(size=224):
    import torch
    import stable_pretraining as spt
    from torchvision.transforms import v2 as T
    return T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=size),
    ])


def get_episodes_length(dataset, episodes, col_name):
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def sample_trajectories(dataset, col_name, seed, num_eval, goal_offset):
    """Replicates eval.py:120-145 — pick `num_eval` (episode, start_step) pairs whose goal
    (start+goal_offset) stays inside the trajectory, seeded by `seed`."""
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices, col_name)
    max_start_idx = {ep: episode_len[i] - goal_offset - 1 for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx[ep] for ep in dataset.get_col_data(col_name)])
    valid_indices = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]

    g = np.random.default_rng(seed)
    chosen = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    rows = np.sort(valid_indices[chosen])
    episodes = dataset.get_row_data(rows)[col_name]
    start_idx = dataset.get_row_data(rows)["step_idx"]
    if len(episodes) < num_eval:
        raise ValueError("Not enough episodes with sufficient length.")
    return episodes.tolist(), start_idx.tolist()


def make_policy(model, horizon, k, device, seed, process, tf):
    import stable_worldmodel as swm
    config = swm.PlanConfig(horizon=horizon, receding_horizon=k, action_block=5)
    solver = swm.solver.CEMSolver(model=model, num_samples=300, n_steps=30, topk=30,
                                  var_scale=1.0, batch_size=1, device=device, seed=seed)
    return swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=tf)


def run_seed(world, model, dataset, col_name, seed, *, num_eval, goal_offset, horizon, k,
             max_budget, device, process, tf):
    """One seed: sample trajectories, set up init/goal, disable termination, roll out to
    max_budget, and return the full per-step state log."""
    from stable_worldmodel.world.world import _extract_init_goal, _apply_callables

    episodes, start_steps = sample_trajectories(dataset, col_name, seed, num_eval, goal_offset)
    n = len(episodes)
    assert n == world.num_envs, f"num_envs {world.num_envs} != num_eval {n}"

    policy = make_policy(model, horizon, k, device, seed, process, tf)
    world.set_policy(policy)

    # --- init/goal setup (mirrors World._evaluate_from_dataset) ---
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

    # --- disable termination so every env steps the full budget (no early freeze) ---
    for i in range(n):
        world.envs.envs[i].unwrapped.eval_state = \
            lambda gs, cs: (False, float(np.linalg.norm(gs - cs)))

    # --- roll out, logging the 7-dim state each step ---
    states_log = []

    def on_step(w):
        w.infos.update(deepcopy(goal_snapshot))                 # keep the goal visible to the policy
        states_log.append(np.asarray(w.infos["state"])[:, 0, :].copy())  # (n, 7)

    t0 = time.time()
    world._run(max_steps=max_budget, mode="wait", on_step=on_step)
    elapsed = time.time() - t0

    states = np.stack(states_log, axis=1)                       # (n, budget, 7)
    goals = np.asarray(goal_state["goal_state"])                # (n, 7)
    return {
        "states": states, "goals": goals,
        "episodes": np.asarray(episodes), "start_steps": np.asarray(start_steps),
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42", help="comma-separated eval seeds")
    ap.add_argument("--num-eval", type=int, default=100)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-budget", type=int, default=200, help="env steps to roll out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--run-dir", default="", help="shared output dir (else auto-stamped)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    torch.set_num_threads(args.threads)
    import stable_worldmodel as swm
    from sklearn import preprocessing

    dataset = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"],
                                   cache_dir=swm.data.utils.get_cache_dir())
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

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

    world = swm.World(env_name="swm/PushT-v1", num_envs=args.num_eval,
                     max_episode_steps=2 * args.max_budget, image_shape=(224, 224))

    run_dir = Path(args.run_dir) if args.run_dir else \
        OUT_ROOT / f"e0-valN{args.num_eval}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        out = run_seed(world, model, dataset, col_name, seed,
                       num_eval=args.num_eval, goal_offset=args.goal_offset,
                       horizon=args.horizon, k=args.k, max_budget=args.max_budget,
                       device=args.device, process=process, tf=tf)
        sd = run_dir / f"seed{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sd / "rollout.npz", states=out["states"], goals=out["goals"],
                            episodes=out["episodes"], start_steps=out["start_steps"])
        meta = {"seed": seed, "num_eval": args.num_eval, "goal_offset": args.goal_offset,
                "horizon": args.horizon, "receding_horizon": args.k, "action_block": 5,
                "max_budget": args.max_budget, "cem": {"num_samples": 300, "n_steps": 30, "topk": 30},
                "elapsed_s": round(out["elapsed_s"], 1),
                "states_shape": list(out["states"].shape)}
        (sd / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[seed {seed}] states {out['states'].shape}  {out['elapsed_s']:.1f}s  -> {sd}")

    print(f"\nDONE {run_dir}")


if __name__ == "__main__":
    main()
