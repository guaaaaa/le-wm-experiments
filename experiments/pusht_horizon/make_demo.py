"""Build clean end-to-end demo videos from the per-condition eval videos.

The eval terminates + freezes an env the moment it succeeds, so raw `env_i.mp4` clips have a long
frozen tail ("agent stopped moving"). This trims the frozen tail (keeps the active push + a short
hold), slows playback, labels each clip SUCCESS/FAIL, and emits:
  - one trimmed end-to-end clip per condition, and
  - a side-by-side comparison (same episode, agent panel, across the hyperparameter).

Uses only the AGENT panel (left third of the 3-panel eval frame). Usage:
  python -m experiments.pusht_horizon.make_demo --env 4 --out outputs/pusht/videos \
      --k-sweep <t2 dir> --h-small <t1o dir> --h-large <t1ol dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PANEL_W = 245        # agent panel width (left third of 736)
FPS = 8              # slow, watchable
HOLD_S = 0.8         # hold the final frame after motion stops


def load_agent_panel(mp4: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :PANEL_W].copy())
    cap.release()
    return frames


def trim_frozen(frames: list[np.ndarray], eps: float = 0.7) -> list[np.ndarray]:
    """Drop the trailing frozen tail: keep up to the last frame that differs from its
    predecessor (motion), plus a short hold."""
    if len(frames) < 2:
        return frames
    last_motion = 0
    for i in range(1, len(frames)):
        if float(np.abs(frames[i].astype(np.int16) - frames[i - 1]).mean()) > eps:
            last_motion = i
    hold = int(round(HOLD_S * FPS))
    end = min(len(frames), last_motion + 1 + hold)
    return frames[:end]


def banner(frame: np.ndarray, text: str, ok: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = np.full((30, w, 3), 255, np.uint8)
    color = (40, 150, 40) if ok else (40, 40, 200)  # BGR: green / red
    cv2.putText(bar, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def write(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    """Encode BGR frames to H.264/yuv420p via ffmpeg (universally playable; cv2's mp4v is
    not decodable by browsers/VSCode). Pads to even dimensions as libx264 requires."""
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed for {path}")


def clip_success(cond_dir: Path, env: int) -> bool:
    r = json.loads((cond_dir / "results.json").read_text())
    return bool(r["episode_successes"][env])


def build(clips: list[tuple[str, Path]], env: int, out: Path, name: str) -> None:
    """clips: list of (label, condition_dir). Emits individual trimmed clips + a side-by-side."""
    panels, labels, oks = [], [], []
    for label, cdir in clips:
        ok = clip_success(cdir, env)
        frames = trim_frozen(load_agent_panel(cdir / f"env_{env}.mp4"))
        write([banner(f, f"{label}  {'SUCCESS' if ok else 'FAIL'}", ok) for f in frames],
              out / f"{name}_{label.replace('=', '').replace(' ', '')}_env{env}.mp4")
        panels.append(frames); labels.append(label); oks.append(ok)
    # side-by-side: pad each to the max active length (hold final frame), banner, hstack
    T = max(len(p) for p in panels)
    rows = []
    for frames, label, ok in zip(panels, labels, oks):
        padded = frames + [frames[-1]] * (T - len(frames))
        rows.append([banner(f, f"{label} {'OK' if ok else 'FAIL'}", ok) for f in padded])
    grid = [np.hstack([rows[c][t] for c in range(len(rows))]) for t in range(T)]
    write(grid, out / f"{name}_compare_env{env}.mp4")
    print(f"wrote {name}: {len(clips)} clips + compare (env {env}, {T} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-sweep", help="T2 sweep dir (videos), builds replan-interval demo")
    ap.add_argument("--h-small", help="T1o sweep dir (H=1..10)")
    ap.add_argument("--h-large", help="T1oL sweep dir (H=20,40)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.k_sweep:
        s = Path(args.k_sweep)
        build([(f"k={k}", s / f"H5-k{k}") for k in (1, 2, 3, 5)], args.env, out, "replan_k")
    if args.h_small and args.h_large:
        so, la = Path(args.h_small), Path(args.h_large)
        clips = [(f"H={h}", so / f"H{h}-k{h}") for h in (1, 3, 5, 10)] + \
                [(f"H={h}", la / f"H{h}-k{h}") for h in (20, 40)]
        build(clips, args.env, out, "horizon_H")


if __name__ == "__main__":
    main()
