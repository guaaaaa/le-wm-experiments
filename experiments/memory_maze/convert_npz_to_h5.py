"""Convert memory-maze npz episodes into ONE swm-schema HDF5 file (see CONTRACTS.md).

Schema: `ep_len` (int32), `ep_offset` (int64), plus flat columns concatenated over episodes:
`pixels` (N,H,W,3) u8 [renamed from npz `image`], `action` (N,A) f32 **shifted by one step**
(npz action[t] leads INTO obs t; LeWM pairs the action DEPARTING obs t, so out[t] = npz[t+1],
zeros at each episode's last row), and unshifted probe labels (agent_pos, agent_dir,
targets_pos, targets_vec, target_pos, target_color, maze_layout, reward).

All datasets Zstd-compressed (hdf5plugin); pixels chunked (chunk_frames, H, W, 3) to match the
reader's dense-span access pattern. Compatible with swm.data.HDF5Dataset(path=...).

Usage:
    python -m experiments.memory_maze.convert_npz_to_h5 \
        --src /mnt/nas3/lewm/memory-maze/9x9/eval --out /mnt/nas3/lewm/memory-maze/memory_maze9_eval.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

# (npz key, out name, out dtype, shifted?)
_COLUMNS = [
    ("image", "pixels", np.uint8, False),
    ("action", "action", np.float32, True),
    ("agent_pos", "agent_pos", np.float32, False),
    ("agent_dir", "agent_dir", np.float32, False),
    ("targets_pos", "targets_pos", np.float32, False),
    ("targets_vec", "targets_vec", np.float32, False),
    ("target_pos", "target_pos", np.float32, False),
    ("target_color", "target_color", np.float32, False),
    ("maze_layout", "maze_layout", np.uint8, False),
    ("reward", "reward", np.float32, False),
]


def _load_episode(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        missing = [k for k, *_ in _COLUMNS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing required key(s) {missing}")
        ep = {k: z[k] for k, *_ in _COLUMNS}
    lengths = {k: len(v) for k, v in ep.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path}: inconsistent row counts across keys: {lengths}")
    return ep


def _shift_action(a: np.ndarray) -> np.ndarray:
    """out[t] = a[t+1] (the action DEPARTING obs t); zeros at the final row."""
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    return out


def convert(src_files, out_path, *, chunk_frames: int = 16, zstd_level: int = 3) -> dict:
    src_files = [Path(f) for f in src_files]
    out_path = Path(out_path)
    if not src_files:
        raise ValueError("no source files given")

    comp = hdf5plugin.Zstd(clevel=zstd_level)
    ep_len: list[int] = []

    with h5py.File(out_path, "w") as f:
        dsets = {}
        for ep_i, path in enumerate(src_files):
            ep = _load_episode(path)
            L = len(ep["image"])
            ep_len.append(L)
            for key, out_name, dtype, shifted in _COLUMNS:
                data = ep[key].astype(dtype, copy=False)
                if shifted:
                    data = _shift_action(data)
                if out_name not in dsets:
                    sample = data[0]
                    chunk0 = chunk_frames if out_name == "pixels" else max(chunk_frames, 256)
                    dsets[out_name] = f.create_dataset(
                        out_name,
                        shape=(0, *sample.shape),
                        maxshape=(None, *sample.shape),
                        dtype=dtype,
                        chunks=(chunk0, *sample.shape),
                        **comp,
                    )
                d = dsets[out_name]
                n0 = d.shape[0]
                d.resize(n0 + L, axis=0)
                d[n0:n0 + L] = data

        lens = np.asarray(ep_len, dtype=np.int32)
        offsets = np.zeros(len(lens), dtype=np.int64)
        offsets[1:] = np.cumsum(lens[:-1])
        f.create_dataset("ep_len", data=lens, dtype=np.int32)
        f.create_dataset("ep_offset", data=offsets, dtype=np.int64)

    return {"episodes": len(ep_len), "rows": int(lens.sum()), "out": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of npz episodes")
    ap.add_argument("--out", required=True, help="output .h5 path")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N files (sorted)")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--chunk-frames", type=int, default=16)
    ap.add_argument("--zstd-level", type=int, default=3)
    args = ap.parse_args()

    files = sorted(Path(args.src).glob(args.pattern))
    if args.limit:
        files = files[: args.limit]
    print(f"converting {len(files)} episodes -> {args.out}")
    res = convert(files, args.out, chunk_frames=args.chunk_frames, zstd_level=args.zstd_level)
    print(res)


if __name__ == "__main__":
    main()
