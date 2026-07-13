"""Blind contract tests for experiments.memory_maze.convert_npz_to_h5.convert.

Written from CONTRACTS.md (section convert_npz_to_h5.convert) ONLY, before any
implementation exists. Tiny synthetic npz episodes, fully deterministic.

Run from the repo root:
    .venv/bin/python -m pytest experiments/memory_maze/tests/test_convert.py -x -q
"""

import os

# stable_worldmodel resolves its home dir at import time.
os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")

import h5py
import hdf5plugin  # noqa: F401  (registers the Zstd filter for h5py reads)
import numpy as np
import pytest

from experiments.memory_maze.convert_npz_to_h5 import convert


# --- Synthetic-episode geometry (contract: L varies; H=W=8, A=6, K=3, M=9) ----
LENGTHS = [12, 15, 9]
E = len(LENGTHS)
N = sum(LENGTHS)  # 36
OFFSETS = [0, LENGTHS[0], LENGTHS[0] + LENGTHS[1]]  # [0, 12, 27]
H = 8
W = 8
A = 6
K = 3
M = 9

NPZ_KEYS = [
    "image",
    "action",
    "agent_pos",
    "agent_dir",
    "targets_pos",
    "targets_vec",
    "target_pos",
    "target_color",
    "maze_layout",
    "reward",
]

# Per-row keys copied through UNSHIFTED (everything except image->pixels and
# the shifted action), mapped h5-name -> (npz-name, h5 dtype).
UNSHIFTED = {
    "agent_pos": ("agent_pos", np.float32),
    "agent_dir": ("agent_dir", np.float32),
    "targets_pos": ("targets_pos", np.float32),
    "targets_vec": ("targets_vec", np.float32),
    "target_pos": ("target_pos", np.float32),
    "target_color": ("target_color", np.float32),
    "maze_layout": ("maze_layout", np.uint8),
    "reward": ("reward", np.float32),
}


def _make_episode(rng, L):
    """One synthetic episode matching the contract's npz key list exactly."""
    action = np.zeros((L, A), dtype=np.float64)
    hot = rng.integers(0, A, size=L - 1)
    action[np.arange(1, L), hot] = 1.0  # one-hot rows; row 0 all-zero reset
    return {
        "image": rng.integers(0, 256, size=(L, H, W, 3), dtype=np.uint8),
        "action": action,
        "agent_pos": rng.standard_normal((L, 2)),
        "agent_dir": rng.standard_normal((L, 2)),
        "targets_pos": rng.standard_normal((L, K, 2)),
        "targets_vec": rng.standard_normal((L, K, 2)),
        "target_pos": rng.standard_normal((L, 2)),
        "target_color": rng.random((L, 3)),
        "maze_layout": rng.integers(0, 2, size=(L, M, M)).astype(np.uint8),
        "reward": rng.standard_normal(L),
    }


@pytest.fixture
def episodes(tmp_path):
    """Write 3 tiny npz episodes (L = 12, 15, 9); return (paths, raw dicts).

    File names are deliberately NOT in sorted order relative to the list order
    (c, a, b) so that any implementation that re-sorts src_files fails: the
    contract fixes episode order to the order of src_files as given.
    """
    rng = np.random.default_rng(20260712)
    names = ["ep_c.npz", "ep_a.npz", "ep_b.npz"]
    paths, raws = [], []
    for name, L in zip(names, LENGTHS):
        ep = _make_episode(rng, L)
        p = tmp_path / name
        np.savez(p, **ep)
        assert sorted(np.load(p).files) == sorted(NPZ_KEYS)
        paths.append(p)
        raws.append(ep)
    return paths, raws


@pytest.fixture
def converted(tmp_path, episodes):
    """convert() the 3 episodes; return (out_path, raw dicts, return value)."""
    paths, raws = episodes
    out = tmp_path / "maze.h5"
    result = convert(paths, out)
    return out, raws, result


# ===========================================================================
# Schema: exact dataset names, shapes, dtypes, ep_len/ep_offset bookkeeping
# ===========================================================================
def test_schema(converted):
    out, _, _ = converted
    expected = {
        "ep_len": ((E,), np.int32),
        "ep_offset": ((E,), np.int64),
        "pixels": ((N, H, W, 3), np.uint8),
        "action": ((N, A), np.float32),
        "agent_pos": ((N, 2), np.float32),
        "agent_dir": ((N, 2), np.float32),
        "targets_pos": ((N, K, 2), np.float32),
        "targets_vec": ((N, K, 2), np.float32),
        "target_pos": ((N, 2), np.float32),
        "target_color": ((N, 3), np.float32),
        "maze_layout": ((N, M, M), np.uint8),
        "reward": ((N,), np.float32),
    }
    with h5py.File(out, "r") as f:
        assert set(f.keys()) == set(expected)
        for name, (shape, dtype) in expected.items():
            ds = f[name]
            assert ds.shape == shape, name
            assert ds.dtype == np.dtype(dtype), name
        assert f["ep_len"][:].tolist() == LENGTHS  # given src_files order
        assert f["ep_offset"][:].tolist() == OFFSETS
        assert int(f["ep_len"][:].sum()) == N == 36


# ===========================================================================
# Action shift (the critical semantic): out row o+t = npz action[t+1] for
# t < L-1, all-zeros at t = L-1; everything else is NOT shifted.
# ===========================================================================
def test_action_shift(converted):
    out, raws, _ = converted
    with h5py.File(out, "r") as f:
        h5_action = f["action"][:]
        for raw, o, L in zip(raws, OFFSETS, LENGTHS):
            # h5 action[o+t] == npz action[t+1] for all t in [0, L-2]
            assert np.array_equal(
                h5_action[o : o + L - 1],
                raw["action"][1:L].astype(np.float32),
            )
            # last row of the episode is the all-zero placeholder
            assert np.array_equal(
                h5_action[o + L - 1], np.zeros(A, dtype=np.float32)
            )
        # per-element spot check of the shift, spelled out row by row
        for raw, o, L in zip(raws, OFFSETS, LENGTHS):
            for t in range(L - 1):
                assert np.array_equal(
                    h5_action[o + t], raw["action"][t + 1].astype(np.float32)
                )


def test_non_action_columns_not_shifted(converted):
    out, raws, _ = converted
    with h5py.File(out, "r") as f:
        pixels = f["pixels"][:]
        for raw, o, L in zip(raws, OFFSETS, LENGTHS):
            assert np.array_equal(pixels[o : o + L], raw["image"])
        for h5_name, (npz_name, dtype) in UNSHIFTED.items():
            data = f[h5_name][:]
            for raw, o, L in zip(raws, OFFSETS, LENGTHS):
                assert np.array_equal(
                    data[o : o + L], raw[npz_name].astype(dtype)
                ), h5_name


# ===========================================================================
# Return value
# ===========================================================================
def test_return_value(tmp_path, episodes):
    paths, _ = episodes
    out = tmp_path / "ret.h5"
    # contract allows list[str] and str out_path too
    result = convert([str(p) for p in paths], str(out))
    assert isinstance(result, dict)
    assert result["episodes"] == E
    assert result["rows"] == N
    assert result["out"] == str(out)


def test_return_value_path_inputs(converted):
    out, _, result = converted
    assert result["episodes"] == 3
    assert result["rows"] == 36
    assert result["out"] == str(out)


# ===========================================================================
# Validation errors
# ===========================================================================
def test_missing_key_raises(tmp_path):
    rng = np.random.default_rng(7)
    ep = _make_episode(rng, 10)
    del ep["action"]
    p = tmp_path / "missing.npz"
    np.savez(p, **ep)
    with pytest.raises(ValueError):
        convert([p], tmp_path / "missing.h5")


def test_inconsistent_rows_raises(tmp_path):
    rng = np.random.default_rng(8)
    ep = _make_episode(rng, 10)
    ep["action"] = ep["action"][:-1]  # image has 10 rows, action has 9
    p = tmp_path / "ragged.npz"
    np.savez(p, **ep)
    with pytest.raises(ValueError):
        convert([p], tmp_path / "ragged.h5")


# ===========================================================================
# Compression / chunking
# ===========================================================================
def _has_filter(ds):
    """True if any HDF5 filter is active on the dataset.

    Zstd via hdf5plugin is a *custom* filter: ``ds.compression`` may be None
    even though the filter pipeline is non-empty, so accept either signal.
    """
    if ds.compression is not None:
        return True
    return ds.id.get_create_plist().get_nfilters() > 0


def test_compression(converted):
    out, _, _ = converted
    with h5py.File(out, "r") as f:
        pixels = f["pixels"]
        assert _has_filter(pixels)
        assert pixels.chunks is not None
        assert pixels.chunks[0] == 16  # default chunk_frames
        assert pixels.chunks == (16, H, W, 3)  # contract: (chunk_frames,H,W,3)
        # all non-pixels per-row datasets are Zstd-compressed too (any chunking)
        for name in ["action", *UNSHIFTED]:
            assert _has_filter(f[name]), name


def test_compression_chunk_frames_override(tmp_path, episodes):
    paths, _ = episodes
    out = tmp_path / "chunked.h5"
    convert(paths, out, chunk_frames=8)
    with h5py.File(out, "r") as f:
        pixels = f["pixels"]
        assert _has_filter(pixels)
        assert pixels.chunks[0] == 8
        assert pixels.chunks == (8, H, W, 3)


# ===========================================================================
# Reader-compatibility guarantee: swm.data.HDF5Dataset round-trip
# ===========================================================================
def test_swm_reader_roundtrip(converted):
    swm = pytest.importorskip("stable_worldmodel")
    out, _, _ = converted
    dataset = swm.data.HDF5Dataset(
        path=out,
        num_steps=3,
        frameskip=2,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    assert list(dataset.lengths) == LENGTHS  # == [12, 15, 9]
    assert dataset.get_dim("action") == A
    item = dataset[0]
    # (num_steps, C, H, W) and (num_steps, frameskip * A)
    assert tuple(item["pixels"].shape) == (3, 3, H, W)
    assert tuple(item["action"].shape) == (3, 2 * A)
