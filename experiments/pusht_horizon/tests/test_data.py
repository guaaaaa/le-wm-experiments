"""Tests for experiments.pusht_horizon.data.

Written from CONTRACTS.md only. Uses FakeDataset in conftest; no real data.
"""

import numpy as np
import pytest
import torch

from experiments.pusht_horizon.data import (
    image_transform,
    load_episode_frames_actions,
    select_episodes,
)

from .conftest import (
    FakeDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    imagenet_normalize_uniform,
)


# ===========================================================================
# image_transform() -> callable
# ===========================================================================
def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def test_image_transform_returns_callable():
    tf = image_transform()
    assert callable(tf)


def test_image_transform_hwc_uniform_exact_values():
    tf = image_transform()
    # uniform image -> resize-to-224 is identity, so per-channel value is exact.
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[..., 0] = 100
    img[..., 1] = 150
    img[..., 2] = 200
    out = _to_np(tf(img))
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32 or out.dtype == np.float64
    exp = imagenet_normalize_uniform(np.array([100.0, 150.0, 200.0]))
    for c in range(3):
        assert np.allclose(out[c], exp[c], atol=1e-4)


def test_image_transform_chw_input_layout():
    tf = image_transform()
    # (...,3,224,224) input layout must also be accepted and yield same result.
    img_hwc = np.zeros((224, 224, 3), dtype=np.uint8)
    img_hwc[..., 0] = 10
    img_hwc[..., 1] = 20
    img_hwc[..., 2] = 30
    img_chw = np.transpose(img_hwc, (2, 0, 1)).copy()  # (3,224,224)
    out = _to_np(tf(img_chw))
    assert out.shape == (3, 224, 224)
    exp = imagenet_normalize_uniform(np.array([10.0, 20.0, 30.0]))
    for c in range(3):
        assert np.allclose(out[c], exp[c], atol=1e-4)


def test_image_transform_output_is_float_tensor():
    tf = image_transform()
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    out = tf(img)
    assert torch.is_tensor(out)
    assert out.dtype == torch.float32
    assert out.shape == (3, 224, 224)
    assert torch.isfinite(out).all()


def test_image_transform_leading_batch_dims_hwc():
    tf = image_transform()
    # (N, 224, 224, 3) -> (N, 3, 224, 224)
    vals = [50, 60, 70]
    batch = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    for i, v in enumerate(vals):
        batch[i] = np.uint8(v)
    out = _to_np(tf(batch))
    assert out.shape == (3, 3, 224, 224)
    for i, v in enumerate(vals):
        exp = imagenet_normalize_uniform(float(v))  # scalar broadcast per channel
        for c in range(3):
            assert np.allclose(out[i, c], exp[c], atol=1e-4)


def test_image_transform_leading_batch_dims_chw():
    tf = image_transform()
    # (N, 3, 224, 224) -> (N, 3, 224, 224)
    batch_hwc = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    batch_hwc[0] = 80
    batch_hwc[1] = 200
    batch_chw = np.transpose(batch_hwc, (0, 3, 1, 2)).copy()
    out = _to_np(tf(batch_chw))
    assert out.shape == (2, 3, 224, 224)
    for i, v in ((0, 80), (1, 200)):
        exp = imagenet_normalize_uniform(float(v))
        for c in range(3):
            assert np.allclose(out[i, c], exp[c], atol=1e-4)


def test_image_transform_deterministic():
    tf = image_transform()
    img = np.full((224, 224, 3), 77, dtype=np.uint8)
    a = _to_np(tf(img))
    b = _to_np(tf(img))
    assert np.array_equal(a, b)


# ===========================================================================
# select_episodes(dataset, n, seed, *, min_length=None) -> np.ndarray[int]
# ===========================================================================
def _ds(lengths):
    return FakeDataset(lengths=lengths, frameskip=5, seed=0)


def test_select_episodes_basic_shape_sorted_unique():
    ds = _ds([10, 20, 5, 30, 15, 8, 12])
    ids = select_episodes(ds, n=4, seed=42)
    assert isinstance(ids, np.ndarray)
    assert np.issubdtype(ids.dtype, np.integer)
    assert ids.shape == (4,)
    assert len(set(ids.tolist())) == 4
    assert list(ids) == sorted(ids)  # ascending
    assert set(ids.tolist()).issubset(set(range(7)))


def test_select_episodes_deterministic_same_seed():
    ds = _ds([10, 20, 5, 30, 15, 8, 12, 25])
    a = select_episodes(ds, n=5, seed=123)
    b = select_episodes(ds, n=5, seed=123)
    assert np.array_equal(a, b)


def test_select_episodes_different_seed_may_differ_but_valid():
    ds = _ds(list(range(10, 110, 10)))  # 10 episodes
    a = select_episodes(ds, n=5, seed=1)
    b = select_episodes(ds, n=5, seed=2)
    # both valid selections
    for ids in (a, b):
        assert ids.shape == (5,)
        assert list(ids) == sorted(ids)
        assert len(set(ids.tolist())) == 5


def test_select_episodes_min_length_filter():
    lengths = [3, 50, 4, 60, 2, 55, 1]
    ds = _ds(lengths)
    ids = select_episodes(ds, n=3, seed=7, min_length=50)
    eligible = {i for i, L in enumerate(lengths) if L >= 50}  # {1,3,5}
    assert set(ids.tolist()).issubset(eligible)
    assert ids.shape == (3,)
    assert list(ids) == sorted(ids)


def test_select_episodes_min_length_all_eligible_selected_when_exact():
    lengths = [3, 50, 4, 60, 2, 55]
    ds = _ds(lengths)
    ids = select_episodes(ds, n=3, seed=99, min_length=50)
    assert set(ids.tolist()) == {1, 3, 5}


def test_select_episodes_too_few_raises():
    ds = _ds([10, 20, 30])
    with pytest.raises(ValueError):
        select_episodes(ds, n=5, seed=0)


def test_select_episodes_too_few_after_min_length_raises():
    lengths = [3, 50, 4, 60, 2]
    ds = _ds(lengths)  # only 2 eligible at min_length=50
    with pytest.raises(ValueError):
        select_episodes(ds, n=3, seed=0, min_length=50)


# ===========================================================================
# load_episode_frames_actions(dataset, ep_id, start, length, *, frameskip=5)
# ===========================================================================
def test_load_full_episode_shapes_dtypes():
    fs = 5
    ds = FakeDataset(lengths=[10, 20, 5], frameskip=fs, seed=1)
    out = load_episode_frames_actions(ds, ep_id=1)  # length=None -> to end
    L = 20
    assert set(out.keys()) >= {"frames_u8", "actions_2d", "state"}
    assert out["frames_u8"].shape == (L, 224, 224, 3)
    assert out["frames_u8"].dtype == np.uint8
    assert out["actions_2d"].shape == (L * fs, 2)
    assert out["actions_2d"].dtype == np.float32
    assert out["state"].shape == (L, 7)


def test_load_partial_slice_shapes():
    fs = 5
    ds = FakeDataset(lengths=[30], frameskip=fs, seed=2)
    start, length = 5, 8
    out = load_episode_frames_actions(ds, ep_id=0, start=start, length=length, frameskip=fs)
    assert out["frames_u8"].shape == (length, 224, 224, 3)
    assert out["actions_2d"].shape == (length * fs, 2)
    assert out["state"].shape == (length, 7)


def test_load_slice_content_alignment():
    # FakeDataset frame block b of episode e has uint8 value (e*7 + b) % 256.
    fs = 5
    e = 2
    ds = FakeDataset(lengths=[40], frameskip=fs, seed=0)
    ds.lengths = np.array([40], dtype=np.int64)
    full = load_episode_frames_actions(ds, ep_id=0, frameskip=fs)
    start, length = 10, 6
    sliced = load_episode_frames_actions(ds, ep_id=0, start=start, length=length, frameskip=fs)
    assert np.array_equal(sliced["frames_u8"], full["frames_u8"][start:start + length])
    assert np.array_equal(
        sliced["actions_2d"], full["actions_2d"][start * fs:(start + length) * fs]
    )
    assert np.array_equal(sliced["state"], full["state"][start:start + length])


def test_load_length_none_to_end():
    fs = 5
    ds = FakeDataset(lengths=[25], frameskip=fs, seed=4)
    start = 7
    out = load_episode_frames_actions(ds, ep_id=0, start=start, length=None, frameskip=fs)
    L = 25 - start
    assert out["frames_u8"].shape[0] == L
    assert out["actions_2d"].shape[0] == L * fs


def test_load_ep_id_out_of_range_raises():
    ds = FakeDataset(lengths=[10, 20], frameskip=5, seed=0)
    with pytest.raises(ValueError):
        load_episode_frames_actions(ds, ep_id=5)
    with pytest.raises(ValueError):
        load_episode_frames_actions(ds, ep_id=-1)


def test_load_negative_start_raises():
    ds = FakeDataset(lengths=[10], frameskip=5, seed=0)
    with pytest.raises(ValueError):
        load_episode_frames_actions(ds, ep_id=0, start=-1)


def test_load_slice_out_of_bounds_raises():
    ds = FakeDataset(lengths=[10], frameskip=5, seed=0)
    # start + length > length of episode (10)
    with pytest.raises(ValueError):
        load_episode_frames_actions(ds, ep_id=0, start=5, length=6)
    with pytest.raises(ValueError):
        load_episode_frames_actions(ds, ep_id=0, start=0, length=11)
