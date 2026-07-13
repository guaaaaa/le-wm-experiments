"""Tests for experiments.pusht_horizon.rollout.open_loop_rollout.

Written from CONTRACTS.md only. Uses the FakeModel/frame helpers in conftest.
"""

import numpy as np
import pytest

from experiments.pusht_horizon.rollout import open_loop_rollout

from .conftest import LinearPerfectModel, make_uniform_frames


# --- input builders ---------------------------------------------------------
def _consecutive_frames(n_blocks, base=0):
    """Frames whose uint8 value is a consecutive run starting at ``base``.

    With LinearPerfectModel, encode(frame with value v) == table[v], so a
    consecutive run makes the true latent sequence line up with predict()'s
    id+1 rule -> the perfect-predictor zero-drift invariant is exact.
    """
    return make_uniform_frames([base + i for i in range(n_blocks)])


def _actions(n_env):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_env, 2)).astype(np.float32)


def _valid_case(context=3, horizon=4, frameskip=5, base=1):
    n_blocks = context + horizon
    n_env = n_blocks * frameskip
    frames = _consecutive_frames(n_blocks, base=base)
    actions = _actions(n_env)
    return frames, actions, context, horizon, frameskip


# --- output shape / dtype / finiteness --------------------------------------
def test_output_keys_shapes_dtypes(perfect_model, D):
    frames, actions, C, H, fs = _valid_case(context=3, horizon=4)
    out = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    assert set(out.keys()) >= {"pred_latents", "true_latents"}
    for key in ("pred_latents", "true_latents"):
        arr = out[key]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (H, D)
        assert np.isfinite(arr).all()


def test_horizon_one_shape(perfect_model, D):
    frames, actions, C, H, fs = _valid_case(context=3, horizon=1)
    out = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    assert out["pred_latents"].shape == (1, D)
    assert out["true_latents"].shape == (1, D)


def test_default_context_and_frameskip(perfect_model, D):
    # defaults: context=3, frameskip=5
    H = 3
    n_blocks = 3 + H
    frames = _consecutive_frames(n_blocks, base=2)
    actions = _actions(n_blocks * 5)
    out = open_loop_rollout(perfect_model, frames, actions, horizon=H)
    assert out["pred_latents"].shape == (H, D)
    assert out["true_latents"].shape == (H, D)


# --- the central invariant: perfect predictor => zero drift ------------------
def test_perfect_predictor_zero_drift(perfect_model):
    frames, actions, C, H, fs = _valid_case(context=3, horizon=5, base=1)
    out = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    assert np.allclose(out["pred_latents"], out["true_latents"], atol=1e-5)
    # drift is genuinely nonzero-capable: pred equals the *true future* latents,
    # which are the table rows for frames C..C+H-1.
    for t in range(H):
        expected = perfect_model.latent_for_frame_value(1 + C + t)
        assert np.allclose(out["true_latents"][t], expected, atol=1e-5)


def test_true_latents_are_encoder_of_future_frames(perfect_model):
    frames, actions, C, H, fs = _valid_case(context=3, horizon=4, base=0)
    out = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    for t in range(H):
        expected = perfect_model.latent_for_frame_value(C + t)
        assert np.allclose(out["true_latents"][t], expected, atol=1e-5)


def test_deterministic_repeatable(perfect_model):
    frames, actions, C, H, fs = _valid_case(context=2, horizon=3)
    o1 = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    o2 = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    assert np.array_equal(o1["pred_latents"], o2["pred_latents"])
    assert np.array_equal(o1["true_latents"], o2["true_latents"])


def test_context_one_minimal(perfect_model, D):
    C, H, fs = 1, 2, 5
    frames = _consecutive_frames(C + H, base=3)
    actions = _actions((C + H) * fs)
    out = open_loop_rollout(
        perfect_model, frames, actions, context=C, horizon=H, frameskip=fs
    )
    assert out["pred_latents"].shape == (H, D)
    assert np.allclose(out["pred_latents"], out["true_latents"], atol=1e-5)


# --- precondition ValueErrors ------------------------------------------------
def test_valueerror_frames_wrong_ndim(perfect_model):
    frames, actions, C, H, fs = _valid_case()
    bad = frames.reshape(frames.shape[0], 224, 224 * 3)  # (n,224,672)
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, bad, actions, context=C, horizon=H, frameskip=fs)


def test_valueerror_frames_wrong_trailing_shape(perfect_model):
    frames, actions, C, H, fs = _valid_case()
    bad = np.zeros((frames.shape[0], 224, 224, 4), dtype=np.uint8)  # 4 channels
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, bad, actions, context=C, horizon=H, frameskip=fs)


def test_valueerror_frames_wrong_dtype(perfect_model):
    frames, actions, C, H, fs = _valid_case()
    bad = frames.astype(np.float32)
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, bad, actions, context=C, horizon=H, frameskip=fs)


def test_valueerror_actions_wrong_ndim(perfect_model):
    frames, actions, C, H, fs = _valid_case()
    bad = actions.reshape(-1)  # 1-D
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, bad, context=C, horizon=H, frameskip=fs)


def test_valueerror_actions_wrong_trailing_shape(perfect_model):
    frames, actions, C, H, fs = _valid_case()
    bad = np.zeros((actions.shape[0], 3), dtype=np.float32)  # last dim != 2
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, bad, context=C, horizon=H, frameskip=fs)


def test_valueerror_actions_wrong_dtype(perfect_model):
    frames, actions, C, H, fs = _valid_case()
    bad = actions.astype(np.float64)
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, bad, context=C, horizon=H, frameskip=fs)


def test_valueerror_too_few_blocks(perfect_model):
    C, H, fs = 3, 4, 5
    frames = _consecutive_frames(C + H - 1)  # one block short
    actions = _actions((C + H) * fs)
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, actions, context=C, horizon=H, frameskip=fs)


def test_valueerror_too_few_env_steps(perfect_model):
    C, H, fs = 3, 4, 5
    frames = _consecutive_frames(C + H)
    actions = _actions((C + H) * fs - 1)  # one env-step short
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, actions, context=C, horizon=H, frameskip=fs)


def test_valueerror_context_less_than_one(perfect_model):
    frames, actions, _, H, fs = _valid_case()
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, actions, context=0, horizon=H, frameskip=fs)


def test_valueerror_horizon_less_than_one(perfect_model):
    frames, actions, C, _, fs = _valid_case()
    with pytest.raises(ValueError):
        open_loop_rollout(perfect_model, frames, actions, context=C, horizon=0, frameskip=fs)
