"""Tests for experiments.pusht_horizon.metrics.

Written from CONTRACTS.md only.
"""

import numpy as np
import pytest

from experiments.pusht_horizon.metrics import (
    latent_drift,
    one_step_error,
    pearson,
)

from .conftest import make_uniform_frames


# ===========================================================================
# latent_drift(pred, true, *, normalize="rms") -> (H,)
# ===========================================================================
def test_latent_drift_shape_and_dtype():
    rng = np.random.default_rng(0)
    H, D = 6, 4
    pred = rng.standard_normal((H, D)).astype(np.float32)
    true = rng.standard_normal((H, D)).astype(np.float32)
    out = latent_drift(pred, true)
    assert isinstance(out, np.ndarray)
    assert out.shape == (H,)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_latent_drift_zero_on_identical():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    out = latent_drift(x, x)
    assert out.shape == (5,)
    assert np.allclose(out, 0.0)


def test_latent_drift_zero_on_identical_no_normalize():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    out = latent_drift(x, x, normalize=None)
    assert np.allclose(out, 0.0)


def test_latent_drift_raw_l2_values():
    # normalize=None -> raw per-step L2 distance
    pred = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    true = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    out = latent_drift(pred, true, normalize=None)
    assert np.allclose(out, [0.0, 5.0, 1.0])


def test_latent_drift_rms_is_scale_free():
    # Scaling BOTH pred and true (and thus the diff and the RMS norm) by the
    # same positive constant leaves the rms-normalized drift unchanged.
    rng = np.random.default_rng(3)
    H, D = 7, 5
    pred = rng.standard_normal((H, D)).astype(np.float32)
    true = rng.standard_normal((H, D)).astype(np.float32)
    base = latent_drift(pred, true, normalize="rms")
    for c in (2.0, 0.1, 37.5):
        scaled = latent_drift((pred * c).astype(np.float32),
                              (true * c).astype(np.float32),
                              normalize="rms")
        assert np.allclose(base, scaled, atol=1e-5)


def test_latent_drift_rms_matches_manual():
    pred = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    true = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    # true is all zeros -> rms norm is 0; still must be finite & >= 0.
    out = latent_drift(pred, true, normalize="rms")
    assert out.shape == (2,)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_latent_drift_rms_normalizes_by_true_rms():
    # Non-degenerate true: verify the rms normalization divisor.
    pred = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # per-step raw L2: step0 |[1,0]|=1 ; step1 |[0,-1]|=1
    # RMS L2-norm of true rows: rows have norm 1 and 1 -> rms = 1
    out = latent_drift(pred, true, normalize="rms")
    assert np.allclose(out, [1.0, 1.0], atol=1e-6)


def test_latent_drift_shape_mismatch_raises():
    a = np.zeros((5, 4), dtype=np.float32)
    b = np.zeros((6, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        latent_drift(a, b)


def test_latent_drift_dim_mismatch_raises():
    a = np.zeros((5, 4), dtype=np.float32)
    b = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        latent_drift(a, b)


# ===========================================================================
# pearson(x, y) -> float
# ===========================================================================
def test_pearson_identical_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    r = pearson(x, x)
    assert isinstance(r, float)
    assert np.isclose(r, 1.0, atol=1e-7)


def test_pearson_perfect_positive_monotone():
    x = np.arange(10, dtype=np.float64)
    y = 3.0 * x + 7.0  # positive linear
    assert np.isclose(pearson(x, y), 1.0, atol=1e-7)


def test_pearson_perfect_negative():
    x = np.arange(10, dtype=np.float64)
    y = -2.0 * x + 1.0
    assert np.isclose(pearson(x, y), -1.0, atol=1e-7)


def test_pearson_sign_positive_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 20.0])  # broadly increasing
    assert pearson(x, y) > 0.0


def test_pearson_in_range():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(30)
    y = rng.standard_normal(30)
    r = pearson(x, y)
    assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9


def test_pearson_constant_input_returns_zero():
    x = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(x, y) == 0.0
    assert pearson(y, x) == 0.0
    # both constant
    assert pearson(x, x) == 0.0


def test_pearson_length_mismatch_raises():
    x = np.arange(5, dtype=np.float64)
    y = np.arange(6, dtype=np.float64)
    with pytest.raises(ValueError):
        pearson(x, y)


def test_pearson_too_short_raises():
    with pytest.raises(ValueError):
        pearson(np.array([1.0]), np.array([2.0]))
    with pytest.raises(ValueError):
        pearson(np.array([]), np.array([]))


# ===========================================================================
# one_step_error(model, frames_u8, actions_2d, ...) -> float
# ===========================================================================
def _consecutive_frames(n, base=0):
    return make_uniform_frames([base + i for i in range(n)])


def test_one_step_error_perfect_predictor_is_zero(perfect_model):
    C, fs = 3, 5
    n_blocks = C + 1
    frames = _consecutive_frames(n_blocks, base=2)
    rng = np.random.default_rng(0)
    actions = rng.standard_normal((n_blocks * fs, 2)).astype(np.float32)
    err = one_step_error(perfect_model, frames, actions, context=C, frameskip=fs)
    assert isinstance(err, float)
    assert np.isfinite(err)
    assert np.isclose(err, 0.0, atol=1e-5)


def test_one_step_error_defaults(perfect_model):
    # defaults context=3, frameskip=5
    n_blocks = 3 + 1
    frames = _consecutive_frames(n_blocks, base=0)
    rng = np.random.default_rng(1)
    actions = rng.standard_normal((n_blocks * 5, 2)).astype(np.float32)
    err = one_step_error(perfect_model, frames, actions)
    assert isinstance(err, float)
    assert np.isfinite(err)


def test_one_step_error_precondition_raises(perfect_model):
    # too few blocks for context+1
    C, fs = 3, 5
    frames = _consecutive_frames(C)  # need C+1
    actions = np.zeros(((C + 1) * fs, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        one_step_error(perfect_model, frames, actions, context=C, frameskip=fs)
