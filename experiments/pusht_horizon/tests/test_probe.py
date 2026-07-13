"""Tests for experiments.pusht_horizon.probe.

Written from CONTRACTS.md only. Synthetic linear data, no real model/data.
"""

import numpy as np
import pytest

from experiments.pusht_horizon.probe import fit_linear_probe, probe_error


def _to_np(x):
    return np.asarray(x)


def _linear_data(N=200, D=4, K=3, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((N, D)).astype(np.float32)
    W = rng.standard_normal((K, D)).astype(np.float32)  # (K, D)
    b = rng.standard_normal((K,)).astype(np.float32)
    targets = latents @ W.T + b
    if noise:
        targets = targets + noise * rng.standard_normal((N, K)).astype(np.float32)
    return latents, targets.astype(np.float32), W, b


# ===========================================================================
# fit_linear_probe(latents, targets) -> probe
# ===========================================================================
def test_fit_recovers_known_linear_map():
    latents, targets, W, b = _linear_data(N=300, D=4, K=3, seed=1, noise=0.0)
    probe = fit_linear_probe(latents, targets)
    # weight (K, D), bias (K,)
    assert _to_np(probe.weight).shape == (3, 4)
    assert _to_np(probe.bias).shape == (3,)
    assert np.allclose(_to_np(probe.weight), W, atol=1e-3)
    assert np.allclose(_to_np(probe.bias), b, atol=1e-3)


def test_predict_shape_and_accuracy():
    latents, targets, W, b = _linear_data(N=150, D=4, K=2, seed=2, noise=0.0)
    probe = fit_linear_probe(latents, targets)
    pred = _to_np(probe.predict(latents))
    assert pred.shape == (150, 2)
    assert np.allclose(pred, targets, atol=1e-3)


def test_predict_on_new_points():
    latents, targets, W, b = _linear_data(N=200, D=5, K=3, seed=3, noise=0.0)
    probe = fit_linear_probe(latents, targets)
    rng = np.random.default_rng(99)
    new = rng.standard_normal((10, 5)).astype(np.float32)
    pred = _to_np(probe.predict(new))
    expected = new @ W.T + b
    assert pred.shape == (10, 3)
    assert np.allclose(pred, expected, atol=1e-3)


def test_fit_deterministic():
    latents, targets, _, _ = _linear_data(N=100, D=4, K=2, seed=5, noise=0.1)
    p1 = fit_linear_probe(latents, targets)
    p2 = fit_linear_probe(latents, targets)
    assert np.allclose(_to_np(p1.weight), _to_np(p2.weight))
    assert np.allclose(_to_np(p1.bias), _to_np(p2.bias))


def test_fit_shape_mismatch_raises():
    latents = np.zeros((50, 4), dtype=np.float32)
    targets = np.zeros((49, 3), dtype=np.float32)  # N mismatch
    with pytest.raises(ValueError):
        fit_linear_probe(latents, targets)


def test_fit_too_few_samples_raises():
    latents = np.zeros((1, 4), dtype=np.float32)
    targets = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_linear_probe(latents, targets)


# ===========================================================================
# probe_error(probe, latents, targets) -> dict
# ===========================================================================
def test_probe_error_keys_and_types():
    latents, targets, _, _ = _linear_data(N=200, D=4, K=3, seed=6, noise=0.05)
    probe = fit_linear_probe(latents, targets)
    err = probe_error(probe, latents, targets)
    assert set(err.keys()) >= {"rmse", "per_dim_rmse", "r2"}
    assert isinstance(err["rmse"], float)
    assert isinstance(err["r2"], float)
    pdr = np.asarray(err["per_dim_rmse"])
    assert pdr.shape == (3,)
    assert pdr.dtype == np.float32
    assert np.isfinite(err["rmse"])
    assert np.isfinite(err["r2"])
    assert np.isfinite(pdr).all()


def test_probe_error_near_zero_on_perfect_fit():
    latents, targets, _, _ = _linear_data(N=300, D=4, K=3, seed=7, noise=0.0)
    probe = fit_linear_probe(latents, targets)
    err = probe_error(probe, latents, targets)
    assert err["rmse"] < 1e-3
    assert np.all(np.asarray(err["per_dim_rmse"]) < 1e-3)
    assert err["r2"] > 1.0 - 1e-4


def test_probe_error_per_dim_rmse_length():
    latents, targets, _, _ = _linear_data(N=120, D=6, K=4, seed=8, noise=0.2)
    probe = fit_linear_probe(latents, targets)
    err = probe_error(probe, latents, targets)
    assert np.asarray(err["per_dim_rmse"]).shape == (4,)
    assert (np.asarray(err["per_dim_rmse"]) >= 0).all()


def test_probe_error_r2_in_reasonable_range():
    latents, targets, _, _ = _linear_data(N=250, D=4, K=2, seed=9, noise=0.3)
    probe = fit_linear_probe(latents, targets)
    err = probe_error(probe, latents, targets)
    # a real (noisy but linear) fit -> r2 should be high but <= 1
    assert err["r2"] <= 1.0 + 1e-6
    assert err["r2"] > 0.5


def test_probe_error_rmse_matches_manual():
    latents, targets, _, _ = _linear_data(N=80, D=3, K=2, seed=10, noise=0.4)
    probe = fit_linear_probe(latents, targets)
    pred = _to_np(probe.predict(latents))
    err = probe_error(probe, latents, targets)
    manual_rmse = float(np.sqrt(np.mean((pred - targets) ** 2)))
    assert np.isclose(err["rmse"], manual_rmse, rtol=1e-4, atol=1e-5)
    manual_pdr = np.sqrt(np.mean((pred - targets) ** 2, axis=0))
    assert np.allclose(np.asarray(err["per_dim_rmse"]), manual_pdr, atol=1e-4)


def test_probe_error_shape_mismatch_raises():
    latents, targets, _, _ = _linear_data(N=60, D=4, K=3, seed=11)
    probe = fit_linear_probe(latents, targets)
    bad_targets = np.zeros((60, 5), dtype=np.float32)  # K mismatch
    with pytest.raises(ValueError):
        probe_error(probe, latents, bad_targets)
    bad_latents = np.zeros((59, 4), dtype=np.float32)  # N mismatch
    with pytest.raises(ValueError):
        probe_error(probe, bad_latents, targets)
