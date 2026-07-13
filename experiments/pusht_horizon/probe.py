"""Linear probe (least-squares with bias) and its error metrics.

See ``CONTRACTS.md`` for the binding I/O contracts.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LinearProbe", "fit_linear_probe", "probe_error"]


class LinearProbe:
    """Affine map ``latents (M,D) -> targets (M,K)``: ``x @ weight.T + bias``.

    Attributes:
        weight: ``(K, D)`` float array.
        bias: ``(K,)`` float array.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = np.asarray(weight, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)

    def predict(self, latents: np.ndarray) -> np.ndarray:
        x = np.asarray(latents, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"latents must be (M,{self.weight.shape[1]}), got {x.shape}."
            )
        return x @ self.weight.T + self.bias


def fit_linear_probe(latents: np.ndarray, targets: np.ndarray) -> LinearProbe:
    """Fit a least-squares affine map ``D -> K`` (with bias) via ``np.linalg.lstsq``.

    ``latents (N,D)``, ``targets (N,K)``. Deterministic. Raises ``ValueError`` on shape
    mismatch or ``N < 2``.
    """
    X = np.asarray(latents, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            f"latents and targets must be 2-D; got {X.shape} and {Y.shape}."
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"latents and targets must share N; got {X.shape[0]} vs {Y.shape[0]}."
        )
    if X.shape[0] < 2:
        raise ValueError(f"fit_linear_probe requires N >= 2, got N={X.shape[0]}.")

    N, D = X.shape
    # Augment with a bias column of ones -> solve (N, D+1) @ (D+1, K) = (N, K).
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)  # (D+1, K)
    weight = coef[:D].T  # (K, D)
    bias = coef[D]  # (K,)
    return LinearProbe(weight, bias)


def probe_error(probe: LinearProbe, latents: np.ndarray, targets: np.ndarray) -> dict:
    """Return ``{"rmse": float, "per_dim_rmse": (K,) float32, "r2": float}`` over
    ``latents (N,D)``, ``targets (N,K)``. ``r2`` is the aggregate coefficient of
    determination. Targets are treated as plain regression.
    """
    Y = np.asarray(targets, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"targets must be 2-D (N,K); got {Y.shape}.")
    pred = probe.predict(latents)  # (N,K)
    if pred.shape != Y.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {pred.shape} vs {Y.shape}."
        )

    resid = pred - Y  # (N,K)
    per_dim_rmse = np.sqrt(np.mean(resid**2, axis=0))  # (K,)
    rmse = float(np.sqrt(np.mean(resid**2)))

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "rmse": rmse,
        "per_dim_rmse": per_dim_rmse.astype(np.float32),
        "r2": float(r2),
    }
