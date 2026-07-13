"""Latent-drift and correlation metrics.

See ``CONTRACTS.md`` for the binding I/O contracts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.pusht_horizon.rollout import open_loop_rollout

__all__ = ["latent_drift", "one_step_error", "pearson"]


def latent_drift(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    normalize: str | None = "rms",
) -> np.ndarray:
    """Per-step latent distance between ``pred (H,D)`` and ``true (H,D)``.

    ``normalize="rms"``: per-step L2 divided by the RMS L2-norm of the ``true`` latents
    (a positive scalar over all steps), yielding a scale-free measure. ``normalize=None``:
    raw per-step L2. Returns ``(H,)``, finite and ``>= 0``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError(
            f"pred and true must share shape (H,D); got {pred.shape} vs {true.shape}."
        )

    per_step = np.linalg.norm(pred - true, axis=1)  # (H,)

    if normalize is None:
        out = per_step
    elif normalize == "rms":
        true_norms = np.linalg.norm(true, axis=1)  # (H,)
        rms = float(np.sqrt(np.mean(true_norms**2)))
        if rms > 0:
            out = per_step / rms
        else:
            out = per_step
    else:
        raise ValueError(f"unknown normalize={normalize!r} (expected 'rms' or None).")

    return out.astype(np.float32)


def one_step_error(
    model: Any,
    frames_u8: np.ndarray,
    actions_2d: np.ndarray,
    *,
    context: int = 3,
    frameskip: int = 5,
    device: str = "cpu",
) -> float:
    """Scalar one-step latent-prediction error: encode ``context`` real frames, predict
    the next latent, and compare (L2) to the encoder latent of the real next frame.
    """
    out = open_loop_rollout(
        model,
        frames_u8,
        actions_2d,
        context=context,
        horizon=1,
        frameskip=frameskip,
        device=device,
    )
    pred = out["pred_latents"][0]
    true = out["true_latents"][0]
    return float(np.linalg.norm(pred.astype(np.float64) - true.astype(np.float64)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of 1-D arrays ``x, y`` (same length ``N >= 2``).

    Returns a value in ``[-1, 1]``; ``pearson(x, x) == 1``. If either input is
    constant, returns ``0.0``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape; got {x.shape} vs {y.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"pearson requires N >= 2, got N={x.shape[0]}.")

    xd = x - x.mean()
    yd = y - y.mean()
    denom = float(np.sqrt(np.sum(xd**2) * np.sum(yd**2)))
    if denom == 0.0:
        return 0.0
    r = float(np.dot(xd, yd) / denom)
    return float(np.clip(r, -1.0, 1.0))
