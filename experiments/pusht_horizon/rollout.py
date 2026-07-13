"""Open-loop latent rollout of a duck-typed world model.

See ``CONTRACTS.md`` for the binding I/O contract of ``open_loop_rollout``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.pusht_horizon.data import image_transform

__all__ = ["open_loop_rollout"]


def _validate(
    frames_u8: np.ndarray,
    actions_2d: np.ndarray,
    context: int,
    horizon: int,
    frameskip: int,
) -> None:
    if context < 1:
        raise ValueError(f"context must be >= 1, got {context}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if frameskip < 1:
        raise ValueError(f"frameskip must be >= 1, got {frameskip}.")

    if not isinstance(frames_u8, np.ndarray) or frames_u8.dtype != np.uint8:
        raise ValueError("frames_u8 must be a np.uint8 ndarray.")
    if frames_u8.ndim != 4 or frames_u8.shape[1:] != (224, 224, 3):
        raise ValueError(
            f"frames_u8 must have shape (n_blocks,224,224,3), got {frames_u8.shape}."
        )
    if not isinstance(actions_2d, np.ndarray) or actions_2d.dtype != np.float32:
        raise ValueError("actions_2d must be a np.float32 ndarray.")
    if actions_2d.ndim != 2 or actions_2d.shape[1] != 2:
        raise ValueError(
            f"actions_2d must have shape (n_env,2), got {actions_2d.shape}."
        )

    n_blocks = frames_u8.shape[0]
    n_env = actions_2d.shape[0]
    if n_blocks < context + horizon:
        raise ValueError(
            f"n_blocks ({n_blocks}) < context+horizon ({context + horizon})."
        )
    if n_env < (context + horizon) * frameskip:
        raise ValueError(
            f"n_env ({n_env}) < (context+horizon)*frameskip "
            f"({(context + horizon) * frameskip})."
        )


def _model_device_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    p = next(model.parameters())
    return p.device, p.dtype


def _encode_frames(
    model: Any,
    frames_float: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode ``(n,3,224,224)`` float frames -> latents ``(n,D)`` (deterministic)."""
    info = {"pixels": frames_float.unsqueeze(0).to(device)}  # (1,n,3,224,224)
    out = model.encode(info)
    return out["emb"][0]  # (n,D)


def open_loop_rollout(
    model: Any,
    frames_u8: np.ndarray,
    actions_2d: np.ndarray,
    *,
    context: int = 3,
    horizon: int,
    frameskip: int = 5,
    device: str = "cpu",
) -> dict:
    """Open-loop rollout: encode ``context`` real frames, then autoregressively
    predict ``horizon`` future latents feeding predictions back into the window.

    Returns ``{"pred_latents": (H,D) float32, "true_latents": (H,D) float32}``.
    """
    _validate(frames_u8, actions_2d, context, horizon, frameskip)

    C, H = int(context), int(horizon)
    n_used = C + H

    model_device, model_dtype = _model_device_dtype(model)
    dev = torch.device(device) if device is not None else model_device

    # 1. Preprocess frames -> (n_used,3,224,224) float.
    transform = image_transform()
    frames_float = transform(frames_u8[:n_used]).to(dtype=model_dtype)

    # 2. Per-block actions: (n_used, frameskip*2).
    blocks = actions_2d[: n_used * frameskip].reshape(n_used, frameskip * 2)
    act_blocks = torch.from_numpy(np.ascontiguousarray(blocks)).to(
        device=dev, dtype=model_dtype
    )

    model.eval()
    with torch.no_grad():
        # 3. Encode context latents + true future latents in one pass.
        all_latents = _encode_frames(model, frames_float, dev)  # (n_used, D)
        context_latents = all_latents[:C]  # (C, D)
        true_latents = all_latents[C : C + H]  # (H, D)

        # Encode all action blocks once: (1, n_used, A_emb).
        act_emb_all = model.action_encoder(act_blocks.unsqueeze(0))[0]  # (n_used,A_emb)

        # 4. Open-loop rollout with a rolling window of the last C latents.
        window = list(context_latents.unbind(dim=0))  # C tensors of (D,)
        preds = []
        for t in range(H):
            win = torch.stack(window[-C:], dim=0).unsqueeze(0)  # (1,C,D)
            act_win = act_emb_all[t : t + C].unsqueeze(0)  # (1,C,A_emb)
            nxt = model.predict(win, act_win)[:, -1]  # (1,D)
            nxt = nxt[0]
            preds.append(nxt)
            window.append(nxt)

        pred_latents = torch.stack(preds, dim=0)  # (H,D)

    pred_np = pred_latents.detach().cpu().numpy().astype(np.float32)
    true_np = true_latents.detach().cpu().numpy().astype(np.float32)
    return {"pred_latents": pred_np, "true_latents": true_np}
