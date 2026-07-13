"""Data primitives: image preprocessing, episode selection, and the bridge from
the real ``stable_worldmodel`` HDF5Dataset to the documented episode dict.

See ``CONTRACTS.md`` for the binding I/O contracts.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms

__all__ = [
    "image_transform",
    "select_episodes",
    "load_episode_frames_actions",
]

_IMG_SIZE = 224


def image_transform() -> Callable[[Any], torch.Tensor]:
    """Return a deterministic callable mapping a uint8 image array/tensor with a
    trailing ``(...,224,224,3)`` or ``(...,3,224,224)`` layout to a float32 tensor
    ``(...,3,224,224)``, ImageNet-normalized and resized to 224.

    Matches ``eval.py:img_transform`` semantics: ToImage -> float scale to [0,1] ->
    Normalize(ImageNet mean/std) -> Resize(224).
    """
    compose = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=_IMG_SIZE),
        ]
    )

    def _transform(images: Any) -> torch.Tensor:
        x = images
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        if x.ndim < 3:
            raise ValueError(
                f"image_transform expects at least 3 dims, got shape {tuple(x.shape)}"
            )
        # Normalize layout to channel-last-of-first: bring channels to (...,3,224,224).
        if x.shape[-1] == 3 and x.shape[-2] == _IMG_SIZE and x.shape[-3] == _IMG_SIZE:
            # (...,224,224,3) -> (...,3,224,224)
            x = x.movedim(-1, -3)
        elif x.shape[-3] == 3 and x.shape[-2] == _IMG_SIZE and x.shape[-1] == _IMG_SIZE:
            pass  # already (...,3,224,224)
        else:
            raise ValueError(
                "image_transform expects a trailing (...,224,224,3) or (...,3,224,224) "
                f"layout, got shape {tuple(x.shape)}"
            )
        x = x.contiguous()
        return compose(x)

    return _transform


def select_episodes(
    dataset: Any,
    n: int,
    seed: int,
    *,
    min_length: int | None = None,
) -> np.ndarray:
    """Deterministically choose ``n`` unique episode ids via ``np.random.default_rng(seed)``.

    Optionally restrict to episodes with ``dataset.lengths >= min_length``. Returns a
    sorted-ascending int array. Raises ``ValueError`` if fewer than ``n`` eligible.
    """
    lengths = np.asarray(dataset.lengths)
    eligible = np.arange(len(lengths))
    if min_length is not None:
        eligible = eligible[lengths[eligible] >= min_length]

    if len(eligible) < n:
        raise ValueError(
            f"Requested {n} episodes but only {len(eligible)} eligible "
            f"(min_length={min_length})."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n, replace=False)
    return np.sort(chosen).astype(int)


def _episode_dict(dataset: Any, ep_id: int) -> dict:
    """Load one episode from the (fake or real) dataset and normalize it to the
    documented layout: pixels ``(L,224,224,3) uint8``, action dense ``(?,2) float32``,
    state ``(L,7)``. Returns numpy arrays and the block count ``L``.
    """
    ep = dataset.load_episode(int(ep_id))

    def _np(v: Any) -> np.ndarray:
        if torch.is_tensor(v):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    pixels = _np(ep["pixels"])
    # Normalize pixels to HWC (L,224,224,3).
    if pixels.ndim != 4:
        raise ValueError(f"pixels must be 4-D, got shape {pixels.shape}")
    if pixels.shape[-1] == 3 and pixels.shape[-2] == 224:
        pass  # already (L,224,224,3)
    elif pixels.shape[1] == 3 and pixels.shape[-1] == 224:
        pixels = np.transpose(pixels, (0, 2, 3, 1))  # (L,3,224,224) -> (L,224,224,3)
    frames = np.ascontiguousarray(pixels).astype(np.uint8, copy=False)

    actions = _np(ep["action"]).astype(np.float32, copy=False)
    state = _np(ep["state"]) if "state" in ep else None
    return {"frames": frames, "actions": actions, "state": state}


def load_episode_frames_actions(
    dataset: Any,
    ep_id: int,
    start: int = 0,
    length: int | None = None,
    *,
    frameskip: int = 5,
) -> dict:
    """Bridge the real ``stable_worldmodel`` HDF5Dataset (or a fake matching the
    documented interface) to the block-oriented episode dict.

    Returns ``{"frames_u8": (L,224,224,3) uint8, "actions_2d": (L*frameskip,2) float32,
    "state": (L,7)}`` for blocks ``[start, start+L)`` of episode ``ep_id``. With
    ``length=None`` the slice runs to the episode end.

    Raises ``ValueError`` if ``ep_id`` is out of range, ``start < 0``, or
    ``start + length`` exceeds the number of available blocks.
    """
    n_ep = len(dataset.lengths)
    if ep_id < 0 or ep_id >= n_ep:
        raise ValueError(f"ep_id {ep_id} out of range [0, {n_ep}).")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}.")

    ep = _episode_dict(dataset, ep_id)
    frames = ep["frames"]  # (L_full,224,224,3)
    actions = ep["actions"]  # dense (?,2)
    state = ep["state"]

    n_blocks = frames.shape[0]  # number of blocks (frames) actually available

    if length is None:
        L = n_blocks - start
    else:
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}.")
        L = length
    if start + L > n_blocks:
        raise ValueError(
            f"start+length ({start + L}) exceeds episode blocks ({n_blocks})."
        )

    frames_slice = frames[start : start + L]

    # Dense actions aligned with these blocks: block b -> actions[b*fs:(b+1)*fs].
    n_env_needed = L * frameskip
    act_start = start * frameskip
    act = actions[act_start : act_start + n_env_needed]
    if act.shape[0] < n_env_needed:
        # Pad the trailing (partial final block) by repeating the last dense action.
        pad_rows = n_env_needed - act.shape[0]
        last = act[-1:] if act.shape[0] > 0 else np.zeros((1, 2), dtype=np.float32)
        act = np.concatenate([act, np.repeat(last, pad_rows, axis=0)], axis=0)
    actions_2d = np.ascontiguousarray(act[:n_env_needed]).astype(np.float32, copy=False)

    out = {
        "frames_u8": np.ascontiguousarray(frames_slice).astype(np.uint8, copy=False),
        "actions_2d": actions_2d,
    }
    if state is not None:
        out["state"] = np.ascontiguousarray(state[start : start + L])
    return out
