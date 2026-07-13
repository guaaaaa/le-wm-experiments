"""Shared fakes and helpers for the pusht_horizon primitive tests.

These tests are written *only* from CONTRACTS.md. They must never import the
implementation modules (rollout/metrics/data/probe) except through the public
functions under test, and they must be fast: tiny synthetic tensors, small D,
fully deterministic. No 18M model, no 46GB dataset.

The fakes below implement the duck-typed ``model`` and ``dataset`` interfaces
described in CONTRACTS.md.
"""

import numpy as np
import pytest
import torch


# --- ImageNet stats (must match eval.py:img_transform -> spt ImageNet) --------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------
class _Param(torch.nn.Module):
    """A trivial module so ``next(model.parameters())`` exposes device/dtype."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype))


class LinearPerfectModel:
    """A toy *linear perfect-predictor* satisfying the duck-typed model API.

    Design (all latents live in R^D):

    * ``encode`` maps each frame to a *known, deterministic* latent. A frame is
      the preprocessed float tensor ``(B, T, 3, 224, 224)``; we collapse it to a
      scalar "frame id" ``f`` = round(mean over pixels * 1000) and produce the
      latent ``phi(f) = A @ base(f)`` where ``base(f) = [f, f^2, .., f^D] `` made
      finite/deterministic. Concretely we just use a fixed pseudo-random but
      deterministic latent per distinct frame content.

    * The *ground-truth latent sequence* used by a test is an arithmetic-style
      progression: frame with id ``k`` encodes to latent ``L[k]``. The perfect
      predictor, given a window of the true latents ending at true index ``j``,
      returns exactly ``L[j+1]`` at its last time position.

    To make this exact and self-contained we hand the model a table
    ``latent_of_frame`` mapping the integer frame-id -> latent, and
    ``next_latent`` mapping a latent (by its frame-id, recovered from an index
    channel we bake into the latent) -> the next latent. We bake the frame-id
    into latent channel 0 so ``predict`` can recover it from the window.
    """

    def __init__(self, D=4, n_frames=32, seed=0, device="cpu", dtype=torch.float32):
        self.D = D
        self.device = torch.device(device)
        self._dtype = dtype
        self._param = _Param(dtype=dtype)
        rng = np.random.default_rng(seed)
        # A deterministic latent table: row k is the latent for frame-id k.
        # Channel 0 holds the frame-id so predict() can recover position.
        tbl = rng.standard_normal((n_frames, D)).astype(np.float32)
        tbl[:, 0] = np.arange(n_frames, dtype=np.float32)
        self._table = tbl
        self.n_frames = n_frames
        self.eval_called = False

    # -- torch-module-ish surface ------------------------------------------
    def parameters(self):
        return self._param.parameters()

    def eval(self):
        self.eval_called = True
        return self

    # -- duck-typed API ----------------------------------------------------
    def _frame_id(self, pixels_bt3hw):
        """Recover the integer frame-id from a preprocessed frame tensor.

        The test harness constructs frames so that frame k has all pixels equal
        to the uint8 value k (before preprocessing). After preprocessing the
        mean over channels/pixels is monotonic in k, but we only need a stable
        deterministic key, so we round the raw (pre-normalization) recoverable
        signal. To keep this robust we instead read an id the harness stamps
        into pixel [0,0] channel 0's *rank*: simplest is to map by nearest
        table row using the mean. Here we take the argmax-safe integer key from
        a monotone statistic.
        """
        # pixels_bt3hw: (B, T, 3, 224, 224) float tensor
        x = pixels_bt3hw
        # Mean over spatial+channel dims -> (B, T)
        stat = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        return stat

    def encode(self, info):
        pixels = info["pixels"]
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        pixels = pixels.to(self.device, self._dtype)
        B, T = pixels.shape[0], pixels.shape[1]
        stat = self._frame_id(pixels)  # (B, T) monotone in uint8 value
        # Map the monotone statistic back to an integer frame-id. The harness
        # builds uniform frames with uint8 value == frame_id, so we invert the
        # (v/255 - mean)/std averaged-over-channels transform to recover v.
        # avg over channels of (v/255 - m_c)/s_c == v/255 * mean(1/s) - mean(m/s)
        inv_s = float((1.0 / IMAGENET_STD).mean())
        m_over_s = float((IMAGENET_MEAN / IMAGENET_STD).mean())
        v = (stat.cpu().numpy() + m_over_s) / inv_s * 255.0
        ids = np.rint(v).astype(int)
        ids = np.clip(ids, 0, self.n_frames - 1)
        emb = self._table[ids]  # (B, T, D)
        out = torch.as_tensor(emb, dtype=self._dtype, device=self.device)
        info = dict(info)
        info["emb"] = out
        return info

    def action_encoder(self, actions):
        """Encode raw action blocks (B, T, A_in) -> (B, T, A_emb).

        The perfect predictor ignores actions, so any deterministic map works.
        We use A_emb == A_in identity to keep shapes simple and finite.
        """
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        return actions.to(self.device, self._dtype)

    def predict(self, emb, act_emb):
        """(B,T,D),(B,T,A_emb) -> (B,T,D); last index = predicted next latent.

        Recovers the frame-id from channel 0 of the *last* window latent and
        returns the next table row (id+1) at the last time position. Earlier
        positions are filled with the input (unused by the caller).
        """
        if not torch.is_tensor(emb):
            emb = torch.as_tensor(emb)
        emb = emb.to(self.device, self._dtype)
        out = emb.clone()
        B, T, D = emb.shape
        last = emb[:, -1, :]  # (B, D)
        ids = torch.round(last[:, 0]).to(torch.int64).cpu().numpy()
        nxt = np.clip(ids + 1, 0, self.n_frames - 1)
        out[:, -1, :] = torch.as_tensor(
            self._table[nxt], dtype=self._dtype, device=self.device
        )
        return out

    # -- test helper: the ground-truth latent for uint8 frame value k -------
    def latent_for_frame_value(self, k):
        return self._table[int(k)].copy()


# ---------------------------------------------------------------------------
# Fake dataset
# ---------------------------------------------------------------------------
class FakeDataset:
    """Implements the duck-typed dataset interface from CONTRACTS.md.

    * ``lengths``: int array (n_episodes,) — per-episode length in action-blocks.
    * ``load_episode(ep_id)`` -> dict with pixels (L,224,224,3) uint8,
      action (L*frameskip, 2) float32, state (L, 7) float32.

    Deterministic content derived from ep_id + block index so tests can check
    slicing bounds without any real data.
    """

    def __init__(self, lengths, frameskip=5, seed=0):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.frameskip = int(frameskip)
        self._seed = int(seed)

    def load_episode(self, ep_id):
        ep_id = int(ep_id)
        if ep_id < 0 or ep_id >= len(self.lengths):
            raise IndexError(ep_id)
        L = int(self.lengths[ep_id])
        rng = np.random.default_rng(self._seed * 100003 + ep_id)
        # Uniform-per-block frames whose uint8 value encodes (ep_id, block).
        pixels = np.zeros((L, 224, 224, 3), dtype=np.uint8)
        for b in range(L):
            pixels[b] = np.uint8((ep_id * 7 + b) % 256)
        action = rng.standard_normal((L * self.frameskip, 2)).astype(np.float32)
        state = rng.standard_normal((L, 7)).astype(np.float32)
        return {"pixels": pixels, "action": action, "state": state}


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def make_uniform_frames(values):
    """Build (n, 224, 224, 3) uint8 frames; frame i is all-``values[i]``."""
    values = list(values)
    frames = np.zeros((len(values), 224, 224, 3), dtype=np.uint8)
    for i, v in enumerate(values):
        frames[i] = np.uint8(v)
    return frames


def imagenet_normalize_uniform(value):
    """Hand-computed per-channel normalized value for a uniform uint8 frame."""
    return (value / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def D():
    return 4


@pytest.fixture
def perfect_model(D):
    return LinearPerfectModel(D=D, n_frames=64, seed=1, device="cpu")


@pytest.fixture
def fake_dataset():
    return FakeDataset(lengths=[10, 20, 5, 30, 15], frameskip=5, seed=3)
