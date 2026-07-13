"""Forward-pass smoke test for the memory-maze training config (plan §3.6).

Composes the real Hydra training config with the memory-maze overrides and pushes a
synthetic batch through encode/predict, asserting every load-bearing shape:
pos-table rows == history_size, emb (B, W, 192), preds (B, C, 192), rollout default
window == history_size.
"""
import os

os.environ.setdefault("STABLEWM_HOME", "/mnt/nas2/lewm")

import hydra
import pytest
import torch
from omegaconf import open_dict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OVERRIDES = [
    "data=memory_maze",
    "img_size=64",
    "model.encoder.patch_size=8",
    "history_size=16",
    "normalize_action=false",
]


@pytest.fixture(scope="module")
def model():
    with hydra.initialize_config_dir(config_dir=os.path.join(REPO, "config", "train"),
                                     version_base=None):
        cfg = hydra.compose(config_name="lewm", overrides=OVERRIDES)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * 6  # 4*6=24
    m = hydra.utils.instantiate(cfg.model)
    m.train()
    return m


def test_config_dims(model):
    assert model.history_size == 16
    assert tuple(model.predictor.pos_embedding.shape) == (1, 16, 192)
    # action encoder first layer takes the 24-D block
    assert model.action_encoder.patch_embed.in_channels == 24
    # ViT configured natively at 64/patch-8
    vc = model.encoder.config
    assert vc.image_size == 64 and vc.patch_size == 8 and vc.hidden_size == 192


def test_forward_shapes(model):
    B, W = 2, 17  # num_steps = history_size + num_preds = 17
    batch = {
        "pixels": torch.rand(B, W, 3, 64, 64),
        "action": torch.zeros(B, W, 24).scatter_(
            2, torch.randint(0, 24, (B, W, 1)), 1.0),  # arbitrary one-hot-ish block
    }
    out = model.encode(dict(batch))
    assert out["emb"].shape == (B, W, 192)
    assert out["act_emb"].shape == (B, W, 192)
    preds = model.predict(out["emb"][:, :16], out["act_emb"][:, :16])
    assert preds.shape == (B, 16, 192)
    tgt = out["emb"][:, 1:]
    assert tgt.shape == preds.shape  # loss alignment W-1 == C


def test_rollout_default_window(model):
    assert getattr(model, "history_size", 3) == 16
    # context > 16 must fail on the pos table (the documented hard cap)
    with pytest.raises(RuntimeError):
        model.predict(torch.rand(1, 17, 192), torch.rand(1, 17, 192))
