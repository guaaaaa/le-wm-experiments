"""Convert the released LeWM checkpoint's ViT encoder keys from the old
HuggingFace ViT layout (transformers <5, `encoder.encoder.layer.N.*`) to the
flattened layout used by transformers>=5 (`encoder.layers.N.*`).

Only the 12 ViT transformer blocks are renamed; all other weights
(embeddings, projector, predictor, action_encoder, pred_proj) already match.
Tensor shapes are identical, so this is a pure key rename.

Usage:
    python convert_checkpoint.py <src_dir> <dst_dir>
    # src_dir holds the ORIGINAL released config.json + weights.pt (download from HF);
    # dst_dir gets the transformers>=5 compatible copy.
    # The already-converted PushT checkpoint lives in models/lewm-pusht/.
"""
import re
import sys
import shutil
from pathlib import Path

import torch

# per-block module renames (old HF ViT -> transformers>=5 flattened ViT)
_MODULE_MAP = {
    "attention.attention.query": "attention.q_proj",
    "attention.attention.key": "attention.k_proj",
    "attention.attention.value": "attention.v_proj",
    "attention.output.dense": "attention.o_proj",
    "intermediate.dense": "mlp.fc1",
    "output.dense": "mlp.fc2",
    # layernorm_before / layernorm_after keep their leaf names
}

_LAYER_RE = re.compile(r"^(?P<pre>.*?)encoder\.encoder\.layer\.(?P<i>\d+)\.(?P<rest>.*)$")


def remap_key(key: str) -> str:
    m = _LAYER_RE.match(key)
    if not m:
        return key  # non-block key: unchanged
    pre, i, rest = m.group("pre"), m.group("i"), m.group("rest")
    for old, new in _MODULE_MAP.items():
        if rest.startswith(old + "."):
            rest = new + rest[len(old):]
            break
    return f"{pre}encoder.layers.{i}.{rest}"


def convert(src_dir: str, dst_dir: str) -> None:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    sd = torch.load(src / "weights.pt", map_location="cpu")
    new_sd = {remap_key(k): v for k, v in sd.items()}
    assert len(new_sd) == len(sd), "key collision during remap"

    torch.save(new_sd, dst / "weights.pt")
    shutil.copy2(src / "config.json", dst / "config.json")
    print(f"wrote {dst/'weights.pt'} ({len(new_sd)} keys) + config.json")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
