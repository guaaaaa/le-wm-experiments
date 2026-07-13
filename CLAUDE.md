# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Running experiments?** Read [EXPERIMENT_RULES.md](EXPERIMENT_RULES.md) first — it is
> binding for output layout, run naming, contract-first/test-driven code changes, and
> shared-machine resource limits (GPU/CPU/disk).

## Overview

This is the research code for **LeWorldModel (LeWM)** — a Joint-Embedding Predictive Architecture (JEPA) that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a Gaussian-isotropy regularizer (SIGReg). The repo is deliberately minimal: it contains *only* the model architecture and training objective. Everything else (environments, datasets, planning, evaluation harness, checkpoint I/O) lives in two external dependencies:

- **`stable_worldmodel` (swm)** — environments (`swm.World`), planning solvers (`swm.solver.*`), MPC policies (`swm.policy.*`), HDF5/Lance datasets (`swm.data.*`), and checkpoint load/save (`swm.wm.utils`).
- **`stable_pretraining` (spt)** — the training `Module`/`Manager`/`DataModule` abstractions, ViT backbones (`spt.backbone.utils.vit_hf`), data transforms, and optimizer/scheduler wiring.

When behavior depends on `swm.*` or `spt.*`, the definition is *not* in this repo — inspect the installed packages in `.venv` rather than searching locally.

## Environment setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

`STABLEWM_HOME` (default `~/.stable-wm/`) is the root for datasets and checkpoints. Dataset names in configs omit the extension and resolve to files under `$STABLEWM_HOME`. `LOCAL_DATASET_DIR` overrides the dataset cache dir for training.

## Commands

Training and evaluation are both Hydra entry points; override any config key on the CLI with `key=value`.

```bash
# Train (data= selects config/train/data/<name>.yaml)
python train.py data=pusht

# Evaluate / plan (--config-name selects config/eval/<name>.yaml)
# policy= is the checkpoint path relative to $STABLEWM_HOME WITHOUT the _object.ckpt suffix
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

There is no test suite, linter, or build step — this is a research repo run directly via the two scripts above. Before training you must set the WandB `entity`/`project` under `wandb.config` in `config/train/lewm.yaml` (or set `wandb.enabled=false`).

## Architecture

### Model (`jepa.py`, `module.py`)

`JEPA` (`jepa.py`) composes five submodules, all injected via Hydra (`config/train/model/lewm.yaml`):
- **encoder** — a ViT (`vit_hf`, tiny, patch 14) mapping pixels → CLS-token embedding.
- **projector** / **pred_proj** — `MLP`s (BatchNorm1d) applied after encoding and after prediction.
- **action_encoder** — `Embedder` (Conv1d + MLP) mapping raw actions → action embeddings. Its `input_dim` is `???` in the config and is filled in at runtime by `train.py` as `frameskip * dataset.action_dim`.
- **predictor** — `ARPredictor`, an autoregressive transformer of `ConditionalBlock`s using **AdaLN-zero** conditioning: the action embedding conditions each block via `modulate(x, shift, scale)`. Attention is **causal** (`is_causal=True`).

Key `JEPA` methods:
- `encode(info)` — flattens `(B,T,...)` pixels, runs the encoder, projects, and reshapes back; also encodes actions if present. Mutates and returns the `info` dict (keys `emb`, `act_emb`).
- `predict(emb, act_emb)` — one predictor step.
- `rollout(info, action_sequence, history_size=3)` — **inference only.** Autoregressively rolls the predictor forward over an action sequence, always feeding the last `history_size` steps back in. Used for planning, not training.
- `get_cost` / `criterion` — planning cost = MSE between the rolled-out final embedding and the goal embedding, per action candidate. This is the interface `swm` planners call.

### Training objective (`train.py`)

`lejepa_forward` is the core loss, passed to `spt.Module` via `partial`. It:
1. Encodes the batch, splits embeddings into context (`[:history_size]`) and target (`[num_preds:]`).
2. `pred_loss` = MSE(predicted next embeddings, target embeddings).
3. `sigreg_loss` = **SIGReg** (`module.py`), the "Sketch Isotropic Gaussian Regularizer" — projects embeddings onto random directions and penalizes deviation from a standard Gaussian via an Epps–Pulley statistic. This single regularizer is what prevents representation collapse (replacing the usual EMA / stop-grad / multi-term losses).
4. `loss = pred_loss + sigreg.weight * sigreg_loss` (one tunable loss hyperparameter, `loss.sigreg.weight`).

Actions are `nan_to_num`'d because NaNs appear at sequence boundaries. Checkpoints are written every epoch by `SaveCkptCallback` (`utils.py`) via `swm`'s `save_pretrained`.

### Config layout (Hydra)

- `config/train/lewm.yaml` — root training config (trainer, optimizer, loss weight, `history_size`, `num_preds`). `defaults` compose in `data/<env>` and `model/lewm`.
- `config/train/data/*.yaml` — per-environment dataset specs (`pusht`, `dmc`, `ogb`, `tworoom`): dataset name, `frameskip`, columns to load/cache.
- `config/eval/*.yaml` — per-environment eval configs; `defaults` compose in a `solver/` (`cem` or `adam`) and a `launcher/`. Non-pixel columns are z-score normalized (fit on the dataset in `train.py`/`eval.py`).

### Checkpoint formats

Two checkpoint flavors coexist (see README "Loading a checkpoint"):
- `<name>_object.ckpt` — a pickled `JEPA` object; what `eval.py` and `swm.policy.AutoCostModel` load.
- `<name>_weight.ckpt` / `weights.pt` — a `state_dict` only. To use HuggingFace-mirrored weights with `eval.py`, first rebuild a `JEPA` from `config.json` and re-save it as `_object.ckpt` (conversion snippet in the README).

## Gotchas

- The Git repo and all code live in the `le-wm-experiments/` subdirectory of the workspace, not the workspace root.
- Tensor shapes are load-bearing and pervasive; `einops.rearrange` conventions are documented in docstrings (`b`=batch, `t`=time, `s`=action-plan samples, `d`=embed dim). Preserve these when editing.
- Normalizers must be picklable (`ZScoreNormalizer` is a class, not a closure) because DataLoader workers are spawned.
