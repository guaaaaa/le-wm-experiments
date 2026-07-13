# Contracts — experiments/pusht_horizon

Binding I/O contracts for the T3/T4 primitives. **Tests are written from this file; the
implementation is written from this file.** The implementer must NOT read the tests; the test
author must NOT read the implementation. Neither edits the other's files. Tests assert this
contract, not implementation details. On an API change, this file is updated first, then tests,
then implementation.

All array shapes use `D` = latent dim, `A_emb` = predictor action-embedding dim,
`A_in` = raw action-block dim (`frameskip * 2` = 10 for LeWM), `H` = horizon, `C` = context.

---

## Duck-typed `model` interface (what the primitives call)
The primitives depend only on these attributes/methods (LeWM satisfies them; tests use a fake):
- `model.encode(info: dict) -> dict`: `info["pixels"]` is a float tensor `(B, T, 3, 224, 224)` on
  the model's device; returns the same dict with `info["emb"]` a tensor `(B, T, D)`. (Encoder runs
  per frame; deterministic in eval mode.)
- `model.predict(emb: (B,T,D), act_emb: (B,T,A_emb)) -> (B,T,D)`: the value at the **last** time
  index is the predicted next latent for the step after the window.
- `model.action_encoder(actions: (B,T,A_in)) -> (B,T,A_emb)`: encodes raw action blocks.
- `next(model.parameters())` exposes the model's device and dtype.

## Duck-typed `dataset` interface
- `dataset.lengths`: int array `(n_episodes,)` — per-episode length in **action-blocks** (= frames).
- `dataset.load_episode(ep_id: int) -> dict` with `"pixels": (L,224,224,3) uint8`,
  `"action": (L*frameskip, 2) float32` (**dense**, not subsampled), `"state": (L,7) float32`.

---

## `rollout.open_loop_rollout(model, frames_u8, actions_2d, *, context=3, horizon, frameskip=5, device="cpu") -> dict`
**Inputs**
- `frames_u8`: `np.uint8 (n_blocks, 224, 224, 3)`, `n_blocks >= context + horizon`.
- `actions_2d`: `np.float32 (n_env, 2)`, `n_env >= (context + horizon) * frameskip`. Dense
  per-env-step actions; action-block `i` = `actions_2d[i*frameskip : (i+1)*frameskip]` flattened → `A_in`.
- `context: int >= 1` (default 3); `horizon: int >= 1`; `frameskip: int >= 1` (default 5).

**Preconditions** — raise `ValueError` if: wrong ndim/dtype/trailing-shape; `n_blocks < context+horizon`;
`n_env < (context+horizon)*frameskip`; `context < 1`; `horizon < 1`.

**Behavior**
1. Preprocess frames via `data.image_transform()` → float tensor `(n, 3, 224, 224)`.
2. Build per-block actions: `blocks = actions_2d[:(context+horizon)*frameskip].reshape(context+horizon, frameskip*2)`.
3. Encode the first `context` real frames → context latents `z[0..C-1]`.
4. Open-loop, feeding predictions back: keep a rolling window of the last `C` latents (seeded with
   the real context latents). For `t` in `0..H-1`: predict the next latent from the window and the
   matching `C` action-blocks (the blocks aligned with those window latents, i.e. ending at block
   `C+t-1`); take `model.predict(window(1,C,D), act_emb(1,C,A_emb))[:, -1]`. Append the prediction
   to the window (drop the oldest). Record `pred_latents[t]`.
5. Ground truth: `true_latents[t] = encode(frames_u8[C + t])` (encoder on the real future frame).

**Output** — `dict`:
- `"pred_latents": np.float32 (H, D)` — open-loop predicted latents.
- `"true_latents": np.float32 (H, D)` — encoder latents of the real future frames.

**Invariants**
- Output shapes exactly `(H, D)`; all finite; deterministic given inputs + model in eval mode.
- **Perfect-predictor ⇒ zero drift:** if `model.predict` returns the exact next true latent for any
  window equal to the true latent sequence, and `encode` is deterministic, then
  `pred_latents == true_latents` (allclose). (Testable with a toy linear model.)

## `metrics.latent_drift(pred, true, *, normalize="rms") -> np.float32 (H,)`
Per-step distance between `pred (H,D)` and `true (H,D)`. `normalize="rms"`: `‖pred_t − true_t‖₂`
divided by the RMS L2-norm of the `true` latents (a positive scalar over all steps) → scale-free;
`normalize=None`: raw L2. `ValueError` on shape mismatch. Output `(H,)`, finite, `>= 0`.
Invariant: `latent_drift(x, x)` is all zeros.

## `metrics.one_step_error(model, frames_u8, actions_2d, *, context=3, frameskip=5, device="cpu") -> float`
Scalar one-step latent-prediction error: encode `context` real frames, predict the next latent,
compare (L2) to `encode(frames_u8[context])`. Same preconditions as `open_loop_rollout` with
`horizon=1`. May reuse `open_loop_rollout`.

## `metrics.pearson(x, y) -> float`
Pearson correlation of 1-D arrays `x, y` (same length `N >= 2`). `ValueError` on mismatch/`N<2`.
Returns a value in `[-1, 1]`; `pearson(x, x) == 1` (allclose). If either input is constant, return `0.0`.

## `data.image_transform() -> callable`
Returns a callable mapping a uint8 image array/tensor with a trailing `(...,224,224,3)` **or**
`(...,3,224,224)` layout to a float32 tensor `(...,3,224,224)`, ImageNet-normalized and resized to
224. Must match `eval.py:img_transform` semantics: ToImage → float scale to [0,1] → Normalize(ImageNet
mean/std) → Resize(224). Deterministic.

## `data.select_episodes(dataset, n, seed, *, min_length=None) -> np.ndarray[int]`
Deterministically choose `n` unique episode ids (optionally restricted to `dataset.lengths >=
min_length`) via `np.random.default_rng(seed)`; return **sorted ascending** int array. `ValueError`
if fewer than `n` eligible. Invariant: same `(dataset, n, seed, min_length)` → identical output.

## `data.load_episode_frames_actions(dataset, ep_id, start=0, length=None, *, frameskip=5) -> dict`
Return `{"frames_u8": (L,224,224,3) uint8, "actions_2d": (L*frameskip,2) float32, "state": (L,7)}`
for blocks `[start, start+L)` of episode `ep_id`; `length=None` ⇒ to episode end
(`L = dataset.lengths[ep_id] - start`). `ValueError` if `ep_id` out of range, `start < 0`, or
`start + length > dataset.lengths[ep_id]`.

## `probe.fit_linear_probe(latents, targets) -> probe`
Least-squares linear map `D -> K` **with bias**, fit on `latents (N,D)`, `targets (N,K)`.
Deterministic. Returns an object with `.predict(latents:(M,D)) -> (M,K)`, `.weight (K,D)`,
`.bias (K,)`. `ValueError` on shape mismatch or `N < 2`.

## `probe.probe_error(probe, latents, targets) -> dict`
`{"rmse": float, "per_dim_rmse": np.float32 (K,), "r2": float}` over `latents (N,D)`, `targets
(N,K)`. `r2` is the standard coefficient of determination (aggregate). Angle handling is the
caller's responsibility (pass wrapped/sincos targets); the probe treats targets as plain regression.
