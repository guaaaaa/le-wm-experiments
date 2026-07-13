# Experiment Rules

Rules for running large-scale experiments in this repo. They govern **structure and
process**, not scientific content: file formats, metrics, models, and configs are chosen
per experiment. Only the invariants below are mandatory.

**MUST** = hard rule, never break. **SHOULD** = strong default; deviate only with a
recorded reason. When unsure, prefer isolation, uniqueness, and leaving the shared
machine healthy.

Core principles: every run is **reproducible**, **self-isolated**, and **safe on a shared
machine**.

---

## 1. Output layout & organization

- **MUST** give every run its own directory. Never write into, overwrite, or reuse
  another run's directory.
- **Default output location = the repo `outputs/` folder** (in the working directory, for
  easy access), organized per task:
  ```
  outputs/<task>/
    runs/<run-name>/     # one dir per run/sweep: results.json / metrics.jsonl,
                         #   config.yaml, meta.json (git SHA, cmd, seed, versions), logs/
    figures/             # generated plots (the publication figures)
    RESULTS.md           # the write-up for this task, referencing figures/
  ```
- **Large raw/generated data stays on the NAS**, never the repo or the near-full home disk:
  source datasets under `/mnt/nas2/lewm/datasets/`; any bulky generated artifact (multi-GB
  checkpoints, videos, big dumps) under `/mnt/nas2/lewm/experiments/<task>/`, leaving a small
  pointer file (its path) in the repo run dir. Rule of thumb: **if a file is > ~100 MB or isn't
  needed for quick inspection, put it on the NAS**, not `outputs/`.
- **SHOULD** keep per-run contents small and inspectable (see "Keep logging lean").
- **MUST NOT** commit large artifacts or datasets to git. The report (`RESULTS.md`) and small
  figures may be tracked; raw run dirs (`outputs/**/runs/`) stay gitignored.

**Keep logging lean.** Log the core results plus only a small, bounded amount of extra
information — just enough to understand and reproduce the run, no more.
- **SHOULD** log scalar/summary metrics, not raw tensors, activations, or per-sample dumps.
- **SHOULD** checkpoint periodically (not every step) and keep only what's useful — e.g.
  best + last (+ a few milestones); prune the rest.
- **SHOULD** cap/rotate text logs; save large artifacts (videos, image grids) only when
  they answer a question, and downsample/compress them.
- **MUST NOT** write verbose per-step debug data, redundant copies, or the dataset into a
  run dir. When in doubt, log less.

## 2. Naming (no duplicate names)

- **MUST** make every run name globally unique and never reuse one. Guarantee uniqueness
  with a timestamp (and a short random/hash suffix if launching in parallel).
- **SHOULD** use: `<task>-<method>-<slug>-<YYYYMMDD-HHMMSS>` — lowercase, `[a-z0-9-]`
  only, no spaces. Example: `pusht-lewm-baseline-20260630-201500`.
- The name `<slug>` describes the *one thing this run changes* (e.g. `lr5e5`, `sigreg0p2`).
- **MUST** override the repo default `output_model_name` (currently fixed to `lewm`) with
  the unique run name — otherwise all runs collide in `checkpoints/lewm/`.
- **SHOULD** append one line per run to a registry (`/mnt/nas2/lewm/experiments/INDEX.md`):
  name, date, purpose, key result, path.

## 3. Reproducibility & provenance

- **MUST** set and record all seeds; **MUST** snapshot the resolved config and the exact
  launch command into the run dir.
- **MUST** record the git commit and, if the tree is dirty, the diff. **SHOULD** avoid
  launching large runs from a dirty tree.
- **SHOULD** record key library/hardware versions (torch, CUDA, GPU, dataset id/hash).
- **MUST** treat source datasets as read-only. Never mutate `/mnt/nas2/lewm/datasets/`;
  derived data goes in the run dir.

## 4. Adding code — contract-first, test-driven, no cheating

Engineering quality (**MUST**): modular, reusable, DRY, concise. Small single-purpose
functions, clear names, type hints, short docstrings. Prefer extending shared utilities
over copy-paste. No dead code, no commented-out blocks.

Development workflow (**MUST**), in order:
1. **Contract first.** Before implementing, write a precise I/O contract for each new
   unit: signature; input types/shapes/ranges/preconditions; output type/shape/semantics;
   errors; invariants. The contract is the **only** interface between the roles below.
2. **Tests from the contract.** A *test author* writes diverse cases from the contract
   alone — typical, boundary, degenerate, error, and invariant/property cases. Tests live
   under `tests/` and are owned by the test author.
3. **Implement blind.** The *implementer* works **only** from the contract and **MUST NOT
   read the tests**. Communication is strictly through the declared API.
4. **Verify.** Run the tests. On failure the implementer fixes the **implementation**,
   guided by the contract and the failing input→output — never by reading or altering the
   test internals.

Anti-cheating invariants (**MUST**):
- The implementer **never** reads, edits, deletes, weakens, skips, or special-cases tests,
  and **never** hardcodes expected outputs or branches on test detection.
- **When an API changes:** update the *contract* first, then the *test author* updates the
  tests to the new contract, then the implementer adapts the implementation. The
  implementer **never** edits tests to make them pass.
- Tests assert contract/behavior, not implementation details.
- In multi-agent runs, use **separate agents** for test-authoring and implementation; they
  share only the contract.

## 5. Resource usage — shared machine (never disrupt others)

GPU (**MUST**):
- Check `nvidia-smi` first. Only use a GPU that is **empty** (no other process / free
  memory). If none is free, **wait** — do not squeeze onto an occupied card.
- Pin to your GPU with `CUDA_VISIBLE_DEVICES=<id>`; cap memory when the framework allows;
  request only what you need.
- **Never** kill, preempt, or crowd another user's job. Clean up your own processes
  (no leftover/zombie GPU memory) when a run ends or aborts.

CPU / memory (**MUST**):
- **Never** use all cores. Cap threads and dataloader workers to a modest share and leave
  headroom (rule of thumb: `≤ cores − 2`, or a set fraction). Set
  `OMP_NUM_THREADS` / `torch.set_num_threads` / `num_workers` explicitly — don't rely on
  defaults that grab everything.
- `nice`/`ionice` heavy or long jobs. Don't exhaust system RAM.

Disk & network (**MUST**):
- Keep large data on the NAS; keep the home disk free. Delete or archive aborted-run junk.
- Don't hammer external services; cache downloads under `/mnt/nas2/lewm/`.

## 6. Lifecycle & hygiene

- **SHOULD** scope each run to one clear hypothesis/change, with intent recorded up front
  and outcome recorded after.
- **SHOULD** launch long runs detached/backgrounded, fully logged, and checkpoint
  periodically so they're resumable.
- **MUST** fail loud: surface and log errors, don't silently swallow them.
- **MUST NOT** touch, move, or delete other users' files or runs. Archive, don't delete,
  your own completed runs.
