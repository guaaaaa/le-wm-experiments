# Memory-Maze experiment package — contracts

## convert_npz_to_h5.convert
```python
convert(src_files: list[Path] | list[str], out_path: Path | str, *,
        chunk_frames: int = 16, zstd_level: int = 3) -> dict
```
Converts memory-maze npz episodes into ONE swm-schema HDF5 file.

**Input npz keys (per episode, L rows each; L may vary per episode):**
`image` (L,H,W,3) uint8 · `action` (L,A) float (one-hot; row 0 is the all-zero reset
placeholder) · `agent_pos` (L,2) · `agent_dir` (L,2) · `targets_pos` (L,K,2) ·
`targets_vec` (L,K,2) · `target_pos` (L,2) · `target_color` (L,3) ·
`maze_layout` (L,M,M) · `reward` (L,)

**Output h5 datasets (episodes concatenated along axis 0, N = Σ L):**
| name | shape | dtype | notes |
|---|---|---|---|
| `ep_len` | (E,) | int32 | per-episode row count |
| `ep_offset` | (E,) | int64 | `[0, cumsum(ep_len)[:-1]]` |
| `pixels` | (N,H,W,3) | uint8 | renamed from `image`; chunked `(chunk_frames,H,W,3)`, Zstd(zstd_level) |
| `action` | (N,A) | float32 | **SHIFTED**: out row `o+t` = npz `action[t+1]` for t < L-1; **all-zeros at t = L-1**. (npz action[t] leads INTO obs t; LeWM pairs the action DEPARTING obs t.) |
| `agent_pos`,`agent_dir` | (N,2) | float32 | unshifted probe labels |
| `targets_pos`,`targets_vec` | (N,K,2) | float32 | unshifted |
| `target_pos` | (N,2) | float32 | unshifted |
| `target_color` | (N,3) | float32 | unshifted |
| `maze_layout` | (N,M,M) | uint8 | unshifted |
| `reward` | (N,) | float32 | unshifted |

All non-pixels datasets Zstd-compressed too (any chunking). Episode order in the file =
the order of `src_files` as given. Returns `{'episodes': E, 'rows': N, 'out': str(out_path)}`.
Raises `ValueError` on an episode missing a required key or with inconsistent row counts
across keys.

**CLI:** `python -m experiments.memory_maze.convert_npz_to_h5 --src DIR --out FILE
[--limit N] [--pattern '*.npz']` — files discovered as `sorted(DIR.glob(pattern))`,
truncated to `--limit` if given.

**Reader-compatibility guarantee:** the file must open with
`swm.data.HDF5Dataset(path=out_path, num_steps=W, frameskip=fs, keys_to_load=['pixels','action'])`
such that `lengths` == ep_len, `get_dim('action') == A`, and `dataset[0]` yields
`pixels` of shape (W, 3, H, W_img) float and `action` of shape (W, fs*A).
