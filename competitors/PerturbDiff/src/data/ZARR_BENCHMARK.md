# `src/data/` — the main adaptation surface

Companion to [../../ZARR_BENCHMARK.md](../../ZARR_BENCHMARK.md). Subfolders have their own
reports: [dataset/](dataset/ZARR_BENCHMARK.md), [data_module/](data_module/ZARR_BENCHMARK.md).

## Files in this folder

> **Phase 1 is implemented.** `zarr_metadata_cache.py`, `zarr_store.py` and
> `dataset/dataset_io_zarr.py` now exist and are wired in behind `data.source_format="zarr"`.
> The sections below describe both the original analysis and what was built.

| File | What it does | Zarr impact |
|---|---|---|
| `metadata_cache.py` | `H5MetadataCache` + `GlobalH5MetadataCache` singleton: reads obs categories/codes once per file | ✅ `get_zarr_cache()` added; see `zarr_metadata_cache.py` |
| `zarr_metadata_cache.py` | **new** — `ZarrMetadataCache` from `obs.parquet` + STATE's `encode_obs_pert` | — |
| `zarr_store.py` | **new** — `ZarrStore` over `X.zarr`/`X.npy`, process-global array cache | — |
| `file_handle.py` | `H5Store`: process-local, LRU-capped `h5py.File` handle cache | ✅ unchanged; `ZarrStore` is the sibling |
| `sampler.py` | `CellSetBatchSampler` / `DistributedCellSetFixPairingBatchSampler`: builds fixed-size cell sets with padding | ✅ arity-tolerant keys, `original_total`, empty-split guard |
| `split_strategy.py` | `split_pbmc`, `split_tahoe100m`, `split_cellxgene`: name-list holdouts | ✅ `split_index_json` added |

---

## 1. `metadata_cache.py` — the duck-typed contract

`H5MetadataCache.__init__` (lines 34-97) opens the h5ad and materializes:

```python
cache.pert_categories        # np.ndarray[str], the categorical levels
cache.pert_codes             # np.ndarray[int32], per-cell code into categories
cache.cell_type_categories   # (read from cell_line_key for Perturb-seq)
cache.cell_type_codes
cache.batch_categories       # from obs[batch_col]/categories, or numeric fallback
cache.batch_codes
cache.n_cells
cache.control_pert_code      # index of control_pert within pert_categories
cache.control_mask           # bool array
```

`GlobalH5MetadataCache` (lines 113-150) is a `Singleton` keyed **only by file path**, and also
carries the global `pert_dict` / `batch_dict` / `cell_type_dict` registered by the datamodule
(lines 123-132).

**Everything downstream reads only these attributes.** `dataset_grouping.py`, `split_strategy.py`,
`sampler.py`, `dataset_core.get_covarates` never touch h5py. Reproducing this object from parquet
is what unlocks the whole pipeline.

### Gotchas

* Line 74: `assert "replogle" in h5_path` in the non-categorical batch branch — a
  dataset-specific escape hatch that will fire on unexpected paths.
* Lines 76-79 assume numeric batch labels are exactly `1..N`.
* The singleton is keyed by path only, so the *first* caller's `pert_col`/`batch_col` wins.
  If a zarr cache is registered under the same key with different columns, the second call
  silently returns the first object (line 146).

### Built — `src/data/zarr_metadata_cache.py`

```python
class ZarrMetadataCache:
    def __init__(self, data_dir, *, cell_type_key="cell_type",
                 batch_col="batch_center", control_pert="empty"):
        obs = pl.read_parquet(Path(data_dir) / "obs.parquet")
        pert_str = [encode_obs_pert(g, m, negative_control=nc) for ...]   # STATE's rule
        pert_str[is_control] = control_pert
        # factorize into sorted categories -> categories + codes, for all three covariates
```

* Port `encode_pert` / `encode_mol_pert` / `encode_obs_pert` verbatim from
  `../../../STATE/state/src/state/tx/data/zarr_perturbation.py:62-120` so the perturbation
  strings are byte-identical to STATE's (this is what makes the two runs comparable).
* Sort categories so codes are deterministic — matches `generate_onehot_map`
  (`zarr_perturbation.py:123-127`) and PerturbDiff's own `sorted(...)` dict construction
  (`data_module_setup.py:143-145`).
* `control_pert_code = categories.index("empty")`, `control_mask = obs["negative_control"]`.
* Expose the **same attribute names**; do not subclass `H5MetadataCache` (its `__init__` opens
  a file unconditionally).
* Register through `GlobalH5MetadataCache.get_cache` behind a `source_format` argument, or add
  a `get_zarr_cache` method — but keep the singleton's `_cache` dict, since
  `dataset_core.py:127, 164` and `sampling_generation_helpers.py:102` look up
  `meta_cache._cache[dataset_path_map[ds_name]]` by path.

---

## 2. `file_handle.py` — `H5Store`

Lines 23-88. Per-process LRU of open `h5py.File` handles, re-opened after a fork
(`_ensure_process_local`, lines 44-51), never pickled into workers (`__getstate__`, lines 102-107).
`max_open` comes from `data_args.max_open_files`.

### Built — `src/data/zarr_store.py`

Same three-method contract (`dataset_file(name)`, `close_all()`, `__getstate__`) so
`dataset_core.py:306, 200, 368` need no change. The returned "handle" should be a small object
holding the materialized matrix:

* `feature_loading="ram"` → `np.load(X.npy)` if present else `np.array(zarr.open(X.zarr))`
* `feature_loading="npy"` → `np.load(X.npy, mmap_mode="r")` (requires the file)
* `feature_loading="zarr"` → `zarr.open(X.zarr, mode="r")` (lazy, slowest)

Mirror `_Source._ensure_open` (`zarr_perturbation.py:278-329`). Critically, **preload in the main
process** for `ram`/`npy` so forked workers share pages copy-on-write — STATE does this explicitly
at `zarr_perturbation.py:804-808`, and skipping it makes every worker re-read the matrix.

Note `zarr.open` objects are not fork-safe to share; keep them out of `__getstate__` exactly as
`_Source.__getstate__` does (`zarr_perturbation.py:331-336`).

---

## 3. `sampler.py` — cell sets (already the right shape)

`DistributedCellSetFixPairingBatchSampler._build_blocks` (lines 68-135):

* iterates `dataset.grouped_pert_data_indices[ds_name]`, whose keys are
  `(pert_code, celltype_code, batch_code)` for Perturb-seq
* pads each group up to a multiple of `use_cell_set` by resampling **with replacement**
  (lines 106-119), recording an `is_padded` flag consumed by the collate and masked out at
  sampling time (`apps/sampling/sampling_generation.py:123, 135-138`)
* yields `(ds_name, local_idx, is_padded)` triplets; `collate_fn` reshapes to `[B, S, G]`

This is functionally STATE's `SentenceBatchSampler` (`zarr_perturbation.py:514-596`):
group by covariates → fixed-length sets → replacement-fill partial sets. `use_cell_set` ≡
`cell_sentence_len`.

> 🚩 **The grouping keys differ, and on this data it matters enormously.** PerturbDiff groups by
> `(pert, cell_type, batch)`; STATE groups sentences by `(cell_type, pert)`. On hepg2 the 3-key
> grouping yields 46,331 groups with a **median of 1 cell** (max 31), so at upstream's own
> `use_cell_set=32` every group under-fills, ~94% of rows are duplicates and the epoch inflates
> 16x. The 2-key grouping yields 1,340 groups, median 53, 1.2x. `is_padded_list` is never
> consulted by the training loss, so those duplicates are trained on as if real.
>
> ✅ Now selectable via `data.cell_set_group_by` (default unchanged) and measured by the
> benchmark A/B — see [../../ZARR_BENCHMARK.md](../../ZARR_BENCHMARK.md) §4 and §8, and §5 below.

### Two edits

* Line 95-99: the `control_type[ds_name] is None` branch raises `NotImplementedError` unless the
  name contains `"cellxgene"`. Only relevant if you add an unperturbed source — the zarr
  datasets all have controls, so they take the `else` branch.
* The `shuffle=False` branch of `__iter__` used to log *"still shuffling within batches for
  validation … do not use this sampler for generation"* — which was backwards: that branch is
  exactly the deterministic one. Block order is `range(n_block)` and the replacement-padding is
  drawn from the seeded RNG in `_build_blocks`, so the sequence is reproducible. The message is
  now an accurate `debug` line. Verified empirically: two sampling runs of one checkpoint produce
  bit-identical predictions and identical obs ordering.
* `_build_blocks` now also stores `self.original_total`, the real cell count before
  replacement-padding, which generation uses to confirm it covered the whole split — see
  [../apps/sampling/ZARR_BENCHMARK.md](../apps/sampling/ZARR_BENCHMARK.md).

Also note line 75: `assert not isinstance(use_cell_set, DictConfig)` — `use_cell_set` must be a
plain int, not per-dataset.

---

## 4. `split_strategy.py` — name lists vs. index JSON

All three functions take `(dm, cache, target_type, key_info, holdout_setname, random_setname)`
and return `{"train": idx, "validation": idx, "test": idx}`.

* `split_pbmc` (6-105): holds out `(holdout_batches × holdout_pert[split])` cells.
* `split_tahoe100m` (107-206): holds out `(holdout_celltype × holdout_pert[split])` cells.
  This is the one the replogle config uses (`holdout_celltype: ["hepg2"]`).
* `split_cellxgene` (208-280): batch/cell-type holdout for unperturbed RNA-seq.

All three read perturbation **names** out of YAML lists and, unless `split_control` is set,
append **all** control indices to every split (lines 101-105, 202-206) — which happens to be the
same convention as STATE's zarr loader.

### Built — `split_index_json`

```python
def split_index_json(dm, cache, target_type, key_info, holdout_setname=None, random_setname=None):
    fold = json.load(open(key_info["split_file"]))["folds"][key_info["split_fold"]]
    test_set, val_set, ft_set = map(set, (fold["test"], fold["validation"], fold["finetune"]))
    pert_rows = np.flatnonzero(~cache.control_mask)
    ctrl_rows = np.flatnonzero(cache.control_mask)
    out = {}
    for name, keep in (("train", ft_set), ("validation", val_set), ("test", test_set)):
        sel = pert_rows[np.isin(pert_rows, list(keep))]
        out[name] = np.concatenate([sel, ctrl_rows]) if len(sel) else np.array([], np.int64)
    return out
```

Match STATE (`zarr_perturbation.py:774-788`) on three points:

1. `finetune` → **train** (not a separate finetune split).
2. Controls are appended to **every non-empty** split; an empty split stays empty.
3. STATE loops per cell type; with one cell type per zarr directory this is equivalent, but keep
   the loop if you later merge cell lines into one directory.

If a dataset has **no** split entry and is marked `[training] = "train"`, return everything as
train (jurkat/k562/rpe1 in the current manifest).

**Empty validation.** Fold 0 of `replogle_nadig_xcell.json` has `validation: []`. `dm.all_split_names`
is `["validation", "test"]` (`data_module.py:182`) and both dataloaders are built unconditionally
(`data_module.py:87-89` asserts the attribute exists). An empty dataset yields zero batches; the
`ModelCheckpoint` monitor `validation_validation_loss_epoch` would then never fire. Decide up
front: monitor the test split, carve a validation slice from `finetune`, or use step-based
checkpointing only.

---

## 5. Implemented: `cell_set_group_by` and a safe grouping cache

Two things landed in `dataset/dataset_grouping.py` beyond the zarr read path.

**`data.cell_set_group_by`** — upstream always groups Perturb-seq cell sets by
`(pert, cell_type, batch)` with no switch. That is degenerate on this data (median 1 cell/group,
max 31, so any `use_cell_set >= 32` under-fills every group). The option adds
`pert_celltype`, STATE's grouping, without changing the default. Both are exercised by the
benchmark A/B; see the root plan §4 and §8.

**Cache identity and concurrency** — the grouping pickle used to be keyed `{ds_name}_{stage}`,
which does not change when the split file, fold, `keep_control_cell` or grouping mode change. Since
`keep_control_cell` alters the *index space* the cached positions refer to, stale reuse produced
wrong cell sets rather than an error (it bit a test during development). The filename now carries
a content hash of all of those, and writes go through a temp file + `os.replace()` so several jobs
starting at once cannot produce a torn read. Duplicated compute across concurrent jobs remains —
wasted CPU, not corruption.
