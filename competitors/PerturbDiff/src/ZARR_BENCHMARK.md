# `src/` — package map and the zarr adaptation surface

Companion to [../ZARR_BENCHMARK.md](../ZARR_BENCHMARK.md). This file covers the top level of
`src/`; each subpackage has its own `ZARR_BENCHMARK.md`.

## Layout

```
src/
├── __init__.py          empty marker
├── apps/                entrypoints + train/sample orchestration   -> apps/ZARR_BENCHMARK.md
├── data/                datasets, samplers, splits, metadata       -> data/ZARR_BENCHMARK.md
├── models/              Cross_DiT, diffusion, lightning module     -> models/ZARR_BENCHMARK.md
└── common/              utils + path constants                     -> common/ZARR_BENCHMARK.md
```

All imports are absolute from the repo root (`from src.…`). Entrypoints prepend the repo root to
`sys.path` themselves (`apps/run/rawdata_diffusion_training.py:11-12`), which is why the project
is not installed as a package — see `../pyproject.toml` (`[tool.uv] package = false`).

Hydra `_target_` strings in `configs/` also resolve against this `src.` root:

* `src.models.lightning.lightning_module.PlModel`
* `src.data.data_module.data_module.PretrainingDataModule`
* `src.common.utils.get_cosine_schedule_with_warmup`

Renaming or moving any of these modules breaks the configs silently at instantiation time.

## Where the data format is actually pinned

Only a handful of places know that the data is HDF5. This is the whole adaptation surface:

| Location | What it assumes |
|---|---|
| `data/metadata_cache.py:52-97` | `h5py.File`, `obs/<col>/{categories,codes}` pandas-categorical layout |
| `data/dataset/dataset_io.py:72-119` | `/X` CSR triplet or dense `X[row]` |
| `data/dataset/dataset_io.py:10-69` | `obsm/<embed_key>` dense or CSR |
| `data/dataset/dataset_grouping.py:229-244` | gene names from `/var/_index` (+4 fallbacks) |
| `data/file_handle.py:53-58` | `h5py.File(..., driver="sec2")` handle cache |
| `data/data_module/data_module_setup.py:113-141, 431-475` | opens every file with `h5py` to enumerate categories |
| `apps/sampling/sampling_generation_helpers.py:118-135` | `anndata.read_h5ad` for control cells |

Everything else — grouping, sampling, splitting, collating, the model, the diffusion process —
operates on numpy/torch and on the duck-typed `cache.*_codes` / `cache.*_categories` attributes.
That is the leverage point: reimplement the seven rows above for zarr/parquet and the rest of the
package is format-agnostic.

## Data flow in one pass

```
configs (hydra)
   ↓
apps/run/rawdata_diffusion_training.py
   ↓ build_datamodule                      apps/training/training_datamodule_builder.py
   ↓
data/data_module/data_module.py            setup(): enumerate categories -> pert/celltype/batch dicts
   ↓ setup_dataset()                       split_strategy.py -> per-split row indices
   ↓
data/dataset/dataset_core.py               H5adSentenceDataset
   ├─ dataset_grouping.split_out_control       control vs perturbed
   ├─ dataset_grouping.register_mapping_indices  (celltype,batch) control pools + (pert,celltype,batch) cell sets
   └─ dataset_io.retrieve_counts               expression row -> selected gene order
   ↓ CellSetBatchSampler                   data/sampler.py  (cell sets + padding mask)
   ↓ collate_fn                            dataset_core.py:36-118  -> reshape to [B, S, G]
   ↓
models/lightning/lightning_module.PlModel  cov encoding -> diffusion.training_losses -> Cross_DiT
```

The sampling entrypoint reuses the same datamodule and takes `val_dataloader()[1]`
(the *test* split) — see `apps/sampling/ZARR_BENCHMARK.md`.

## What was added (the h5ad path is untouched)

All of it is additive, switched by `data.source_format` / `data.data_name`:

```
src/data/zarr_metadata_cache.py     ZarrMetadataCache (duck-typed like H5MetadataCache)
src/data/zarr_store.py              ZarrStore (same contract as H5Store)
src/data/dataset/dataset_io_zarr.py row fetch for X.zarr / X.npy + var.parquet gene panel
src/common/gene_id_map.py           Ensembl <-> symbol, for the symbol-keyed embeddings
src/apps/sampling/sampling_eval_celleval.py   cell-eval scoring matched to STATE
```

plus, in existing files: `split_strategy.split_index_json`,
`data_module.ZarrPerturbationDataModule` (+ `zarr_setup` / `zarr_setup_dataset`), the
`data.cell_set_group_by` switch in `dataset_grouping`, three dispatch points in
`H5adSentenceDataset` (store, `_retrieve_counts`, `_get_selected_gene_vars`) and the `data_name`
dispatch in the two builders.

The h5ad path is regression-checked after every change: identical validation loss to 16 digits
(`23.42843246459961`) and 56/56 modules importing.
