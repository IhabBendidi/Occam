# `src/data/data_module/` — LightningDataModules and setup

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Role |
|---|---|
| `data_module.py` | dataloader builders + the datamodule class hierarchy |
| `data_module_setup.py` | the actual `setup()` / `setup_dataset()` logic, extracted as free functions |

---

## 1. `data_module.py`

### Dataloaders

* `build_train_dataloader` (39-75): `DistributedCellSetFixPairingBatchSampler` when DDP is
  initialized **and** `use_fixed_pairing`, else `CellSetBatchSampler`. `drop_last=True`.
* `build_val_dataloaders` (78-120): one loader **per name in `dm.all_split_names`**, in order.
  `drop_last=False`.

Consequence used by the sampling entrypoint: with `all_split_names = ["validation", "test"]`,
`val_dataloader()[1]` is the **test** loader (`apps/sampling/sampling_generation.py:45`).

### Class hierarchy

```
PretrainingDataModule            (123-156)  train/val loaders + pretraining_setup*
└── PBMCPerturbationDataModule   (158-238)  __init__; all_split_names=["validation","test"];
    │                                       get_dataset_names_and_paths; split_pbmc
    └── Tahoe100mPerturbationDataModule (241-253)  swaps in split_tahoe100m
        └── CellxGeneDataModule          (256-263)  adds split_cellxgene
            └── PerturbationPretrainingDataModule (267-301)  multi-source; pert-dict rebuild
```

Only `PBMCPerturbationDataModule` defines `__init__` (168-182): `seed`, `micro_batch_size`,
`data_args`, `py_logger`.

### `get_dataset_names_and_paths` (184-226)

Returns five parallel lists: `dataset_name_list, file_list, data_type_list, key_info_list,
selected_genes_list`. `key_info` is a dict of the column names + holdout definitions pulled from
the data config. When `cfg.dataset_name is None` it globs `*.h5ad` and **asserts the file count**
(23 for cellxgene, 14 for tahoe100m — lines 215-222).

`selected_genes` is loaded here from a **pickle** (205-209).

---

## 2. `data_module_setup.py`

Two phases, both duplicated across a "pretraining" and a "perturbation pretraining" variant:

### `setup()` — build the global covariate vocabularies

`pretraining_setup` (89-153) / `perturbation_pretraining_setup` (409-553):

* opens **every** file with `h5py` and reads `obs/<col>/categories` (113-141, 431-475)
* builds `pert_dict`, `batch_dict`, `cell_type_dict` as `{name: i}` over the **sorted** union
  (143-145, 477-479)
* batch names are namespaced `f"{dataset_name}_{batch}"` (140, 451, 474)
* applies the `sample_*_only` / `skip_*` filters (489-549) — pure substring matching on the
  dataset name
* optionally rebuilds `pert_dict` from scratch when `replace_pert_dict` (551-553)

### `setup_dataset()` — build the split indices and the datasets

`perturbation_pretraining_setup_dataset` (156-310):

* `control_type[ds] = None if RNA-seq else key_info["control_pert"]` (175-178)
* dispatches to a split strategy by **substring of the path**: `"pbmc" in path` →
  `split_pbmc`, `"tahoe100m"/"replogle" in path` → `split_tahoe100m` (220-229)
* caches the per-split index arrays as pickles under `indices_cache_dir` (242-254), only for
  `data_name == "Tahoe100mPBMCReplogleCellxGenePretrain"` (181)
* constructs one `H5adSentenceDataset` per split (294-310)

---

> **Phase 2 is implemented.** `ZarrPerturbationDataModule` exists in `data_module.py`, backed by
> `zarr_setup` / `zarr_setup_dataset` in `data_module_setup.py` and `split_index_json` in
> `../split_strategy.py`. Verified over all four cell lines; see
> [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md) §4. The section below is the original
> design note, which the implementation follows.

## 3. Plan: a `ZarrPerturbationDataModule`

Subclass `PBMCPerturbationDataModule` (it already has the right `__init__` and
`all_split_names`) and override three things.

### 3.1 `get_dataset_names_and_paths` — read the STATE TOML

Use **the same manifest format** so the two models provably see the same data:

```toml
[datasets]
ihabxcell_hepg2  = "/rxrx/scratch/hooke_tx/public/ihabxcell_replogle_nadig_hepg2"
ihabxcell_jurkat = "/rxrx/scratch/hooke_tx/public/ihabxcell_replogle_nadig_jurkat"
ihabxcell_k562   = "/rxrx/scratch/hooke_tx/public/ihabxcell_replogle_nadig_k562"
ihabxcell_rpe1   = "/rxrx/scratch/hooke_tx/public/ihabxcell_replogle_nadig_rpe1"

[training]
ihabxcell_hepg2 = "train"   # (and the other three)

[splits]
ihabxcell_hepg2 = "/rxrx/data/valence/external_benchmarking/cell_eval/splits/replogle_nadig_xcell.json"

[split_fold]
ihabxcell_hepg2 = 0
```

Return, per dataset: name, **directory** path (not a file), `"Perturb-seq"`, and a `key_info`
carrying `cell_type_key="cell_type"`, `batch_col="batch_center"`, `control_pert="empty"`,
`split_file`, `split_fold`, `is_training`.

`tomllib` is stdlib on Python 3.11 (which you need anyway for zarr v3); on 3.9 use `tomli`.

The dataset **names must keep the substring `replogle`** so `get_short_dsname`
(`src/common/utils.py:34-45`) resolves them — `ihabxcell_replogle_nadig_hepg2` already does.
Using a bare name like `hepg2` would hit its `assert 0`.

### 3.2 `setup()` — vocabularies from parquet

Replace the h5py loop with `ZarrMetadataCache` construction per dataset, then:

```python
all_perts.update(cache.pert_categories)                       # Ensembl ids + "empty"
all_celltypes.update(cache.cell_type_categories)              # "HepG2", ...
all_batches.update(f"{ds_name}_{b}" for b in cache.batch_categories)
```

Keep the `f"{ds}_{batch}"` namespacing — `dataset_core.get_covarates:186` and
`sampling_generation_helpers.py:111` both reconstruct the key that way. (`batch_center` is already
globally unique, so this is redundant but must stay *consistent*.)

Scale check for the full manifest: ~2000 perts × 4 cell types × (60-ish batches each) — the
one-hot `nn.Embedding` tables are fine, but see the grouping cost note in
[../dataset/ZARR_BENCHMARK.md](../dataset/ZARR_BENCHMARK.md).

### 3.3 `setup_dataset()` — index-JSON splits

Copy `perturbation_pretraining_setup_dataset` and replace the path-substring dispatch (220-229)
with a single call to `split_index_json` (see [../ZARR_BENCHMARK.md](../ZARR_BENCHMARK.md) §4).
Datasets without a `[splits]` entry return everything as train.

Then build `H5adSentenceDataset` as today — with `data_args.source_format = "zarr"` so the
dataset's IO methods dispatch to the zarr readers.

### 3.4 Register it

```python
# src/apps/training/training_datamodule_builder.py:13-39   (currently ends in `assert 0`)
elif cfg.data.data_name in ["IhabxcellZarrFinetune"]:
    datamodule = data_module.ZarrPerturbationDataModule(...)
# and the same in src/apps/sampling/sampling_setup.py:9-57
```

### 3.5 Empty-validation handling

Fold 0 has `validation: []`. `build_val_dataloaders` (78-120) still builds a loader for it, which
yields nothing, so `validation_validation_loss_epoch` — the `ModelCheckpoint` monitor in
`configs/lightning/base.yaml` — is never logged. Pick one:

* monitor `validation_test_loss_epoch` instead, or
* hold out a slice of `finetune` as validation (deviates from STATE's split), or
* drop the monitor and checkpoint purely on `every_n_train_steps` (closest to STATE's
  `ckpt_every_n_steps=2000`).

STATE handles this by falling back to the test set inside `val_dataloader()`
(`../../../../STATE/state/src/state/tx/data/zarr_perturbation.py:853-858`).
