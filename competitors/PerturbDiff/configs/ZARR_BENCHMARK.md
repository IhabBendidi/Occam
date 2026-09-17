# `configs/` — the Hydra tree and the new zarr configs

Companion to [../ZARR_BENCHMARK.md](../ZARR_BENCHMARK.md). This file covers **all** config
subfolders (`data/`, `path/`, `model/`, `trainer/`, `lightning/`, `optimization/`,
`cov_encoding/`) rather than scattering one-line stubs into each.

## Tree

```
configs/
├── rawdata_diffusion_training.yaml     top level: defaults + run_name + profiler
├── rawdata_diffusion_sampling.yaml     top level: defaults + model_checkpoint_path + sampling.*
├── trainer/base.yaml                   expanded verbatim into pl.Trainer(**kwargs)
├── lightning/base.yaml                 callbacks (checkpoint/lr/progress), EMA, WandbLogger
├── model/base.yaml                     Cross_DiT + diffusion hyperparameters
├── optimization/base.yaml              micro_batch_size, seed, loss switches
│   ├── optimizer/adam.yaml
│   └── scheduler/cosine.yaml
├── cov_encoding/trixie_onehot.yaml     encoder selection + embedding pickle paths
├── path/trixie_path.yaml               all filesystem roots
└── data/
    ├── pbmc_finetune.yaml   tahoe100m_finetune.yaml   replogle_finetune.yaml
    ├── tahoe100m_pbmc_replogle_pretrain_cellxgene.yaml
    ├── perturb_data/{pbmc,tahoe100m,replogle}.yaml     per-source metadata
    └── rna_data/cellxgene.yaml
```

Default composition (both entrypoints): `trainer: base`, `lightning: base`, `optimization: base`,
`path: trixie_path`, `model: base`, `cov_encoding: trixie_onehot`; `data:` differs
(`pbmc` for training — note that file does **not** exist, so `data=` must always be passed on the
CLI; `pbmc_finetune` for sampling).

---

## What each group pins today

**`path/trixie_path.yaml`** — every root is a `"/path/to/…"` placeholder. Defines
`wandb.logging_dir`, `diffusion.save_dir`, `tmp_dir`, per-dataset dirs, `selected_gene_dir`,
`indices_cache_dir`, `gene_mapping_file`, and the three `*_ctrl_h5ad` files used by sampling.

**`data/replogle_finetune.yaml`** (the closest analogue to the benchmark) —
`data_name: ReplogleFinetune`; `pert_col: gene`; `perturbseq_batch_col: gem_group`;
`cell_line_key: cell_line`; `control_pert: "non-targeting"`; `embed_key: X_hvg`;
`pad_length: 2000`; `use_cell_set: 16`; `normalize_counts: false`;
`holdout_celltype: ["hepg2"]`; and **~100 validation + ~700 test gene symbols inlined**
in `data/perturb_data/replogle.yaml`.

That inlined name-list is exactly what the index-JSON split replaces.

**`cov_encoding/trixie_onehot.yaml`** — selects the encoder branches and points at five pickles
(gene embeddings ×4, pert/celltype/drug/replogle-gene embeddings). Defaults here are
`celltype_encoding: llm` and `replogle_gene_encoding: onehot`, but the README's run blocks
override `replogle_gene_encoding=genept` for Replogle.

**`model/base.yaml`** — `input_dim: 2000`, `hidden_num: [2000,512]`, `steps: 1000`,
`predict_xstart: True`, `separate_embedder: false`, plus the finetune layer-replacement switches.

**`lightning/base.yaml`** — `ModelCheckpoint` monitors **`validation_validation_loss_epoch`**
(a problem with the empty validation fold), `WandbLogger` by default.

**`trainer/base.yaml`** — `devices: -1`, `precision: 32-true`, `max_steps: 1560300`,
`deterministic: true`, `reload_dataloaders_every_n_epochs: 1`.

---

> **All of these now exist** and are verified by end-to-end runs on the real zarr data. The text
> below is kept as design rationale; where a shipped file differs from the sketch, the shipped file
> wins — notably `use_cell_set: 32` (upstream's Replogle value, not the 64 proposed here) and the
> added `cell_set_group_by` switch.
>
> Shipped for the benchmark:
>
> | file | what |
> |---|---|
> | `path/hooke_tx_path.yaml` | zarr roots, writable `indices_cache_dir`, embedding assets |
> | `data/ihabxcell_zarr.yaml` | `IhabxcellZarrFinetune`, canonical obs columns, Replogle hyperparameters |
> | `cov_encoding/hooke_tx_zarr.yaml` | one-hot by default; `pert_key_type: ensembl`, `control_pert_key: empty` for genept/llm |
> | `manifests/ihabxcell_hepg2_zarr.toml` | copy of STATE's manifest, verified identical modulo comments |
> | `metadata/gene_id_map.csv` | mygene Ensembl↔symbol, 53,653 pairs |
> | `metadata/vocab_id2name.csv` | upstream's id→name table, 25,133 rows |
> | `rawdata_diffusion_sampling.yaml` | `num_sampled_batches: null`, `run_cell_eval`, `cell_eval_profile` |

## New config 1 — `configs/path/hooke_tx_path.yaml`

```yaml
wandb:
  logging_dir: "/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_wandb"
diffusion:
  save_dir: "/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/rawdata_diffusion_model"

# hooke-tx canonical zarr root
tmp_dir: "/rxrx/scratch/hooke_tx/public"

# writable scratch for the grouping pickles (must NOT be the read-only data root)
indices_cache_dir: "/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/indices_cache/"

# benchmark assets
selected_gene_file: "/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/configs/metadata/replogle_nadig_selected_ens_ids.txt"
split_file: "/rxrx/data/valence/external_benchmarking/cell_eval/splits/replogle_nadig_xcell.json"
```

`indices_cache_dir` **must be writable** — `dataset_grouping.register_mapping_indices`
(`../src/data/dataset/dataset_grouping.py:189-192`) pickles into it unconditionally, and
`/rxrx/scratch/hooke_tx/public` is not yours to write to.

Keep `diffusion.save_dir` defined: `setup_loggings` derives `cfg.save_dir_path` from
`trainer.default_root_dir`, which resolves to it.

## New config 2 — `configs/manifests/ihabxcell_hepg2_zarr.toml`

Copy `../../STATE/state/ihabxcell_hepg2_zarr.toml` **verbatim**. Using literally the same file (or
a byte-identical copy) is the cheapest guarantee that the two models see the same datasets, the
same split file, and the same fold.

## New config 3 — `configs/data/ihabxcell_zarr.yaml`

```yaml
data_name: IhabxcellZarrFinetune

# --- native-zarr specifics (mirrors STATE's configs/data/perturbation_zarr.yaml) ---
source_format: zarr            # zarr | h5ad
feature_loading: ram           # ram | npy | zarr
toml_config_path: ${..path.manifest_file}
selected_gene_file: ${..path.selected_gene_file}   # 2000 Ensembl ids, one per line (.txt)
indices_cache_dir: ${..path.indices_cache_dir}
split_fold: 0

# --- canonical obs columns ---
pert_col: gene                 # derived from gene_pert; kept for the output adata's obs
cell_line_key: cell_type       # PerturbDiff reads cell types through cell_line_key
cell_type_key: cell_type
perturbseq_batch_col: batch_center
rnaseq_batch_col: null
control_pert: empty

# --- expression space ---
embed_key: X_hvg
output_space: gene
embed_shape: ${..model.input_dim}    # 2000
pad_length: 2000
normalize_counts: false        # X.zarr is already log1p(1e4)

# --- cell sets (STATE: cell_set_len=64) ---
use_cell_set: 64
optimizer_batch_size: ${..optimization.micro_batch_size}
use_fixed_pairing: true
basal_mapping_strategy: random
mapping_strategy: random
keep_control_cell: false
split_control: false

# --- dataloader ---
num_workers: 4
prefetch_factor: 16
persistent_workers: false
pin_memory: true
max_open_files: 16

select_strategy: "sum"
sort_gene_names: false
skip_cached_indices: false
```

Field-by-field correspondence with STATE's `data.kwargs`:

| STATE | PerturbDiff |
|---|---|
| `toml_config_path` | `data.toml_config_path` |
| `selected_ensembl_gene_ids_path` | `data.selected_gene_file` |
| `embed_key: X_hvg` | `data.embed_key: X_hvg` |
| `output_space: gene` | `data.output_space: gene` |
| `batch_col: batch_center` | `data.perturbseq_batch_col: batch_center` |
| `cell_type_key: cell_type` | `data.cell_line_key: cell_type` |
| `control_pert: empty` | `data.control_pert: empty` |
| `target_sum: null`, `log1p: false` | `data.normalize_counts: false` |
| `feature_loading: ram` | `data.feature_loading: ram` |
| `cell_sentence_len` / `model.kwargs.cell_set_len: 64` | `data.use_cell_set: 64` |
| `num_workers: 4` | `data.num_workers: 4` |

## Overrides to apply on the CLI

```bash
# neutralize the name-keyed embedding pickles for the first run
cov_encoding.celltype_encoding=onehot
cov_encoding.replogle_gene_encoding=onehot
cov_encoding.pert_encoding=onehot
cov_encoding.batch_encoding=onehot
model.use_gene_embedding=False

# STATE-matched cadence
trainer.max_steps=80000
lightning.callbacks.checkpoint.every_n_train_steps=2000

# empty validation fold -> do not monitor it
lightning.callbacks.checkpoint.monitor=validation_test_loss_epoch

# full test coverage at sampling time
sampling.num_sampled_batches=null
```

plus the README's `$NO_WANDB` block to swap `WandbLogger` for `DummyLogger`.

## Gotchas

* `configs/rawdata_diffusion_training.yaml` defaults to `data: pbmc`, and **no `data/pbmc.yaml`
  exists** — always pass `data=…` explicitly.
* `configs/.DS_Store` is committed; harmless.
* Hydra interpolations in `data/*.yaml` use relative refs (`${..path.…}`, `${...path.…}`); copy
  the exact depth from a sibling file rather than guessing.
* Changing `use_cell_set`, the split, or the covariate columns invalidates the pickles under
  `indices_cache_dir` — the filenames only encode `ds_name` + stage
  (`../src/data/dataset/dataset_grouping.py:146-147`). Clear the directory or set
  `data.skip_cached_indices=true`.
