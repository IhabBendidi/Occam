# PerturbDiff on the hooke-tx zarr benchmark — end-to-end plan

**Scope.** How to train and evaluate PerturbDiff on the same benchmark we already run STATE on
(`../STATE/state`), reading the **hooke-tx canonical zarr** datasets directly, and what has to
change in this repo to get there.

**Status:** Phases **0-5 are done and verified** — the training entrypoint runs end to end on the
real zarr data, and evaluation produces cell-eval CSVs with the same 27 metric columns as STATE's
(see §4 and §5).

Per-folder companion reports (each named `ZARR_BENCHMARK.md`) are indexed in [CLAUDE.md](CLAUDE.md).

---

## 1. The reference workflow (how STATE is used today)

Two scripts define the whole thing:

| File | Role |
|---|---|
| `../STATE/state/train_state_zarr.sh` | train |
| `../STATE/state/predict_state_zarr.sh` | predict + cell-eval |
| `../STATE/state/ihabxcell_hepg2_zarr.toml` | dataset/split manifest |
| `../STATE/state/src/state/tx/data/zarr_perturbation.py` | the native-zarr data module (the thing we must mirror) |
| `../STATE/state/src/state/configs/data/perturbation_zarr.yaml` | its config defaults |

### Train

```bash
uv run state tx train \
    data=perturbation_zarr \
    data.kwargs.toml_config_path="${STATE_DIR}/ihabxcell_hepg2_zarr.toml" \
    data.kwargs.selected_ensembl_gene_ids_path="${GENES}" \
    data.kwargs.num_workers=4 \
    data.kwargs.embed_key='X_hvg' \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="batch_center" \
    data.kwargs.cell_type_key="cell_type" \
    data.kwargs.control_pert="empty" \
    data.kwargs.target_sum=null \
    data.kwargs.log1p=false \
    data.kwargs.feature_loading=ram \
    training.max_steps=80000 training.ckpt_every_n_steps=2000 \
    training.batch_size=64 training.lr=1e-3 \
    model.kwargs.cell_set_len=64 model.kwargs.hidden_dim=336 \
    model.kwargs.batch_encoder=True model=state \
    output_dir="test_replogle_zarr" name="hepg2_holdout_ram"
```

`GENES = /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/configs/metadata/replogle_nadig_selected_ens_ids.txt`
(2000 Ensembl ids, one per line — verified).

### Evaluate

```bash
uv run state tx predict \
  --output-dir "test_replogle_zarr/hepg2_holdout" \
  --checkpoint "snapshot_step.ckpt"
```

Writes `eval_<ckpt>/{adata_pred.h5ad, adata_real.h5ad, <CellType>_results.csv, <CellType>_{pred,real}_de.csv}`.
Confirmed on disk at `../STATE/state/test_replogle_zarr/hepg2_holdout_ram/eval_last.ckpt/`.

---

## 2. The data contract we must satisfy

### 2.1 On-disk layout

`/rxrx/scratch/hooke_tx/public/ihabxcell_replogle_nadig_{hepg2,jurkat,k562,rpe1}/`

```
X.zarr/          zarr **format v3**, (96616, 2000) float32, chunks (1000, 2000)   [hepg2]
obs.parquet      canonical schema (below)
var.parquet      ensembl_gene_id, gene_symbol, seqid, start, end, strand, ...
obs_raw.parquet  source-specific columns (not used by the canonical path)
var_raw.parquet
X.npy            optional; present on some datasets (e.g. parse_pbmc) for mmap/RAM loading
```

`X` is **already `log1p(1e4)`-normalized** — hence `target_sum=null, log1p=false`.

### 2.2 `obs.parquet` canonical columns (verified on hepg2, 96616 rows)

```
cell_type            'HepG2' (single value here)
batch_center         'ihabxcell_replogle_nadig_hepg2_gem33', ...   <- the batch key
negative_control     bool; 4976 True / 91640 False
gene_pert            list[[kind, ensembl_id, direction]] e.g. [["genetic","ENSG00000163807","inhibition"]]
mol_pert             list (empty for this dataset)
experiment_label, well_address, assay_type, perturbations, unused,
positive_control, gene_only, mol_only, multi_pert
```

**Perturbation identity is derived, not stored as a string.** STATE's rule
(`zarr_perturbation.py:103-120`, `encode_obs_pert`):

* `negative_control == True` or no perts → `"empty"` (the control sentinel, `control_pert="empty"`)
* one genetic pert → its Ensembl id
* several → sorted ids joined with `"+"`
* gene+mol on the same cell → rejected

### 2.3 Splits are **index-based**

`/rxrx/data/valence/external_benchmarking/cell_eval/splits/replogle_nadig_xcell.json`

```json
{"folds": [{"finetune": [...], "validation": [...], "test": [...]}]}
```

Verified: 1 fold; `finetune` 69738 rows, `validation` 0 rows, `test` 26878 rows. These are **row
indices into `obs.parquet`**, not perturbation names.

STATE's split semantics (`zarr_perturbation.py:756-799`), per dataset **per cell type**:

* `finetune` → train, `validation` → val, `test` → test (perturbed rows only)
* **all control rows are added to every non-empty split** (cell_load semantics)
* datasets with no `[splits]` entry and `[training] = "train"` contribute *all* rows to train
  (that is why jurkat/k562/rpe1 are train-only in `ihabxcell_hepg2_zarr.toml`)

### 2.4 Batching

`SentenceBatchSampler` (`zarr_perturbation.py:514-596`): group cells by `(cell_type, pert)` codes
(plus `batch` when `basal_mapping_strategy="batch"`), cut into sentences of `cell_sentence_len`,
pack into flat meta-batches of `batch_size * cell_sentence_len`; partial sentences are filled by
sampling **with replacement**, except in test mode (`batch_size` forced to 1, no fill).

This maps almost 1:1 onto PerturbDiff's `CellSetBatchSampler` — see §4.4.

---

## 3. What PerturbDiff did before this work (and where it collided)

> Historical: this table is the *starting point*. Every row is addressed in §4; the code now
> supports both paths, switched by `data.source_format`.

| Concern | STATE zarr path | PerturbDiff today |
|---|---|---|
| Storage | `zarr.open(X.zarr)` / `np.load(X.npy)` | `h5py.File(...)`, `/X` CSR or `obsm/<embed_key>` (`src/data/dataset/dataset_io.py`) |
| Metadata | `polars.read_parquet(obs.parquet)` | `h5py` reads `obs/<col>/{categories,codes}` (`src/data/metadata_cache.py:52-97`) |
| Pert identity | derived from `gene_pert` → Ensembl / `"empty"` | a categorical **column value**, e.g. `"non-targeting"` (gene symbols) |
| Gene identity | `var.parquet.ensembl_gene_id` | gene **symbols** from `/var/_index` etc. (`dataset_grouping.py:233-244`), selected-gene **pickle** |
| Splits | index JSON, fold 0 | hard-coded **name lists** in YAML (`configs/data/perturb_data/replogle.yaml`) |
| Control rows | shared across splits, `negative_control` | `split_out_control` on a pert-category match (`dataset_grouping.py:12-48`) |
| Normalization | `target_sum`/`log1p` config | `normalize_counts` divisor in collate (`dataset_core.py:80-82`) |
| Eval | cell-eval `MetricsEvaluator` | R² + energy-MMD only (`sampling_generation.py:148-178`) |
| Python | 3.11 (`../STATE/state/.python-version`) | was 3.9; **now 3.11** (Phase 0) |

### 3.1 Two hard blockers *(resolved in Phase 0)*

1. **`X.zarr` is zarr format v3.** Its `zarr.json` uses `chunk_grid`/`codecs` — only
   zarr-python ≥ 3 reads it, and zarr-python 3 requires **Python ≥ 3.11**. The venv was
   Python 3.9.
2. **`tomllib` is stdlib only from 3.11.** STATE's loader uses it directly
   (`zarr_perturbation.py:41`). On 3.9 you would need `tomli`.

Both pointed the same direction, and Phase 0 acted on it: the venv is now Python 3.11.14 with
`zarr>=3`, `polars`, `pyarrow`, `cell-eval==0.7.2`, and both blockers are verified gone against
the real hepg2 dataset.

### 3.2 Lucky alignments worth exploiting

* `get_short_dsname` (`src/common/utils.py:34-45`) matches on substrings; the dataset names
  `ihabxcell_replogle_nadig_hepg2` contain `"replogle"`, so `separate_embedder=by_name`,
  the per-dataset loss keys, and the Cross_DiT embedder dict all work **unchanged**.
* The zarr hepg2 dataset has exactly **4976** `negative_control` rows, matching the existing
  `assert len(ctrl_adata) == 4976` for replogle/hepg2 non-targeting controls
  (`sampling_generation_helpers.py:125`) — same underlying cells.
* The gene panel is 2000 wide, matching PerturbDiff's `assert len(_genes) == 2000`
  (`sampling_generation.py:94`) and `pad_length: 2000`.
* PerturbDiff already concatenates **real** control rows into both `pred_adata` and
  `true_adata` (`sampling_generation.py:219-220`) — the same convention as STATE's
  `--inject-real-controls`.

---

## 4. Adaptation plan

Design rule: **do not rewrite the pipeline.** Everything downstream of the metadata cache
(`dataset_grouping`, `sampler`, `split_out_control`, `collate_fn`, model, diffusion) consumes a
small duck-typed surface:

```
cache.pert_categories / pert_codes
cache.cell_type_categories / cell_type_codes
cache.batch_categories / batch_codes
cache.n_cells, cache.control_pert_code, cache.control_mask
```

Provide that from parquet and most of the repo keeps working untouched.

### Phase 0 — environment ✅ DONE

* `.venv` rebuilt on **Python 3.11.14** (same interpreter as `../STATE/state`);
  `requires-python = ">=3.11,<3.13"`, `.python-version` pins 3.11.
* Added `zarr>=3`, `polars`, `pyarrow`, `cell-eval>=0.7.0`; re-locked (137 packages).
* Verified against the **real** benchmark data from this env: `X.zarr` (96616, 2000) reads,
  `obs`/`var.parquet` load, `tomllib` parses STATE's manifest, the split fold reports
  69738/0/26878, all 2000 panel genes map into `var.parquet`, the perturbation encoding
  reproduces STATE's strings, and `cell_eval.MetricsEvaluator` imports.
* Verified PerturbDiff itself still runs: all 52 `src/` modules import, and both entrypoints
  complete end-to-end on synthetic replogle-shaped data (train → checkpoint → validate → sample
  → write `diffusion_{predict,true}_*.h5ad`).

Two pre-existing bugs had to be fixed to get there — see §7.

### Phase 1 — read the zarr data ✅ DONE

Three new modules plus a `data.source_format` switch; the h5ad path is untouched and verified
bit-identical (same validation loss to 16 digits before and after).

| Module | Role |
|---|---|
| `src/data/zarr_metadata_cache.py` | `ZarrMetadataCache` — duck-typed twin of `H5MetadataCache`, built from `obs.parquet`; ports STATE's `encode_obs_pert` verbatim |
| `src/data/zarr_store.py` | `ZarrStore` — `H5Store`'s contract over `X.zarr`/`X.npy`, with a **process-global** array cache |
| `src/data/dataset/dataset_io_zarr.py` | row fetch + gene-panel resolution through `var.parquet` |

`GlobalH5MetadataCache.get_zarr_cache()` registers into the same `_cache` dict (keyed by
directory), so `meta_cache._cache[dataset_path_map[ds_name]]` lookups in `get_covarates` /
`mapping_cells` / `collect_batch_covariates` keep working.

Verified on real hepg2: cache builds in 0.6 s (1341 perts / 1 cell type / 56 batches, 4976
controls), pert codes round-trip to STATE's derived strings, expression rows are **exactly** equal
to a direct `zarr` read in panel order, and a full batch satisfies the contract — `[B, 16, 2000]`,
one `(pert, celltype, batch)` per cell set, `cont_emb` rows are real controls, `col_genes` is the
panel in order.

Two design notes worth carrying forward:

* The array cache is module-level, not per-instance, because one `H5adSentenceDataset` is built
  per split over the same files — per-instance storage would hold three copies (hepg2 alone is
  0.77 GB; the four cell lines would cost ~9 GB instead of ~3). It is excluded from pickling, so
  forked workers share it copy-on-write.
* A gene missing from `var.parquet` raises instead of being zero-filled (the h5ad path
  zero-fills silently) — a partial panel would change the benchmark with no signal.

### 🚩 Found in Phase 1 — cell-set padding is degenerate on this data *(now a config switch)*

`register_mapping_indices` groups perturbed cells by **(pert, cell_type, batch)** and the sampler
pads every group up to a multiple of `use_cell_set` by sampling *with replacement*. On hepg2 that
grouping is far too fine:

| grouping | non-empty groups | cells/group (median) | `use_cell_set=16` | `use_cell_set=64` |
|---|---|---|---|---|
| `(pert, cell_type, batch)` — PerturbDiff today | 46331 | **1** | 742,240 (8.1× real) | **2,965,184 (32.4× real)** |
| `(pert, cell_type)` — STATE's sentences | 1340 | 53 | 101,920 (1.1×) | 128,256 (1.4×) |

At `use_cell_set=64` — the value that matches STATE's `cell_set_len=64` — **100 % of groups hold
fewer cells than one set**, so every cell set is one real cell replicated 64 times. And
`is_padded_list` is consumed *only* by sampling (`sampling_generation.py:148`); the training loss
never sees it. So the energy-MMD term — the distribution-level objective that is PerturbDiff's
entire contribution — would be matching a predicted cloud against 64 copies of a single point,
on a 32× inflated epoch.

**Resolved as a configurable switch rather than a silent default change**, since upstream has no
such option and both behaviours are worth measuring: `data.cell_set_group_by` is
`pert_celltype_batch` (upstream, the default) or `pert_celltype` (STATE's sentence grouping,
where batch stays a per-cell covariate on the `CovEncoder` and only set *membership* stops being
batch-homogeneous).

At upstream's own Replogle `use_cell_set=32`, measured on hepg2:

| grouping | groups | cells/group (median) | epoch | duplicate rows |
|---|---|---|---|---|
| `pert_celltype_batch` | 46,331 | 1 (max **31**) | 16.2x real | 93.8% |
| `pert_celltype` | 1,340 | 53 (max 1,166) | 1.2x real | 19.6% |

The max 3-key group holds 31 cells, so **any** `use_cell_set >= 32` — including upstream's own
Replogle value — leaves 100% of groups under-filled. There is no operating point at which the
3-key grouping produces a naturally full cell set on this data.

The A/B is the first benchmark experiment (§5). Compare at **matched step counts**: the two are
~13x apart in samples per epoch, so equal epochs are not equal compute.

### Phase 1 — original plan (for reference)

* **New** `src/data/zarr_metadata_cache.py`: `ZarrMetadataCache` exposing the duck-typed
  attributes above, built from `obs.parquet` via a local copy of `encode_obs_pert`
  (port from `zarr_perturbation.py:62-120` so the pert strings are byte-identical to STATE's).
  Register it in `GlobalH5MetadataCache` behind a `source_format` switch.
* **New** `src/data/dataset/dataset_io_zarr.py`: row fetch with `feature_loading ∈ {ram, npy, zarr}`
  (mirror `_Source._ensure_open`, `zarr_perturbation.py:278-329`), plus gene selection through
  `var.parquet.ensembl_gene_id`.
* **New** `src/data/zarr_store.py`: a `ZarrStore` with the same `dataset_file(name)` /
  `close_all()` / `__getstate__` contract as `H5Store` so worker forking stays safe.
* `H5adSentenceDataset` gains a format switch on `_retrieve_counts` and
  `_get_selected_gene_vars` — the *only* two format-specific methods.

Details: [src/data/ZARR_BENCHMARK.md](src/data/ZARR_BENCHMARK.md),
[src/data/dataset/ZARR_BENCHMARK.md](src/data/dataset/ZARR_BENCHMARK.md).

### Phase 2 — index-based splits + a TOML manifest ✅ DONE

* `split_strategy.split_index_json` — reads `{"folds":[{finetune, validation, test}]}`,
  restricts each split to **perturbed** rows and adds **all** control rows to every non-empty
  split; a dataset with no `[splits]` entry and `[training] = "train"` contributes all its rows
  to train.
* `data_module.ZarrPerturbationDataModule` (+ `zarr_setup` / `zarr_setup_dataset`) — covariate
  vocabularies from `obs.parquet`, dataset list from the TOML manifest, expression preloaded in
  the main process for copy-on-write sharing.
* `configs/manifests/ihabxcell_hepg2_zarr.toml` — verified identical to
  `../STATE/state/ihabxcell_hepg2_zarr.toml` ignoring comments.

Verified across all four cell lines (643,413 cells, 5.15 GB at `feature_loading=ram`):

| split | perturbed | control | total | fold entry |
|---|---|---|---|---|
| hepg2 test | 26,878 | 4,976 | **31,854** | `test` lists 26,878 |
| hepg2 train | 64,762 | 4,976 | **69,738** | `finetune` lists 69,738 |
| hepg2 validation | 0 | 0 | 0 | empty by design |

Both totals equal STATE's `test_p + ctrl` / `finetune_p + ctrl` exactly. jurkat / k562 / rpe1 are
train-only whole datasets (577,370 training cells in total). Vocabularies: 2024 perturbations,
4 cell types, 215 batches.

Two behaviours worth remembering:

* `split_out_control` mutates `num_cell` **in place** on the datamodule's own dict, so
  `dm.<split>_num_cell` counts *perturbed* cells only; the raw arrays live in
  `dm.<split>_indices`. (This tripped up the first version of the acceptance test.)
* Dataset names come from the **directory basename**, not the manifest label —
  `get_short_dsname` requires a tahoe/pbmc/replogle/cellxgene substring and STATE's labels
  (`ihabxcell_hepg2`) do not have one. This keeps the manifest byte-identical to STATE's.
* The empty validation split produced `np.concatenate([])` in the sampler; it now yields zero
  batches instead of raising.

### Phase 2 — original plan (for reference)

* **New** `split_index_json(...)` in `src/data/split_strategy.py` implementing §2.3 exactly
  (including "controls in every non-empty split" and the `keep_control_cell` interaction).
* **New** `ZarrPerturbationDataModule` in `src/data/data_module/data_module.py` whose
  `get_dataset_names_and_paths` parses **the same TOML** STATE uses, and whose `setup()`
  builds `pert_dict` / `cell_type_dict` / `batch_dict` from parquet instead of h5.
* Keep the `ds_name + "_" + batch` namespacing (`dataset_core.py:186`) — `batch_center` is
  already globally unique, so this is harmless but must be consistent between train and sampling.

Details: [src/data/data_module/ZARR_BENCHMARK.md](src/data/data_module/ZARR_BENCHMARK.md).

### Phase 3 — config + dispatch ✅ DONE

* `configs/path/hooke_tx_path.yaml` — zarr roots, a **writable** `indices_cache_dir`, and
  absolute paths to the manifest and the gene panel.
* `configs/data/ihabxcell_zarr.yaml` — `data_name: IhabxcellZarrFinetune`, `source_format: zarr`,
  canonical obs columns, and upstream's Replogle hyperparameters (`use_cell_set=32`).
* `IhabxcellZarrFinetune` registered in `build_datamodule` and `build_sampling_datamodule`
  (the former's bare `assert 0` is now a message listing the known names), and
  `ZarrPerturbationDataModule` exported from the `data_module` package `__init__`.

Verified: the **real training entrypoint** runs end to end on the real zarr data, both grouping
modes selectable from the CLI. Measured on the test split (26,878 real cells):
`pert_celltype_batch` expands it to 423,008 samples, `pert_celltype` to 33,408 — 12.7x fewer.
The empty validation split yields 0 batches and the run still completes.

One shared-code fix was needed: the dataloader builders passed `prefetch_factor` unconditionally,
which torch rejects when `num_workers=0`, breaking every debug run. It is now passed only when
workers are enabled.

### Phase 3 — original plan (for reference)

* **New** `configs/path/hooke_tx_path.yaml` (points at `/rxrx/scratch/hooke_tx/public`, the
  split JSON, and a writable `save_dir`/`indices_cache_dir`).
* **New** `configs/data/ihabxcell_zarr.yaml` with `data_name: IhabxcellZarrFinetune`,
  `source_format: zarr`, `toml_config_path`, `selected_gene_file` (the Ensembl `.txt`),
  `cell_type_key: cell_type`, `batch_col: batch_center`, `control_pert: empty`,
  `embed_key: X_hvg`, `normalize_counts: false`, `use_cell_set: 64`, `pad_length: 2000`.
* Register the new `data_name` in `build_datamodule` (`training_datamodule_builder.py:13-39`,
  which currently ends in `assert 0`) and `build_sampling_datamodule`
  (`sampling_setup.py:9-57`).

Details: [configs/ZARR_BENCHMARK.md](configs/ZARR_BENCHMARK.md),
[src/apps/training/ZARR_BENCHMARK.md](src/apps/training/ZARR_BENCHMARK.md).

### Phase 4 — covariate encoders ✅ DONE

All-one-hot is the benchmark default and needs no assets. `genept` and `llm` also work now:

* `configs/metadata/gene_id_map.csv` (mygene, 53,653 pairs) and
  `configs/metadata/vocab_id2name.csv` (upstream, 25,133 rows), chained upstream-first by
  `src/common/gene_id_map.py`. Measured against the real genept pickle: mygene alone 1965/2023,
  upstream alone 1979, **chained 1980 (97.9%)**. mygene returns current HGNC symbols where genept
  used older ones, so upstream's table is *better* here and both are kept.
* `covariate_encoding` translates when `cov_encoding.pert_key_type=ensembl` and takes the control
  key from `control_pert_key`. It refuses to build if >`max_zero_embedding_frac` (50%) of
  perturbations lack an embedding, and warns when the control key is absent.
* `celltype_encoding=llm` resolves case-insensitively (upstream keys are `hepg2`, obs has `HepG2`).
* Assets fetched with a targeted ~75 MB download, not the 3 TB release; `path.tmp_dir` points at
  them. Regenerate the mygene table with `tools/build_gene_id_map.py` (`--extra assets`).

Verified end to end: `genept perturbation embeddings: 1980/2023 resolved (2.1% zero-filled)`.
Residual 42 misses are recent "-1 suffix" renames (`AARS1` vs genept's `AARS`).

### Phase 4 — original plan (for reference)

* `celltype_encoding=llm` needs `new_all_emb.pkl` keyed by `"HepG2"`/`"Jurkat"`/… . If the keys
  don't match the zarr `cell_type` values, use `celltype_encoding=onehot` for the first run.
* `replogle_gene_encoding=genept` is keyed by **gene symbol** with control key
  `"non-targeting"` (`covariate_encoding.py:173-177`); the zarr pert strings are **Ensembl ids**
  with control `"empty"`. Either translate via `var.parquet` or start with
  `replogle_gene_encoding=onehot`.

Details: [src/models/ZARR_BENCHMARK.md](src/models/ZARR_BENCHMARK.md).

### Phase 5 — cell-eval parity ✅ DONE

* `src/apps/sampling/sampling_eval_celleval.py` — a port of
  `_predict.py:896-946`: `[0, 14]` clip, per-cell-type split, `MetricsEvaluator` with
  `pdex_kwargs={"exp_post_agg": True, "is_log1p": True}`, cgroup-aware thread count, and
  `skip_metrics=["pearson_edistance", "clustering_agreement"]`.
* Controls now come from the zarr test split (`build_zarr_controls`) instead of a separate
  control `.h5ad`, and obs uses STATE's column names (`gene` / `cell_type` / `batch_center`)
  via `build_obs_data_zarr`.
* Enabled with `sampling.run_cell_eval=true` (default off).
* **cell-eval pinned to `==0.7.2`**, the version STATE has installed, so metric implementations
  cannot drift between the two competitors.

Verified against STATE's own outputs: identical obs columns, Ensembl pert ids, `HepG2`, 4976
controls on both sides, and `HepG2_results.csv` with **the same 27 metric columns**. Our row count
differs only because the check ran on 3 batches; at `num_sampled_batches=null` the adata is
26,878 + 4,976 = 31,854 rows, exactly STATE's.

> Real controls are concatenated into *both* adatas, which is STATE's
> `--inject-real-controls` convention. Compare against STATE runs that used that flag.

### Phase 5 — original plan (for reference)

* **New** `src/apps/sampling/sampling_eval_celleval.py` reproducing `_predict.py:896-946`:
  clip X to `[0, 14]`, split by cell type, `MetricsEvaluator(control_pert=..., pert_col=...,
  pdex_kwargs={"exp_post_agg": True, "is_log1p": True}, num_threads=len(os.sched_getaffinity(0)))`,
  `skip_metrics=["pearson_edistance", "clustering_agreement"]`.
* Make `load_ctrl_adata` / `build_obs_data` zarr-aware — take control rows from the zarr test
  split instead of the separate `replogle.h5ad`.
* ~~Make sampling cover the whole test split and make the order deterministic.~~ **Done** —
  `num_sampled_batches` defaults to `null`, truncation is warned about loudly, batch counts are
  capped, and coverage is cross-checked against the sampler's non-padded cell count. Two runs of
  one checkpoint are bit-identical. See
  [src/apps/sampling/ZARR_BENCHMARK.md](src/apps/sampling/ZARR_BENCHMARK.md) §2.1.

Details: [src/apps/sampling/ZARR_BENCHMARK.md](src/apps/sampling/ZARR_BENCHMARK.md).

---

## 5. Running it

On slurm, use the committed scripts (conventions copied from
`../STATE/state/train_state_zarr.sh` and `../../launch/*.sh`):

```bash
sbatch launch/perturbdiff_zarr_canary.sh          # short GPU smoke, run this first
TRAIN=$(sbatch --parsable launch/perturbdiff_zarr_train.sh)
sbatch --dependency=aftercorr:$TRAIN launch/perturbdiff_zarr_eval.sh
```

`launch/_pd_common.sh` holds the grid and the shared hyperparameters; `aftercorr` pairs eval task
*N* with train task *N*. Measured on an H100: ~11.3 it/s (80k steps ≈ 2 h), peak RSS 7.5 GB.

The equivalent by hand, if you want to vary something not in the grid:

```bash
cd /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff

NO_WANDB="lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger
~lightning.logger.project ~lightning.logger.save_dir ~lightning.logger.name"

.venv/bin/python ./src/apps/run/rawdata_diffusion_training.py \
  data=ihabxcell_zarr path=hooke_tx_path \
  run_name=perturbdiff_hepg2_zarr \
  trainer.devices=[0] trainer.max_steps=80000 \
  optimization.micro_batch_size=128 optimization.optimizer.lr=0.002 \
  data.use_cell_set=32 data.num_workers=4 \
  data.cell_set_group_by=pert_celltype_batch \
  cov_encoding.celltype_encoding=onehot \
  cov_encoding.replogle_gene_encoding=onehot \
  lightning.callbacks.checkpoint.monitor=null \
  lightning.callbacks.checkpoint.save_top_k=1 \
  lightning.callbacks.checkpoint.every_n_train_steps=2000 \
  $NO_WANDB
```

`use_cell_set=32`, `micro_batch_size=128` and `lr=0.002` are upstream's own Replogle recipe
(README `COMMON_REPLOGLE_TRAIN`), not the PBMC/Tahoe values of 256/2048.

`monitor=null` is deliberate: the benchmark fold has no validation cells, so a monitored
`ModelCheckpoint` would never fire. The final validation and the evaluation both use the final
checkpoint at `${path.diffusion.save_dir}/${run_name}` (§7.3).

**The grouping A/B** is one flag — everything else identical:

```bash
data.cell_set_group_by=pert_celltype_batch   # upstream; test split 26,878 -> 423,008 samples
data.cell_set_group_by=pert_celltype         # STATE-style; 26,878 -> 33,408
```

Note these are *not* equal-compute at equal epochs (~13x apart), so compare at matched step counts.

Evaluate — **Phase 5 still missing**: the sampler runs, but `load_ctrl_adata` / `build_obs_data`
have no zarr branch yet and there is no cell-eval step.

```bash
.venv/bin/python ./src/apps/run/rawdata_diffusion_sampling.py \
  data=ihabxcell_zarr path=hooke_tx_path \
  model_checkpoint_path=/…/perturbdiff_out/runs/perturbdiff_hepg2_zarr \
  optimization.micro_batch_size=128 data.use_cell_set=32 \
  cov_encoding.celltype_encoding=onehot \
  sampling.num_sampled_batches=null \
  sampling.output_dir=/…/eval_perturbdiff_hepg2_zarr \
  $NO_WANDB
```

To keep the comparison against STATE fair: same gene panel (2000 Ensembl ids), same split JSON
and fold, same held-out cell type, real controls in both adatas, same cell-eval configuration.

---

## 6. Risks / open questions

* **`validation` is empty in fold 0** (0 rows). PerturbDiff's checkpoint callback monitors
  `validation_validation_loss_epoch` (`configs/lightning/base.yaml`), which that fold never
  produces. This no longer breaks the run — the final validation uses the final checkpoint rather
  than `"best"` (§7.3) — but it does mean **no checkpoint is selected by validation score**: you
  evaluate the last model. To select by score instead, point the monitor at a metric the run
  emits (e.g. `validation_test_loss_epoch`) or carve a validation slice out of `finetune`.
  STATE sidesteps the same issue by falling back to the test set in `val_dataloader()`
  (`zarr_perturbation.py:853-858`).
* **Cross-dataset training.** The TOML lists 4 cell lines; only hepg2 has a split. PerturbDiff
  namespaces batches by dataset and can hold a per-dataset embedder, so multi-source training
  should work — but `grouped_pert_data_indices` is grouped by `(pert, cell_type, batch)` and the
  cache files are keyed by `ds_name`+stage (`dataset_grouping.py:146-147`); clear
  `indices_cache_dir` between config changes or set `skip_cached_indices=true`.
* **Memory.** `feature_loading=ram` on hepg2 is 96616 × 2000 × 4 B ≈ 0.77 GB — fine. All four
  cell lines together, or `parse_pbmc` (`X.npy` is 77 GB), are not: use `npy` (mmap) or `zarr`.
* **Gene ordering.** PerturbDiff reorders columns to the selected-gene order at read time
  (`dataset_io.py:144-160`) and again in sampling (`sampling_generation.py:210-215`). With zarr,
  do the mapping **once** through `var.parquet` and drop the second reorder.
* **`normalize_counts`.** The zarr X is already log1p-normalized, so keep
  `normalize_counts: false`; note the collate divides and sampling multiplies back
  (`dataset_core.py:80-82`, `sampling_generation.py:192-194`) — a non-`false` value would need
  both sides changed together.


## 7. Pre-existing bugs found and fixed during Phase 0

Both were latent in the vendored code and are **not** caused by the Python upgrade — the old 3.9
env had the same `omegaconf` 2.3.1 and a torch (2.8) with the same `weights_only` default.

### 7.1 `configs/cov_encoding/trixie_onehot.yaml` — interpolation depth in a list

`gene_embedding_path` used `${..path.tmp_dir}` inside a **list**. List elements sit one level
deeper (the enclosing node is the `ListConfig`), so it resolved to `cov_encoding.path.tmp_dir` and
`OmegaConf.resolve(cfg)` (`src/apps/run/rawdata_diffusion_training.py:33`) raised
`InterpolationKeyError` before *any* run could start. Fixed to `${...path.tmp_dir}`.

Reproduce the semantics standalone:

```python
OmegaConf.create({"path": {"tmp_dir": "/T"},
                  "cov": {"scalar": "${..path.tmp_dir}",     # -> /T
                          "lst":  ["${..path.tmp_dir}"]}})   # -> InterpolationKeyError
```

### 7.2 `src/apps/training/training_pipeline.py` — checkpoint reload under torch ≥ 2.6

torch 2.6 flipped `torch.load` to `weights_only=True`, which cannot unpickle the OmegaConf
objects `save_hyperparameters()` writes into every checkpoint. `load_plmodel_checkpoint` has its
own `weights_only=False` fallback (`training_model_checkpoint.py:100-113`), but Lightning's
**internal** reloads do not: resuming from `cfg.model.ckpt_path`, and the final
`trainer.validate(...)` at the end of every run. An 80k-step run would train fine and then die on
the last line.

Fixed by passing `weights_only=False` to `trainer.fit()` / `trainer.validate()` — these are our own
checkpoints, written seconds earlier by the same process. Requires pytorch-lightning ≥ 2.5, so the
floor in `pyproject.toml` was raised from `>=2.0`.

> Extending `allow_omegaconf_checkpoint_unpickling()`'s allowlist instead is a dead end: the
> unpickler then stops successively on `typing.Any`, `dict`, `collections.defaultdict`,
> `numpy._core.multiarray.scalar`, … — unbounded whack-a-mole.

### 7.3 Final validation now uses the final checkpoint, not `"best"`

`run_pipeline` used to end with `trainer.validate(ckpt_path="best")`. That requires the
`ModelCheckpoint` monitor (`lightning.callbacks.checkpoint.monitor`, default
`validation_validation_loss_epoch`) to have fired at least once. When it has not — an empty
validation split, or a run shorter than `every_n_train_steps` — no best checkpoint exists and
Lightning raises `ValueError: ... ModelCheckpoint is not configured to save the best model`
*after* training has already completed.

It now validates `cfg.save_dir_path`, the final checkpoint written by the immediately preceding
`trainer.save_checkpoint(...)`. This has no dependency on the monitor, so the empty-validation-fold
of the benchmark split (§6) no longer breaks the end of a run.

Verified: with the **default** checkpoint config (`monitor=validation_validation_loss_epoch`,
`save_top_k=3`, `every_n_train_steps=20000`) a 2-step run now completes, logging
`Loaded model weights from the checkpoint at <save_dir_path>/<run_name>`.

The same file is what you pass to the evaluation entrypoint:

```bash
python ./src/apps/run/rawdata_diffusion_sampling.py \
  model_checkpoint_path=${SAVE_DIR}/${RUN_NAME} ...
```

which is confirmed to load it (it is a Lightning `weights_only=True` save, so it still carries the
`hyper_parameters` that `load_sampling_model` needs).

> Choosing the checkpoint *by validation score* is still possible — set
> `lightning.callbacks.checkpoint.monitor` to a metric the run actually produces (e.g.
> `validation_test_loss_epoch`) and pass that file to the sampler explicitly.

---

## 8. Benchmark runs launched

First experiment: the cell-set grouping A/B (§4) crossed with one lr, 4 runs, everything else
upstream's Replogle recipe at STATE's 80k-step budget.

| task | run | `cell_set_group_by` | lr |
|---|---|---|---|
| 0 | `pd_full_lr0.002_genept` | `pert_celltype_batch` | 0.002 |
| 1 | `pd_full_lr0.0005_genept` | `pert_celltype_batch` | 0.0005 |
| 2 | `pd_noBatch_lr0.002_genept` | `pert_celltype` | 0.002 |
| 3 | `pd_noBatch_lr0.0005_genept` | `pert_celltype` | 0.0005 |

Fixed: `use_cell_set=32`, `micro_batch_size=128`, `replogle_gene_encoding=genept`,
`max_steps=80000`, checkpoint every 2000, `monitor=null` (fold 0 has no validation cells, so the
final checkpoint is what gets evaluated — §7.3).

Submitted as array `16988030` (train) + `16988031` (eval, `aftercorr`), `%2` concurrency. Those
two were submitted under `qos=vco`; the committed scripts use `qos`/`wckey=hooke-predict`.

All 8 jobs completed with exit 0. Training: ~2 h each, 80k steps reached, no NaN/Inf, no
tracebacks, loss falling monotonically (21.1 → ~6 for `pert_celltype_batch`, 13.4 → ~0.9 for
`pert_celltype`). Every eval reported `Full test-split coverage: 26878 cells` + 4,976 controls =
31,854 rows, matching STATE exactly. `genept perturbation embeddings: 1980/2023 resolved (2.1%
zero-filled)` in all four.

### Results — cell-eval, 380 perturbations (same set STATE scores)

| run | pearson_delta | mse | discrimination_l1 | overlap_at_N |
|---|---|---|---|---|
| `pd_full_lr0.002_genept` | 0.031 | 0.1191 | 0.501 | 0.083 |
| `pd_full_lr0.0005_genept` | 0.121 | 0.3789 | 0.501 | 0.218 |
| **`pd_noBatch_lr0.002_genept`** | **0.350** | **0.0122** | **0.728** | 0.133 |
| `pd_noBatch_lr0.0005_genept` | 0.196 | 0.0614 | 0.611 | 0.129 |
| **STATE** (`hepg2_holdout_ram`) | **0.431** | **0.0060** | **0.797** | 0.200 |

**The A/B is decisive: `pert_celltype` beats `pert_celltype_batch`** — 0.350 vs 0.031 pearson_delta
at the same lr — which is what §4 predicts from training on ~94% duplicated rows. Within
`pert_celltype`, lr 0.002 beats 0.0005. STATE still leads on every metric.

The comparison is fair on the control convention: STATE's `adata_pred` control rows are
bit-identical to its `adata_real` (max diff 0.0), i.e. it used `--inject-real-controls`, same as
us. Same cell-eval 0.7.2, same 380 perturbations, same 27 columns.

Two things these numbers are *not*:

* The training-log validation losses (7.45 for `full` vs 1.13 for `noBatch`) are **not comparable
  across grouping modes** — the val dataloader uses the same grouping, so each mode is scored
  against a differently-constructed target. Only the cell-eval table above is comparable.
* **EMA was off in this sweep.** `configs/lightning/base.yaml` ships `ema.decay: 0`, which disables
  the callback entirely, while upstream's `COMMON_TRAIN_RUNTIME` sets `decay=0.99,
  update_steps=10` for every published recipe. The sweep carried the Replogle block but not the
  shared runtime block. So these are upstream's Replogle hyperparameters *minus EMA*, plausibly a
  real handicap for a diffusion model. `launch/_pd_common.sh` now defaults to `0.99`/`10`.

### Follow-up 1: EMA (array 17014939/17014940)

`configs/lightning/base.yaml` ships `ema.decay: 0`, which disables the callback outright, while
upstream's `COMMON_TRAIN_RUNTIME` sets `0.99` / `10` for every published recipe. The first sweep
therefore ran **without EMA** — an unintended deviation. Re-running the two `pert_celltype` arms
with it on:

| run | pearson_delta | mse | discrim_l1 | overlap_at_N |
|---|---|---|---|---|
| `noBatch_lr0.002` | 0.3495 | 0.0122 | 0.7276 | 0.1331 |
| `noBatch_lr0.002_ema` | **0.3857** | **0.0109** | **0.7357** | **0.1468** |
| `noBatch_lr0.0005` | 0.1956 | 0.0614 | 0.6110 | 0.1292 |
| `noBatch_lr0.0005_ema` | **0.2202** | **0.0559** | **0.6147** | **0.1348** |

EMA improved **every metric in both arms** — 12 independent comparisons, all the same direction,
which is more convincing than the size of any single delta. `pert_celltype_batch` was deliberately
not re-run: its deficit is structural (~94% duplicated rows), not something weight averaging fixes.
`launch/_pd_common.sh` now defaults EMA on.

### Follow-up 2: learning rate (array 17017897/17017898)

The first sweep tried only 0.002 and 0.0005, and 0.002 — the *top* of that range — won, so the
optimum could have been higher. Five more points, all `pert_celltype` + EMA + genept:

| lr | pearson_delta | mse | mae | discrim_l1 | overlap_at_N |
|---|---|---|---|---|---|
| 0.0005 | 0.2202 | 0.0559 | 0.1471 | 0.6147 | 0.1348 |
| 0.001 | 0.3142 | 0.0149 | 0.0784 | 0.7113 | **0.1628** |
| **0.002** | **0.3857** | 0.0109 | 0.0758 | **0.7357** | 0.1468 |
| 0.003 | 0.3549 | **0.0084** | **0.0589** | 0.5716 | 0.0912 |
| 0.005 | 0.3625 | 0.0090 | 0.0606 | 0.5192 | 0.0873 |
| 0.008 | 0.3681 | 0.0110 | 0.0656 | 0.5032 | 0.1152 |
| 0.012 | 0.3387 | 0.0095 | 0.0619 | 0.5053 | 0.0741 |
| **STATE** | **0.4313** | **0.0060** | **0.0535** | **0.7973** | **0.2004** |

**0.002 is a genuine optimum**, not an untested edge — and it is upstream's own Replogle value.
0.001-0.002 is a broad plateau rather than a knife-edge.

Above 0.002 the metrics **disagree**: `mse`/`mae` keep improving to a best at 0.003
(0.0084 / 0.0589, the closest any run gets to STATE) while `pearson_delta`, `discrimination` and
`overlap_at_N` degrade sharply — discrimination drops 0.736 → 0.572 and never recovers. That is
what predictions regressing toward the per-gene mean look like: good absolute error, poor at
ranking *which* genes move and at telling perturbations apart. For a perturbation benchmark the
delta/discrimination metrics are the ones that matter, so 0.002 is the pick; 0.003 is defensible
if raw reconstruction error is what you care about.

### Standing comparison

| | pearson_delta | mse | discrim_l1 |
|---|---|---|---|
| PerturbDiff best (`pert_celltype`, EMA, genept, lr 0.002) | 0.386 | 0.0109 | 0.736 |
| STATE (`hepg2_holdout_ram`) | **0.431** | **0.0060** | **0.797** |

STATE still leads on every metric. The gap has narrowed from 0.031 → 0.386 pearson_delta as the
grouping, EMA and lr issues were fixed; ~0.045 remains.

### Follow-up 3: `use_cell_set` (arrays 17074743/17074744, 17076392, 17079181)

All at the best config (`pert_celltype`, EMA, genept, lr 0.002). `micro_batch_size` stays 128, so
sets-per-batch is `128 / use_cell_set`.

| `use_cell_set` | pearson_delta | mse | mae | discrim_l1 | overlap_at_N |
|---|---|---|---|---|---|
| 2 | 0.1567 | 0.0908 | 0.1809 | 0.5882 | 0.1574 |
| 4 | 0.3580 | 0.0219 | 0.0943 | 0.6652 | 0.1687 |
| 8 | 0.4117 | 0.0099 | 0.0672 | 0.7295 | 0.1680 |
| **16** | **0.4298** | **0.0071** | **0.0559** | **0.7678** | **0.1998** |
| 32 (upstream Replogle) | 0.3857 | 0.0109 | 0.0758 | 0.7357 | 0.1468 |
| 64 (STATE's `cell_set_len`) | 0.2718 | 0.0126 | 0.0742 | 0.5534 | 0.1111 |
| 128 | 0.3545 | 0.0163 | 0.0870 | 0.5011 | 0.1663 |
| **STATE** | **0.4313** | **0.0060** | **0.0535** | **0.7973** | **0.2004** |

**An inverted U peaking at 16**, not a monotone trend — cs2 is the worst point on the board. That
matters for interpretation: a "smaller is always better" curve would have suggested PerturbDiff
scores well by degenerating toward a pointwise predictor and abandoning the distribution-level
objective that is its whole contribution. The collapse at 2 says the opposite — the energy-MMD
needs a real population, just a smaller one than 64 on this data.

Mechanistically consistent with the group sizes: the median `(pert, cell_type)` group holds 53
cells, so 16 fills from real cells while 64 and 128 need heavy replacement-padding, and 2-4 give
the MMD too few points to estimate a distribution from.

### Follow-up 4: perturbation encoding (array 17074745/17074746)

At the best config, `use_cell_set=32`:

| encoding | pearson_delta | mse | mae | discrim_l1 | overlap_at_N |
|---|---|---|---|---|---|
| `genept` | **0.3857** | **0.0109** | **0.0758** | 0.7357 | **0.1468** |
| `onehot` | 0.3349 | 0.0151 | 0.0867 | 0.7353 | 0.1333 |

genept is worth ~0.05 pearson_delta, so the Ensembl->symbol translation in Phase 4 paid for
itself. Note discrimination is identical to four decimal places (0.7357 vs 0.7353) — the gain sits
entirely in the delta and error metrics. onehot is the closer analogue of STATE's perturbation
encoding, so it is the fairer *architectural* comparison even though genept is upstream's recipe.

### Follow-up 5: seed replication at the best config (arrays 17097101/17097102)

The section above closed by saying no number here was quotable without an error bar. This
measures it: three fresh seeds (43, 44, 45) at the best config — `pert_celltype`, EMA 0.99,
lr 0.002, `use_cell_set=16`, 80k steps — for **both** perturbation encodings. The existing
seed-42 runs join as a fourth genept replicate. `optimization.seed` feeds
`pl.seed_everything` in *both* entrypoints, so each replicate varies training init and ordering
**and** the sampling-time diffusion noise and control pairing: the spread below is the error bar
for rerunning the whole pipeline, which is exactly what the single-seed numbers needed.

| encoding | seed | pearson_delta | mse | mae | discrim_l1 | overlap_at_N |
|---|---|---|---|---|---|---|
| genept | 42 | 0.4298 | 0.0071 | 0.0559 | 0.7678 | 0.1998 |
| genept | 43 | 0.4004 | 0.0072 | 0.0575 | 0.7571 | 0.1741 |
| genept | 44 | 0.4293 | 0.0074 | 0.0577 | 0.7516 | 0.1948 |
| genept | 45 | 0.3822 | 0.0099 | 0.0683 | 0.7508 | 0.1638 |
| onehot | 43 | 0.4254 | 0.0072 | 0.0600 | 0.7644 | 0.1589 |
| onehot | 44 | 0.3812 | 0.0096 | 0.0667 | 0.7741 | 0.1755 |
| onehot | 45 | 0.4081 | 0.0077 | 0.0593 | 0.7860 | 0.1837 |

| | pearson_delta | mse | mae | discrim_l1 | overlap_at_N |
|---|---|---|---|---|---|
| genept, mean ± sd (n=4) | 0.4104 ± 0.0233 | 0.0079 ± 0.0013 | 0.0598 ± 0.0057 | 0.7568 ± 0.0078 | 0.1831 ± 0.0170 |
| onehot, mean ± sd (n=3) | 0.4049 ± 0.0223 | 0.0082 ± 0.0013 | 0.0620 ± 0.0041 | 0.7748 ± 0.0108 | 0.1727 ± 0.0126 |
| STATE (n=1) | 0.4313 | 0.0060 | 0.0535 | 0.7973 | 0.2004 |

**Three conclusions recorded above have to be revised.**

1. **The 0.4298 headline was the best of four draws, not a typical run.** The mean is 0.4104 and
   the seed sd is 0.0233 — an observed range of 0.048 across four seeds. The predicted spread of
   ~0.08 was the right order of magnitude.
2. **PerturbDiff does not match STATE.** STATE's 0.4313 sits 0.9 sd above the genept mean on
   pearson_delta — suggestive but not separating on that metric alone. The other four separate it
   clearly, and `discrim_l1` decisively: STATE's 0.7973 is **5.2 sd** above the genept mean of
   0.7568, and above *every individual* PerturbDiff run including the best onehot seed (0.7860).
   mse is 1.5 sd, overlap_at_N 1.0 sd, all favouring STATE. STATE is itself a single seed, so
   these are one-sided distances, not tests — but a 5 sd gap does not come from seed noise.
3. **The genept advantage does not survive replication.** Follow-up 4 measured genept − onehot =
   +0.051 pearson_delta from one seed each at cs32. At cs16 with n=4 vs n=3 the gap is **+0.0055**,
   under a quarter of one sd — nothing. And it *reverses* on discrimination, where onehot leads by
   0.018 (2.3 sd). The Ensembl→symbol translation in Phase 4 was still necessary to make genept
   work at all, but it buys no measurable accuracy. Since onehot is also the closer analogue of
   STATE's perturbation encoding, it is the better default for the architectural comparison.

What the seed sd does **not** explain is the `use_cell_set=64` dip: 0.2718 is 6 sd below the cs16
mean. That is a real effect of the configuration, not a bad draw — so the inverted U stands, and
whatever happens at cs64 needs its own explanation. Conversely cs16-vs-cs32 (0.4104 vs a
single-seed 0.3857) is only ~1.1 sd, so the choice of 16 over 32 is weakly supported at best.

**Cost of the correction:** every effect in Follow-ups 1–4 was measured at n=1 against a noise
floor of sd 0.023. Only the grouping A/B (0.319), the cs64 dip and the lr collapse above 0.005
are larger than ~3 sd. EMA (+0.036, 1.5 sd) and the cs16 peak are plausible but unconfirmed.

### Standing comparison

| | pearson_delta | mse | discrim_l1 | overlap_at_N |
|---|---|---|---|---|
| PerturbDiff, session start (upstream grouping, n=1) | 0.031 | 0.119 | 0.501 | 0.083 |
| PerturbDiff best config, mean ± sd (`pert_celltype`, EMA, genept, lr 0.002, cs16, n=4) | 0.4104 ± 0.0233 | 0.0079 | 0.7568 ± 0.0078 | 0.1831 |
| PerturbDiff best *single run* (genept seed 42) | 0.4298 | 0.0071 | 0.7678 | 0.1998 |
| STATE (`hepg2_holdout_ram`, n=1) | **0.4313** | **0.0060** | **0.7973** | **0.2004** |

**STATE wins this benchmark.** Quote the mean ± sd row, not the best run. The honest summary is
that correct configuration moved PerturbDiff from 0.031 to 0.41 — two orders of magnitude — and
left it consistently but not enormously behind STATE, with the clearest margin on discrimination.

Remaining caveat: STATE is a single seed. Its own replication error bar is unknown, so the gap is
measured against a point estimate on one side. Three STATE seeds would close that hole, and is the
one measurement still missing from the comparison.

### Operational notes

* One training task (`use_cell_set=16`, job 17074743_1) **hung** at step 10 of epoch 0: zero bytes
  written for 63 minutes at `0.00 it/s`, while a sibling task on the same node wrote 132 KB/10 s.
  Cancelled and resubmitted; the retry completed normally, so it was a flake rather than anything
  specific to that configuration. One hang in ~26 training runs. A suspect worth keeping in mind if
  it recurs: `ZarrStore` guards its process-global array cache with a `threading.RLock`, and a lock
  held at `fork()` is inherited locked by DataLoader workers — though the timing (a hang *after*
  workers had produced batches) does not fit that explanation well.
* Node contention is visible in wall time: tasks sharing `hop02` with another user's 57-task array
  ran at ~7 it/s versus ~10.5 elsewhere, turning a 2 h job into 3 h 10 m. Not a correctness issue.
* **`ModelCheckpoint` used to write every run's intermediates into one shared directory.**
  `configs/lightning/base.yaml` ships `dirpath: null`, which Lightning resolves to
  `<default_root_dir>/checkpoints` — identical for every run in a concurrent array. They collide on
  the `{epoch}-{step}` filename and are separated only by a `-v<N>` suffix chosen from a
  non-atomic existence check, so two runs saving simultaneously can write the same path and
  `save_top_k=1` can delete another run's file. **No recorded result was affected**: evaluation
  reads the final checkpoint `trainer.save_checkpoint()` writes to the unique
  `<runs>/<run_name>`, never these. But 57 GB of unread intermediates had accumulated. The train
  script now passes a per-run `dirpath`.
* Slurm snapshots the batch script at submit time, so editing `perturbdiff_zarr_train.sh` does
  **not** reach already-queued jobs — but `_pd_common.sh` is sourced at runtime, so edits there
  do land on pending tasks. Cancel and resubmit if a script change must apply to a queued array.
