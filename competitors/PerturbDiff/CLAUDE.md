# CLAUDE.md — PerturbDiff (hooke-tx competitor eval)

PyTorch implementation of **PerturbDiff**, a functional-diffusion model for single-cell
perturbation prediction. Vendored here as a competitor baseline alongside
[`../STATE`](../STATE), which we already run on the hooke-tx zarr benchmark.

Upstream docs: [README.md](README.md) (install, download, all published run recipes).

## Provenance

Vendored from upstream, whose own `.git` was removed so this lives directly in the hooke repo
(not as a submodule):

| | |
|---|---|
| Upstream | `git@github.com:DeepGraphLearning/PerturbDiff.git` |
| Commit | `f4e27c155be5325418c4cb3182453d4022754e91` (`main`) |
| Date | 2026-04-07 — *"[issue #2]: small bug fix"* |

That commit is the exact baseline; everything since is ours and is listed under
[Changes made to the vendored code](#changes-made-to-the-vendored-code). To diff against
upstream, clone that repo at that commit and compare.

## Working goal

Train and evaluate PerturbDiff on **the same benchmark and the same native-zarr data** we use for
STATE (`../STATE/state/train_state_zarr.sh`, `predict_state_zarr.sh`), so the two are directly
comparable under cell-eval.

## Planning documents

Every folder that matters carries a `ZARR_BENCHMARK.md`. Start at the root one.

| Document | Covers |
|---|---|
| **[ZARR_BENCHMARK.md](ZARR_BENCHMARK.md)** | **Start here.** The STATE reference workflow, the zarr/parquet data contract, the gap analysis, and the phased adaptation plan |
| [configs/ZARR_BENCHMARK.md](configs/ZARR_BENCHMARK.md) | The whole Hydra tree (data/path/model/trainer/lightning/optimization/cov_encoding) and the three new configs to write |
| [src/ZARR_BENCHMARK.md](src/ZARR_BENCHMARK.md) | Package map; the seven places that hard-code HDF5 |
| [src/data/ZARR_BENCHMARK.md](src/data/ZARR_BENCHMARK.md) | `metadata_cache`, `file_handle`, `sampler`, `split_strategy` — the main work |
| [src/data/dataset/ZARR_BENCHMARK.md](src/data/dataset/ZARR_BENCHMARK.md) | `H5adSentenceDataset`, expression IO, grouping and gene selection |
| [src/data/data_module/ZARR_BENCHMARK.md](src/data/data_module/ZARR_BENCHMARK.md) | The datamodule hierarchy and the proposed `ZarrPerturbationDataModule` |
| [src/apps/ZARR_BENCHMARK.md](src/apps/ZARR_BENCHMARK.md) | Train vs. sample paths, side by side with `state tx train` / `state tx predict` |
| [src/apps/run/ZARR_BENCHMARK.md](src/apps/run/ZARR_BENCHMARK.md) | The two Hydra entrypoints |
| [src/apps/training/ZARR_BENCHMARK.md](src/apps/training/ZARR_BENCHMARK.md) | Builders, trainer assembly, the `data_name` dispatch to extend |
| [src/apps/sampling/ZARR_BENCHMARK.md](src/apps/sampling/ZARR_BENCHMARK.md) | Generation loop and the cell-eval parity work |
| [src/models/ZARR_BENCHMARK.md](src/models/ZARR_BENCHMARK.md) | `CovEncoder` — the one model file affected by Ensembl-vs-symbol identifiers |
| [src/models/cross_dit/ZARR_BENCHMARK.md](src/models/cross_dit/ZARR_BENCHMARK.md) | Backbone shapes and the dataset-naming coupling |
| [src/models/diffusion/ZARR_BENCHMARK.md](src/models/diffusion/ZARR_BENCHMARK.md) | Diffusion process, losses, DDIM/DDPM sampling knobs |
| [src/models/lightning/ZARR_BENCHMARK.md](src/models/lightning/ZARR_BENCHMARK.md) | `PlModel` batch contract and the checkpoint-monitor problem |
| [src/common/ZARR_BENCHMARK.md](src/common/ZARR_BENCHMARK.md) | `get_short_dsname` naming constraint; the CWD trap in `paths.py` |

`asset/` holds only paper figures — nothing to document.

## Running the zarr benchmark

All five phases are implemented: training and cell-eval scoring both work.

```bash
.venv/bin/python ./src/apps/run/rawdata_diffusion_training.py \
  data=ihabxcell_zarr path=hooke_tx_path run_name=perturbdiff_hepg2_zarr \
  trainer.devices=[0] trainer.max_steps=80000 \
  optimization.micro_batch_size=128 optimization.optimizer.lr=0.002 \
  data.use_cell_set=32 data.num_workers=4 \
  cov_encoding.celltype_encoding=onehot cov_encoding.replogle_gene_encoding=onehot \
  lightning.callbacks.checkpoint.monitor=null \
  lightning.callbacks.checkpoint.save_top_k=1 \
  lightning.callbacks.checkpoint.every_n_train_steps=2000 \
  lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger \
  '~lightning.logger.project' '~lightning.logger.save_dir' '~lightning.logger.name'
```

Then evaluate, which writes STATE-comparable cell-eval CSVs:

```bash
.venv/bin/python ./src/apps/run/rawdata_diffusion_sampling.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  model_checkpoint_path=<save_dir>/perturbdiff_hepg2_zarr \
  optimization.micro_batch_size=128 data.use_cell_set=32 \
  sampling.num_sampled_batches=null sampling.run_cell_eval=true \
  sampling.output_dir=<out>/eval_perturbdiff_hepg2_zarr \
  <NO_WANDB overrides>
```

Hyperparameters are upstream's own Replogle recipe. The cell-set grouping A/B is one flag:
`data.cell_set_group_by=pert_celltype_batch` (upstream) or `pert_celltype` (STATE-style).
Full details in [ZARR_BENCHMARK.md](ZARR_BENCHMARK.md) §5.

## Orientation

```
src/apps/run/rawdata_diffusion_training.py    training entrypoint  (hydra)
src/apps/run/rawdata_diffusion_sampling.py    sampling entrypoint  (hydra)
src/data/                                     datasets, cell-set sampler, splits
src/models/                                   Cross_DiT backbone + Gaussian diffusion
configs/                                      Hydra config groups
launch/                                       slurm scripts for the benchmark runs
tools/                                        one-off asset builders

added for the zarr benchmark (the h5ad path is untouched, switched by
data.source_format):
src/data/zarr_metadata_cache.py               obs.parquet -> covariate codes
src/data/zarr_store.py                        X.zarr / X.npy handles
src/data/dataset/dataset_io_zarr.py           row fetch + gene panel
src/data/split_strategy.py:split_index_json   index-JSON splits
src/data/data_module/...ZarrPerturbationDataModule
src/common/gene_id_map.py                     Ensembl <-> symbol
src/apps/sampling/sampling_eval_celleval.py   cell-eval, matched to STATE
tools/build_gene_id_map.py                    regenerates the mygene table
```

Run from the **repo root** — `src/common/paths.py` reads `configs/path/trixie_path.yaml` by a
relative path, and the entrypoints add the root to `sys.path` themselves.

## Environment

`.venv` is a **Python 3.11.14** environment built with `uv` — the same interpreter as
`../STATE/state`, pinned by `.python-version`. See [pyproject.toml](pyproject.toml) and `uv.lock`
(137 packages).

It carries the README's package list **plus** the zarr-benchmark deps: `zarr>=3` (X.zarr is zarr
format v3), `polars` + `pyarrow` (obs/var parquet), and `cell-eval==0.7.2` (pinned to STATE's installed version so metric
implementations cannot drift between the two competitors).
The upstream README's `python=3.9` no longer applies — 3.9 cannot read the benchmark data.

```bash
uv sync                     # reproduce exactly
.venv/bin/python -c "import hydra,pytorch_lightning,torch,anndata,scanpy,zarr,polars,cell_eval"
```

Verified working on 3.11: all 56 `src/` modules import, and both entrypoints run end to end on
the real zarr data (train → checkpoint → validate → sample → cell-eval CSVs), on CPU and on an
H100.

Optional extras: `--extra download` (huggingface_hub, for the pretrained embedding pickles) and
`--extra assets` (mygene, only to regenerate `configs/metadata/gene_id_map.csv`).

> The README's `pip install -U "huggingface_hub[cli]"` is stale — the `cli` extra was dropped in
> huggingface_hub 1.x and the `hf` command ships with the base package.

## Sharp edges (details in the linked docs)

* **Dataset names must contain** `tahoe`/`pbmc`/`replogle`/`cellxgene` — `get_short_dsname`
  (`src/common/utils.py:34-45`) does `assert 0` otherwise. `ihabxcell_replogle_nadig_hepg2` passes.
* **`genept` / `llm` embeddings are keyed by gene symbol and lowercase cell-type name**, with
  control key `"non-targeting"`; zarr perturbations are Ensembl ids with control `"empty"`.
  Handled by `cov_encoding.pert_key_type=ensembl` + `control_pert_key` and a case-insensitive
  cell-type match (`configs/cov_encoding/hooke_tx_zarr.yaml`). Without the translation **every**
  lookup misses and the table silently zero-fills, so the encoder now refuses to build past
  `max_zero_embedding_frac` (50%). Current coverage: 1980/2023 (2.1% zero-filled).
* **Cell-set grouping is a real modelling choice on this data**, exposed as
  `data.cell_set_group_by`. `pert_celltype_batch` (upstream) gives a median of 1 cell per group
  here, so at `use_cell_set=32` ~94% of training rows are replacement-drawn duplicates and the
  epoch inflates ~12x; `pert_celltype` (STATE-style) gives median 53 and ~1.2x. Padding is masked
  only in sampling, never in the loss. Compare the two at **matched step counts**.
* **Fold 0 of the benchmark split has an empty validation set**, so the `ModelCheckpoint` monitor
  `validation_validation_loss_epoch` never fires. Runs no longer break on this (final validation
  uses the final checkpoint), but nothing is selected by validation score — you evaluate the last
  model.
* **`indices_cache_dir` pickles are now content-hashed** (split indices + grouping mode +
  `keep_control_cell` + control type) and written atomically, so a changed configuration is a
  different file and concurrent launches cannot produce a torn read. They used to be keyed only
  by dataset name + stage, which silently reused stale groupings.
* **Sampling now covers the whole test split by default**
  (`sampling.num_sampled_batches: null`; it used to be `4`, silently truncating every eval).
  Setting an integer still works for quick debugging and logs a `PARTIAL EVAL` warning.
  Generation order is deterministic — two runs of the same checkpoint are bit-identical.
* **R² and energy-MMD are PerturbDiff-native and not comparable across models.** Use
  `sampling.run_cell_eval=true`, which runs the same cell-eval call STATE does (cell-eval pinned
  to `==0.7.2` to match STATE's installed version).
* **Real controls go into both adatas**, i.e. STATE's `--inject-real-controls` convention —
  compare against STATE runs that used that flag.

## Changes made to the vendored code

Kept deliberately minimal. Phase 0 required two fixes for **pre-existing** bugs (neither caused by
the Python upgrade — the old 3.9 env had the same omegaconf and the same torch `weights_only`
default). Full write-ups in [ZARR_BENCHMARK.md](ZARR_BENCHMARK.md) §7.

| File | Change | Why |
|---|---|---|
| `configs/cov_encoding/trixie_onehot.yaml` | `${..path.tmp_dir}` → `${...path.tmp_dir}` in the `gene_embedding_path` **list** | list elements are one level deeper; `OmegaConf.resolve()` raised `InterpolationKeyError` before any run could start |
| `src/apps/training/training_pipeline.py` | `weights_only=False` on `trainer.fit()` / `trainer.validate()` | torch ≥ 2.6 cannot unpickle the OmegaConf hparams in our own checkpoints; Lightning's internal reloads have no fallback, so runs died on the final validate |
| `src/apps/training/training_pipeline.py` | final validation uses `ckpt_path=cfg.save_dir_path` (the final checkpoint) instead of `"best"` | `"best"` needs the `ModelCheckpoint` monitor to have fired; fold 0's empty validation split means it never does |
| `configs/rawdata_diffusion_sampling.yaml` | `num_sampled_batches: 4` → `null` | the old default silently evaluated ~4 batches of the test split and reported metrics as if complete |
| `src/apps/sampling/sampling_generation.py` | `PARTIAL EVAL` warnings, batch-count capping, post-hoc coverage check, shuffle guard | make partial coverage impossible to miss, stop `StopIteration` when over-requesting batches, and assert the eval order is reproducible |
| `src/data/sampler.py` | expose `original_total`; correct the `shuffle=False` log message | the old message claimed the sampler "still shuffles … do not use for generation" in exactly the branch that *is* deterministic |
| `pyproject.toml` | pytorch-lightning floor `>=2.0` → `>=2.5`; Python `>=3.11,<3.13`; added `zarr`/`polars`/`pyarrow`/`cell-eval==0.7.2` | the `weights_only` kwarg needs 2.5; zarr v3 needs 3.11 |
| `src/data/dataset/dataset_core.py` | store + `_retrieve_counts` + `_get_selected_gene_vars` dispatch on `data.source_format` | the only three format-specific points; everything downstream is shared |
| `src/data/dataset/dataset_grouping.py` | `data.cell_set_group_by`; cache filename carries a content hash; atomic writes; `keep_control_cell` index-space fix | see the sharp edges above — the last one put non-control cells into "control" cell sets |
| `src/data/sampler.py` | tolerant group-key arity; `original_total`; empty-split guard; corrected a backwards log line | 2-key grouping, coverage checks, and fold 0's empty validation split |
| `src/data/data_module/data_module.py` | `ZarrPerturbationDataModule`; `prefetch_factor` only when `num_workers>0` | the latter broke every `num_workers=0` debug run |
| `src/models/covariate_encoding.py` | Ensembl→symbol translation, `control_pert_key`, zero-fill guard, case-insensitive cell types | `genept`/`llm` would otherwise silently zero-fill |
| `src/apps/sampling/sampling_generation.py` | zarr controls + obs, coverage/PARTIAL-EVAL warnings, batch capping, cell-eval hook | see the sharp edges above |
| `configs/` | `data/ihabxcell_zarr.yaml`, `path/hooke_tx_path.yaml`, `cov_encoding/hooke_tx_zarr.yaml`, `manifests/`, `metadata/` | the benchmark configuration |

## Benchmark runs

`launch/` holds the slurm scripts, following the conventions in
`../STATE/state/train_state_zarr.sh` and `../../launch/*.sh` (sbatch array,
`partition=def`, `h100:1`, `qos`/`wckey=hooke-predict`):

| script | what |
|---|---|
| `launch/_pd_common.sh` | shared paths + the combo grid |
| `launch/perturbdiff_zarr_canary.sh` | short GPU smoke; run this before any long job |
| `launch/perturbdiff_zarr_train.sh` | 2x2 = 4 runs: grouping A/B x lr, all tasks concurrent |
| `launch/perturbdiff_zarr_eval.sh` | cell-eval on the full test split |

```bash
TRAIN=$(sbatch --parsable launch/perturbdiff_zarr_train.sh)
sbatch --dependency=aftercorr:$TRAIN launch/perturbdiff_zarr_eval.sh
```

Array tasks run concurrently by default. To be polite on a busy cluster add a throttle
(`--array=0-3%2`), or change one already submitted:
`scontrol update JobId=<id> ArrayTaskThrottle=<n>` (`0` = unlimited).

`aftercorr` pairs eval task *N* with train task *N*, so each eval starts as soon as its own
training finishes.

The grid (`_pd_common.sh`): `cell_set_group_by` ∈ {`pert_celltype_batch`, `pert_celltype`} x
`optimizer.lr` ∈ {0.002, 0.0005}, everything else upstream's Replogle recipe
(`use_cell_set=32`, `micro_batch_size=128`, `replogle_gene_encoding=genept`) at STATE's budget
(80k steps, checkpoint every 2000).

Measured on an H100: ~11.3 it/s, so 80k steps is ~2 h; peak RSS 7.5 GB. Outputs land under
`$PD_OUT` (default `/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out`) —
`runs/<name>` for the final checkpoint, `eval/<name>/cell_eval_<ts>/` for the CSVs, `logs/` for
slurm output.

> Reruns with the same grid **overwrite** previous checkpoints and eval outputs, because run
> names are derived from the grid. Set `PD_RUN_SUFFIX` (or `PD_OUT`) to keep an arm.

Every axis is overridable from the environment, so a follow-up sweep reuses the same scripts:

```bash
export PD_GROUPS_CSV=pert_celltype PD_LRS_CSV=0.001,0.003,0.005 PD_RUN_SUFFIX=_ema
sbatch --export=ALL --array=0-2%2 launch/perturbdiff_zarr_train.sh
```

> Export in the submitting shell and pass a bare `--export=ALL`. Inlining them as
> `--export=ALL,VAR=a,b,c` does **not** work: sbatch splits that string on commas, so a
> comma-containing value is truncated to its first element and the remainder are read as
> variable names.

**Results** (cell-eval, 380 perturbations; full tables and caveats in
[ZARR_BENCHMARK.md](ZARR_BENCHMARK.md) §8):

| configuration | pearson_delta | mse | discrim_l1 |
|---|---|---|---|
| session start: `pert_celltype_batch`, no EMA, cs32 (n=1) | 0.031 | 0.119 | 0.501 |
| + `pert_celltype` grouping (n=1) | 0.350 | 0.0122 | 0.728 |
| + EMA (n=1) | 0.386 | 0.0109 | 0.736 |
| + `use_cell_set=16` — best config, **mean ± sd over 4 seeds** | 0.4104 ± 0.0233 | 0.0079 | 0.7568 ± 0.0078 |
| best *single* run (seed 42) | 0.4298 | 0.0071 | 0.7678 |
| **STATE** (n=1) | **0.4313** | **0.0060** | **0.7973** |

**STATE wins.** Quote the mean ± sd row, not the best run. Correct configuration moved PerturbDiff
from 0.031 to 0.41 — two orders of magnitude — and left it consistently behind STATE, most clearly
on discrimination (5.2 sd).

What each sweep showed:

* **Grouping** (the A/B): decisive — `pert_celltype` 0.350 vs `pert_celltype_batch` 0.031.
* **Seed replication** at the best config, 3 fresh seeds x 2 encodings: **seed sd is 0.0233
  pearson_delta**, range 0.048 over 4 seeds. This is the noise floor every other row is measured
  against.
* **EMA**: improved all 12 metric comparisons it was tested on (+0.036, ~1.5 sd — plausible, not
  confirmed). `configs/lightning/base.yaml` ships `decay: 0`, which disables it; upstream uses 0.99.
* **lr**, 7 points: 0.002 is a genuine optimum (upstream's own Replogle value). Above it
  `mse`/`mae` keep improving while `pearson_delta`/discrimination collapse.
* **`use_cell_set`**, 7 points: an inverted U. The cs64 dip (0.272) is **6 sd** below the cs16 mean,
  so it is real and unexplained; cs2 is the worst point, so the method does need a real population.
  But cs16-over-cs32 is only ~1.1 sd — weakly supported.
* **Perturbation encoding**: the earlier "genept beats onehot by ~0.05" was a single-seed artefact.
  Replicated at cs16, the gap is **+0.0055** (a quarter of one sd) and *reverses* on discrimination,
  where onehot leads by 2.3 sd. genept buys no measurable accuracy; onehot is also the closer
  analogue of STATE's encoding, so prefer it for the architectural comparison.

> ⚠️ Effects in the list above other than the grouping A/B, the cs64 dip and the lr collapse were
> measured at **n=1 against a noise floor of sd 0.023** — treat them as provisional.
> **STATE is also a single seed**, so the gap is measured against a point estimate on one side;
> 3 STATE seeds is the one measurement still missing.

## Conventions

* All five phases are implemented. Each `ZARR_BENCHMARK.md` now mixes **what was built** (marked
  DONE) with the original design note kept for rationale — where they disagree, the code wins.
* Keep the h5ad path working: add zarr modules alongside, switched by `data.source_format`.
* Match STATE's data decisions exactly (gene panel, split file + fold, control convention,
  cell-eval arguments) or the comparison is not meaningful.
