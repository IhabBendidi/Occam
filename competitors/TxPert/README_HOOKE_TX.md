# TxPert on the hooke-tx replogle_nadig benchmark

TxPert (upstream codebase name **GSPP**, `valence-labs/GSP`) as a competitor arm alongside
[`../STATE`](../STATE) and [`../PerturbDiff`](../PerturbDiff), scored with the same cell-eval
protocol so all four models land in one table.

`README.md` is upstream's. This file records what was changed and why.

## Which TxPert this is

TxPert has two orthogonal axes, and it is easy to conflate them:

* **data mode** — `datamodule.xcell` / `datamodule.xcell_v2`
* **perturbation model** — `model.pert_model.model_type`, ~10 options upstream

"xcell mode" does **not** imply the adjacency map; upstream ships `config-x-cell-exphormer`,
`config-x-cell-multilayer` and `config-x-cell-baseline` too. This arm is specifically
**xcell_v2 data + `simple_adjacency_map`**, the combination that scored best on this benchmark.

## What was removed, and why

Upstream is a research monorepo. Everything unreachable from the adjacency-map config is gone,
which is what makes a `uv` environment tractable — the deleted models were the only importers of
`torch_scatter`, `torch_sparse` and `molfeat`, all of which are painful or impossible to resolve
with uv.

| Removed | Reason |
|---|---|
| 9 pert models (`exphormer`, `node2vec`, `hybrid_gnn`, `hybrid_wrapper`, `multi_graph`, `node_feat_mlp`, `basic_gnn`, `basic_gnn_comp_gene`, `comp_pert_simple_models`) | Not selected by the config; sole importers of `torch_scatter` / `torch_sparse` |
| `basal_state_models/{moe,vae}.py` | Config uses `cntr_model.model_type: mlp` |
| `models/baselines.py` | hooke-tx has its own baselines (`hooke_tx/eval/baselines/`) for this benchmark |
| `src/GSP/` (Tx-Evaluation) | Only reachable from `ihab_et_al_benchmark()`, which runs under `slow_benchmark: true`; the config sets `false` |
| `norman_export.py` | Hardcoded to `scMAE-main/` paths; superseded by the cell-eval export |
| Molecular-perturbation path | Benchmark is CRISPRi-only. It also eagerly `torch.load`-ed a molgps pickle from another user's `outgoing/` on **every** run |
| `read_filter_adata` + the multi-h5ad loading chain | Read per-cell-type h5ads from a hardcoded registry under three different users' directories |
| `INTERNAL_CELL_TYPE_DATASETS` (HUVEC Perturb-seq) | Reachable only with `use_added_data: true`; benchmark sets `false` |
| `*.bak` / `*.bak2`, `run_sclambda.py`, dead configs & sweeps | Not part of this arm |

`gspp/` went from 28 modules to 13.

## External dependencies

Only two paths outside this directory, both read-only:

| What | Where | Note |
|---|---|---|
| Knowledge graphs (`go`, `string`, `pxmap-top1`, `txmap-top1`) | `/rxrx/data/valence/pathfinder/modules/gspp/graphs` | **Byte-identical** to the files hooke-tx's own `AdjacencyMapEmbedder` reads (`hooke_tx/architecture/embedding.py`). Not copied: this arm is no more exposed than hooke-tx already is. Traversal needs the `D-Valence-R` group. |
| Expression data | zarr datasets named in a manifest TOML | `X.zarr` (v3) + `obs/var.parquet`, shared with STATE and PerturbDiff |

Deliberately **not** used: `/rxrx/data/valence/pathfinder/modules/gspp/cache_v2` (2.8 TB, another
user's). The preprocessing cache is written to `./cache/`, which is gitignored.

The two perturbation/gene vocabularies in `data/` are tracked. Upstream's `.gitignore` excludes
`/data/`, so they are force-added — `_initialize_gene_sets` cannot run without them.

## Environment

```bash
cd src/hooke-tx/competitors/TxPert
export UV_CACHE_DIR=/rxrx/scratch/sandbox/$USER/.uv-cache        # see below -- do not skip
export UV_PYTHON_INSTALL_DIR=/rxrx/scratch/sandbox/$USER/.uv-python
uv sync
```

Python 3.11–3.12, matching `../STATE/state` and `../PerturbDiff`. `cell-eval==0.7.2` and
`pdex==0.2.2` are pinned to the versions hooke-tx and STATE resolve, so DE metrics are
comparable across arms (PerturbDiff currently resolves pdex 0.3.0).

`gspp` had no `__init__.py` upstream and was imported as an implicit namespace package; it is now
a declared package so `uv sync` installs it properly.

### Gotcha: keep the uv cache off NFS

uv's default cache is `~/.cache/uv`, and `/mnt/ps/home` is NFS (`172.20.2.21:/home_bh2`) while
the venv lives on GPFS. Installing torch means unpacking ~13,000 small files; doing that onto NFS
**and** across a filesystem boundary (so uv byte-copies instead of hardlinking) took over an hour
without finishing — the process sat at ~0.5 s of CPU per hour, entirely I/O-blocked. Pointing
`UV_CACHE_DIR` at GPFS brings a cold, full build down to **~1 minute**:

```
Prepared 113 packages in 1m 01s
Installed 113 packages in 1.40s
```

### Gotcha: pandas 3

Upstream pinned `pandas==2.2.3`; we take a floor instead, and pandas 3 broke this code in
**three** places. Each was rewritten to a version-independent equivalent and each equivalence
was checked against the old behaviour rather than assumed.

| Where | Symptom | Cause |
|---|---|---|
| `graphmodule.py` `process_graph` | `KeyError: 'target'` on every graph load | `groupby("target").apply(...)` no longer passes the grouping column in, so `"target"` vanished from the result |
| `datamodule.py` `_process_treatment_data` | `treatment_data` silently became **0 rows** | `categorical.str.split("+")` returns the *repr* `"['ABCB7', 'ctrl']"` instead of a list; iterating it yields characters, so every pert2id lookup failed |
| `metrics.py` `compute_mean_control` | `operands could not be broadcast (70,2000) (70,1999)` | `groupby('group').apply(lambda x: x.iloc[:, :-1].mean(0))` sliced off the grouping column that pandas 3 no longer adds — dropping a real gene |

The middle one is the cautionary tale: it failed **silently**, producing an empty dataset rather
than an exception. If you bump pandas again, re-run the loader assertions in
`tests/test_zarr_pipeline.py` rather than trusting that training still starts.

## Running the benchmark

```bash
uv run python main.py --config-name=config-txpert-zarr-adjmap   # single run
sbatch launch/txpert_zarr_train.sh                              # 3 seeds on SLURM
uv run pytest tests/test_zarr_pipeline.py                       # loader regression tests
```

### Data path

`gspp/data/zarr_loader.py` reads the datasets named in
`configs/manifests/ihabxcell_hepg2_zarr.toml` (byte-identical to STATE's copy, enforced by a
test) and hands `PertDataModule` an AnnData with gspp's obs schema. Everything downstream --
control matching, the GEARS-derivative model, the metrics -- is upstream's.

Three things are genuinely new, and each has a test:

1. **Ensembl to symbol.** The zarr is Ensembl throughout; gspp and all four knowledge graphs are
   symbol-keyed. Translation uses the map vendored at `data/gene_ensembl_map.parquet` -- the same
   table hooke-tx's `GeneIDMapper` uses, so both arms resolve graph nodes identically.
2. **Index-based splits.** hooke-tx stores the fold as row indices; gspp splits by perturbation
   name. `load_split_perturbations` converts, and *verifies the two sides are disjoint* -- if a
   perturbation straddled the split, the name-based exclusion downstream would silently leak.
3. **Vocabulary extension.** See below.

### Deviation from upstream: the perturbation vocabulary

`pert2id` is a fixed 9,853-symbol list shipped in `data/gears_gene_set.csv`, not derived from the
data. 33 of the benchmark's 2,023 perturbed genes are missing from it -- mostly HGNC renames the
GEARS list predates (`HARS1`<-`HARS`, `LARS1`<-`LARS`, `MARS1`<-`MARS`, `H2AZ1`<-`H2AFZ`, ...) --
and **9 of those are in the 380-perturbation HepG2 test set**. `_process_treatment_data` drops
cells outside the vocabulary, so stock behaviour would score TxPert on 371 perturbations while
STATE / PerturbDiff / hooke-tx score 380.

`_extend_pert_vocabulary` appends them (9,854 -> 9,887 entries, adjacency +0.3%). This is a
deliberate departure from upstream, made so the scored set is identical across arms.

### No normalization

`X.zarr` is already `log1p(target_sum=1e4)` and is fed to the model untouched, matching STATE and
PerturbDiff. `target_sum`, `hvg_number`, `min_cells`, `min_de_genes` and `knockdown_threshold` are
inert on this path. Upstream's exporter renormalized predictions to a magic `target_sum=823`
before scoring; that is gone.

### Scoring

`gspp/celleval_eval.py` is copied verbatim from PerturbDiff's port of STATE's `tx predict`
scoring, so all three arms share one protocol: clip to `[0, 14]`, per-cell-type split,
`skip_metrics=[pearson_edistance, clustering_agreement]`, cgroup-aware threads.
`export_for_celleval` builds row-aligned pred/real AnnData with the real HepG2 controls injected
into both, so DE in each is measured against the same baseline.

Output CSVs land in `outputs/celleval/<run-id>/` and mirror STATE's layout. Note the filename
prefix is the cell-type string, which is `hepg2` here versus `HepG2` in STATE's own outputs --
cosmetic, but it means the files are not name-identical across arms.

## The few-shot arm: a real validation split, and selection on it

Everything above describes the ORIGINAL arm (`config-txpert-zarr-adjmap`), which has no validation
split. `config-txpert-zarr-fewshot` adds one and brings TxPert under the same
validate -> select -> test protocol as `../STATE`, `../PerturbDiff` and hooke-tx itself.

### What was wrong without it

| | |
|---|---|
| Upstream monitored `test_pearson_delta` | checkpoint selected on the test set |
| The vendored arm used `save_last` instead | honest, but the 50-epoch budget is then unchosen |
| `on_validation_epoch_end` scored the TEST loader every epoch | 50 test evaluations per run, so test metrics were visible throughout training |
| `val_dataloader()` returned `test_dataloader()` when no validation dataset existed | every `val_*` metric on this arm was a TEST metric. STATE shipped the identical fallback |
| Hyperparameters came from `sweep-xcell-v2-training-params` | test-selected, and still are -- see the caveat above. Only *checkpoint* selection is fixed here |

### Why a name-based split can express this one

gspp splits by perturbation **name**, so a validation set built by sampling cells could not be
represented at all. The fewshot split happens to be name-shaped, and this is asserted rather than
assumed (`tests/test_zarr_pipeline.py::test_fewshot_split_resolves_to_whole_perturbations`):

```
validation = 60 whole perturbations = 4,825 HepG2 cells
validation & finetune overlap = 0        validation & test overlap = 0
every HepG2 cell of those 60 genes is in validation  (4,825 == 4,825)
```

Validation is held out of HepG2 training **only**; those 60 perturbations keep training in
jurkat/k562/rpe1. That is the fairness condition, not a convenience: hooke-tx, STATE and
PerturbDiff all carve validation out of the HepG2 rows, so excluding the names everywhere would
train TxPert on strictly less data than the arms it is compared against. Both held-out groups go
into one `_xcell_v2_hepg2_exclude` set and `condition_group["train"]` still lists them -- the row
mask, not the name list, is what holds them out.

⚠ One line in `zarr_loader.load_split_perturbations` was the leakage mechanism:
`train_names |= (all_names - test_names)`, which exists so the three unsplit cell lines become
training data but made "not test" mean "train". A populated validation list would have been swept
back into training silently, since every name is observed somewhere. Validation is now subtracted
there too.

### The two control conventions, and why no retraining was needed

Test is reported under both conventions, from one checkpoint and one forward pass:

| arm | controls on the pred side |
|---|---|
| `real` | the measured controls, swapped in by the shared scorer |
| `decoded` | **model outputs**, produced with perturbation id `-1` |

The `decoded` arm needed no training change, because TxPert already trains for it:

* `embed_cntr: true` puts `ctrl -> -1` in `pert2id` (`gspp/data/datamodule.py:220-238`)
* control cells are injected as training **targets** with condition `['ctrl']` (`:1449-1463`)
* `gears_derivative.py:256-273` maps `ctrl`/`-1` to `z_p[-1]`

`z_p[-1]` is Python negative indexing onto row `num_perts - 1`. That row works as a control token
only because **no real gene owns that id**: it is isolated in every knowledge graph, so its
embedding is the MLP bias. This holds by the accident that `len(pert2id)` counts the `ctrl: -1`
entry -- with `embed_cntr: false`, `-1` would collide with the last real gene. There is a test
(`test_control_token_is_not_a_graph_node`), and a knowledge-graph edge whose symbol were literally
`ctrl` would map to `-1` and silently contaminate that row.

Note `omit_cntr: false`, so `z_p[-1]` is added to *every* sample: a perturbed cell gets
`z_p[pert] + z_p[-1]` and a control gets `z_p[-1]` alone. Controls stay distinguishable. This is
the tuned configuration -- do not "fix" it.

Like STATE, and unlike hooke-tx, `decoded` here is **not** the PCA-decode sense of "reconstruct":
TxPert emits a genuinely predicted control, and there is no codec.

### How it is scored

`export_for_celleval` can only ever produce the inject-real convention -- it injects the real
controls into pred *and* real itself. So `gspp/export_artifacts.py` writes `adata_pred.h5ad` +
`adata_real.h5ad` once and the **shared** scorer (`../../hooke_tx/eval/celleval_artifacts.py`)
scores them twice into `results_real/` and `results_decoded/`. Predicting twice into one directory
is what left STATE's `test_replogle_zarr/` with two agg CSVs that survive only because someone
renamed them by hand.

The controls are predicted through `XcelltypePerturbDataset` and `evaluate()`, the same classes
that build the test batches, so the basal input, dose encoding and `p` embedding cannot drift from
what the perturbed rows get.

### Running it

```bash
sbatch launch/txpert_fewshot_train.sh                      # 5 seeds, 42-46
sbatch --dependency=aftercorr:<jobid> launch/txpert_fewshot_eval.sh

EPOCHS=2 CHECK_VAL=1 TAG_SUFFIX=_smoke sbatch --array=0 \
  --export=ALL,EPOCHS,CHECK_VAL,TAG_SUFFIX launch/txpert_fewshot_train.sh
```

Seeds are **42-46**, not TxPert's historical 1-3, so the arm shares its seed set with the others.
`TAG_SUFFIX` keeps a smoke run out of the real run's directories: `save_top_k=1` only prunes files
it tracked itself, so a seed-42 smoke would otherwise leave a checkpoint sitting next to the real
one.

TxPert already uses `every_n_epochs`, not `every_n_train_steps`, so the Lightning hook-ordering bug
that made PerturbDiff keep a stale checkpoint (`every_n_train_steps` saves in `on_train_batch_end`,
before the validation loop) does not apply here.

### Hyperparameters, and a caveat

`config-txpert-zarr-adjmap.yaml` uses the optimum recovered from
`sweep-xcell-v2-training-params` (217 runs): `dropout 0.1 / hidden 512 / latent 128 / lr 3e-4 /
warmup 5 / wd 0`, `test_pearson_delta` 0.4779 versus 0.4660 for the shipped defaults.

**That sweep maximised a test metric** -- xcell_v2 has no validation split -- so the
hyperparameters are test-selected and should be reported with that caveat. Upstream compounded
this by also pointing `ModelCheckpoint(monitor=...)` at `test_pearson_delta`; here checkpointing
is `save_last` instead, so at least model selection is not test-driven.

### Reference numbers

A 1-epoch canary (H100, ~3 min/epoch) reached cell-eval `pearson_delta` **0.3883** over all 380
held-out perturbations, against 0.4779 for the tuned 50-epoch run under TxPert's own metric. The
canary's internal `test_pearson_delta` was 0.3881 -- computed by a completely different code path
from cell-eval's, which is a useful check that the export is faithful.
