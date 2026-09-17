# CellFlow (competitor baseline)

Vendored copy of [theislab/CellFlow](https://github.com/theislab/cellflow) for
fair comparison against Hooke-Tx — same pattern as `competitors/STATE/`.

| Path | Role |
|------|------|
| `cellflow/` | Upstream package (`cellflow-tools`; own `.venv`) |
| `hooke_bridge/` | Hooke dump → AnnData + splits + PCA |
| `configs/` | `norman.yaml`, `replogle_rpe1.yaml`, `replogle_k562_gwps.yaml`, `replogle_nadig_state.yaml` |
| `train_cellflow.py` / `train_cellflow.sh` | Train entrypoint + shared sbatch body |
| `train_{norman,replogle_rpe1,replogle_k562,replogle_nadig_state}.sh` | Dataset wrappers |
| `sweep_replogle_nadig_state.sh` | Tutorial hparam deltas from the selected STATE default |
| `predict_cellflow.py` / `predict_cellflow.sh` | Predict + CellEval |
| `make_state_split.py` | Rebuild the STATE HepG2 split from existing dumps |
| `tests/` | Bridge unit tests |
| `smoke_train{,_phase2,_multisource}.py` | Phase 0 / 2 / 4 CPU smokes |
| `.python-version` | Pin **3.12** |

Vendored from `main` @ `446ed6073c60` (2026-07-12).

## Setup

```bash
cd src/hooke-tx/competitors/CellFlow/cellflow
export UV_CACHE_DIR=/path/to/bc-hooke/.uv-cache   # optional
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 HATCH_VCS_PRETEND_VERSION=0.1.0

uv sync --python 3.12
uv pip install requests                 # eager import; only in [embedding]
uv pip install "jax[cuda12]==0.11.0"     # GPU nodes; pin to locked jax
uv pip install 'cell-eval==0.8.1'        # Match Hooke-Tx Phase 3 scoring

uv run --frozen --no-sync python ../smoke_train.py          # Phase 0
uv run --frozen --no-sync python ../smoke_train_phase2.py   # Phase 2
uv run --frozen --no-sync pytest ../tests/test_hooke_bridge.py -q
```

Login nodes fall back to CPU JAX (`CUDA error 303`). On Hopper expect
`[CudaDevice(id=0)]`. Always `unset VIRTUAL_ENV` and `uv run` from `cellflow/`.

## Datasets

| Config | Dump | Split | Combos | Scale |
|--------|------|-------|--------|-------|
| `norman.yaml` | `.../norman2019` | `norman.json` | 2 | ~111k cells |
| `replogle_rpe1.yaml` | `.../replogle2022_rpe1` | `replogle_rpe1.json` | 1 | ~248k |
| `replogle_k562_gwps.yaml` | `.../replogle2022` | `replogle_k562_gwps.json` | 1 | ~2.0M (512G sbatch) |
| `replogle_nadig_state.yaml` | `.../ihabxcell_replogle_nadig_*` (4 lines) | `replogle_nadig_state.json` | 1 | ~643k |

Preprocess for the first three: `target_sum=300000`, `log1p`, trek gene list with
`on_missing_genes=intersect`. The Replogle-Nadig dumps are already
`log1p(target_sum=10000)`, so that config sets `target_sum: null, log1p: false`.

Train/predict load **fold row indices only** (needed for K562).

### Gene holdout vs. few-shot

The two split families need different gene representations, and mixing them up
produces `Representation for 'ENSG…' not found in adata.uns['gene_embeddings']`:

- **Strict holdout** (`replogle_rpe1`, `replogle_k562_gwps`): test genes never
  appear in training, so one-hot embeddings cannot represent them. These need
  ESM2-style embeddings to be meaningful — the configs are wired but the one-hot
  schema will fail at predict time.
- **Few-shot** (`replogle_nadig_state.yaml`): HepG2 **STATE** holdout
  (`replogle_nadig_state.json`, 945 test perts + non-targeting). Trains on
  HepG2 finetune + Jurkat + K562 + RPE1. HepG2 test is 70,718 cells / 945
  perturbations with **all 4,976 NCs in test**. Held-out genes are unseen in
  HepG2 finetune but KO'd on the other lines, so one-hot embeddings cover test
  (unlike the 380-pert xcell list, which has 8 tokens never seen anywhere).
  `replogle_nadig_xcell.json` is a different protocol — do not mix numbers.

Beware the name collision: hooke-tx `tasks/replogle_rpe1.yaml` is the standalone
`replogle2022_rpe1` benchmark, while `DATA_SOURCES["replogle_rpe1"]` is the
Replogle-Nadig RPE1 *training* source. Only the latter feeds the few-shot run.

### Cross-cell-line setup

`replogle_nadig_state.yaml` uses the multi-source path: `data_dir` is the
evaluated line (contributing only `finetune` rows) and `train_sources` lists the
supporting lines (contributing every cell). `sample_covariates`/`split_covariates`
of `[cell_type]` make CellFlow condition on the line and flow each perturbation
from its own controls.

STATE puts **all** HepG2 NCs in `test` (0 in finetune). `control_from_split_key:
test` copies only those DMSO rows into training as OT sources; test perts stay
out. Predict/CellEval uses the NCs already in the eval split (no inject).

The canonical `cell_eval/splits/replogle_nadig_state.json` is not writable; rebuild
with `python ../make_state_split.py` into the scratch path in the config.

## Train

```bash
cd competitors/CellFlow
NUM_ITERATIONS=50 RUN_NAME=norman_fold0_smoke sbatch train_norman.sh
NUM_ITERATIONS=50 RUN_NAME=replogle_rpe1_fold0_smoke sbatch train_replogle_rpe1.sh
# full defaults (20k iters):
sbatch train_norman.sh
sbatch train_replogle_rpe1.sh
# K562 needs 512G / 24h:
sbatch train_replogle_k562.sh
# Selected few-shot cross-cell-line default: 100k iters, tutorial VF width,
# and linear projection. Execute directly; it self-submits on qos=normal.
./train_replogle_nadig_state.sh
# Tutorial hparam deltas from that default:
DRY_RUN=1 ./sweep_replogle_nadig_state.sh all
./sweep_replogle_nadig_state.sh noise15 attn no_linproj
```

Outputs: `OUTPUT_DIR/RUN_NAME/{CellFlow.pkl,pca/,genes.txt,run_meta.json,config.yaml}`.
`CellFlow.save` pickles the train AnnData (~6 GB Norman; much larger on K562).
The STATE launcher derives paths from the clone and `$USER`; if the per-user
STATE split is absent, it builds it automatically from the shared dumps. SLURM
logs are written to the directory from which the launcher is executed.

The linear-projection default was selected from the September 2026 exploratory
sweep: it had the best Pearson delta (`0.4044`) and DE direction match (`0.7767`).
Because selection used test CellEval, confirm it on another seed/fold before
reporting it as an unbiased final benchmark.

Norman smoke verified: job `17039417` (50 iters, H100, ~2 min).

## Predict + CellEval

```bash
cd cellflow && uv pip install 'cell-eval==0.8.1'   # match Hooke-Tx

cd competitors/CellFlow
RUN_DIR=/rxrx/scratch/sandbox/$USER/cellflow/norman/norman_fold0_smoke \
  PROFILE=minimal MAX_CONTROL_CELLS=256 \
  sbatch predict_cellflow.sh

# Re-score existing h5ads:
EVAL_ONLY=1 RUN_DIR=... PROFILE=full sbatch predict_cellflow.sh

# few-shot HepG2 STATE:
sbatch predict_replogle_nadig_state.sh
```

Artifacts: `pred_{split}.h5ad`, `real_{split}.h5ad`, `celleval_{split}_metrics.json`.
STATE prediction uses 512 control source cells per perturbation and clips
reconstructed expression to Hooke-Tx's CellEval range, `[0, 14]`.

Smoke metrics (50-iter Norman, `minimal`, job `17041935`): `pearson_delta≈0.57`,
`mae≈0.079` — pipeline check only.

## Bridge API (sketch)

```python
from hooke_bridge import (
    load_hooke_anndata, apply_cellflow_gene_schema,
    fit_and_project_pca, save_pca_artifacts, load_fold, reconstruct_from_pca,
)

adata = load_hooke_anndata(dump, selected_genes=genes, on_missing_genes="intersect",
                           target_sum=300_000, log1p=True, row_indices=fold_rows)
apply_cellflow_gene_schema(adata, max_combination_length=1)  # 2 for Norman
```

| Hooke | CellFlow |
|-------|----------|
| `negative_control` / empty `gene_pert` | `obs["is_control"]` |
| `gene_pert` IDs | `obs["gene_1"]`… (+ `NULL` pad) |
| (derived) | `obs["pert_str"]` |
| — | `uns["gene_embeddings"]` one-hot |
| `cell_type` | `uns["cell_type_embeddings"]` one-hot + split covariate |
| `X` | `obsm["X_pca"]` (train-fit) |

## Status

- [x] Phase 0 — env + smoke train
- [x] Phase 1 — Hooke → CellFlow AnnData bridge
- [x] Phase 2 — train wrapper + sbatch
- [x] Phase 3 — predict + CellEval
- [x] Phase 4 — Replogle/Norman configs + docs
- [x] Phase 5 — cross-cell-line few-shot (Replogle-Nadig STATE)

**Later:** Tahoe streaming, ESM2 embeddings (needed for the strict holdouts),
GENOT, mol_pert, `cross_cell_line_trek` compound baseline.
