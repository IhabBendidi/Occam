# Reproducing the two main Occam vcell results

## What these scripts are for

Four numbered commands per benchmark: **train → evaluate → inspect the metrics → predict on
your own cells**. Every script is runnable as-is from the repo root, with no `--set` overrides,
because `configs/rhj_fewshot.yaml` and `configs/norman_combo.yaml` now ship the default recipe
(`docs/default_recipe.md`) and resolve to exactly the published arms.

| | `rhj_fewshot` | `norman_combo` |
|---|---|---|
| task | unseen perturbation, seen cell line family | unseen PAIR from seen singles |
| data | HepG2 + Jurkat + RPE1 | K562 (Norman 2019, GEO rebuild) |
| published arm | `Y06` | `NQ02` |
| headline | `pearson_delta` **0.4043** (bar: `transfer` 0.2323) | **0.8169** / **0.7380** (bar: `sum_of_singles` 0.8824 / 0.7757) |
| train | `rhj_train.sh` | `norman_train.sh` |
| evaluate | `rhj_eval.sh` | `norman_eval.sh` |
| predict | `occam_predict.py` | `occam_predict.py --compose decode_seq_mean` |

## Before anything

```bash
cd <repo root>
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Two things must be on disk, and neither is in git (both are gitignored, ~6 GB):

| what | where it goes | how to get it |
|---|---|---|
| the pretrained encoder | `checkpoints/release/occam_small_encoder_decoder.pth` | the `checkpoints/` folder of the HF repo (`docs/hf_release.md`) |
| the benchmark artifacts | `data/benchmarks/<benchmark>/` and `data/eval/` | the `benchmarks/` and `eval/` folders of the same repo |

Inside the research cluster, `bash scripts/bootstrap_assets.sh <SOURCE_REPO>` copies both out
of the source tree instead. Training starts from the frozen encoder; the published checkpoints
already contain it, so **inference and evaluation need only the checkpoint and the benchmark
artifacts**, not the encoder file.

## 1. Train

```bash
sbatch scripts/repro/rhj_train.sh        # ~4.5 h on one H100
sbatch scripts/repro/norman_train.sh     # ~3 h on one H100
```

Each writes `runs/<benchmark>_repro__<suffix>/vcell/checkpoints/{best_model,final_model}.pth`.
**`final_model.pth` is written only at the very end**, so its presence is the completion marker;
`best_model.pth` appears as soon as validation improves and is the one to evaluate.

⚠️ `best_model.pth` is selected on **validation** `pearson_delta`. Validation picks the epoch
well but **misranks arms** (finding F-ckpt4, held on 5 consecutive grids), so do not use the
validation number to compare two configurations — evaluate them.

## 2. Evaluate

```bash
sbatch scripts/repro/rhj_eval.sh      # 5 splits x 1 arm
sbatch scripts/repro/norman_eval.sh   # 5 splits x 1 arm, --compose decode_seq_mean
```

Both default to the **published** checkpoint; pass a path as `$1` to score your own. Set
`ARMS=reference` to (re)score the baselines instead of a model.

⚠️ The array throttle is `%6` and is **not** cosmetic: `pdex` (cell-eval's Wilcoxon backend)
races on `/dev/shm` and `%12` produced ~50% marginal task failures.

⚠️ All evaluation is `--calibrate none` (`nocal`). Calibrated arms are not reported.

## 3. Read every metric, per split

```bash
python scripts/export_split_metrics.py --out data/release/metrics
```

Writes, for every arm and split already evaluated, **all** cell-eval metrics — not only the
four headline ones — as both a tidy long CSV and wide per-benchmark CSVs. See
`data/release/metrics/README.md` for the schema and which direction is good per metric.

## 4. Predict on your own cells

```bash
python scripts/repro/occam_predict.py \
    --checkpoint <best_model.pth> \
    --controls my_controls.h5ad \
    --perts NRF1 "CBL+CNN1" \
    --out predictions.h5ad
```

One prediction per (control cell × perturbation). `"A+B"` composes a pair using
`--compose` (default `decode_seq_mean`). Output `.X` is in the model's own **log1p** space at
the checkpoint's `target_sum`; `--raw-counts` inverts it back to counts.

⚠️ The gene axis is taken from the checkpoint, not from your file: genes it does not know are
dropped, genes missing from your file are zero-filled, and the script prints both counts.
A zero-fill rate above a few percent means the prediction is being made from a mostly empty
input — check your `--gene-col` first.
