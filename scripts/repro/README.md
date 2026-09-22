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
| 0. build the data | `build_benchmark_rhj.sh` | `build_benchmark_norman.sh` |
| 1. train | `rhj_train.sh` | `norman_train.sh` |
| 2. evaluate | `rhj_eval.sh` | `norman_eval.sh` |
| 2b. baselines | `baselines.sh` | `BENCHMARK=norman_combo baselines.sh` |
| 2c. competitors | `ARM=txpert competitors_rhj.sh` | — (single-cell-line panel) |
| 3. read metrics | `../export_split_metrics.py` | same |
| 4. predict | `occam_predict.py` | `occam_predict.py` (`decode_seq_mean` by default) |

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

## 0. Rebuild the data and splits exactly as we use them

```bash
bash scripts/repro/build_benchmark_rhj.sh        # ~25 min, ~120 GB RAM, no GPU
bash scripts/repro/build_benchmark_norman.sh     # ~15 min, ~60 GB RAM, no GPU
```

Each rebuilds the gene axis, the cohort, the splits, the 5 seeded split-halves and (for
Norman) the additivity strata from the published `.h5ad` files, then runs
`verify_benchmark.py`, which compares **every scientific quantity** against the published
artifact: the axis digest, the builder's own per-file content digests, the perturbation
counts, the input `.h5ad` checksums, the cohort digest and the membership of all seven
panels, pair by pair. Exit 3 naming each difference; exit 0 prints *REPRODUCED*.

Measured on 2026-09-22: **both** rebuilds reproduce the published benchmark exactly —
Norman (all 7 panel memberships, every digest) and RHJ (3/3 source checksums,
`artifact_sha256`, `counts`).

⚠️ **The verification is semantic, not a byte hash, and it has to be.** `benchmark.json`
embeds `git_sha`, the build date and the output path, so hashing the file reports a
mismatch on a rebuild that is bit-identical in content — which is exactly what the first
version of these scripts did. `axis.npz` is worse: it is a zip and carries per-entry
timestamps.

✅ **A rebuilt panel CAN score a published checkpoint** — this is the normal thing to do
and it needs no flag. `manifest.benchmark_sha256` is `sha256(benchmark.json)`, which
embeds `git_sha`, the build date and the output path, so a verified rebuild can never
reproduce it. Rather than refuse, the guard checks the property that digest stands in for,
which is stronger than the digest:

1. the **scoring axis** is identical (`gene_axis_sha256` is a content hash), and the
   benchmark `id`, `n_axis_genes` and `axis_scope` agree; **and**
2. **every perturbation the panel will score is held out of that checkpoint** — absent
   from its `split_info.train` and present in its `benchmark.excluded_perts`.

A different axis fails (1); a different split fails (2). The acceptance is printed, never
silent. Pass `--require-exact-benchmark` to demand byte equality instead.

**Measured 2026-09-22.** The published Norman checkpoint scored on a rebuilt panel, split 0:

| | rebuilt panel | published run |
|---|---|---|
| `pearson_delta` | 0.7782 | 0.7782 |
| `overlap_at_N` | 0.3065 | 0.3065 |
| `mae` | 0.0137 | 0.0137 |

Identical on all three.

⚠️ Both scripts **refuse to write into a directory that already holds a
`benchmark.json`**. Rebuild into a scratch dir (`OUT=/tmp/check bash …`), or
`FORCE=1` to replace the frozen artifact deliberately.

⚠️ Skip this entirely if you downloaded `benchmarks/rhj_fewshot/` and
`benchmarks/norman_combo/` from the Hub — those ARE the frozen artifacts.

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

## 2b. The baselines — the scale every score is read against

```bash
sbatch scripts/repro/baselines.sh                        # rhj_fewshot
BENCHMARK=norman_combo sbatch scripts/repro/baselines.sh
```

Deterministic given the panel and the seed (the control draw is
`default_rng(42 + 1000 * split)`, the same one the model arms use). A baseline that does
not reproduce means the **panel** differs, not the model — rebuild it with step 0.

Two of them are the real tests: `meanpert` (predict one average delta for everything —
a model at `meanpert` learned nothing perturbation-specific) and, on `norman_combo`,
`sum_of_singles` (`Δ_A + Δ_B`, which IS this architecture's hypothesis in closed form).

## 2c. The competitors on `rhj_fewshot`

```bash
ARM=txpert bash scripts/repro/competitors_rhj.sh   # also state | cellflow | perturbdiff
```

Submits train → export → score as three Slurm legs with `afterok` dependencies and
returns immediately. Measured: TxPert 0.3463 · PerturbDiff 0.2449 · STATE 0.2362 ·
CellFlow 0.2154, against our 0.4043.

⚠️ None reproduces its method's published figure and none is meant to: each is retrained
on our data at our axis and carries deliberate, recorded deviations. The script's header
lists them per arm; the full accounting is `docs/rhj_competitors_results.md`.

## 3. Read every metric, per split

```bash
python scripts/export_split_metrics.py                        # published scope (default)
python scripts/export_split_metrics.py --scope all \
    --out data/release/metrics_all                            # every arm ever evaluated
```

Writes **all 35** cell-eval metrics — not only the four headline ones — per arm, per
split, per panel, as a tidy long CSV plus wide (mean + SEM) and one-column-per-split
views, and copies cell-eval's own per-perturbation rows.

`--scope published` (the default, and what is tracked in git) keeps **our published
arms** (`Y06`; `NQ02` under all 12 composition modes) plus **every baseline** and
**every competitor arm**. `--scope all` adds the campaign's other ~360 eval roots and is
published to the Hub as `metrics_all/`.

See `data/release/metrics/README.md` for the schema, which direction is good per metric,
which metrics must not be ranked on, and which competitor arms are withdrawn.

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
