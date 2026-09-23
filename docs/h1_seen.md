# h1_seen — RHJ checkpoints on H1 hESC (seen perturbation, UNSEEN assay, UNSEEN cell line)

> The live panel has **80 perturbations** (since 2026-09-23). The earlier 99-perturbation panel
> and every number measured on it are archived in `archive/docs/h1_seen_99_results.md` and must
> not be quoted.

## Context

**Standing objective.** Predict a perturbation's effect in a context where it was never
measured, from a single control cell.

**This benchmark's question.** Models trained on `rhj_fewshot` (HepG2 + Jurkat + RPE1) have
never seen H1 hESC (VCC 2025): not its perturbations, not its controls, not its assay. For
perturbations they HAVE seen in training, how much of their RHJ score survives?

**Read it together with `k562_seen_h1set`** (docs/k562_seen.md): the SAME 80 perturbations in
K562, in a seen assay. The K562 → H1 gap is the assay shift; K562 → RHJ is the cell line.

**Varied:** the evaluated cell line and assay. **Fixed:** the checkpoints (no training), the
split-half protocol (5 splits, halves A/B from the H1 cells only, model input = one half-B
control cell, cell-eval 0.6.0, nocal), the 4 headline metrics: `pearson_delta` ↑, `mae` ↓
(error), `overlap_at_N` ↑, `discrimination_score_l1` ↑ (chance 0.5).

**What each outcome means.**
- model `pearson_delta` near or above `transfer`, well above `ctrlcell` → what the models
  learned about a perturbation carries over to a new cell line and assay;
- near `ctrlcell` / `meanpert` → it does not leave the training cell lines.

## Panel (built 2026-09-23, job 18367618)

| item | value |
|---|---|
| source | VCC 2025 H1: `adata_{Training,Validation,Test}.h5ad` (paths + sha256 in `benchmark.json`) |
| local copy (on HF) | `data/benchmarks/h1_seen/h1_seen_raw.h5ad` — raw counts, 105,338 cells × 18,080 genes = the 80 perts + 38,176 controls; sha256 `ea72830a…` |
| perturbations | **80** = `data/benchmarks/shared/h1_k562_perts.json`: H1 perts eligible in rhj_fewshot with ≥ 10 cells in H1 AND in the K562 essential screen. 56 RHJ-`train` (trained in HepG2, Jurkat, RPE1), 22 `test` + 2 `val` (withheld from HepG2) |
| controls | 38,176 (identical in all 3 VCC files, read once) |
| per split | 33,559 perturbed + 19,088 control cells per half |
| gene axis | **6,535** of the 6,983 RHJ axis genes (448 absent in H1 → 0 at input, not scored) |
| normalisation | log1p(count / rowsum × 2000), rowsum over the 8,910 H1 genes also in the RHJ HepG2 file |
| depth | median 53,674 UMI (not thinned; part of the shift) |
| `meanpert` | RHJ HepG2 training grand-mean DELTA placed on the H1 control mean |
| `transfer` | cell-weighted mean delta of the same pert over the RHJ lines where it was TRAINED |

## Gene axis — identical for every arm

| arm | model input | scored output |
|---|---|---|
| Occam | the 6,535 panel genes at their encoder columns; the 448 absent genes and all non-axis columns are 0 (as in RHJ training) | 6,535 |
| TxPert / STATE / CellFlow / PerturbDiff | 6,983-wide RHJ axis with the same 448 at 0; emission gathered back to the 6,535 | 6,535 |
| baselines | the panel | 6,535 |

Competitors with a cell-type label are given **HEPG2**; STATE / PerturbDiff also get a seeded
per-cell draw of HepG2 batches (a constant batch distorts STATE by 0.37). Recorded in each
export manifest as `context_proxy`. Occam has no cell-line input.

## Rebuilding

`OUT=/tmp/h1_check bash scripts/repro/build_benchmark_h1.sh` — checks the local copy's sha256,
rebuilds cohort, splits and baselines, compares every quantity with the published panel.
**Verified 2026-09-23 (job 18368097):** rebuilt from the 80-pert HF copy → **MATCH** on axis, cohort, every cell of `X`, labels, batches, all 5 A/B splits, `meanpert`, `transfer`.

## Arms (launched 2026-09-23, `BENCH=h1_seen bash scripts/repro/ood_seen.sh`)

| arm | job |
|---|---|
| baselines, control-draw seeds 42–46 | 18368098–102 |
| Occam Y06 / Q02 / M06 / M03 | 18368103 |
| TxPert (single-cell ctrl) | export 18368104 → score 18368105 |
| STATE | 18368106 → 18368107 |
| CellFlow | 18368108 → 18368109 |
| PerturbDiff genept | 18368110 → 18368111 |

## Results

⏳ Pending. Written here as each arm's 5 splits land, before being reported.
