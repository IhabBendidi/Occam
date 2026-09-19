# `rhj_fewshot` competitors — results

**Objective.** Predict a perturbation's effect in a context where that perturbation was never measured, from a single control cell.

**The question.** Where does Occam sit against the field on an instrument we control — same 611-perturbation cohort, same 5 splits, same 6,983-gene axis, same drawn half-B controls, same cell-eval 0.6.0, same five baselines?

**Varied:** the model × the seed (42–46). **Held fixed:** the frozen benchmark artifact, the cohort, the gene axis, the control draw, the emission space and the scoring path.

**How to read it.** `transfer` **0.2323** is the bar — a model below it is not earning its parameters. `experimental` **0.3722** is replicate agreement at `m` = 22 and is *not* a ceiling: a model can beat it from the two-sided noise asymmetry alone. `ctrlcell` **−0.0018** is the floor. `meanpert` discriminates at **chance** by construction, which is what makes "it learned something perturbation-specific" testable.

Metrics: `pearson_delta`, `mae` ↓, `overlap_at_N`, `discrimination_score_l1` (**l1**, not l2; chance 0.5). Uncalibrated (`model_nocal`) throughout — on our own model calibration cost 0.047 `pearson_delta` and quadrupled `mae` (F12).

4/20 declared (arm, seed) runs complete, 0 partial, 16 never ran.

## Baselines, all 5 splits

| arm | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| `ctrlcell` | -0.0018 | 0.0296 | 0.0000 | 0.5038 |
| `ctrlmean` | 0.0016 | 0.0231 | 0.0081 | 0.5014 |
| `meanpert` | 0.1324 | 0.0290 | 0.0108 | 0.5037 |
| `transfer` | 0.2323 | 0.0299 | 0.0351 | 0.6404 |
| `experimental` | 0.3722 | 0.0257 | 0.1942 | 0.7821 |

## Competitors

| arm | seed | status | splits | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|---|---|---|
| `txpert` | 42 | complete | 5 | 0.3463 ± 0.0035 | 0.0227 | 0.0582 | 0.6704 |
| `txpert` | 43 | MISSING | 0 | — | — | — | — |
| `txpert` | 44 | MISSING | 0 | — | — | — | — |
| `txpert` | 45 | MISSING | 0 | — | — | — | — |
| `txpert` | 46 | MISSING | 0 | — | — | — | — |
| `state` | 42 | complete | 5 | 0.2362 ± 0.0019 | 0.0291 | 0.0849 | 0.6639 |
| `state` | 43 | MISSING | 0 | — | — | — | — |
| `state` | 44 | MISSING | 0 | — | — | — | — |
| `state` | 45 | MISSING | 0 | — | — | — | — |
| `state` | 46 | MISSING | 0 | — | — | — | — |
| `cellflow` | 42 | complete | 5 | 0.2154 ± 0.0039 | 0.0251 | 0.0315 | 0.5436 |
| `cellflow` | 43 | MISSING | 0 | — | — | — | — |
| `cellflow` | 44 | MISSING | 0 | — | — | — | — |
| `cellflow` | 45 | MISSING | 0 | — | — | — | — |
| `cellflow` | 46 | MISSING | 0 | — | — | — | — |
| `perturbdiff` | 42 | complete | 5 | 0.2449 ± 0.0038 | 0.0287 | 0.0734 | 0.6650 |
| `perturbdiff` | 43 | MISSING | 0 | — | — | — | — |
| `perturbdiff` | 44 | MISSING | 0 | — | — | — | — |
| `perturbdiff` | 45 | MISSING | 0 | — | — | — | — |
| `perturbdiff` | 46 | MISSING | 0 | — | — | — | — |

## Diagnostic A/B arms — seed 42, NOT headline numbers

Each isolates ONE deviation, so the headline can be *attributed* rather than merely reported. Read every row against `txpert` = **0.3463** above.

| arm | splits | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | what it isolates |
|---|---|---|---|---|---|---|
| `txpert_mean` | 5 | 0.3343 ± 0.0044 | 0.0231 | 0.0589 | 0.6584 | TxPert trained+inferenced on (cell_type, batch) MEAN controls — the upstream `match_cntr/avg_cntr: true` regime. Measures what the fairness decision (single-cell controls, to match every other arm) costs or buys. |
| `txpert_ep20` | 5 | 0.3038 ± 0.0037 | 0.0231 | 0.0486 | 0.6450 | TxPert single-cell regime, epoch-20 checkpoint instead of the selected one. Measures checkpoint-selection sensitivity. |
| ~~`txpert`~~ ⛔ | 5 | 0.3077 ± 0.0032 | 0.0228 | 0.0522 | 0.6458 | ⛔ WITHDRAWN original: mean-control training, single-cell inference, and the missing `-1` token. Kept so the retracted 0.3077 stays visible. |

**Why `txpert` 0.3077 was withdrawn.** C28: exported rows omitted the `-1` perturbation token the model was trained with (`omit_cntr: false`), the basal input was a single control cell while the checkpoint was trained and selected on the (cell_type, batch) MEAN control, and the frozen manifest recorded `kg_list: []`. Retraining on single-cell controls with `omit_cntr: true`, plus an A/B against the mean-control regime.

## Ours, for comparison

`OCCAM adjmap_s315 (model_nocal)` — `pearson_delta` **0.3439** over 4 split(s).

## ⚠️ Their published numbers, and why they are not in the table above

`the upstream benchmark`'s `replogle_nadig_fewshot` is **2,000 genes / 380 test perturbations / 4 cell lines**, built from a different preprocessing of the same experiment — their own task config warns the two are not comparable. For orientation only, their inject-real `pearson_delta` was:

| arm | their panel | note |
|---|---|---|
| `txpert` | 0.50787 | different axis, cohort and cell-line count |
| `cellflow` | 0.44015 | different axis, cohort and cell-line count |
| `state` | 0.42574 | different axis, cohort and cell-line count |
| `perturbdiff` | 0.40306 | different axis, cohort and cell-line count |

Their half-split reproducibility ceiling was 0.38600 and their train-perturbation mean 0.35339 — usefully close to our `experimental` 0.3722 and far from our `meanpert` 0.1324, which is itself a reminder that the two panels measure different things.

⚠️ Two of their arms' HYPERPARAMETERS were selected on a test metric upstream (TxPert's and CellFlow's `RESULTS.md` both say so). We reuse those hyperparameters unchanged rather than re-sweeping, so that inheritance is carried here and stated rather than hidden.

⚠️ TxPert runs on **GO + STRING only** (decision 2026-09-17), matching Occam's `graph_types: [go, string]`. Their 0.50787 was measured with four KGs (`go, string, pxmap-top1, txmap-top1`).

Emission space, and the guard every arm passes through: `docs/rhj_competitors.md` C1. Reader: C2. What was copied: C3.

