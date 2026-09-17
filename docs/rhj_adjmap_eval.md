# `rhj_fewshot` adjacency-map sweep — validation vs cell-eval

**The question:** does the validation metric used for checkpoint selection predict the benchmark result? One GNN arm showed val 0.3368 against cell-eval 0.0020 (`docs/rhj_splithalf.md` F11), so they are not interchangeable.

Comparison metrics (decision 2026-09-17): `pearson_delta`, `mae` ↓, `overlap_at_N`, `discrimination_score_l1` *(chance 0.5)*. Both calibration settings are shown because they win different metrics of the four (F12).

`val` is group-mean `pearson_delta` on the 103 HepG2 **validation** perturbations — the checkpoint-selection metric, read at the epoch that scored best. The cell-eval columns are the 611 held-out **test** perturbations under the split-half protocol.

1 of 164 completed arms scored so far.

## Baselines, all 5 splits

| arm | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| `ctrlcell` | -0.0018 | 0.0296 | 0.0000 | 0.5038 |
| `ctrlmean` | 0.0016 | 0.0231 | 0.0081 | 0.5014 |
| `meanpert` | 0.1324 | 0.0290 | 0.0108 | 0.5037 |
| `transfer` | 0.2323 | 0.0299 | 0.0351 | 0.6404 |
| `experimental` | 0.3722 | 0.0257 | 0.1942 | 0.7821 |

## Rank correlation: validation vs cell-eval

Spearman rho over the arms scored so far. **Positive and high** = validation is a usable proxy. **Near zero** = validation is uninformative and arms must be chosen on cell-eval. **Negative** = validation is actively harmful.

| cell-eval metric | arm | Spearman rho vs val | n |
|---|---|---|---|
| `pearson_delta` | `model_nocal` | — | 1 |
| `mae` | `model_nocal` | — | 1 |
| `overlap_at_N` | `model_nocal` | — | 1 |
| `discrimination_score_l1` | `model_nocal` | — | 1 |
| `pearson_delta` | `model_cal` | — | 1 |
| `mae` | `model_cal` | — | 1 |
| `overlap_at_N` | `model_cal` | — | 1 |
| `discrimination_score_l1` | `model_cal` | — | 1 |

## Arms by cell-eval `pearson_delta` — `model_nocal`

| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits | lr | dlr | cos | dpo | film_mode | fh | fmf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s129` | 0.3676 | 0.3348 | 0.0210 | 0.0618 | 0.6479 | 5 | 3e-4 | 1e-3 | 0.25 | 0 | full | 256 | 1 |

## Arms by cell-eval `pearson_delta` — `model_cal`

| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits | lr | dlr | cos | dpo | film_mode | fh | fmf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s129` | 0.3676 | 0.2883 | 0.0856 | 0.0250 | 0.7528 | 5 | 3e-4 | 1e-3 | 0.25 | 0 | full | 256 | 1 |

