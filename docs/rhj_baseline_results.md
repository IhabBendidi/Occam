# `rhj_fewshot` split-half benchmark — BASELINE results

**Baselines only.** No model score appears in this file: `scripts/write_baseline_results.py` filters to the five model-independent reference arms by name. Regenerate with that script.

## What this benchmark is

**Objective.** Predict a perturbation's effect in a context where that perturbation was never measured, from a single control cell.

**The protocol** (a reviewer's request, implemented verbatim). Within each perturbation, split its HepG2 cells into balanced halves A and B. Half A is the **reference**: the scored target file holds half-A perturbed cells plus half-A controls. Each arm's prediction is written with **half-B controls**. Because cell-eval's `pearson_delta` uses *each file's own* control pseudobulk, that one substitution makes the reference delta A-relative and every predicted delta B-relative — so reference and prediction share no cells.

**What is varied:** which cells land in half A and half B — 5 independent, batch-stratified splits. **Held fixed:** the cohort (611 perturbations, `86c91f938024b2a2…`), the 6,983-gene axis, and the per-perturbation row counts. `m(p) = floor(n_p/2)` depends only on `n_p`, so every arm in every split emits exactly the same rows for the same perturbation.

| | value |
|---|---|
| source | `nadig_2024_hepg2.h5ad`, sha256 `f31a85d8c90f5a2e…` |
| cohort | **611** of 621 test perturbations (`min_cells` 10) |
| half size `m` | median **22**; `m≥5` 611 · `m≥10` 568 · `m≥25` 278 |
| rows per arm | 18,575 |
| controls per half | 2,488 |
| gene axis | 6,983, sha256 `ef64f2ba7ff7816e…` |
| normalisation | full-panel row sum over all file genes, then axis restriction, then log1p -- matches stream_v2.load_chunk_sparse |

Scored with ARC cell-eval v1, profile `full`, `--allow-discrete`, `--batch-size 2000`, 16 threads.

`↓` marks metrics where **lower is better**. **Chance is 0.5**, not 0, for `discrimination_score_*`, `de_direction_match` and `roc_auc`.

---

## The five baselines

### `ctrlmean` — Zero-signal anchor — a constant

**What it predicts.** The mean of all 2,488 half-B control cells, tiled identically across every one of the 18,575 predicted rows. Every perturbation gets the same vector.

**Why it is here.** It carries no biology by construction, so whatever it scores is what the metric reads from the emission format alone. Its within-group variance is **zero**, which makes the Wilcoxon DE tests over-call massively — it is the upper bracket on `de_nsig_counts_pred`.

**Coverage:** splits [0, 1, 2, 3, 4] (5 of 5).

| metric | split 0 | split 1 | split 2 | split 3 | split 4 | mean | std | n |
|---|---|---|---|---|---|---|---|---|
| `clustering_agreement` | 0.0011 | 0.0087 | 0.0093 | 0.0108 | 0.0054 | **0.0070** | 0.0039 | 5 |
| `de_direction_match` *(chance 0.5)* | 0.4885 | 0.4449 | 0.4719 | 0.4624 | 0.5049 | **0.4745** | 0.0232 | 5 |
| `de_nsig_counts_pred` ↓ | 3,026.8 | 3,059.6 | 3,041.6 | 3,035.8 | 3,052.3 | **3,043.2** | 13.002 | 5 |
| `de_nsig_counts_real` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** | 1.967 | 5 |
| `de_sig_genes_recall` | 0.3080 | 0.3135 | 0.3058 | 0.3136 | 0.3162 | **0.3114** | 0.0043 | 5 |
| `de_spearman_lfc_sig` | — | — | -0.0649 | — | — | **-0.0649** | — | 1 |
| `de_spearman_sig` | 0.4048 | 0.3974 | 0.3870 | 0.4076 | 0.4061 | **0.4006** | 0.0085 | 5 |
| `discrimination_score_cosine` *(chance 0.5)* | 0.4996 | 0.5238 | 0.4965 | 0.5010 | 0.4717 | **0.4985** | 0.0185 | 5 |
| `discrimination_score_l1` *(chance 0.5)* | 0.5015 | 0.5015 | 0.5014 | 0.5015 | 0.5014 | **0.5014** | 0.0000 | 5 |
| `discrimination_score_l2` *(chance 0.5)* | 0.5040 | 0.5040 | 0.5040 | 0.5038 | 0.5038 | **0.5039** | 0.0001 | 5 |
| `mae` ↓ | 0.0231 | 0.0231 | 0.0231 | 0.0230 | 0.0231 | **0.0231** | 0.0000 | 5 |
| `mae_delta` ↓ | 0.0231 | 0.0231 | 0.0231 | 0.0230 | 0.0231 | **0.0231** | 0.0000 | 5 |
| `mse` ↓ | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0016 | **0.0016** | 0.0000 | 5 |
| `mse_delta` ↓ | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0016 | **0.0016** | 0.0000 | 5 |
| `overlap_at_100` | 0.0040 | 0.0054 | 0.0051 | 0.0049 | 0.0067 | **0.0052** | 0.0010 | 5 |
| `overlap_at_200` | 0.0053 | 0.0070 | 0.0062 | 0.0062 | 0.0078 | **0.0065** | 0.0009 | 5 |
| `overlap_at_50` | 0.0028 | 0.0034 | 0.0042 | 0.0040 | 0.0050 | **0.0039** | 0.0008 | 5 |
| `overlap_at_500` | 0.0067 | 0.0078 | 0.0073 | 0.0077 | 0.0091 | **0.0077** | 0.0009 | 5 |
| `overlap_at_N` | 0.0071 | 0.0083 | 0.0077 | 0.0077 | 0.0094 | **0.0081** | 0.0009 | 5 |
| `pearson_delta` | 0.0006 | 0.0021 | 0.0003 | -0.0010 | 0.0061 | **0.0016** | 0.0028 | 5 |
| `pearson_edistance` | -0.1263 | 0.0990 | 0.0182 | 0.0471 | -0.2166 | **-0.0357** | 0.1312 | 5 |
| `pr_auc` | 0.0120 | 0.0123 | 0.0125 | 0.0120 | — | **0.0122** | 0.0003 | 4 |
| `precision_at_100` | 0.0041 | 0.0057 | 0.0053 | 0.0050 | 0.0061 | **0.0052** | 0.0008 | 5 |
| `precision_at_200` | 0.0054 | 0.0075 | 0.0065 | 0.0062 | 0.0071 | **0.0065** | 0.0008 | 5 |
| `precision_at_50` | 0.0028 | 0.0035 | 0.0044 | 0.0042 | 0.0044 | **0.0038** | 0.0007 | 5 |
| `precision_at_500` | 0.0068 | 0.0080 | 0.0074 | 0.0075 | 0.0080 | **0.0075** | 0.0005 | 5 |
| `precision_at_N` | 0.0065 | 0.0070 | 0.0066 | 0.0066 | 0.0070 | **0.0068** | 0.0002 | 5 |
| `roc_auc` *(chance 0.5)* | 0.3902 | 0.3929 | 0.3912 | 0.3971 | — | **0.3929** | 0.0031 | 4 |

### `ctrlcell` — Zero-signal anchor — the floor

**What it predicts.** The drawn half-B control cell itself, unchanged. Row *i*'s prediction is exactly the control cell that the model would have been fed for row *i*.

**Why it is here.** **The load-bearing floor.** Its delta is the mean of *m* drawn half-B controls minus the full half-B control mean, i.e. pure control-sampling noise at *m* cells, which scales correctly with cell count. Unlike `ctrlmean` it has *natural* single-cell variance, so it calls **zero** genes significant — the lower bracket on the DE metrics.

**Coverage:** splits [0, 1, 2, 3, 4] (5 of 5).

| metric | split 0 | split 1 | split 2 | split 3 | split 4 | mean | std | n |
|---|---|---|---|---|---|---|---|---|
| `clustering_agreement` | 0.0017 | 0.0045 | 0.0067 | 0.0035 | 0.0148 | **0.0062** | 0.0051 | 5 |
| `de_direction_match` *(chance 0.5)* | 0.4837 | 0.4823 | 0.4878 | 0.4791 | 0.4783 | **0.4822** | 0.0038 | 5 |
| `de_nsig_counts_pred` ↓ | 0.0000 | 0.0033 | 0.0000 | 0.0065 | 0.0065 | **0.0033** | 0.0033 | 5 |
| `de_nsig_counts_real` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** | 1.967 | 5 |
| `de_sig_genes_recall` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `de_spearman_lfc_sig` | — | — | — | — | — | **—** | — | 0 |
| `de_spearman_sig` | — | -0.0293 | — | -0.0019 | -0.0383 | **-0.0232** | 0.0190 | 3 |
| `discrimination_score_cosine` *(chance 0.5)* | 0.4985 | 0.4892 | 0.5023 | 0.5082 | 0.4793 | **0.4955** | 0.0114 | 5 |
| `discrimination_score_l1` *(chance 0.5)* | 0.5046 | 0.5030 | 0.5053 | 0.5038 | 0.5025 | **0.5038** | 0.0012 | 5 |
| `discrimination_score_l2` *(chance 0.5)* | 0.5059 | 0.5036 | 0.5063 | 0.5042 | 0.5031 | **0.5046** | 0.0014 | 5 |
| `mae` ↓ | 0.0296 | 0.0296 | 0.0296 | 0.0295 | 0.0296 | **0.0296** | 0.0000 | 5 |
| `mae_delta` ↓ | 0.0295 | 0.0296 | 0.0295 | 0.0295 | 0.0296 | **0.0296** | 0.0001 | 5 |
| `mse` ↓ | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0023 | **0.0023** | 0.0000 | 5 |
| `mse_delta` ↓ | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0023 | **0.0023** | 0.0000 | 5 |
| `overlap_at_100` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `overlap_at_200` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `overlap_at_50` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `overlap_at_500` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `overlap_at_N` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `pearson_delta` | -0.0003 | -0.0009 | -0.0019 | 0.0012 | -0.0070 | **-0.0018** | 0.0031 | 5 |
| `pearson_edistance` | 0.6300 | 0.5954 | 0.6082 | 0.6139 | 0.6232 | **0.6142** | 0.0134 | 5 |
| `pr_auc` | 0.0188 | 0.0294 | 0.0198 | 0.0208 | — | **0.0222** | 0.0049 | 4 |
| `precision_at_100` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `precision_at_200` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `precision_at_50` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `precision_at_500` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `precision_at_N` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 5 |
| `roc_auc` *(chance 0.5)* | 0.5051 | 0.5043 | 0.4953 | 0.5036 | — | **0.5021** | 0.0045 | 4 |

### `meanpert` — Baseline — "it learned the average knockdown"

**What it predicts.** `ctrl_used + (training grand mean − half-B control mean)`. The grand mean is the equal-weight-per-perturbation average of the 1,657 HepG2 **training** perturbations' mean profiles (95,527 cells). One single delta, added to every row regardless of which perturbation it is.

**Why it is here.** It has no per-perturbation information at all. Beating it is the minimum bar for claiming a model learned anything perturbation-specific — and its `discrimination_score_l2` should sit at chance, which is the check that it is constructed correctly. Averaged per *perturbation*, not per cell: per-cell would be dominated by TFAM's 1,213 cells.

**Coverage:** splits [0, 1, 2, 3, 4] (5 of 5).

| metric | split 0 | split 1 | split 2 | split 3 | split 4 | mean | std | n |
|---|---|---|---|---|---|---|---|---|
| `clustering_agreement` | 0.0000 | 0.0023 | 0.0038 | 0.0043 | 0.0153 | **0.0051** | 0.0059 | 5 |
| `de_direction_match` *(chance 0.5)* | 0.5786 | 0.5708 | 0.5818 | 0.5794 | 0.5718 | **0.5765** | 0.0049 | 5 |
| `de_nsig_counts_pred` ↓ | 2,226.0 | 2,227.4 | 2,225.7 | 2,188.6 | 2,192.0 | **2,212.0** | 19.803 | 5 |
| `de_nsig_counts_real` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** | 1.967 | 5 |
| `de_sig_genes_recall` | 0.2710 | 0.2662 | 0.2760 | 0.2681 | 0.2699 | **0.2702** | 0.0037 | 5 |
| `de_spearman_lfc_sig` | — | — | 0.1523 | — | — | **0.1523** | — | 1 |
| `de_spearman_sig` | 0.4103 | 0.3906 | 0.3790 | 0.4109 | 0.3972 | **0.3976** | 0.0135 | 5 |
| `discrimination_score_cosine` *(chance 0.5)* | 0.5062 | 0.4987 | 0.4986 | 0.5033 | 0.5038 | **0.5021** | 0.0034 | 5 |
| `discrimination_score_l1` *(chance 0.5)* | 0.5050 | 0.5023 | 0.5050 | 0.5046 | 0.5017 | **0.5037** | 0.0016 | 5 |
| `discrimination_score_l2` *(chance 0.5)* | 0.5065 | 0.5045 | 0.5073 | 0.5056 | 0.5021 | **0.5052** | 0.0020 | 5 |
| `mae` ↓ | 0.0290 | 0.0290 | 0.0289 | 0.0289 | 0.0290 | **0.0290** | 0.0001 | 5 |
| `mae_delta` ↓ | 0.0290 | 0.0291 | 0.0290 | 0.0290 | 0.0291 | **0.0290** | 0.0001 | 5 |
| `mse` ↓ | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | **0.0021** | 0.0000 | 5 |
| `mse_delta` ↓ | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | **0.0021** | 0.0000 | 5 |
| `overlap_at_100` | 0.0059 | 0.0059 | 0.0060 | 0.0054 | 0.0050 | **0.0056** | 0.0004 | 5 |
| `overlap_at_200` | 0.0079 | 0.0078 | 0.0076 | 0.0072 | 0.0069 | **0.0075** | 0.0004 | 5 |
| `overlap_at_50` | 0.0048 | 0.0044 | 0.0047 | 0.0042 | 0.0038 | **0.0044** | 0.0004 | 5 |
| `overlap_at_500` | 0.0100 | 0.0104 | 0.0097 | 0.0098 | 0.0090 | **0.0098** | 0.0005 | 5 |
| `overlap_at_N` | 0.0109 | 0.0114 | 0.0107 | 0.0107 | 0.0100 | **0.0108** | 0.0005 | 5 |
| `pearson_delta` | 0.1335 | 0.1351 | 0.1307 | 0.1347 | 0.1279 | **0.1324** | 0.0030 | 5 |
| `pearson_edistance` | 0.6239 | 0.5871 | 0.6041 | 0.6085 | 0.6222 | **0.6091** | 0.0150 | 5 |
| `pr_auc` | 0.0144 | 0.0148 | 0.0150 | 0.0145 | — | **0.0147** | 0.0003 | 4 |
| `precision_at_100` | 0.0054 | 0.0054 | 0.0056 | 0.0051 | 0.0057 | **0.0054** | 0.0002 | 5 |
| `precision_at_200` | 0.0075 | 0.0073 | 0.0074 | 0.0072 | 0.0075 | **0.0074** | 0.0001 | 5 |
| `precision_at_50` | 0.0045 | 0.0039 | 0.0043 | 0.0037 | 0.0044 | **0.0042** | 0.0003 | 5 |
| `precision_at_500` | 0.0100 | 0.0102 | 0.0098 | 0.0101 | 0.0099 | **0.0100** | 0.0002 | 5 |
| `precision_at_N` | 0.0098 | 0.0101 | 0.0102 | 0.0098 | 0.0099 | **0.0100** | 0.0002 | 5 |
| `roc_auc` *(chance 0.5)* | 0.4834 | 0.4786 | 0.4833 | 0.4855 | — | **0.4827** | 0.0029 | 4 |

### `transfer` — ⭐ Baseline — the non-learned version of the task itself

**What it predicts.** `ctrl_used + delta[this row's own perturbation]`, where `delta[p]` is the count-weighted mean of *p*'s (perturbed mean − that file's control mean) over **Jurkat and RPE1 only** — median 158 source cells per perturbation. HepG2 is deliberately excluded: it is the held-out context and using it would read the answer.

**Why it is here.** This is what the model is asked to do, done by averaging instead of learning: take the same perturbation's measured effect in two other cell lines and add it to a HepG2 control cell. **It is the bar.** A model that cannot beat it is not earning its parameters. On the *validation* side its analogue beat all 24 grid arms by 0.20 (`docs/rhj_fewshot.md` F5).

**Coverage:** splits [0, 1, 2, 3, 4] (5 of 5).

| metric | split 0 | split 1 | split 2 | split 3 | split 4 | mean | std | n |
|---|---|---|---|---|---|---|---|---|
| `clustering_agreement` | 0.3136 | 0.3780 | 0.3472 | 0.3494 | 0.4147 | **0.3606** | 0.0379 | 5 |
| `de_direction_match` *(chance 0.5)* | 0.6865 | 0.6921 | 0.6964 | 0.6813 | 0.6790 | **0.6870** | 0.0073 | 5 |
| `de_nsig_counts_pred` ↓ | 2,171.1 | 2,180.0 | 2,176.5 | 2,174.2 | 2,166.7 | **2,173.7** | 5.079 | 5 |
| `de_nsig_counts_real` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** | 1.967 | 5 |
| `de_sig_genes_recall` | 0.3991 | 0.4025 | 0.4050 | 0.4005 | 0.4027 | **0.4020** | 0.0023 | 5 |
| `de_spearman_lfc_sig` | — | — | 0.3514 | — | — | **0.3514** | — | 1 |
| `de_spearman_sig` | 0.6032 | 0.5913 | 0.5959 | 0.6143 | 0.6213 | **0.6052** | 0.0125 | 5 |
| `discrimination_score_cosine` *(chance 0.5)* | 0.7552 | 0.7521 | 0.7596 | 0.7526 | 0.7517 | **0.7543** | 0.0033 | 5 |
| `discrimination_score_l1` *(chance 0.5)* | 0.6416 | 0.6381 | 0.6435 | 0.6415 | 0.6373 | **0.6404** | 0.0026 | 5 |
| `discrimination_score_l2` *(chance 0.5)* | 0.7018 | 0.6984 | 0.7065 | 0.7026 | 0.6984 | **0.7015** | 0.0034 | 5 |
| `mae` ↓ | 0.0299 | 0.0299 | 0.0298 | 0.0299 | 0.0299 | **0.0299** | 0.0000 | 5 |
| `mae_delta` ↓ | 0.0299 | 0.0299 | 0.0298 | 0.0298 | 0.0299 | **0.0299** | 0.0000 | 5 |
| `mse` ↓ | 0.0022 | 0.0022 | 0.0021 | 0.0021 | 0.0021 | **0.0021** | 0.0000 | 5 |
| `mse_delta` ↓ | 0.0022 | 0.0022 | 0.0021 | 0.0021 | 0.0021 | **0.0021** | 0.0000 | 5 |
| `overlap_at_100` | 0.0306 | 0.0275 | 0.0286 | 0.0281 | 0.0279 | **0.0285** | 0.0013 | 5 |
| `overlap_at_200` | 0.0330 | 0.0302 | 0.0312 | 0.0309 | 0.0311 | **0.0313** | 0.0010 | 5 |
| `overlap_at_50` | 0.0282 | 0.0247 | 0.0265 | 0.0264 | 0.0259 | **0.0263** | 0.0013 | 5 |
| `overlap_at_500` | 0.0353 | 0.0333 | 0.0337 | 0.0338 | 0.0334 | **0.0339** | 0.0008 | 5 |
| `overlap_at_N` | 0.0363 | 0.0345 | 0.0348 | 0.0348 | 0.0348 | **0.0351** | 0.0007 | 5 |
| `pearson_delta` | 0.2338 | 0.2340 | 0.2318 | 0.2333 | 0.2288 | **0.2323** | 0.0021 | 5 |
| `pearson_edistance` | 0.8315 | 0.8182 | 0.8239 | 0.8181 | 0.8349 | **0.8253** | 0.0076 | 5 |
| `pr_auc` | 0.0177 | 0.0188 | 0.0191 | 0.0182 | — | **0.0185** | 0.0006 | 4 |
| `precision_at_100` | 0.0228 | 0.0222 | 0.0228 | 0.0221 | 0.0216 | **0.0223** | 0.0005 | 5 |
| `precision_at_200` | 0.0232 | 0.0233 | 0.0233 | 0.0232 | 0.0231 | **0.0232** | 0.0001 | 5 |
| `precision_at_50` | 0.0222 | 0.0217 | 0.0229 | 0.0224 | 0.0216 | **0.0222** | 0.0005 | 5 |
| `precision_at_500` | 0.0231 | 0.0242 | 0.0236 | 0.0238 | 0.0233 | **0.0236** | 0.0004 | 5 |
| `precision_at_N` | 0.0186 | 0.0192 | 0.0189 | 0.0188 | 0.0190 | **0.0189** | 0.0002 | 5 |
| `roc_auc` *(chance 0.5)* | 0.5536 | 0.5551 | 0.5574 | 0.5546 | — | **0.5551** | 0.0016 | 4 |

### `experimental` — Reference — replicate agreement, NOT a ceiling

**What it predicts.** The real half-B **perturbed** cells, as the prediction. Same perturbation, same cell line, different cells — scored against the half-A reference exactly as any other arm is.

**Why it is here.** ⚠️ **Not an upper bound.** At median *m* = 22 both this arm and the half-A reference are noisy pseudobulks, while a low-variance predictor is attenuated on one side only — so a model can beat it from noise alone on perturbations whose true effect is small relative to σ/√m. Read it as *"agreement between two 22-cell replicates"* and check the count-stratified gap.

**Coverage:** splits [0, 1, 2, 3, 4] (5 of 5).

| metric | split 0 | split 1 | split 2 | split 3 | split 4 | mean | std | n |
|---|---|---|---|---|---|---|---|---|
| `clustering_agreement` | 0.6084 | 0.5625 | 0.6385 | 0.5547 | 0.6174 | **0.5963** | 0.0362 | 5 |
| `de_direction_match` *(chance 0.5)* | 0.7604 | 0.7507 | 0.7582 | 0.7559 | 0.7486 | **0.7548** | 0.0050 | 5 |
| `de_nsig_counts_pred` ↓ | 114.1 | 108.7 | 112.9 | 112.1 | 110.8 | **111.7** | 2.097 | 5 |
| `de_nsig_counts_real` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** | 1.967 | 5 |
| `de_sig_genes_recall` | 0.2359 | 0.2226 | 0.2327 | 0.2301 | 0.2239 | **0.2290** | 0.0057 | 5 |
| `de_spearman_lfc_sig` | — | — | 0.4562 | — | — | **0.4562** | — | 1 |
| `de_spearman_sig` | 0.8872 | 0.8761 | 0.8845 | 0.8838 | 0.8752 | **0.8813** | 0.0054 | 5 |
| `discrimination_score_cosine` *(chance 0.5)* | 0.8659 | 0.8720 | 0.8715 | 0.8698 | 0.8717 | **0.8702** | 0.0026 | 5 |
| `discrimination_score_l1` *(chance 0.5)* | 0.7820 | 0.7815 | 0.7817 | 0.7821 | 0.7832 | **0.7821** | 0.0007 | 5 |
| `discrimination_score_l2` *(chance 0.5)* | 0.8392 | 0.8369 | 0.8402 | 0.8405 | 0.8421 | **0.8398** | 0.0019 | 5 |
| `mae` ↓ | 0.0257 | 0.0257 | 0.0256 | 0.0256 | 0.0257 | **0.0257** | 0.0000 | 5 |
| `mae_delta` ↓ | 0.0258 | 0.0258 | 0.0258 | 0.0258 | 0.0258 | **0.0258** | 0.0000 | 5 |
| `mse` ↓ | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0015 | **0.0015** | 0.0000 | 5 |
| `mse_delta` ↓ | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0015 | **0.0015** | 0.0000 | 5 |
| `overlap_at_100` | 0.1771 | 0.1756 | 0.1774 | 0.1788 | 0.1736 | **0.1765** | 0.0020 | 5 |
| `overlap_at_200` | 0.1884 | 0.1865 | 0.1884 | 0.1899 | 0.1853 | **0.1877** | 0.0018 | 5 |
| `overlap_at_50` | 0.1543 | 0.1533 | 0.1567 | 0.1567 | 0.1524 | **0.1547** | 0.0020 | 5 |
| `overlap_at_500` | 0.1934 | 0.1911 | 0.1930 | 0.1949 | 0.1895 | **0.1924** | 0.0021 | 5 |
| `overlap_at_N` | 0.1949 | 0.1930 | 0.1948 | 0.1969 | 0.1916 | **0.1942** | 0.0020 | 5 |
| `pearson_delta` | 0.3717 | 0.3709 | 0.3735 | 0.3722 | 0.3727 | **0.3722** | 0.0010 | 5 |
| `pearson_edistance` | 0.9548 | 0.9528 | 0.9458 | 0.9658 | 0.9437 | **0.9526** | 0.0087 | 5 |
| `pr_auc` | 0.2109 | 0.2115 | 0.2124 | 0.2106 | — | **0.2113** | 0.0008 | 4 |
| `precision_at_100` | 0.1768 | 0.1801 | 0.1818 | 0.1786 | 0.1763 | **0.1787** | 0.0023 | 5 |
| `precision_at_200` | 0.1884 | 0.1901 | 0.1937 | 0.1903 | 0.1883 | **0.1901** | 0.0022 | 5 |
| `precision_at_50` | 0.1543 | 0.1585 | 0.1611 | 0.1568 | 0.1550 | **0.1572** | 0.0028 | 5 |
| `precision_at_500` | 0.1933 | 0.1942 | 0.1982 | 0.1956 | 0.1924 | **0.1947** | 0.0023 | 5 |
| `precision_at_N` | 0.1950 | 0.1960 | 0.1997 | 0.1978 | 0.1941 | **0.1965** | 0.0023 | 5 |
| `roc_auc` *(chance 0.5)* | 0.7002 | 0.6990 | 0.6997 | 0.6970 | — | **0.6990** | 0.0014 | 4 |

---

## All five baselines side by side

Mean ± std across the splits each arm has, `ddof=1`. `n` differs per arm where coverage does — see above.

### `clustering_agreement`

Panel-level, not per perturbation. Not re-aggregable.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0011 | 0.0087 | 0.0093 | 0.0108 | 0.0054 | **0.0070** ± 0.0039 | 5 |
| `ctrlcell` | 0.0017 | 0.0045 | 0.0067 | 0.0035 | 0.0148 | **0.0062** ± 0.0051 | 5 |
| `meanpert` | 0.0000 | 0.0023 | 0.0038 | 0.0043 | 0.0153 | **0.0051** ± 0.0059 | 5 |
| `transfer` | 0.3136 | 0.3780 | 0.3472 | 0.3494 | 0.4147 | **0.3606** ± 0.0379 | 5 |
| `experimental` | 0.6084 | 0.5625 | 0.6385 | 0.5547 | 0.6174 | **0.5963** ± 0.0362 | 5 |

### `de_direction_match` *(chance 0.5)*

Fraction of DE genes where prediction and truth agree on up/down.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.4885 | 0.4449 | 0.4719 | 0.4624 | 0.5049 | **0.4745** ± 0.0232 | 5 |
| `ctrlcell` | 0.4837 | 0.4823 | 0.4878 | 0.4791 | 0.4783 | **0.4822** ± 0.0038 | 5 |
| `meanpert` | 0.5786 | 0.5708 | 0.5818 | 0.5794 | 0.5718 | **0.5765** ± 0.0049 | 5 |
| `transfer` | 0.6865 | 0.6921 | 0.6964 | 0.6813 | 0.6790 | **0.6870** ± 0.0073 | 5 |
| `experimental` | 0.7604 | 0.7507 | 0.7582 | 0.7559 | 0.7486 | **0.7548** ± 0.0050 | 5 |

### `de_nsig_counts_pred` ↓

Genes FDR-significant on the predicted side. Read against BOTH anchors: `ctrlcell` 0 and `ctrlmean` ~3,000 bracket it.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 3,026.8 | 3,059.6 | 3,041.6 | 3,035.8 | 3,052.3 | **3,043.2** ± 13.002 | 5 |
| `ctrlcell` | 0.0000 | 0.0033 | 0.0000 | 0.0065 | 0.0065 | **0.0033** ± 0.0033 | 5 |
| `meanpert` | 2,226.0 | 2,227.4 | 2,225.7 | 2,188.6 | 2,192.0 | **2,212.0** ± 19.803 | 5 |
| `transfer` | 2,171.1 | 2,180.0 | 2,176.5 | 2,174.2 | 2,166.7 | **2,173.7** ± 5.079 | 5 |
| `experimental` | 114.1 | 108.7 | 112.9 | 112.1 | 110.8 | **111.7** ± 2.097 | 5 |

### `de_nsig_counts_real`

Genes FDR-significant on the real side — the reference count.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** ± 1.967 | 5 |
| `ctrlcell` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** ± 1.967 | 5 |
| `meanpert` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** ± 1.967 | 5 |
| `transfer` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** ± 1.967 | 5 |
| `experimental` | 111.6 | 116.5 | 112.9 | 112.5 | 114.7 | **113.7** ± 1.967 | 5 |

### `de_sig_genes_recall`

Fraction of the real significant genes the arm also calls significant. Trivially raised by calling everything.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.3080 | 0.3135 | 0.3058 | 0.3136 | 0.3162 | **0.3114** ± 0.0043 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.2710 | 0.2662 | 0.2760 | 0.2681 | 0.2699 | **0.2702** ± 0.0037 | 5 |
| `transfer` | 0.3991 | 0.4025 | 0.4050 | 0.4005 | 0.4027 | **0.4020** ± 0.0023 | 5 |
| `experimental` | 0.2359 | 0.2226 | 0.2327 | 0.2301 | 0.2239 | **0.2290** ± 0.0057 | 5 |

### `de_spearman_lfc_sig`

Spearman of log fold changes over significant genes.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | — | — | -0.0649 | — | — | **-0.0649** ± — | 1 |
| `ctrlcell` | — | — | — | — | — | — | 0 |
| `meanpert` | — | — | 0.1523 | — | — | **0.1523** ± — | 1 |
| `transfer` | — | — | 0.3514 | — | — | **0.3514** ± — | 1 |
| `experimental` | — | — | 0.4562 | — | — | **0.4562** ± — | 1 |

### `de_spearman_sig`

Spearman of effect sizes over significant genes.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.4048 | 0.3974 | 0.3870 | 0.4076 | 0.4061 | **0.4006** ± 0.0085 | 5 |
| `ctrlcell` | — | -0.0293 | — | -0.0019 | -0.0383 | **-0.0232** ± 0.0190 | 3 |
| `meanpert` | 0.4103 | 0.3906 | 0.3790 | 0.4109 | 0.3972 | **0.3976** ± 0.0135 | 5 |
| `transfer` | 0.6032 | 0.5913 | 0.5959 | 0.6143 | 0.6213 | **0.6052** ± 0.0125 | 5 |
| `experimental` | 0.8872 | 0.8761 | 0.8845 | 0.8838 | 0.8752 | **0.8813** ± 0.0054 | 5 |

### `discrimination_score_cosine` *(chance 0.5)*

Same, cosine distance.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.4996 | 0.5238 | 0.4965 | 0.5010 | 0.4717 | **0.4985** ± 0.0185 | 5 |
| `ctrlcell` | 0.4985 | 0.4892 | 0.5023 | 0.5082 | 0.4793 | **0.4955** ± 0.0114 | 5 |
| `meanpert` | 0.5062 | 0.4987 | 0.4986 | 0.5033 | 0.5038 | **0.5021** ± 0.0034 | 5 |
| `transfer` | 0.7552 | 0.7521 | 0.7596 | 0.7526 | 0.7517 | **0.7543** ± 0.0033 | 5 |
| `experimental` | 0.8659 | 0.8720 | 0.8715 | 0.8698 | 0.8717 | **0.8702** ± 0.0026 | 5 |

### `discrimination_score_l1` *(chance 0.5)*

Same, L1 distance.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.5015 | 0.5015 | 0.5014 | 0.5015 | 0.5014 | **0.5014** ± 0.0000 | 5 |
| `ctrlcell` | 0.5046 | 0.5030 | 0.5053 | 0.5038 | 0.5025 | **0.5038** ± 0.0012 | 5 |
| `meanpert` | 0.5050 | 0.5023 | 0.5050 | 0.5046 | 0.5017 | **0.5037** ± 0.0016 | 5 |
| `transfer` | 0.6416 | 0.6381 | 0.6435 | 0.6415 | 0.6373 | **0.6404** ± 0.0026 | 5 |
| `experimental` | 0.7820 | 0.7815 | 0.7817 | 0.7821 | 0.7832 | **0.7821** ± 0.0007 | 5 |

### `discrimination_score_l2` *(chance 0.5)*

1 − rank/n_perts over the 611-perturbation cohort, L2 distance. Excludes each perturbation's own gene by SYMBOL.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.5040 | 0.5040 | 0.5040 | 0.5038 | 0.5038 | **0.5039** ± 0.0001 | 5 |
| `ctrlcell` | 0.5059 | 0.5036 | 0.5063 | 0.5042 | 0.5031 | **0.5046** ± 0.0014 | 5 |
| `meanpert` | 0.5065 | 0.5045 | 0.5073 | 0.5056 | 0.5021 | **0.5052** ± 0.0020 | 5 |
| `transfer` | 0.7018 | 0.6984 | 0.7065 | 0.7026 | 0.6984 | **0.7015** ± 0.0034 | 5 |
| `experimental` | 0.8392 | 0.8369 | 0.8402 | 0.8405 | 0.8421 | **0.8398** ± 0.0019 | 5 |

### `mae` ↓

Control-INDEPENDENT mean absolute error against the real pseudobulk.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0231 | 0.0231 | 0.0231 | 0.0230 | 0.0231 | **0.0231** ± 0.0000 | 5 |
| `ctrlcell` | 0.0296 | 0.0296 | 0.0296 | 0.0295 | 0.0296 | **0.0296** ± 0.0000 | 5 |
| `meanpert` | 0.0290 | 0.0290 | 0.0289 | 0.0289 | 0.0290 | **0.0290** ± 0.0001 | 5 |
| `transfer` | 0.0299 | 0.0299 | 0.0298 | 0.0299 | 0.0299 | **0.0299** ± 0.0000 | 5 |
| `experimental` | 0.0257 | 0.0257 | 0.0256 | 0.0256 | 0.0257 | **0.0257** ± 0.0000 | 5 |

### `mae_delta` ↓

Control-DEPENDENT: error on the delta. The headline error metric here.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0231 | 0.0231 | 0.0231 | 0.0230 | 0.0231 | **0.0231** ± 0.0000 | 5 |
| `ctrlcell` | 0.0295 | 0.0296 | 0.0295 | 0.0295 | 0.0296 | **0.0296** ± 0.0001 | 5 |
| `meanpert` | 0.0290 | 0.0291 | 0.0290 | 0.0290 | 0.0291 | **0.0290** ± 0.0001 | 5 |
| `transfer` | 0.0299 | 0.0299 | 0.0298 | 0.0298 | 0.0299 | **0.0299** ± 0.0000 | 5 |
| `experimental` | 0.0258 | 0.0258 | 0.0258 | 0.0258 | 0.0258 | **0.0258** ± 0.0000 | 5 |

### `mse` ↓

Control-independent squared error.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0016 | **0.0016** ± 0.0000 | 5 |
| `ctrlcell` | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0023 | **0.0023** ± 0.0000 | 5 |
| `meanpert` | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | **0.0021** ± 0.0000 | 5 |
| `transfer` | 0.0022 | 0.0022 | 0.0021 | 0.0021 | 0.0021 | **0.0021** ± 0.0000 | 5 |
| `experimental` | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0015 | **0.0015** ± 0.0000 | 5 |

### `mse_delta` ↓

Control-dependent squared error on the delta.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0016 | **0.0016** ± 0.0000 | 5 |
| `ctrlcell` | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0023 | **0.0023** ± 0.0000 | 5 |
| `meanpert` | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | **0.0021** ± 0.0000 | 5 |
| `transfer` | 0.0022 | 0.0022 | 0.0021 | 0.0021 | 0.0021 | **0.0021** ± 0.0000 | 5 |
| `experimental` | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0015 | **0.0015** ± 0.0000 | 5 |

### `overlap_at_100`

Same at fixed N = 100.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0040 | 0.0054 | 0.0051 | 0.0049 | 0.0067 | **0.0052** ± 0.0010 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0059 | 0.0059 | 0.0060 | 0.0054 | 0.0050 | **0.0056** ± 0.0004 | 5 |
| `transfer` | 0.0306 | 0.0275 | 0.0286 | 0.0281 | 0.0279 | **0.0285** ± 0.0013 | 5 |
| `experimental` | 0.1771 | 0.1756 | 0.1774 | 0.1788 | 0.1736 | **0.1765** ± 0.0020 | 5 |

### `overlap_at_200`

Same at fixed N = 200.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0053 | 0.0070 | 0.0062 | 0.0062 | 0.0078 | **0.0065** ± 0.0009 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0079 | 0.0078 | 0.0076 | 0.0072 | 0.0069 | **0.0075** ± 0.0004 | 5 |
| `transfer` | 0.0330 | 0.0302 | 0.0312 | 0.0309 | 0.0311 | **0.0313** ± 0.0010 | 5 |
| `experimental` | 0.1884 | 0.1865 | 0.1884 | 0.1899 | 0.1853 | **0.1877** ± 0.0018 | 5 |

### `overlap_at_50`

Same at fixed N = 50.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0028 | 0.0034 | 0.0042 | 0.0040 | 0.0050 | **0.0039** ± 0.0008 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0048 | 0.0044 | 0.0047 | 0.0042 | 0.0038 | **0.0044** ± 0.0004 | 5 |
| `transfer` | 0.0282 | 0.0247 | 0.0265 | 0.0264 | 0.0259 | **0.0263** ± 0.0013 | 5 |
| `experimental` | 0.1543 | 0.1533 | 0.1567 | 0.1567 | 0.1524 | **0.1547** ± 0.0020 | 5 |

### `overlap_at_500`

Same at fixed N = 500.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0067 | 0.0078 | 0.0073 | 0.0077 | 0.0091 | **0.0077** ± 0.0009 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0100 | 0.0104 | 0.0097 | 0.0098 | 0.0090 | **0.0098** ± 0.0005 | 5 |
| `transfer` | 0.0353 | 0.0333 | 0.0337 | 0.0338 | 0.0334 | **0.0339** ± 0.0008 | 5 |
| `experimental` | 0.1934 | 0.1911 | 0.1930 | 0.1949 | 0.1895 | **0.1924** ± 0.0021 | 5 |

### `overlap_at_N`

Top-N DE gene overlap, N = the number of FDR-significant genes on the REAL side. Exactly 0 when the arm calls nothing significant.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0071 | 0.0083 | 0.0077 | 0.0077 | 0.0094 | **0.0081** ± 0.0009 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0109 | 0.0114 | 0.0107 | 0.0107 | 0.0100 | **0.0108** ± 0.0005 | 5 |
| `transfer` | 0.0363 | 0.0345 | 0.0348 | 0.0348 | 0.0348 | **0.0351** ± 0.0007 | 5 |
| `experimental` | 0.1949 | 0.1930 | 0.1948 | 0.1969 | 0.1916 | **0.1942** ± 0.0020 | 5 |

### `pearson_delta`

Field convention. Pearson r between (pred − pred-file control mean) and (real − real-file control mean), per perturbation, averaged. The A/B asymmetry lives here: the target file carries half-A controls, the prediction file half-B.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0006 | 0.0021 | 0.0003 | -0.0010 | 0.0061 | **0.0016** ± 0.0028 | 5 |
| `ctrlcell` | -0.0003 | -0.0009 | -0.0019 | 0.0012 | -0.0070 | **-0.0018** ± 0.0031 | 5 |
| `meanpert` | 0.1335 | 0.1351 | 0.1307 | 0.1347 | 0.1279 | **0.1324** ± 0.0030 | 5 |
| `transfer` | 0.2338 | 0.2340 | 0.2318 | 0.2333 | 0.2288 | **0.2323** ± 0.0021 | 5 |
| `experimental` | 0.3717 | 0.3709 | 0.3735 | 0.3722 | 0.3727 | **0.3722** ± 0.0010 | 5 |

### `pearson_edistance`

Panel-level energy-distance correlation. Not re-aggregable.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | -0.1263 | 0.0990 | 0.0182 | 0.0471 | -0.2166 | **-0.0357** ± 0.1312 | 5 |
| `ctrlcell` | 0.6300 | 0.5954 | 0.6082 | 0.6139 | 0.6232 | **0.6142** ± 0.0134 | 5 |
| `meanpert` | 0.6239 | 0.5871 | 0.6041 | 0.6085 | 0.6222 | **0.6091** ± 0.0150 | 5 |
| `transfer` | 0.8315 | 0.8182 | 0.8239 | 0.8181 | 0.8349 | **0.8253** ± 0.0076 | 5 |
| `experimental` | 0.9548 | 0.9528 | 0.9458 | 0.9658 | 0.9437 | **0.9526** ± 0.0087 | 5 |

### `pr_auc`

Precision-recall AUC over the DE calls.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0120 | 0.0123 | 0.0125 | 0.0120 | — | **0.0122** ± 0.0003 | 4 |
| `ctrlcell` | 0.0188 | 0.0294 | 0.0198 | 0.0208 | — | **0.0222** ± 0.0049 | 4 |
| `meanpert` | 0.0144 | 0.0148 | 0.0150 | 0.0145 | — | **0.0147** ± 0.0003 | 4 |
| `transfer` | 0.0177 | 0.0188 | 0.0191 | 0.0182 | — | **0.0185** ± 0.0006 | 4 |
| `experimental` | 0.2109 | 0.2115 | 0.2124 | 0.2106 | — | **0.2113** ± 0.0008 | 4 |

### `precision_at_100`

Precision at fixed N = 100.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0041 | 0.0057 | 0.0053 | 0.0050 | 0.0061 | **0.0052** ± 0.0008 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0054 | 0.0054 | 0.0056 | 0.0051 | 0.0057 | **0.0054** ± 0.0002 | 5 |
| `transfer` | 0.0228 | 0.0222 | 0.0228 | 0.0221 | 0.0216 | **0.0223** ± 0.0005 | 5 |
| `experimental` | 0.1768 | 0.1801 | 0.1818 | 0.1786 | 0.1763 | **0.1787** ± 0.0023 | 5 |

### `precision_at_200`

Precision at fixed N = 200.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0054 | 0.0075 | 0.0065 | 0.0062 | 0.0071 | **0.0065** ± 0.0008 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0075 | 0.0073 | 0.0074 | 0.0072 | 0.0075 | **0.0074** ± 0.0001 | 5 |
| `transfer` | 0.0232 | 0.0233 | 0.0233 | 0.0232 | 0.0231 | **0.0232** ± 0.0001 | 5 |
| `experimental` | 0.1884 | 0.1901 | 0.1937 | 0.1903 | 0.1883 | **0.1901** ± 0.0022 | 5 |

### `precision_at_50`

Precision at fixed N = 50.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0028 | 0.0035 | 0.0044 | 0.0042 | 0.0044 | **0.0038** ± 0.0007 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0045 | 0.0039 | 0.0043 | 0.0037 | 0.0044 | **0.0042** ± 0.0003 | 5 |
| `transfer` | 0.0222 | 0.0217 | 0.0229 | 0.0224 | 0.0216 | **0.0222** ± 0.0005 | 5 |
| `experimental` | 0.1543 | 0.1585 | 0.1611 | 0.1568 | 0.1550 | **0.1572** ± 0.0028 | 5 |

### `precision_at_500`

Precision at fixed N = 500.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0068 | 0.0080 | 0.0074 | 0.0075 | 0.0080 | **0.0075** ± 0.0005 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0100 | 0.0102 | 0.0098 | 0.0101 | 0.0099 | **0.0100** ± 0.0002 | 5 |
| `transfer` | 0.0231 | 0.0242 | 0.0236 | 0.0238 | 0.0233 | **0.0236** ± 0.0004 | 5 |
| `experimental` | 0.1933 | 0.1942 | 0.1982 | 0.1956 | 0.1924 | **0.1947** ± 0.0023 | 5 |

### `precision_at_N`

Precision form of `overlap_at_N`.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.0065 | 0.0070 | 0.0066 | 0.0066 | 0.0070 | **0.0068** ± 0.0002 | 5 |
| `ctrlcell` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** ± 0.0000 | 5 |
| `meanpert` | 0.0098 | 0.0101 | 0.0102 | 0.0098 | 0.0099 | **0.0100** ± 0.0002 | 5 |
| `transfer` | 0.0186 | 0.0192 | 0.0189 | 0.0188 | 0.0190 | **0.0189** ± 0.0002 | 5 |
| `experimental` | 0.1950 | 0.1960 | 0.1997 | 0.1978 | 0.1941 | **0.1965** ± 0.0023 | 5 |

### `roc_auc` *(chance 0.5)*

ROC AUC over the DE calls.

| arm | split 0 | split 1 | split 2 | split 3 | split 4 | mean ± std | n |
|---|---|---|---|---|---|---|---|
| `ctrlmean` | 0.3902 | 0.3929 | 0.3912 | 0.3971 | — | **0.3929** ± 0.0031 | 4 |
| `ctrlcell` | 0.5051 | 0.5043 | 0.4953 | 0.5036 | — | **0.5021** ± 0.0045 | 4 |
| `meanpert` | 0.4834 | 0.4786 | 0.4833 | 0.4855 | — | **0.4827** ± 0.0029 | 4 |
| `transfer` | 0.5536 | 0.5551 | 0.5574 | 0.5546 | — | **0.5551** ± 0.0016 | 4 |
| `experimental` | 0.7002 | 0.6990 | 0.6997 | 0.6970 | — | **0.6990** ± 0.0014 | 4 |

---

## How to read a model score against these

1. **`ctrlcell` is the floor** for both metric families — `pearson_delta` ~0 and `overlap_at_N` exactly 0. A model at the floor has learned nothing.
2. **`meanpert` is the minimum bar for perturbation-specificity.** It predicts one delta for everything, so its `discrimination_score_l2` sits at chance. A model above it on `pearson_delta` but at chance on discrimination has learned the average response, not the perturbations.
3. **`transfer` is the bar that matters.** It is this task done by averaging two other cell lines instead of learning. A model below it is not earning its parameters.
4. **`experimental` is not a ceiling.** Beating it is possible from noise alone; check whether the gap shrinks as `m` grows.
5. **The split-to-split std is the resolution limit.** It is 0.001–0.002 on `pearson_delta` here, so a model-vs-baseline difference above ~0.005 is real and anything smaller is not.

Panel definition: `data/benchmarks/rhj_fewshot/splithalf/CARD.md`. Protocol findings: `docs/rhj_splithalf.md`.
