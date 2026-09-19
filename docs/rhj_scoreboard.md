# `rhj_fewshot` split-half — the scoreboard

**Objective.** Predict a perturbation's effect in a cellular context where that perturbation was never measured, from a single control cell.

**This table's question.** Across our loss/DCOS arms, the four competitor ports, the post-hoc emission treatments and the model-independent baselines — what does each actually score, on the same instrument, with no metric omitted?

**Varied:** the arm. **Held fixed:** the 611-perturbation cohort, the 6,983-gene axis, the seeded per-split half-B control draw, cell-eval 0.6.0 / pdex 0.1.24.

**Directions:** `pearson_delta` ↑ · `mae` **↓ (it is an ERROR)** · `overlap_at_N` ↑ · `discrimination_score_l1` ↑ (chance 0.5).

⚠️ **`n` is load-bearing.** A pdex shared-memory race (F-infra1) made cell-eval return nothing for some (arm, split, mode) triples while the SLURM task still exited 0. **Any row with n < 5 is PARTIAL**, marked ⚠️, and its mean may move.

⚠️ **Never compare across the two calibration tables.** Calibration costs 0.008–0.068 `pearson_delta` and quadruples `mae` (F-loss3); the uncalibrated table is the reportable one.

## Baselines — model-independent anchors

Read every model number against these, never against 0.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| ctrlcell | 5 | **-0.0018** ± 0.0031 | 0.0296 | 0.0000 | 0.5038 |  |
| ctrlmean | 5 | **0.0016** ± 0.0028 | 0.0231 | 0.0081 | 0.5014 |  |
| meanpert | 5 | **0.1324** ± 0.0030 | 0.0290 | 0.0108 | 0.5037 |  |
| transfer | 5 | **0.2323** ± 0.0021 | 0.0299 | 0.0351 | 0.6404 |  |
| experimental | 5 | **0.3722** ± 0.0010 | 0.0257 | 0.1942 | 0.7821 |  |

## Competitors — retrained on our data, our axis, our splits

⚠️ None of these reproduces the method's published figure; see `docs/rhj_competitors_results.md` for each arm's deliberate deviations.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| **TxPert** (single-cell controls) | 5 | **0.3463** ± 0.0035 | 0.0227 | 0.0582 | 0.6704 | headline |
| TxPert mean-controls | 5 | **0.3343** ± 0.0044 | 0.0231 | 0.0589 | 0.6584 | A/B diagnostic |
| TxPert epoch-20 | 5 | **0.3038** ± 0.0037 | 0.0231 | 0.0486 | 0.6450 | A/B diagnostic |
| ~~TxPert original~~ | 5 | **0.3077** ± 0.0032 | 0.0228 | 0.0522 | 0.6458 | ⛔ WITHDRAWN (C28) |
| **PerturbDiff** (genept) | 5 | **0.2449** ± 0.0038 | 0.0287 | 0.0734 | 0.6650 | headline |
| **STATE** | 5 | **0.2362** ± 0.0019 | 0.0291 | 0.0849 | 0.6639 | headline |
| **CellFlow** | 0 | — | — | — | — | not scored yet |

## Our arms — UNCALIBRATED (`model_nocal`), the reportable table

Loss composition and delta-cosine sweep, seed 42.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `D02` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.3661** ± 0.0023 | 0.0277 | 0.0568 | 0.7180 |  |
| `L11` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 |  |
| `L13` smooth_l1_nbw0_nbs1.0_dcos1.0 | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 |  |
| `L12` smooth_l1_nbw0_nbs1.0_dcos0.5 | 1 | **0.3414** | 0.0210 | 0.0642 | 0.6538 |  ⚠️ PARTIAL |
| `L10` smooth_l1_nbw0_nbs1.0_dcos0.1 | 2 | **0.3396** ± 0.0025 | 0.0210 | 0.0635 | 0.6533 |  ⚠️ PARTIAL |
| `L02` nb_smooth_l1_nbw0.1_nbs1.0_dcos0.0 | 5 | **0.3184** ± 0.0033 | 0.0211 | 0.0621 | 0.6483 |  |
| `L14` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.25 | 3 | **0.3134** ± 0.0024 | 0.0211 | 0.0650 | 0.6434 |  ⚠️ PARTIAL |
| `L08` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.0 | 2 | **0.3129** ± 0.0033 | 0.0211 | 0.0643 | 0.6434 |  ⚠️ PARTIAL |
| `L07` nb_smooth_l1_nbw0.25_nbs6.5_dcos0.0 | 4 | **0.3098** ± 0.0028 | 0.0212 | 0.0608 | 0.6412 |  ⚠️ PARTIAL |
| `L03` nb_smooth_l1_nbw0.25_nbs1.0_dcos0.0 | 5 | **0.2989** ± 0.0029 | 0.0214 | 0.0663 | 0.6440 |  |
| `L09` nb_smooth_l1_nbw0.75_nbs6.5_dcos0.0 | 4 | **0.2976** ± 0.0025 | 0.0216 | 0.0629 | 0.6447 |  ⚠️ PARTIAL |
| `L05` nb_smooth_l1_nbw0.75_nbs1.0_dcos0.0 | 5 | **0.2790** ± 0.0027 | 0.0226 | 0.0539 | 0.6472 |  |
| `L06` nb_smooth_l1_nbw0.9_nbs1.0_dcos0.0 | 5 | **0.2774** ± 0.0027 | 0.0221 | 0.0660 | 0.6409 |  |
| `L01` nb_nbw0_nbs1.0_dcos0.0 | 5 | **0.2766** ± 0.0027 | 0.0220 | 0.0651 | 0.6425 |  |
| `L04` nb_smooth_l1_nbw0.5_nbs1.0_dcos0.0 | 5 | **0.2740** ± 0.0029 | 0.0219 | 0.0600 | 0.6357 |  |

## Our arms — CALIBRATED (`model_cal`), same checkpoints

Shown because it was measured, NOT because it should be used.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `D02` smooth_l1_nbw0_nbs1.0_dcos0.25 | 4 | **0.2993** ± 0.0013 | 0.1021 | 0.0289 | 0.8099 |  ⚠️ PARTIAL |
| `D03` smooth_l1_nbw0_nbs1.0_dcos0.5 | 1 | **0.2990** | 0.0925 | 0.0217 | 0.7753 |  ⚠️ PARTIAL |
| `L02` nb_smooth_l1_nbw0.1_nbs1.0_dcos0.0 | 5 | **0.2789** ± 0.0024 | 0.0917 | 0.0240 | 0.7575 |  |
| `L08` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.0 | 4 | **0.2773** ± 0.0013 | 0.0871 | 0.0229 | 0.7546 |  ⚠️ PARTIAL |
| `L14` nb_smooth_l1_nbw0.5_nbs6.5_dcos0.25 | 4 | **0.2773** ± 0.0013 | 0.0871 | 0.0229 | 0.7546 |  ⚠️ PARTIAL |
| `L11` smooth_l1_nbw0_nbs1.0_dcos0.25 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L12` smooth_l1_nbw0_nbs1.0_dcos0.5 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L13` smooth_l1_nbw0_nbs1.0_dcos1.0 | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 |  |
| `L10` smooth_l1_nbw0_nbs1.0_dcos0.1 | 1 | **0.2715** | 0.0880 | 0.0239 | 0.7805 |  ⚠️ PARTIAL |
| `L09` nb_smooth_l1_nbw0.75_nbs6.5_dcos0.0 | 4 | **0.2703** ± 0.0031 | 0.0896 | 0.0243 | 0.7619 |  ⚠️ PARTIAL |
| `L06` nb_smooth_l1_nbw0.9_nbs1.0_dcos0.0 | 4 | **0.2691** ± 0.0019 | 0.0917 | 0.0275 | 0.7595 |  ⚠️ PARTIAL |
| `L01` nb_nbw0_nbs1.0_dcos0.0 | 4 | **0.2627** ± 0.0014 | 0.0902 | 0.0246 | 0.7708 |  ⚠️ PARTIAL |
| `L07` nb_smooth_l1_nbw0.25_nbs6.5_dcos0.0 | 4 | **0.2619** ± 0.0013 | 0.0867 | 0.0324 | 0.7610 |  ⚠️ PARTIAL |
| `L03` nb_smooth_l1_nbw0.25_nbs1.0_dcos0.0 | 5 | **0.2612** ± 0.0021 | 0.0624 | 0.0208 | 0.7728 |  |
| `L05` nb_smooth_l1_nbw0.75_nbs1.0_dcos0.0 | 4 | **0.2590** ± 0.0016 | 0.0923 | 0.0283 | 0.7699 |  ⚠️ PARTIAL |
| `L04` nb_smooth_l1_nbw0.5_nbs1.0_dcos0.0 | 5 | **0.2372** ± 0.0043 | 0.0842 | 0.0262 | 0.7754 |  |

## Post-hoc emission treatments on `s313`

⚠️ split 00 only for the NB rows (n=1).

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| none (`model_nocal`) | 5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | the default |
| calibration only (`model_cal`) | 5 | **0.2732** ± 0.0021 | 0.0890 | 0.0243 | 0.7766 | control-fit quantile |
| NB emission + `exact_colsum` | 1 | **0.2027** | 0.0276 | 0.0961 | 0.6393 | fair theta, lib 17,500 ⚠️ PARTIAL |
| NB emission alone | 1 | **0.1775** | 0.0331 | 0.0614 | 0.6475 | fair theta, lib 17,500 ⚠️ PARTIAL |
| NB emission + `exact_colsum` | 1 | **0.2478** | 0.0295 | 0.0314 | 0.6511 | ⛔ BROKEN theta/lib (F-emit4) ⚠️ PARTIAL |
| NB emission alone | 1 | **0.1611** | 0.0490 | 0.0079 | 0.6700 | ⛔ BROKEN theta/lib (F-emit4) ⚠️ PARTIAL |

