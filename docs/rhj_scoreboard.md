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
| **PerturbDiff** (genept) | 5 | **0.2449** ± 0.0038 | 0.0287 | 0.0734 | 0.6650 | headline |
| **STATE** | 5 | **0.2362** ± 0.0019 | 0.0291 | 0.0849 | 0.6639 | headline |
| **CellFlow** | 5 | **0.2154** ± 0.0039 | 0.0251 | 0.0315 | 0.5436 | headline |

## Schedule arms — epochs x `eta_min`

F-sched1/2/3. `epochs` is the strongest lever in the campaign; `eta_min` is noise inside 1e-7…1e-4 and a real gain at 3e-4. The campaign BEST lives here.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `Y06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em3e-4 | 5 | **0.4043** ± 0.0019 | 0.0215 | 0.0946 | 0.7288 | extended schedule (F-sched3) |
| `Y03` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e120_em3e-4 | 5 | **0.4015** ± 0.0018 | 0.0217 | 0.0919 | 0.7247 | extended schedule (F-sched3) |
| `Y05` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em1e-4 | 5 | **0.3983** ± 0.0023 | 0.0217 | 0.0931 | 0.7223 | extended schedule (F-sched3) |
| `Y01` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e120_em1e-5 | 5 | **0.3978** ± 0.0028 | 0.0217 | 0.0889 | 0.7141 | extended schedule (F-sched3) |
| `Y04` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e150_em1e-5 | 5 | **0.3963** ± 0.0021 | 0.0217 | 0.0913 | 0.7176 | extended schedule (F-sched3) |
| `Y02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044_e120_em1e-4 | 5 | **0.3960** ± 0.0021 | 0.0223 | 0.0904 | 0.7221 | extended schedule (F-sched3) |

## Loss-composition arms — latent, scale-match, split reconstruction

F-latent4/5/6, F-loss7/8, F-recon3/4. These produced the three bases M03/M06/Q02.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `Q02` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363_lmse0.0044 | 5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 | latent losses on M06 (F-latent4/6) |
| `M06` smooth_l1_nbw0_nbs1.0_dcos0.5_mag0.01363 | 5 | **0.3681** ± 0.0023 | 0.0232 | 0.0703 | 0.7047 | dcos/delta_mag scale-match (F-loss7/8) |
| `M03` smooth_l1_nbw0_nbs1.0_dcos0.01 | 5 | **0.3679** ± 0.0017 | 0.0212 | 0.0722 | 0.6715 | dcos/delta_mag scale-match (F-loss7/8) |

## Post-hoc emission treatments on `s313`

⚠️ split 00 only for the NB rows (n=1).

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| none (`model_nocal`) | 0 | — | — | — | — | not scored yet |
| calibration only (`model_cal`) | 0 | — | — | — | — | not scored yet |
| NB emission + `exact_colsum` | 1 | **0.2027** | 0.0276 | 0.0961 | 0.6393 | fair theta, lib 17,500 ⚠️ PARTIAL |
| NB emission alone | 1 | **0.1775** | 0.0331 | 0.0614 | 0.6475 | fair theta, lib 17,500 ⚠️ PARTIAL |
| NB emission + `exact_colsum` | 1 | **0.2478** | 0.0295 | 0.0314 | 0.6511 | ⛔ BROKEN theta/lib (F-emit4) ⚠️ PARTIAL |
| NB emission alone | 1 | **0.1611** | 0.0490 | 0.0079 | 0.6700 | ⛔ BROKEN theta/lib (F-emit4) ⚠️ PARTIAL |

