# `rhj_fewshot` split-half — EVERY measured arm

**Objective.** Predict a perturbation's effect in a cellular context where that perturbation was never measured, from a single control cell. Headline: split-half paired `pearson_delta`.

**Varied:** the arm. **Held fixed:** the 611-perturbation cohort, the 6,983-gene axis, the seeded per-split half-B control draw, cell-eval 0.6.0 / pdex 0.1.24, uncalibrated (`model_nocal`).

**Directions:** `pearson_delta` ↑ · `mae` **↓ (it is an ERROR)** · `overlap_at_N` ↑ · `discrimination_score_l1` ↑ (chance 0.5).

⚠️ **`n` is load-bearing.** Rows below 5/5 are PARTIAL — a pdex shared-memory race (F-infra1) drops individual (arm, split, mode) scores. Their means may move.

⚠️ **Never compare a calibrated number to an uncalibrated one.** Calibration costs 0.008–0.068 `pearson_delta` and quadruples `mae` (F-loss3). Everything here is uncalibrated.

⚠️ **Generated from disk on 2026-09-20.** Arm inventory: `python scripts/collect_all_results.py`. The per-sweep tables below are grouped by hand because the grouping IS the interpretation — which arms belong to which question, and what each was varying. A flat dump of every directory would be complete and unreadable.


## Baselines — model-independent anchors

Read every model number against these, never against 0.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | what it is |
|---|---|---|---|---|---|---|
| `ctrlcell` | 5/5 | **-0.0018** ± 0.0031 | 0.0296 | 0.0000 | 0.5038 | predict a control cell — the zero-signal floor |
| `ctrlmean` | 5/5 | **0.0016** ± 0.0028 | 0.0231 | 0.0081 | 0.5014 | predict the control mean |
| `meanpert` | 5/5 | **0.1324** ± 0.0030 | 0.0290 | 0.0108 | 0.5037 | mean perturbation effect, IGNORING identity. Discriminates at chance by construction |
| `transfer` | 5/5 | **0.2323** ± 0.0021 | 0.0299 | 0.0351 | 0.6404 | the same perturbation in a DIFFERENT cell line — **the bar** |
| `experimental` | 5/5 | **0.3722** ± 0.0010 | 0.0257 | 0.1942 | 0.7821 | half-A vs half-B of the SAME experiment. NOT a ceiling: beatable via the two-sided noise asymmetry |

## Competitors — retrained on our data, our axis, our splits

⚠️ None reproduces the method's published figure; deliberate deviations per arm are in `docs/rhj_competitors_results.md`. F-comp1: CellFlow is the only one below `transfer`, and its `disc_l1` 0.5436 is near the 0.5 chance floor while `pearson_delta` 0.2154 sits well above `meanpert` — the signature `meanpert` exists to expose. Its `mae` is the BEST of any competitor, which is the same fact from the other side: **`mae` must never be read alone.**

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| **TxPert** (single-cell controls) | 5/5 | **0.3463** ± 0.0035 | 0.0227 | 0.0582 | 0.6704 | headline |
| TxPert (mean controls) | 5/5 | **0.3343** ± 0.0044 | 0.0231 | 0.0589 | 0.6584 | A/B: single-cell training is worth +0.0120 |
| ~~TxPert original~~ | 5/5 | **0.3077** ± 0.0032 | 0.0228 | 0.0522 | 0.6458 | ⛔ WITHDRAWN (C28) |
| TxPert epoch-20 | 5/5 | **0.3038** ± 0.0037 | 0.0231 | 0.0486 | 0.6450 | A/B: checkpoint-selection sensitivity |
| **PerturbDiff** (genept) | 5/5 | **0.2449** ± 0.0038 | 0.0287 | 0.0734 | 0.6650 | headline |
| **STATE** | 5/5 | **0.2362** ± 0.0019 | 0.0291 | 0.0849 | 0.6639 | headline; best competitor `ovl@N` |
| **CellFlow** | 5/5 | **0.2154** ± 0.0039 | 0.0251 | 0.0315 | 0.5436 | headline; only arm below `transfer` |

## Loss composition — NB vs smooth_l1 (job 18179618)

F-loss3: **10/10 NB arms lose to plain smooth_l1**, including the scale-matched `nb_scale` 6.5 arms, so it is not a gradient-magnitude artefact. NB's optimum is at `nb_weight` → 0 — an ENDPOINT optimum, i.e. 'do not use NB'. ⚠️ L10–L13 are bit-identical because `delta_cos_weight` ran INERT (F-loss1); they are smooth_l1 replicates, not DCOS arms, and that identity is what exposed the bug.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `L11` smooth_l1 | 5/5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | previous best; dcos INERT |
| `L13` smooth_l1 | 5/5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | dcos INERT |
| `L12` smooth_l1 | 5/5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | dcos INERT |
| `L10` smooth_l1 | 5/5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | dcos INERT |
| `L02` nb_sl1 w0.1 | 5/5 | **0.3184** ± 0.0033 | 0.0211 | 0.0621 | 0.6483 | best NB arm |
| `L08` nb_sl1 w0.5 s6.5 | 5/5 | **0.3149** ± 0.0028 | 0.0211 | 0.0652 | 0.6437 | scale-matched |
| `L14` nb_sl1 w0.5 s6.5 | 5/5 | **0.3149** ± 0.0028 | 0.0211 | 0.0652 | 0.6437 | = L08; dcos INERT |
| `L07` nb_sl1 w0.25 s6.5 | 5/5 | **0.3094** ± 0.0025 | 0.0212 | 0.0609 | 0.6412 | scale-matched |
| `L03` nb_sl1 w0.25 | 5/5 | **0.2989** ± 0.0029 | 0.0214 | 0.0663 | 0.6440 |  |
| `L09` nb_sl1 w0.75 s6.5 | 5/5 | **0.2979** ± 0.0022 | 0.0216 | 0.0629 | 0.6451 | scale-matched |
| `L05` nb_sl1 w0.75 | 5/5 | **0.2790** ± 0.0027 | 0.0226 | 0.0539 | 0.6472 |  |
| `L06` nb_sl1 w0.9 | 5/5 | **0.2774** ± 0.0027 | 0.0221 | 0.0660 | 0.6409 |  |
| `L01` nb (pure) | 5/5 | **0.2766** ± 0.0027 | 0.0220 | 0.0651 | 0.6425 |  |
| `L04` nb_sl1 w0.5 | 5/5 | **0.2740** ± 0.0029 | 0.0219 | 0.0600 | 0.6357 |  |

## Delta-cosine (DCOS) — scale-match sweep (jobs 18179643, 18186699, 18181379)

F-loss5 measured the raw terms: `smooth_l1` 0.008443, `dcos` 0.796206 — a **~100×** gap, so `delta_cos_weight` was never a blend knob but an off-switch for reconstruction (at 0.25 the recon term is 3.7% of the loss; at 0.5, 2.1%). F-loss7: the optimum is INTERIOR and lands exactly at the arithmetic scale-match point 0.008443/0.796206 = **0.01060**. ⚠️ Validation `best_metric` ranked this sweep BACKWARDS — do not select DCOS arms on it.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `D03` dcos 0.5 | 5/5 | **0.3686** ± 0.0021 | 0.0321 | 0.0718 | 0.7263 | best `disc_l1` of any arm |
| `M06` dcos 0.5 + `delta_mag` 0.01363 | 5/5 | **0.3681** ± 0.0023 | 0.0232 | 0.0703 | 0.7047 | one of the two preferred bases |
| **`M03` dcos 0.01 (scale-matched)** | 5/5 | **0.3679** ± 0.0017 | 0.0212 | 0.0722 | 0.6715 | **preferred base**; best `mae`+`ovl@N` of the family |
| `D02` dcos 0.25 | 5/5 | **0.3661** ± 0.0023 | 0.0277 | 0.0568 | 0.7180 |  |
| `M04` dcos 0.02 (65%) | 5/5 | **0.3642** ± 0.0026 | 0.0217 | 0.0703 | 0.6704 |  |
| `M05` dcos 0.05 (82%) | 5/5 | **0.3624** ± 0.0021 | 0.0228 | 0.0693 | 0.6921 |  |
| `M02` dcos 0.005 (32%) | 5/5 | **0.3553** ± 0.0030 | 0.0212 | 0.0598 | 0.6624 |  |
| `M01` dcos 0.002 (16%) | 5/5 | **0.3457** ± 0.0022 | 0.0211 | 0.0658 | 0.6522 | ≈ pure smooth_l1 |

## Delta-magnitude on the M03 base (job 18215120)

F-loss9: **strictly harmful on all four metrics, monotone in weight.** `N01` carries M06's EXACT weight on M03's setup and still loses 0.0482 `pearson_delta`, so the M06 win was SETUP-driven, not weight-driven. Mechanism (F-loss5): in D03/M06's regime `smooth_l1` is 2.1% of the loss so magnitude is unsupervised and `mag_loss` restores it; in M03's it is 51.5%, so the term is redundant and its unbounded ratio-penalty gradient distorts training.

⚠️ **The magnitude-ratio probe is DEAD as a selection signal**: `N01` has the BEST ratio of any arm (4.683 vs M03 6.584) and is worst on all four metrics.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `N01` mag 0.01363 (1×) | 5/5 | **0.3197** ± 0.0037 | 0.0230 | 0.0586 | 0.6264 |  |
| `N02` mag 0.1363 (10×) | 5/5 | **0.2703** ± 0.0038 | 0.0248 | 0.0473 | 0.5726 |  |
| `N03` mag 0.6815 (50×) | 5/5 | **0.2476** ± 0.0033 | 0.0252 | 0.0439 | 0.5671 |  |

## Latent-space losses — M03 base (job 18215244)

F-latent3: **no arm improves M03.** The only positive is `P06` at +0.0013, inside M03's own ±0.0017 sd and on 4/5 splits. `disc_l1` is LOWER in all 11. `mae` is INERT — every arm lands in 0.0211–0.0216, a range of 0.0005 against the 0.0109 M03-vs-D03 gap, which independently corroborates F-recon1.

✅ **Pre-registered outcome confirmed:** the weight absorbs the scale and the cosine TARGET does not matter — `P07` (`z_hat`) 0.3570 vs `P05` (`delta_z`) 0.3578, inside the sds, despite raw terms differing **18.4×** (0.9733 vs 0.0528).

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `P06` latcos(delta_z) 10× | 4/5 ⚠️ | **0.3692** ± 0.0033 | 0.0213 | 0.0720 | 0.6655 | only positive; inside sd |
| `P11` both 0.1× | 5/5 | **0.3670** ± 0.0023 | 0.0211 | 0.0738 | 0.6653 | user-added; best `ovl@N` in the table |
| `P10` latMSE 10× + cos 1× | 4/5 ⚠️ | **0.3643** ± 0.0033 | 0.0212 | 0.0723 | 0.6599 |  |
| `P08` both 1× | 5/5 | **0.3640** ± 0.0019 | 0.0211 | 0.0660 | 0.6521 |  |
| `P09` both 10× | 5/5 | **0.3633** ± 0.0022 | 0.0212 | 0.0667 | 0.6568 |  |
| `P02` latMSE 1× | 5/5 | **0.3606** ± 0.0032 | 0.0213 | 0.0710 | 0.6612 |  |
| `P01` latMSE 0.1× | 5/5 | **0.3600** ± 0.0019 | 0.0213 | 0.0693 | 0.6589 |  |
| `P03` latMSE 10× | 4/5 ⚠️ | **0.3592** ± 0.0033 | 0.0215 | 0.0687 | 0.6571 |  |
| `P05` latcos(delta_z) 1× | 5/5 | **0.3578** ± 0.0015 | 0.0212 | 0.0675 | 0.6637 |  |
| `P07` latcos(z_hat) 1× | 5/5 | **0.3570** ± 0.0030 | 0.0214 | 0.0702 | 0.6547 | the one z_hat arm |
| `P04` latcos(delta_z) 0.1× | 4/5 ⚠️ | **0.3551** ± 0.0025 | 0.0216 | 0.0673 | 0.6663 |  |
| `T01` both 0.01× | 5/5 | **0.3595** ± 0.0027 | 0.0214 | 0.0700 | 0.6653 | rescored via 18217089; WORSE than 0.1× |

## Latent-space losses — M06 base (job 18215244)

F-latent4: **the latent loss HELPS on M06 and HURTS on M03 — identical intervention, opposite sign, several times the sds.** This is the THIRD setup-dependent lever after `delta_mag_weight`, and the pattern is now statable: *a term that repairs a DIRECTION-DOMINATED loss (M06/D03, `smooth_l1` at 2.1%) tends to HARM a balanced one (M03, at 51.5%)*. **Stop generalising any loss-term result across bases without testing both.**

⚠️ `experimental` is 0.3722. Several arms here exceed it; that is NOT super-experimental accuracy — `experimental` is replicate agreement and is beatable via the two-sided noise asymmetry.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `Q05` latcos(delta_z) 1× | 4/5 ⚠️ | **0.3787** ± 0.0032 | 0.0225 | 0.0763 | 0.7078 | highest number in the campaign, but 4/5 |
| **`Q02` latMSE 1×** | 5/5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 | **best 5/5 arm in the campaign** |
| `Q04` latcos(delta_z) 0.1× | 4/5 ⚠️ | **0.3764** ± 0.0023 | 0.0223 | 0.0693 | 0.6921 | best `mae` of the M06 family |
| `Q09` both 10× | 4/5 ⚠️ | **0.3754** ± 0.0017 | 0.0221 | 0.0684 | 0.6927 |  |
| `Q01` latMSE 0.1× | 5/5 | **0.3725** ± 0.0024 | 0.0227 | 0.0794 | 0.7008 | best `ovl@N` of the M06 family |
| `Q03` latMSE 10× | 4/5 ⚠️ | **0.3721** ± 0.0028 | 0.0229 | 0.0755 | 0.6968 |  |
| `Q08` both 1× | 5/5 | **0.3717** ± 0.0018 | 0.0227 | 0.0742 | 0.6992 |  |
| `Q06` latcos(delta_z) 10× | 4/5 ⚠️ | **0.3690** ± 0.0023 | 0.0230 | 0.0720 | 0.7022 |  |
| `Q07` latcos(z_hat) 1× | 4/5 ⚠️ | **0.3681** ± 0.0020 | 0.0225 | 0.0663 | 0.6890 | the one z_hat arm |
| `Q10` latMSE 10× + cos 1× | 0/5 | — | — | — | — | not scored |
| `Q11` both 0.1× | 0/5 | — | — | — | — | not scored |
| `T02` both 0.01× | 5/5 | **0.3617** ± 0.0023 | 0.0235 | 0.0714 | 0.7026 | rescored via 18217089; first latent arm to HURT M06 |

## Post-hoc emission treatments on `s313` — ⚠️ split 00 only (n=1)

F-emit6. `pearson_delta` and `overlap_at_N` rank these OPPOSITELY, so it is a trade, not a dominance. `exact_colsum` is worth +0.025 `pearson_delta` and +0.035 `overlap_at_N` WITHIN emission — never emit without it.

| treatment | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|
| none (`model_nocal`) — the default | **0.3415** | 0.0210 | 0.0642 | 0.6534 |
| calibration only (`model_cal`) | **0.2732** | 0.0890 | 0.0243 | 0.7766 |
| NB emission + `exact_colsum` (fair theta) | **0.2027** | 0.0276 | 0.0961 | 0.6393 |
| NB emission alone (fair theta) | **0.1775** | 0.0331 | 0.0614 | 0.6475 |
| NB emission + `exact_colsum` ⛔ BROKEN theta/lib | **0.2478** | 0.0295 | 0.0314 | 0.6511 |
| NB emission alone ⛔ BROKEN theta/lib | **0.1611** | 0.0490 | 0.0079 | 0.6700 |

---

**Still running at time of writing:** `18217040` (26-arm split-reconstruction grid) → `18217067` (its 130 cell-eval tasks); `18215244` (the last M06 latent arms); `18217089` (the orphaned 0.01× arms T01/T02).


## 🧭 THE THREE BASES (user direction, 2026-09-20)

Future optimisation starts from one of these, not from `adjmap_s313`. All test, uncalibrated,
5/5 splits. `mae` is an ERROR: LOWER IS BETTER.

| base | config | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | what it is best for |
|---|---|---|---|---|---|---|
| **`Q02`** | M06 + latent MSE at 1× parity | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 | **highest `pearson_delta` at 5/5** |
| **`M06`** | `dcos` 0.5 + `delta_mag` 0.01363 | 0.3681 ± 0.0023 | 0.0232 | 0.0703 | **0.7047** | best `disc_l1` of the three |
| **`M03`** | `dcos` 0.01 (scale-matched) | 0.3679 ± 0.0017 | **0.0212** | 0.0722 | 0.6715 | best `mae`; the balanced-loss regime |

⚠️ **These are NOT interchangeable starting points, and a result on one does not transfer.**
Three separate levers have now shown opposite signs between M03 and M06: `delta_mag_weight`
(F-loss9), the latent losses (F-latent4/F-latent5), and the split reconstruction (F-recon3, M03
side). The working rule from F-latent4: *a term that repairs a DIRECTION-DOMINATED loss (M06/D03,
`smooth_l1` at 2.1% of the total) tends to HARM a balanced one (M03, at 51.5%)*. `Q02` inherits
M06's regime, so expect it to behave like M06, not like M03 — but that is an expectation, not a
measurement.

⚠️ `Q05` (latent cosine 1×) scores 0.3787, marginally above `Q02`, but on **4/5 splits**. `Q02`
is the defensible choice until `Q05` completes.


## Split reconstruction — M03 base (job 18217067, PARTIAL: 26 arms, most still scoring)

F-recon3: **F-recon1's prediction is FALSIFIED.** The split is the first lever that touches the control-reconstruction term directly, and `mae` got WORSE in 5/5 arms (+0.0012…+0.0036). The mechanism was right and measured; the prediction from it was wrong.

**Phase 2 loses to phase 1 on both modes** — `raw` 0.3282 vs 0.3424, `cmean` 0.3540 vs 0.3575. Denoising the delta target HURTS: the single-cell target's noise was not the binding constraint. `raw` is the WORST mode despite being the most faithful, because reconstructing a specific control cell is dominated by that cell's own unlearnable noise — which is why its balance stayed FLAT at 32% while `cmean`'s decayed. **The flat balance was the warning sign, not the feature.**

✅ The one thing the split buys: **`disc_l1` rises in all five arms** (+0.016…+0.027).

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `G05` `delta_only` / group_mean | 2/5 ⚠️ | **0.3601** ± 0.0016 | 0.0226 | 0.0718 | 0.6985 | 2/5 — partial |
| `G03` `cmean` / cell | 5/5 | **0.3575** ± 0.0023 | 0.0224 | 0.0663 | 0.6871 | phase 1 |
| `G04` `cmean` / group_mean | 5/5 | **0.3540** ± 0.0022 | 0.0225 | 0.0689 | 0.6931 | **PHASE 2 PROPER** |
| `G01` `raw` / cell | 5/5 | **0.3424** ± 0.0033 | 0.0248 | 0.0632 | 0.6932 | phase 1 |
| `G02` `raw` / group_mean | 5/5 | **0.3282** ± 0.0024 | 0.0240 | 0.0590 | 0.6922 | phase 2 |

## Latent-space losses — M06 base, COMPLETE (job 18215244)

F-latent4 / F-latent5: **all 11 arms beat the M06 base**, and the same interventions HURT M03 — identical change, opposite sign, several times the sds. Third setup-dependent lever after `delta_mag_weight`. The useful range is around **parity**; at 0.01× (`T02`) the term stops helping and starts hurting, which is the lower bound.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `Q05` latcos(delta_z) 1× | 4/5 ⚠️ | **0.3787** ± 0.0032 | 0.0225 | 0.0763 | 0.7078 | highest number, but 4/5 |
| **`Q02` latMSE 1×** | 5/5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 | **best 5/5 arm in the campaign** |
| `Q10` latMSE 10× + cos 1× | 5/5 | **0.3772** ± 0.0017 | 0.0223 | 0.0768 | 0.6961 |  |
| `Q04` latcos(delta_z) 0.1× | 4/5 ⚠️ | **0.3764** ± 0.0023 | 0.0223 | 0.0693 | 0.6921 | best `mae` of the family |
| `Q09` both 10× | 5/5 | **0.3755** ± 0.0015 | 0.0221 | 0.0683 | 0.6931 |  |
| `Q01` latMSE 0.1× | 5/5 | **0.3725** ± 0.0024 | 0.0227 | 0.0794 | 0.7008 | best `ovl@N` of the family |
| `Q03` latMSE 10× | 4/5 ⚠️ | **0.3721** ± 0.0028 | 0.0229 | 0.0755 | 0.6968 |  |
| `Q08` both 1× | 5/5 | **0.3717** ± 0.0018 | 0.0227 | 0.0742 | 0.6992 |  |
| `Q11` both 0.1× | 5/5 | **0.3708** ± 0.0024 | 0.0226 | 0.0697 | 0.7016 |  |
| `Q06` latcos(delta_z) 10× | 4/5 ⚠️ | **0.3690** ± 0.0023 | 0.0230 | 0.0720 | 0.7022 |  |
| `Q07` latcos(z_hat) 1× | 4/5 ⚠️ | **0.3681** ± 0.0020 | 0.0225 | 0.0663 | 0.6890 | the one z_hat arm |
| `T02` both 0.01× | 5/5 | **0.3617** ± 0.0023 | 0.0235 | 0.0714 | 0.7026 | ⛔ only latent arm to HURT M06 |

## Adjacency-map hyperparameter arms (earlier sweep)

The pre-DCOS population. `adjmap_s313` is the arm every later base descends from; its 0.3415 is L11's number and was the campaign headline before DCOS.

| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |
|---|---|---|---|---|---|---|
| `adjmap_s313` | 5/5 | **0.3415** ± 0.0031 | 0.0210 | 0.0642 | 0.6534 | s313 = the ancestor of M03/M06 |
| `adjmap_s145` | 5/5 | **0.3405** ± 0.0029 | 0.0210 | 0.0666 | 0.6506 |  |
| `adjmap_s306` | 5/5 | **0.3396** ± 0.0029 | 0.0209 | 0.0657 | 0.6543 |  |
| `adjmap_s469` | 5/5 | **0.3392** ± 0.0034 | 0.0210 | 0.0669 | 0.6533 |  |
| `adjmap_s321` | 5/5 | **0.3373** ± 0.0038 | 0.0209 | 0.0656 | 0.6489 |  |
| `adjmap_s129` | 5/5 | **0.3348** ± 0.0032 | 0.0210 | 0.0618 | 0.6479 |  |

## Split reconstruction — the COMPLETE 26-arm grid

F-recon3 / F-recon4. **No arm beats its base on `pearson_delta`.** Best is `G10` (M06, `delta_only`) at 0.3607 against M06's 0.3681. `delta_only` — dropping `L_ctrl` entirely — is the best split mode on BOTH bases, so the control-reconstruction term is not mis-weighted, it is unwanted. Phase 2 (`group_mean` delta target) loses to phase 1 on both modes.

✅ The one thing the split buys is **`disc_l1`**: `G10` 0.7138, `G06` 0.7129, `G08` 0.7127, `G19` 0.7074, `G16` 0.7072 — all above both bases (0.7047 / 0.6715).

⚠️ The ratio ladder is monotonically destructive on M03 (`G01`→`G11`→`G12`→`G13`: 0.3424 → 0.3236 → 0.2933 → 0.2706) but NOT on M06 (`G06`→`G14`→`G15`→`G16`: 0.3499 → 0.3423 → 0.3248 → 0.3399) — M06 recovers at `w_delta` 30. Setup-dependence again.

| arm | base | mode / weights | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|---|
| `G23` | M06 | delta_only w_d 3 | 4/5 ⚠️ | **0.3628** ± 0.0019 | 0.0238 | 0.0734 | 0.7108 |
| `G10` | M06 | delta_only 1:1 | 5/5 | **0.3607** ± 0.0026 | 0.0247 | 0.0691 | 0.7138 |
| `G05` | M03 | delta_only 1:1 | 5/5 | **0.3582** ± 0.0025 | 0.0227 | 0.0711 | 0.6979 |
| `G19` | M06 | cmean/cell w_d 3 | 5/5 | **0.3576** ± 0.0022 | 0.0239 | 0.0661 | 0.7074 |
| `G03` | M03 | cmean/cell 1:1 | 5/5 | **0.3575** ± 0.0023 | 0.0224 | 0.0663 | 0.6871 |
| `G21` | M03 | delta_only w_d 3 | 5/5 | **0.3562** ± 0.0021 | 0.0220 | 0.0699 | 0.6830 |
| `G04` | M03 | cmean/group_mean 1:1 | 5/5 | **0.3540** ± 0.0022 | 0.0225 | 0.0689 | 0.6931 |
| `G26` | M06 | raw scale-matched | 5/5 | **0.3532** ± 0.0025 | 0.0244 | 0.0654 | 0.7057 |
| `G22` | M03 | delta_only w_d 10 | 5/5 | **0.3529** ± 0.0018 | 0.0215 | 0.0692 | 0.6725 |
| `G18` | M03 | cmean/cell w_d 10 | 5/5 | **0.3529** ± 0.0026 | 0.0216 | 0.0678 | 0.6694 |
| `G17` | M03 | cmean/cell w_d 3 | 5/5 | **0.3522** ± 0.0025 | 0.0219 | 0.0648 | 0.6789 |
| `G09` | M06 | cmean/group_mean 1:1 | 5/5 | **0.3512** ± 0.0027 | 0.0249 | 0.0600 | 0.7038 |
| `G08` | M06 | cmean/cell 1:1 | 5/5 | **0.3511** ± 0.0027 | 0.0247 | 0.0636 | 0.7127 |
| `G06` | M06 | raw/cell 1:1 | 5/5 | **0.3499** ± 0.0025 | 0.0242 | 0.0653 | 0.7129 |
| `G24` | M06 | delta_only w_d 10 | 5/5 | **0.3481** ± 0.0027 | 0.0243 | 0.0563 | 0.6841 |
| `G07` | M06 | raw/group_mean 1:1 | 4/5 ⚠️ | **0.3456** ± 0.0019 | 0.0237 | 0.0634 | 0.7020 |
| `G01` | M03 | raw/cell 1:1 | 5/5 | **0.3424** ± 0.0033 | 0.0248 | 0.0632 | 0.6932 |
| `G14` | M06 | raw/cell w_d 3 | 5/5 | **0.3423** ± 0.0022 | 0.0242 | 0.0587 | 0.7010 |
| `G20` | M06 | cmean/cell w_d 10 | 4/5 ⚠️ | **0.3415** ± 0.0024 | 0.0242 | 0.0604 | 0.7037 |
| `G16` | M06 | raw/cell w_d 30 | 5/5 | **0.3399** ± 0.0025 | 0.0252 | 0.0555 | 0.7072 |
| `G25` | M03 | raw scale-matched | 5/5 | **0.3362** ± 0.0032 | 0.0244 | 0.0595 | 0.6760 |
| `G02` | M03 | raw/group_mean 1:1 | 5/5 | **0.3282** ± 0.0024 | 0.0240 | 0.0590 | 0.6922 |
| `G15` | M06 | raw/cell w_d 10 | 5/5 | **0.3248** ± 0.0029 | 0.0260 | 0.0559 | 0.7006 |
| `G11` | M03 | raw/cell w_d 3 | 5/5 | **0.3231** ± 0.0030 | 0.0248 | 0.0555 | 0.6778 |
| `G12` | M03 | raw/cell w_d 10 | 5/5 | **0.2930** ± 0.0035 | 0.0304 | 0.0577 | 0.6109 |
| `G13` | M03 | raw/cell w_d 30 | 5/5 | **0.2699** ± 0.0036 | 0.0371 | 0.0501 | 0.5740 |

**Bases for reference:** M06 0.3681 / 0.0232 / 0.0703 / 0.7047 · M03 0.3679 / 0.0212 / 0.0722 / 0.6715.

