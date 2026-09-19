# `rhj_fewshot` split-half paired evaluation — findings

**This file owns every number that comes out of the split-half protocol.** Written as it is
read, partial results labelled partial, superseded numbers kept with the correction beside
them.

## ⛔ F-latent1 — the latent loss has NEVER been swept, and its diagnostics were never read

**Asked directly, so checked rather than recalled.**

| | status |
|---|---|
| runs using `latent_loss_type != "none"` | **0**, across every run in `runs/` |
| findings docs mentioning it | **none** |
| declared in `configs/rhj_fewshot.yaml` | **NO** — so `--set stages.vcell.loss.latent_loss_type=...` would fail exactly as `nb_weight` did, with *"not an existing key"* |
| registered in `VCELL_SCHEMA` | yes (`schema.py:344`) |

So it is half-plumbed: the dataclass has `latent_loss_type` (`none / delta_z / z_hat / both /
projected`), `latent_loss_weight`, `latent_loss_alpha`, `latent_proj_dim`, and the schema knows
them — but the config never declares them, so **the axis is unreachable from a launcher** and
has never been run.

### ⚠️ Worse: the diagnostics that would justify sweeping it were never read either

`config.py` documents why they exist, and it is a strong claim:

> Run REGARDLESS of the latent loss, because **the arms without one are exactly the ones that
> can collapse to the identity** (`delta_z_ratio` -> 0, which scores at the identity floor and
> so reads as "weak", not "broken") or drift off the decoder's manifold (`offmanifold_ratio`
> >> 1, i.e. the score is decoder extrapolation).

`latent_diag_every: 50` is on by default, and `_compute_latent_diagnostics` emits
`delta_z_ratio`, `offmanifold_ratio`, `z_hat_offmanifold`, `z_ctrl_offmanifold`,
`delta_z_cosine_w_target`, `gate_active_dims`. **But they are written only to `train_log`, i.e.
to wandb** (`pipeline.py:4053-4060`) — never to stdout. Grepping every run log for
`delta_z_ratio` or `offmanifold` returns **0 hits**, and the arms have no local wandb
directories, so the values are not on disk at all.

**Consequence: we have never checked whether our models are collapsing to the identity or
predicting off-manifold** — the two failure modes the code says look like "a weak model" rather
than a broken one. Every arm in this campaign, including `s313`, could be sitting at the
identity floor and the score alone would not distinguish that from a genuinely weak model.

### What would make this answerable, cheaply

1. **Print the diagnostics to stdout**, not only wandb — one line per eval epoch. They are
   already computed; only the sink is wrong, and wandb is now `offline` for competitor runs
   anyway.
2. **Declare the four `latent_*` keys in the config** so the axis is reachable at all.
3. Read `delta_z_ratio` on the existing best arm FIRST. If it is healthy (not ~0), a latent
   loss is solving a problem we do not have and the sweep is not worth GPUs; if it is near
   zero, that reframes the whole campaign — the model would be scoring at the identity floor
   and every loss-composition result above would be measuring the wrong thing.

⚠️ Step 3 is the pre-registration: **the diagnostic decides whether the sweep is worth running**,
rather than the sweep being run and the diagnostic quoted afterwards.

## ⛔ F-infra1 — a pdex shared-memory race silently dropped **31 (arm, split, mode) scores** while every SLURM task exited 0

**What broke.** `pdex 0.1.24`'s multiprocessing path fails with
`FileNotFoundError: [Errno 2] No such file or directory: '/psm_<hex>'` at
`pdex/_single_cell.py:140` (`SharedMemory(name=shm_name)`) when several cell-eval processes run
concurrently on one node. Measured in **18 of 60** tasks of job `18179618`.

**Why it was invisible.** `src/occam/vcell/panel/run.py:158` catches the empty result and prints
`[panel] <tag>: cell-eval returned nothing; skipping`, then **continues**. The task finishes the
other arms, writes `metrics.json`, and exits 0 — so `sacct` reports `COMPLETED`, the
`celleval_model_<mode>/` directory exists, and only `agg_results.csv` is absent. **24 skips**
across those 18 tasks.

**Damage: 31 missing (arm, split, mode) scores**, so several published means are over fewer
than 5 splits:

| arm | uncalibrated splits present | calibrated |
|---|---|---|
| `L12` | **1/5** | 5/5 |
| `L10` | 2/5 | 1/5 |
| `L08` | 2/5 | 4/5 |
| `L14` | 3/5 | 4/5 |
| `L07`, `L09` | 4/5 | 4/5 |
| `L01`, `L05`, `L06`, `L13` | 5/5 | 4/5 |
| `D03` | 2/5 | 3/5 |
| `D02` | **5/5** ✅ | 4/5 |

**What this does and does not invalidate.**
- ✅ **F-loss4 stands.** `D02` has all 5 uncalibrated splits and `L11` has all 5, so the
  +0.0246 / +0.0646 paired comparison is complete and unaffected.
- ✅ **F-loss3's ordering stands.** Every NB arm's uncalibrated mean rests on 4–5 splits, and
  the across-split sd is ~0.003 against gaps of 0.02–0.07.
- ⚠️ **`D03` (the validation optimum, `delta_cos_weight` 0.5) cannot be reported.** Its control
  `L12` has 1/5 uncalibrated splits and `D03` itself has 2/5. That pair is the one this bug
  actually blocks.

**Why this is the dangerous class of failure.** The job says COMPLETED, the directory exists,
and the number is merely absent — so a mean silently becomes a mean-over-fewer-splits with no
error anywhere. This is the same shape as F-loss1 (DCOS ran inert with no error) and the STATE
sentence-collapse (S1, no error, no warning). **A guard that checks a count is not a guard that
checks a result.**

**Fix applied:** `run.py` now RAISES on an empty cell-eval result instead of printing and
continuing, so the task fails loudly and the arm-split is re-runnable. Opt out with
`OCCAM_ALLOW_EMPTY_CELLEVAL=1` for the case where a genuinely degenerate arm is expected to
produce nothing.

**Not yet done:** re-running the 31 missing scores. Cheap (~10 min each, embarrassingly
parallel) but not launched — it was not requested.

## ⭐⭐⭐ F-loss4 — DCOS CONFIRMED ON TEST: **`pearson_delta` +0.0246 AND `disc_l1` +0.0646**, at the cost of `mae` and `overlap_at_N`

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline is split-half
paired `pearson_delta` on `rhj_fewshot`. Anchors (test, 5 splits): `ctrlcell` −0.0018 ·
`ctrlmean` 0.0016 · `meanpert` 0.1324 · **`transfer` 0.2323** · **previous best 0.3415** ·
best competitor (TxPert) 0.3463 · `experimental` 0.3722.

**Question this answers.** F-loss2 measured +0.028…+0.034 for DCOS on *validation*
`best_metric`. Validation and cell-eval have ranked models OPPOSITELY in this project before.
Does the gain survive to the test metric the 0.3415 headline is quoted on?

**What is varied:** the delta-cosine term, live vs inert. **Held fixed: everything else, exactly.**
`D02` and `L11` have the **same `_provenance.stage_hashes.vcell`** (`8e4af0d2…`) — byte-identical
configs, both `delta_cos_weight: 0.25`. The only difference is `git_sha`: `L11` ran before the
F-loss1 group-mean gate fix with the term silently ZERO, `D02` after it with the term live. Same
cohort, same axis, same seeded per-split control draw, so the comparison is **paired per split**.

**What each outcome means (pre-registered in F-loss2).** Positive ⇒ DCOS is a real lever and the
weight is adoptable. Null ⇒ `best_metric` and cell-eval disagree again and DCOS is a
validation-metric artefact.

### Measured — uncalibrated (`model_nocal`), paired per split, job `18181379`

`mae` is an ERROR: **lower is better.** `pearson_delta` and `overlap_at_N`: higher is better.

| split | `pearson_delta` L11 → D02 | Δ | `mae` ↓ L11 → D02 | `overlap_at_N` L11 → D02 |
|---|---|---|---|---|
| 0 | 0.3411 → **0.3656** | **+0.0245** | 0.0210 → 0.0277 | 0.0646 → 0.0567 |
| 1 | 0.3464 → **0.3691** | **+0.0227** | 0.0210 → 0.0277 | 0.0655 → 0.0577 |
| 2 | 0.3407 → **0.3674** | **+0.0267** | 0.0210 → 0.0277 | 0.0640 → 0.0572 |
| 3 | 0.3378 → **0.3652** | **+0.0274** | 0.0210 → 0.0277 | 0.0628 → 0.0562 |
| 4 | 0.3414 → **0.3631** | **+0.0217** | 0.0210 → 0.0278 | 0.0642 → 0.0563 |
| **mean** | **0.3415 → 0.3661** | **+0.0246** | 0.0210 → 0.0277 (**+32%, worse**) | 0.0642 → 0.0568 (**−12%, worse**) |

**`discrimination_score_l1` (chance 0.5, higher is better) — the fourth metric, and it IMPROVES:**

| split | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| L11 (inert) | 0.6541 | 0.6525 | 0.6537 | 0.6527 | 0.6538 | 0.6534 |
| **D02 (live)** | **0.7181** | **0.7181** | **0.7191** | **0.7165** | **0.7181** | **0.7180** |
| Δ | +0.0640 | +0.0656 | +0.0654 | +0.0638 | +0.0643 | **+0.0646** |

### Reading

1. **DCOS survives to test. 5/5 splits positive**, range +0.0217…+0.0274 — the whole range sits
   ~7× outside the across-split sd of ~0.003. The validation signal (+0.033 at this weight) and
   the test signal (+0.0246) agree in sign and are close in size, so this is not a
   validation-metric artefact.
2. **0.3661 is a new best.** It clears the previous 0.3415 headline and also clears the best
   competitor, TxPert at 0.3463 (+0.0198). It sits 0.0061 below `experimental` 0.3722.
3. **`disc_l1` gains +0.0646, 5/5 splits, range +0.0638…+0.0656** — a 2.6× larger move than
   `pearson_delta`'s, and even tighter. `disc_l1` measures how well the prediction identifies
   WHICH perturbation it is, so a delta-direction term improving it is mechanistically the
   expected result, not a coincidence.
4. **⚠️ It is a TRADE: two metrics up, two down.** UP: `pearson_delta` +0.0246, `disc_l1`
   +0.0646. DOWN: `mae` 0.0210 → 0.0277 (+32%, worse) and `overlap_at_N` 0.0642 → 0.0568
   (−12%, worse), consistently on all 5 splits. That is the expected signature of a term that
   supervises delta DIRECTION with magnitude divided out: direction and identity improve,
   calibrated magnitude degrades. **Anyone quoting 0.3661 must quote all four.** Which arm is
   preferable depends on the metric the submission is scored on; that is a decision, not a
   measurement.

   ⚠️ **Correction to my own first report of this finding:** I reported the `mae` and
   `overlap_at_N` costs but gave D02's `disc_l1` as 0.6534 — that is L11's value. D02's is
   0.7180. The trade is 2-up/2-down, not 1-up/2-down, and the largest single move in the whole
   comparison is a GAIN.
4. The `disc_l1` column and `D03` (`delta_cos_weight` 0.5, the validation optimum at 0.4108, vs
   0.25's 0.4100) are still scoring — `D03`'s control `L12` currently has 1/5 uncalibrated
   splits, so that pair is **not yet comparable**.

⚠️ **n = 1 seed (42), 1 weight confirmed.** The per-split pairing is tight, but nothing here
separates "DCOS helps" from "DCOS helps at seed 42". No seed replicate has been run.

⚠️ **Prior launch defect, corrected:** job `18181054` scored these arms with the launcher's
default `CALIBRATE=quantile`, which emits `model_cal` ONLY. Calibration costs 0.068
`pearson_delta` (F-loss3), so that job could not answer this question. Cancelled and relaunched
as `18181379` with `CALIBRATE=both`. The one salvageable number from it agrees:
D02 split 0 calibrated 0.3009 vs L11 calibrated 0.2732, **+0.0277**.


### ⏳ PARTIAL — `D03` (`delta_cos_weight` 0.5): 3 of 5 splits, and it looks BETTER than `D02` on 3 of 4 metrics

⚠️ **n = 3 (splits 1, 2, 3). Not a result yet.** Splits 0 and 4 lost their uncalibrated arm to
the F-infra1 shm race and are being re-run. Recorded now because F-loss2 pre-registered 0.5 as
the validation optimum (0.4108 vs 0.25's 0.4100), so this is the arm that prediction points at.

⚠️ **The control is `L11`, not `L12`.** `D03`'s exact same-config-hash twin is `L12`, which has
only **1 of 5** uncalibrated splits (F-infra1). `L11`, `L12` and `L13` are all smooth_l1
replicates and agree to four decimals where both exist (0.3415 / 0.3414 / 0.3415), so `L11` is a
sound stand-in — but it is a **stand-in, stated as such**, not the exact pairing used for `D02`.

| | split 1 | split 2 | split 3 | mean (n=3) |
|---|---|---|---|---|
| `L11` control `pearson_delta` | 0.3464 | 0.3407 | 0.3378 | 0.3416 |
| **`D03` `pearson_delta`** | **0.3721** | **0.3672** | **0.3670** | **0.3688** |
| `L11` control `overlap_at_N` | 0.0655 | 0.0640 | 0.0628 | 0.0641 |
| **`D03` `overlap_at_N`** | **0.0732** | **0.0716** | **0.0699** | **0.0716** |

**Paired `D03` − `L11` over those 3 splits, all four metrics:**

| metric | Δ | direction |
|---|---|---|
| `pearson_delta` ↑ | **+0.0271** | better (`D02` was +0.0246) |
| `discrimination_score_l1` ↑ | **+0.0732** | better (`D02` was +0.0646) |
| `overlap_at_N` ↑ | **+0.0075** | **better — and `D02` was −0.0074, i.e. the sign FLIPS** |
| `mae` ↓ | +0.0112 | worse (`D02` was +0.0067, so `D03` is worse here) |

**Reading, if it holds at n=5.** At weight 0.5 the trade is no longer 2-up/2-down: **three
metrics improve and only `mae` degrades.** `overlap_at_N` changing sign between weight 0.25
(−0.0074) and 0.5 (+0.0075) is the single most interesting thing in the sweep, because
`overlap_at_N` was the metric DCOS appeared to cost. If real, the right `delta_cos_weight` is
0.5, not 0.25, and the "DCOS trades DE-overlap for correlation" framing from `D02` is an
artefact of using too small a weight.

⚠️ **Do not act on this yet.** n=3, one seed, and the two missing splits are not missing at
random — they are the ones the shm race hit. `D02`'s per-split spread is ±0.0023 and its worst
split (4) is its lowest, so a mean over splits 1–3 is plausibly optimistic. The `D02` numbers in
the table above ARE complete (5/5) and are the ones to quote today.

**Config that produced it:** `rhj_fewshot_D02__smooth_l1_nbw0_nbs1.0_dcos0.25_s42`, checkpoint
epoch 21 `best_metric` 0.4100, git `e7c3a78`, `configs/rhj_fewshot.yaml` +
`--set stages.vcell.loss.delta_cos_weight=0.25`. Scored by `configs/rhj_dcos_eval.tsv` through
`scripts/slurm/rhj_splithalf_models.sbatch` with `CALIBRATE=both`.

## ⭐⭐ F-emit6 — the 4-way emission/calibration comparison on s313, **split 00 only**: doing NOTHING wins `pearson_delta`; emission buys `overlap_at_N`

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline is split-half
paired `pearson_delta` on `rhj_fewshot`. Anchors (test, 5 splits): `meanpert` 0.1324 ·
**`transfer` 0.2323** · **ours 0.3415** · `experimental` 0.3722.

**Question this answers.** On ONE fixed checkpoint, which post-hoc emission treatment should be
the default: nothing, control-fit quantile calibration, NB count emission, or NB emission with
per-perturbation exact column sums? Phrased so every outcome is meaningful: if one arm wins all
four metrics it is simply the default; if the arms trade metrics, the choice depends on which
metric the submission is scored on, and that has to be said explicitly.

**What is varied:** the post-hoc treatment only. **Held fixed:** the checkpoint (`s313`,
`rhj_fewshot_s313__lr1e-3_dlr1e-3_cos0.5_dpo0_fmode-mult_fh256_fmf1_s42`, epoch 25,
`best_metric` 0.3769), the split (00), the cohort, the gene axis, the seeded control draw.
**Nothing is retrained** — this is engineering on a finished prediction, not modelling.

### Measured — `s313`, split 00

`mae` is an ERROR: **lower is better.**

| treatment | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | emission config |
|---|---|---|---|---|---|
| **none (`model_nocal`) — the default** | **0.3411** | **0.0210** | 0.0646 | 0.6541 | — |
| calibration only (`model_cal`) | 0.2740 | 0.0886 | 0.0248 | **0.7750** | control-fit quantile |
| NB emission, fair theta | 0.1775 | 0.0331 | 0.0614 | 0.6475 | theta from checkpoint `log_theta` via `axis_ensembl`, lib 17,500 |
| **NB emission + `exact_colsum`, fair theta** | 0.2027 | 0.0276 | **0.0961** | 0.6393 | same, per-perturbation column sums pinned |
| NB emission, **broken** theta/lib | 0.1611 | 0.0490 | 0.0079 | 0.6700 | ⛔ MoM theta collapsed to the 10,000 cap, lib 2,000 (8.7× too shallow) |
| NB emission + `exact_colsum`, **broken** | 0.2478 | 0.0295 | 0.0314 | 0.6511 | ⛔ same broken inputs |

### Reading

1. **On `pearson_delta`, doing nothing wins by a wide margin.** 0.3411 vs 0.2740 (calibration),
   0.2027 (NB + colsum), 0.1775 (NB alone). Every treatment costs between 0.067 and 0.164. If
   `pearson_delta` is the reported metric, the answer is: apply none of them.
2. **On `overlap_at_N`, NB emission + `exact_colsum` wins by 49%** — 0.0961 vs 0.0646 for the
   raw model. That is F-emit5's result, reproduced here in the same table as its alternatives.
   The two metrics point in **opposite directions**, so this is a genuine trade, not a
   dominance.
3. **`exact_colsum` is worth +0.025 `pearson_delta` and +0.035 `overlap_at_N` within emission**
   (0.1775→0.2027, 0.0614→0.0961) and improves `mae` 0.0331→0.0276. It is variance reduction on
   noise we injected ourselves, and it moves what pinning a column sum should move. Never use NB
   emission without it.
4. **Calibration is the worst of both:** it loses 0.067 `pearson_delta`, quadruples `mae`
   (0.0210→0.0886) and cuts `overlap_at_N` by 2.6× (0.0646→0.0248). Its only gain is `disc_l1`
   (0.6541→0.7750). Consistent with F-loss3, where calibration lost on 14/14 loss arms.
5. **The broken-input rows are kept deliberately** (F-emit4 → F-emit5). They show the failure is
   not subtle: with theta collapsed to the cap and lib 8.7× too shallow, `overlap_at_N` falls to
   0.0079 — below the baseline — and a conclusion was drawn from it before the zero-fraction
   sanity check (0.877 vs the reference 0.515) was read.

⚠️ **SPLIT 00 ONLY, n = 1.** Every number in this table is a single split. The loss arms show an
across-split sd of ~0.003 on `pearson_delta`, so the large gaps here (0.07–0.16) are far outside
it, but the small ones — `exact_colsum`'s +0.025, and anything on `overlap_at_N`, whose
across-split variability has not been measured — are **not** established. Splits 1–4 were
deferred by the user ("nah lets just wait for others").

⚠️ `s313_fairtheta` emitted split 00 but its cell-eval never produced `agg_results.csv`; it is
not in the table.

## ⭐⭐ F-loss3 — TEST cell-eval on all 14 loss arms: **every NB arm loses to plain smooth_l1, and calibration hurts 14/14**

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline is split-half
paired `pearson_delta` on `rhj_fewshot`. Anchors (test, all 5 splits): `ctrlcell` −0.0018 ·
`ctrlmean` 0.0016 · `meanpert` 0.1324 · **`transfer` 0.2323** · **ours 0.3415** ·
`experimental` 0.3722.

**Questions this answers.** (a) Does an NB likelihood term beat plain smooth_l1 on the *test*
metric, at any weight, and does scale-matching rescue it? (b) Does control-fit quantile
calibration help on the loss arms, as it was assumed to?

**What is varied:** `loss_type` ∈ {smooth_l1, nb, nb_smooth_l1}, `nb_weight` ∈ {0, 0.1, 0.25,
0.5, 0.75, 0.9}, `nb_scale` ∈ {1.0, 6.5}, × calibration {off, on}. **Held fixed:** lr 1e-3,
decoder_lr 1e-3, cos 0.5, `fmode-mult`, `fh256`, `fmf1`, seed 42, adjacency-map pert model, the
611-perturbation cohort, the 6,983-gene axis, the seeded per-split control draw.

**What each outcome means (pre-registered).** NB above smooth_l1 ⇒ the count likelihood is worth
its complexity and `nb_scale` matters. NB below at *every* weight including the scale-matched
ones ⇒ F6's NB penalty is real on adjacency-map arms and is not a scale artefact. Calibration
positive ⇒ keep it on. Calibration negative ⇒ the 0.3415 headline must stay `model_nocal`.

⚠️ `nb_scale` 6.5 is the **scale-matched** arm: NB and smooth_l1 have different natural
magnitudes, so a raw `nb_weight` confounds "how much NB" with "how big NB's gradient is". 6.5 is
the measured ratio.

### Measured — jobs `18179608` (L01/L02) and `18179618` (14 arms × 5 splits), seed 42

Uncalibrated (`model_nocal`) is the reportable column. `mae` and the calibrated `mae` are
**ERRORS: lower is better.**

| arm | loss | `nb_weight` | `nb_scale` | n | **`pearson_delta` uncal** | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|---|---|
| **L11** | smooth_l1 | 0 | — | 5 | **0.3415 ± 0.0031** | 0.0210 | 0.0642 | 0.6534 |
| **L13** | smooth_l1 | 0 | — | 5 | **0.3415 ± 0.0031** | 0.0210 | 0.0642 | 0.6534 |
| L12 | smooth_l1 | 0 | — | 1 | 0.3414 | 0.0210 | 0.0642 | 0.6538 |
| L10 | smooth_l1 | 0 | — | 2 | 0.3396 ± 0.0025 | 0.0210 | 0.0635 | 0.6533 |
| L02 | nb_smooth_l1 | 0.1 | 1.0 | 5 | 0.3184 ± 0.0033 | 0.0211 | 0.0621 | 0.6483 |
| L08 | nb_smooth_l1 | 0.5 | **6.5** | 2 | 0.3129 ± 0.0033 | 0.0211 | 0.0643 | 0.6434 |
| L14 | nb_smooth_l1 | 0.5 | **6.5** | 2 | 0.3129 ± 0.0033 | 0.0211 | 0.0643 | 0.6434 |
| L07 | nb_smooth_l1 | 0.25 | **6.5** | 4 | 0.3098 ± 0.0028 | 0.0212 | 0.0608 | 0.6412 |
| L03 | nb_smooth_l1 | 0.25 | 1.0 | 5 | 0.2989 ± 0.0029 | 0.0214 | 0.0663 | 0.6440 |
| L09 | nb_smooth_l1 | 0.75 | **6.5** | 4 | 0.2976 ± 0.0025 | 0.0216 | 0.0629 | 0.6447 |
| L05 | nb_smooth_l1 | 0.75 | 1.0 | 5 | 0.2790 ± 0.0027 | 0.0226 | 0.0539 | 0.6472 |
| L06 | nb_smooth_l1 | 0.9 | 1.0 | 5 | 0.2774 ± 0.0027 | 0.0221 | 0.0660 | 0.6409 |
| L01 | **nb (pure)** | 1.0 | 1.0 | 5 | **0.2766 ± 0.0027** | 0.0220 | 0.0651 | 0.6425 |
| L04 | nb_smooth_l1 | 0.5 | 1.0 | 5 | **0.2740 ± 0.0029** | 0.0219 | 0.0600 | 0.6357 |

### Calibration: control-fit quantile, same arms

| arm | uncal | **cal** | Δ | `mae` uncal ↓ | `mae` cal ↓ | `ovl@N` uncal | `ovl@N` cal | `disc_l1` uncal | `disc_l1` cal |
|---|---|---|---|---|---|---|---|---|---|
| L11 | 0.3415 | 0.2732 | **−0.0683** | 0.0210 | 0.0890 | 0.0642 | 0.0243 | 0.6534 | 0.7766 |
| L13 | 0.3415 | 0.2736 | −0.0679 | 0.0210 | 0.0893 | 0.0642 | 0.0244 | 0.6534 | 0.7757 |
| L12 | 0.3414 | 0.2732 | −0.0682 | 0.0210 | 0.0890 | 0.0642 | 0.0243 | 0.6538 | 0.7766 |
| L10 | 0.3396 | 0.2715 | −0.0681 | 0.0210 | 0.0880 | 0.0635 | 0.0239 | 0.6533 | 0.7805 |
| L02 | 0.3184 | 0.2789 | −0.0395 | 0.0211 | 0.0917 | 0.0621 | 0.0240 | 0.6483 | 0.7575 |
| L08 | 0.3129 | 0.2773 | −0.0356 | 0.0211 | 0.0871 | 0.0643 | 0.0229 | 0.6434 | 0.7546 |
| L14 | 0.3129 | 0.2771 | −0.0358 | 0.0211 | 0.0869 | 0.0643 | 0.0226 | 0.6434 | 0.7556 |
| L07 | 0.3098 | 0.2619 | −0.0479 | 0.0212 | 0.0867 | 0.0608 | 0.0324 | 0.6412 | 0.7610 |
| L03 | 0.2989 | 0.2612 | −0.0377 | 0.0214 | 0.0624 | 0.0663 | 0.0208 | 0.6440 | 0.7728 |
| L09 | 0.2976 | 0.2703 | −0.0273 | 0.0216 | 0.0896 | 0.0629 | 0.0243 | 0.6447 | 0.7619 |
| L05 | 0.2790 | 0.2590 | −0.0200 | 0.0226 | 0.0923 | 0.0539 | 0.0283 | 0.6472 | 0.7699 |
| L06 | 0.2774 | 0.2691 | −0.0083 | 0.0221 | 0.0917 | 0.0660 | 0.0275 | 0.6409 | 0.7595 |
| L01 | 0.2766 | 0.2627 | −0.0139 | 0.0220 | 0.0902 | 0.0651 | 0.0246 | 0.6425 | 0.7708 |
| L04 | 0.2740 | 0.2372 | −0.0368 | 0.0219 | 0.0842 | 0.0600 | 0.0262 | 0.6357 | 0.7754 |

### Reading

1. **NB loses at every weight and every scale — 10/10 NB arms below plain smooth_l1.** Best NB is
   `L02` (`nb_weight` 0.1) at 0.3184, −0.0231 from 0.3415; pure NB (`L01`) is −0.0649. This
   reproduces F6 on adjacency-map arms with the confound removed, and the **scale-matched**
   arms (`nb_scale` 6.5: L07/L08/L09 at 0.3098/0.3129/0.2976) rule out "NB only lost because its
   gradient was the wrong size". Scale-matching *does* help NB relative to `nb_scale` 1.0 at the
   same weight (L08 0.3129 > L04 0.2740; L07 0.3098 > L03 0.2989) — it just never reaches
   smooth_l1. **NB's optimum is at `nb_weight` → 0**, which is an endpoint optimum and means
   "do not use NB", not "tune it lower".
2. **⛔ Calibration costs `pearson_delta` on 14/14 arms**, by 0.008 to 0.068, and makes `mae`
   **4× worse** (0.021 → 0.089). It buys `disc_l1` (0.64 → 0.78) and destroys `overlap_at_N`
   (0.064 → 0.024, a 2.6× fall). The headline must stay `model_nocal`. This extends F12 from one
   checkpoint to the whole loss family, and the damage is LARGEST on the best arms
   (−0.068 on the smooth_l1 group vs −0.008…−0.014 on the weakest NB arms), so calibration is
   not a safe default that merely fails to help.
3. **L11 = L13 = 0.3415 to four decimals, and L12 = 0.3414** — the bit-identical-curve signature
   that detected F-loss1. All four "DCOS" arms here ran with the term INERT, so they are
   smooth_l1 replicates. That makes them the **exact controls** for the live-DCOS D-series:
   `L11` is `D02`'s control and `L12` is `D03`'s (same config hash, different git_sha).
4. **The 0.3415 headline is reproduced** by an independent arm (`L13`) at the same value with the
   same sd, so it is not a one-run number.

⚠️ **Incomplete at time of writing:** 3 of 60 tasks in `18179618` still RUNNING. `L08`, `L10`,
`L14` have 2/5 splits and `L12` has 1/5 on the uncalibrated side; their means are **partial** and
labelled `n` above. Across-split sd is ~0.003 everywhere, so the partial means are unlikely to
move by more than ~0.005, but they are not final.

## ⭐⭐ F-loss2 — DCOS, once actually live, adds **+0.028…+0.034 validation** at every weight tested (5/5)

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline is split-half
paired `pearson_delta` on `rhj_fewshot`. Anchors: `meanpert` 0.1324 · `transfer` 0.2323 ·
**ours 0.3415** · `experimental` 0.3722.

**Question this answers.** Does a delta-cosine term — cosine between the predicted and true
perturbation *delta*, i.e. direction with magnitude divided out — improve the model, and at what
weight? Phrased so both outcomes are meaningful: a null says the smooth-L1 objective already
captures delta direction and the term is redundant; a positive says magnitude error was
dominating the gradient and direction was under-supervised.

**What is varied:** `stages.vcell.loss.delta_cos_weight` ∈ {0.1, 0.25, 0.5, 1.0, 2.0}, plus one
arm combining it with NB. **Held fixed:** everything else — lr 1e-3, decoder_lr 1e-3, cos 0.5,
`fmode-mult`, `fh256`, `fmf1`, `smooth_l1`, seed 42, adjacency-map pert model. Judged on
**validation `best_metric` only** (test cell-eval not yet run on these arms).

**Why the control is exact, not approximate.** `D01` and `L10` have the **same
`_provenance.stage_hashes.vcell`** (`07e9bfc6…`), as do `D02` and `L11` (`8e4af0d2…`). The
configs are byte-identical. The ONLY difference is `git_sha`: L10/L11 ran before the F-loss1 fix
with the term INERT, D01/D02 ran after it with the term LIVE. This is a clean code-level A/B.

### Measured — validation `best_metric`, seed 42

| arm | `delta_cos_weight` | loss | epoch | **val `best_metric`** | Δ vs inert 0.3769 |
|---|---|---|---|---|---|
| `L10` / `L11` / `s313` (INERT — the baseline) | 0.1 / 0.25 / — | smooth_l1 | 25 | **0.3769** | — |
| `D01` | 0.1 | smooth_l1 | 25 | 0.4049 | **+0.0280** |
| `D02` | 0.25 | smooth_l1 | 21 | 0.4100 | **+0.0331** |
| **`D03`** | **0.5** | smooth_l1 | 13 | **0.4108** | **+0.0339** ← best |
| `D04` | 1.0 | smooth_l1 | 17 | 0.4038 | +0.0269 |
| `D05` | 2.0 | smooth_l1 | 17 | 0.4077 | +0.0308 |
| `D06` | 0.25 | nb_smooth_l1, `nb_weight` 0.1 | 23 | 0.4070 | +0.0301 |

**Reading: 5/5 weights beat the baseline, and the optimum is interior and flat.** The spread
across a 20× range of weights (0.1 → 2.0) is 0.0070, smaller than the 0.0280 gap to baseline —
so the gain comes from *having* the term, not from tuning it. `0.5` is the best single value.
⚠️ The optimum being interior (not at 0.1 or 2.0) is what makes this a real optimum rather than
an endpoint artefact.

Adding NB on top (`D06`, `nb_weight` 0.1) does **not** help: 0.4070 vs `D02`'s 0.4100 at the same
`dcos` 0.25. Consistent with every other NB arm in the L-series sitting below plain smooth_l1.

**The best arms converge EARLIER** — `D03` peaks at epoch 13 vs the baseline's 25. Direction
supervision reaches its best state in about half the epochs.

⚠️ **This is validation only.** `best_metric` is not the split-half test `pearson_delta` that
0.3415 refers to, and the two have diverged before. The result is not comparable to the anchors
above until the test cell-eval is run on `D03`.

**Config that produced it:** `rhj_fewshot_D03__smooth_l1_nbw0_nbs1.0_dcos0.5_s42`, job
`18179643`, git `e7c3a78`, `configs/rhj_fewshot.yaml` + `--set stages.vcell.loss.delta_cos_weight=0.5`.

## ⭐ F-latent1 — the latent is NOT collapsed: `delta_z_ratio` 0.160, and 98% of the move is perturbation-SPECIFIC

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline is split-half
paired `pearson_delta` on the `rhj_fewshot` panel. Anchors: `ctrlcell` −0.0018 · `ctrlmean`
0.0016 · **`meanpert` 0.1324** · `transfer` 0.2323 · **ours 0.3415** · `experimental` 0.3722.

**Question this answers.** Is the perturbation head degenerate — i.e. does the model score where
it scores because `delta_z → 0` (it reproduces its control input, the identity floor), or because
it emits a single shared shift for every perturbation? Both failure modes are invisible in
`pearson_delta`, which a near-identity model can still pass at the `meanpert` anchor.

**What is varied, what is held fixed.** Nothing is varied — this is a diagnostic read on ONE
trained checkpoint. Inputs are the REAL HepG2 control mean from `cache/pert_means/hepg2.npz`
(model input space, encoder axis, 44,728 genes), held fixed across all 256 rows, so the ONLY
thing differing between rows is the perturbation id. 256 real HepG2 perturbations, all present
in `pert2id`.

**What each outcome means (pre-registered).**
- `delta_z_ratio ≈ 0` ⇒ COLLAPSED to identity; the pert head is dead weight and the score is the
  identity floor. Loss weighting / latent loss becomes the priority.
- pert-specific share ≈ 0 ⇒ one shared shift for every perturbation; the model has learned
  "something happened", not "which". `meanpert` is then the ceiling.
- Both healthy ⇒ the head works; the loss is not the bottleneck, and the remaining gap to
  `experimental` 0.3722 is a decoder/magnitude problem, not a collapse problem.

### Measured — checkpoint `s313` (`rhj_fewshot_s313__lr1e-3_dlr1e-3_cos0.5_dpo0_fmode-mult_fh256_fmf1_s42`, `best_model.pth`, epoch 25, `best_metric` 0.3769, pert model `adjacency_map`, 18,216 nodes)

| quantity | value | reads as |
|---|---|---|
| `‖delta_z‖` mean | 8.574 | — |
| `‖z_ctrl‖` mean | 53.593 | — |
| **`delta_z_ratio` mean** | **0.1600** | **NOT collapsed** (0 = identity) |
| `delta_z_ratio` median | 0.1359 | |
| `delta_z_ratio` p10 / p90 | 0.0808 / 0.2542 | 3.1× spread across perturbations |
| **pert-specific share** `‖dz − mean(dz)‖ / ‖dz‖` | **0.9824** | **NOT a shared shift** — 98% of the move is specific |

**Reading: both pre-registered failure modes are ruled out.** The perturbation moves the latent
by 16% of the control-state norm, and only 1.8% of that move is a bias common to all 256
perturbations. The `p10 → p90` spread (0.081 → 0.254) says the model also assigns *different
magnitudes* to different perturbations rather than one fixed step length.

⇒ The remaining gap to `experimental` 0.3722 is not a latent-collapse problem, and the latent
loss (never swept — 0 runs, see below) is not an obvious lever. Nothing here says the head is
*correct*, only that it is neither dead nor constant.

**Caveat that changes what happens next:** this is a single checkpoint and a single cell line's
control mean. A *real* control CELL (not the group mean) has more input variance; `delta_z_ratio`
measured against single cells could differ. Not measured.

**Reproduce:** `scripts/probe_latent.py <checkpoint>` (probe kept in the scratchpad as written;
the diagnostics themselves are now printed to stdout every eval epoch by
`src/occam/vcell/pipeline.py`, so future runs need no probe).

## ⭐⭐ F-emit5 — CORRECTION: with a fair theta and the right depth, emission RAISES `overlap_at_N` by 50%

⚠️ **THIS OVERTURNS F-emit4's RECOMMENDATION, WHICH WAS MINE AND WAS WRONG.** F-emit4 concluded
"recommend against pursuing emission for `overlap_at_N`". That conclusion was drawn from a run
with **two broken settings**, both mine, and it does not survive fixing them.

### What was broken in F-emit4

| | F-emit4 (broken) | F-emit5 (fair) |
|---|---|---|
| theta | method-of-moments on **library-normalised** data -> collapsed to the 10,000 CAP for every gene, i.e. **Poisson** | model-learned `log_theta` from the released decoder checkpoint, aligned on the **ENSEMBL** axis (symbol matching found 2 of 6,983), panel median **80.7** |
| emitted depth | `lib=2000` -> **0.29 counts/gene** over 6,983 genes -> zero-fraction **0.8767** | `lib=17500`, calibrated so the zero-fraction matches the reference's own **0.5153** |
| resulting zero-fraction | 0.8767 (target 0.5153) | **0.5053 / 0.5076** |

### The corrected numbers — `s313`, split 0, `model_nocal` base

| arm | `pearson_delta` ↑ | `mae` ↓ | **`overlap_at_N` ↑** | `disc_l1` ↑ |
|---|---|---|---|---|
| **mean-only (published)** | **0.3415** | **0.0210** | 0.0642 | **0.6534** |
| NB emission (fair) | 0.1775 | 0.0331 | 0.0614 | 0.6475 |
| **NB emission + `exact_colsum`** | 0.2027 | 0.0276 | **0.0961** | 0.6393 |

**`overlap_at_N` rises 0.0642 -> 0.0961, +50%.** That is the metric the exercise was for, and it
is the OPPOSITE of what F-emit4 measured (0.0079, an 8x fall) and of what I predicted from the
VCC `jac` evidence. The prediction was reasonable and it was wrong; the mechanism I reasoned
from does not dominate once the emitted distribution actually resembles the data.

⚠️ **And `exact_colsum` is doing the work, not emission alone.** Emission by itself is
essentially neutral on `overlap_at_N` (0.0614 vs 0.0642). The +50% appears only with the column
sum pinned — consistent with F-emit2/F-emit3: the pin removes the sampling scatter, leaving the
dispersion that the Wilcoxon test needs without the noise that dilutes the ranking.

### ⚠️ It is a TRADE-OFF, not a free win

| metric | change vs mean-only | direction |
|---|---|---|
| `overlap_at_N` | **+0.0319 (+50%)** | ✅ better |
| `pearson_delta` | **−0.1388** | ⛔ much worse |
| `mae` | +0.0066 | ⛔ worse |
| `disc_l1` | −0.0141 | ⛔ slightly worse |

`pearson_delta` is our headline metric and it falls by more than the entire `meanpert`→
`transfer` gap. **So this is not an adopt-by-default change.** It is a per-metric emission
choice: if `overlap_at_N` is the target, emission + `exact_colsum` buys 50%; if
`pearson_delta` is, it costs 0.14.

⚠️ **n=1 split.** The other four splits are needed before this is quoted as a result, and the
depth `lib=17500` was tuned to match the reference's zero fraction — legitimate under the
project's "match the reference's own statistics" rule, but it IS a fitted quantity and should
be stated as one.

### The lesson

Two configuration errors — an ID-space mismatch and a depth 8.7x too shallow — produced a
confident, wrong, and *plausible* negative result, which I then converted into a
recommendation. The zero-fraction diagnostic (0.8767 vs a target of 0.5153) was visible in the
F-emit4 log and I read past it. **A sanity statistic that disagrees with the reference by 70%
is not a detail to note; it invalidates the run.**

## ⭐⭐ F-emit4 — MEASURED on s313: emission HURTS. `exact_colsum` helps a lot, but only within emission.

**Question.** Does NB count emission raise `overlap_at_N`, and does `exact_colsum` on top of it
help? Split 0 of our best arm (`s313`), post-hoc on the saved predictions, so the mean-only row
is byte-identical to the published number.

| arm | `pearson_delta` ↑ | `mae` ↓ | **`overlap_at_N` ↑** | `disc_l1` ↑ |
|---|---|---|---|---|
| **mean-only (published)** | **0.3415** | **0.0210** | **0.0642** | 0.6534 |
| emission only | 0.1611 | 0.0490 | **0.0079** | **0.6700** |
| emission + `exact_colsum` | 0.2478 | 0.0295 | 0.0314 | 0.6511 |

### The answer to the question that motivated this: NO

`overlap_at_N` was the target, and it **fell 8x** with emission (0.0642 → 0.0079) and is still
**halved** with `exact_colsum` (0.0314). `pearson_delta` and `mae` fall too. **The
pre-registered expectation held**: VCC measured colsum at `jac` −0.002…−0.004, `overlap_at_N`
is in the same Wilcoxon family, and the mechanism predicted little or nothing there. It is
worse than nothing.

### `exact_colsum` is strongly confirmed — *within* emission

| | emission | + `exact_colsum` | gain |
|---|---|---|---|
| `pearson_delta` | 0.1611 | 0.2478 | **+0.0867** |
| `mae` | 0.0490 | 0.0295 | **−40%** |
| `overlap_at_N` | 0.0079 | 0.0314 | **4x** |

That is exactly the F-emit2 prediction: it removes the scatter the sampler introduced, so it
recovers most — **but not all** — of what emission cost. Net against the published arm it is
still **−0.094 `pearson_delta`**. The two operations are a pair, and the pair loses.

### ⚠️ THIS TESTED POISSON EMISSION, NOT NB — the theta estimate collapsed

```
[emit] split 0: 2488 model-visible controls -> theta median 10000.000 p10 10000.000 p90 10000.000
```

10,000 is the estimator's **cap**, returned when `Var - mu <= 0` i.e. no evidence of
over-dispersion. It was returned for essentially every gene, so `NB(mu, theta→∞)` degenerated
to Poisson and the run tested Poisson emission.

The cause is a space error in the estimator, and it is mine: the control rows are **library-
normalised** (log1p at 2000), and method-of-moments on `expm1` of normalised data does not
recover a count dispersion — normalisation to a fixed library size compresses the variance and
drives `Var` below `mu`. The emitted zero fraction confirms the regime is off: **0.8767 against
the target's 0.5170.**

**So the honest status of the headline result:** emission as tested is Poisson at the wrong
dispersion, and it hurts. A fair NB test needs a real per-gene dispersion, and there are two
sources:
1. a model-learned `log_theta` from an NB-trained checkpoint — the principled one, and L01-L09
   are training now (note L01, clean NB, validates at 0.2614 against smooth_l1's 0.3558, so it
   is not a strong checkpoint to borrow from);
2. method of moments on **raw counts**, before library normalisation — which the frozen panel
   does not carry, so it would have to come from the source h5ads.

⚠️ Either way, the `overlap_at_N` result is unlikely to reverse: the gap is 8x, the mechanism
argues against it, and a better theta changes the dispersion, not the fact that a Wilcoxon test
on emitted counts sees a different distribution than one on a mean profile. **Recommend against
pursuing emission for `overlap_at_N` unless a model-learned theta is available cheaply.**

## ⛔ F-loss1 — DCOS ran INERT: I added the consumer without the gate that loads its inputs

**Caught by the data, not by reading the code.** Three DCOS arms at `delta_cos_weight`
0.25 / 0.5 / 1.0 produced **bit-identical validation curves** — 0.2893, 0.3460, 0.2712 for
their last three epochs, to four decimals. A term that changes nothing at three different
weights is not applied.

### Two causes, one mine and one pre-existing

**1. The loader gate (mine).** `pipeline.py:1678` only loads the group-mean table when
something declares it needs it:

```python
if (getattr(config, "target_pb_k", 1) != 1) or float(getattr(config, "delta_mag_weight", 0.0)) > 0:
    ... load_group_means(...)
```

`delta_cos_weight` is not in that condition. So `gmeans` stayed `None`, my guard
(`st.get("gmeans") is not None`) failed exactly as written, and `dcos_loss` stayed at its
initialised `0.0` — **silently**, with no warning, for every DCOS arm. Fixed by adding
`delta_cos_weight` to the gate, with a comment saying every consumer must be listed there.

**2. The table does not exist in this repo (pre-existing).**
`cache/pert_means` is absent, and `scripts/pert_group_means.py` — the builder that
`stream_v2.py:789` and `config.py:164` both name — **was never ported**. It exists only in the
research tree (128 lines, keyed to that tree's file layout and encoder axis).

⚠️ **Consequence beyond DCOS: `delta_mag_weight` has never been usable in this repo either.**
Setting it >0 would open the gate and then raise `FileNotFoundError: no group-mean tables in
cache/pert_means`. It is dead config, and it has been dead since the port.

### What this does and does not invalidate

| | status |
|---|---|
| L01-L09 (the NB axis) | ✅ **VALID** — they set `delta_cos_weight=0.0`, so the inert term was correctly zero |
| L10-L15 (the DCOS axis) | ⛔ **VOID as DCOS arms** — they are plain `smooth_l1` / `nb_smooth_l1` runs with a no-op term. L14/L15 remain valid as `nb_smooth_l1` replicates |

**DCOS needs the builder ported before it can be swept at all.** That is a bounded task —
adapt the 128-line script to our three frozen h5ads and the 6,983-gene axis — but it is a
prerequisite, not a detail.

### ⚠️ The lesson, which is the same one as C28

The audit's conclusion was that every check in place was about the CONTENT of inputs and none
about their GROUPING or their COVARIATES. This is the third variant: **a term whose inputs were
never loaded**. The guard was correct, the weight was plumbed end-to-end, the config declared
it, the schema registered it, the launcher passed it — and the term was still zero, because one
`or` clause in an unrelated gate did not mention it.

**A weight that is set and has no effect should be an ERROR, not a silent zero.** The fix
should not stop at the gate: a term with a non-zero weight whose inputs are missing must
refuse to train.

## ⭐ F-emit3 — NB count emission implemented at library norm 2000; `exact_colsum` now measurable

**Question.** F-emit2 showed `exact_colsum` is a no-op without a count emitter. Build the
emitter, and check the coupling actually holds.

**What was built** — `occam.vcell.panel.celleval_v1.nb_emit`, opt-in via
`score_arm(..., emit={...})`. `emit=None` is the default and reproduces every number to date.

The space chain, which is the part that goes wrong:

| step | operation |
|---|---|
| in | `pred_log1p`, log1p at **2000** on the FULL-panel denominator |
| 1 | `mu_2000 = expm1(pred_log1p)` — expected counts per 2000-library |
| 2 | `mu_cell = mu_2000 * lib / 2000` — at `lib=2000` this is a no-op by construction |
| 3 | `counts ~ NB(mu_cell, theta)` as Gamma-Poisson, so `Var = mu + mu²/theta` |
| 4 | optional `exact_colsum`, pinned **per perturbation** |
| 5 | **`log1p(counts / rowsum * 2000)` — back to the eval space** |

⚠️ **Step 5 is not optional.** Skipping it hands cell-eval a count matrix while the target is
log1p, and every metric silently becomes a comparison between two different spaces — the
failure this whole campaign has been chasing.

⚠️ **Emission applies to the PERTURBED ROWS ONLY.** The control block is real half-B cells that
already carry real dispersion; re-emitting them would replace measurements with samples and
destroy what makes `pearson_delta` a paired comparison.

### Measured on a synthetic panel (40 cells, 2 perturbations, 300 genes, theta 2.0)

| | mean-only (current) | NB emit | **NB + `exact_colsum`** |
|---|---|---|---|
| zero fraction | **0.0000** | 0.5174 | 0.5234 |
| within-group std (count space) | 0.147 | **4.84** | 4.73 |
| mean rel-error vs the input profile | — | 0.0935 | **0.0060** |
| rowsum in eval space | 2000.2 | 2000.0 | 2000.0 |

**Two things this shows.**

1. **The structural defect is real and large.** A mean-only prediction has a zero fraction of
   **0.0000** against a real-data ~0.52, and essentially no within-group dispersion. The
   Wilcoxon test behind `overlap_at_N` reads exactly that dispersion, so it has been calling
   significance on a distribution no real data has. This is the same pseudobulk-variance
   artifact documented for XCellv2.
2. **`exact_colsum` is worth 15× on mean fidelity — but only now.** It cuts the mean
   reconstruction error 0.0935 → **0.0060**, and that gain exists *because* step 3 introduced
   scatter for it to remove. Run on the current mean-only path it would have had a residual of
   zero and done nothing, exactly as F-emit2 predicted.

### ⚠️ What this does NOT yet establish

- **No `overlap_at_N` number yet.** The measurements above are distributional, on synthetic
  input. Whether restoring dispersion actually raises `overlap_at_N` on the real panel is the
  open question, and VCC's evidence cuts against it (`jac` −0.002…−0.004 from colsum, and
  `overlap_at_N` is in the same Wilcoxon family).
- **`theta` has to come from somewhere.** Per-gene `log_theta` from an NB-trained checkpoint is
  the principled source; a scalar broadcast is a weaker, different claim and `nb_emit` prints
  which was used. **Our best arm (`s313`) is `smooth_l1` and carries no dispersion tensor** —
  so the NB arms now training (L01–L09) are also the prerequisite for a principled `theta`.
- This is **engineering, not modelling**: the emitter adds no information and leaves the mean
  unchanged in expectation. It changes the shape of the predicted distribution so the scorer's
  assumptions hold. Label it that way next to any number.

## ⛔ F-emit2 — `exact_colsum` CANNOT go first: it is a no-op without count emission

**Question.** Port `exact_colsum` and run it on `s313` to push `overlap_at_N`.

**Finding: there is nothing for it to pin.** It is not a matter of effort — the operation is
undefined in Occam's current emission path.

### What `exact_colsum` actually does

`occam/scripts/build_submission.py:142` takes **sampled integer counts** and forces each gene's
per-perturbation column sum onto `round(sum_i mu[i,g])`, scattering the residual over random
cells:

```python
def exact_colsum(counts, mu, n_per, rng):
    tgt = np.rint(mu[s0:e].sum(0)).astype(np.int64)
    cur = blk.sum(0, dtype=np.int64)
    d = tgt - cur   # the SAMPLING scatter, and the only thing removed
```

Its own docstring says what the variance is: *"Sampling each cell independently leaves the
realised column sum scattered around the profile we actually predicted, and that scatter is pure
self-inflicted loss."* **The scatter it removes is created by the sampler.**

### Occam's panel emits no counts

`celleval_v1.py:64` writes the prediction straight through:

```python
ad.AnnData(X=pred_X, ...).write_h5ad(ppath)
```

`pred_X` is the continuous log1p profile — **one predicted row per drawn control cell, no
sampler, no `mu`, no integers.** `calibrate.py`'s header already records why: *"the second,
count-space scorer and its emission machinery are gone."*

So the group-sum pseudobulk the metric reads is **already exactly the sum of the model's own
predictions**. There is no sampling residual `d`, so `exact_colsum` would compute `d ≈ 0` on
rounded values and, if forced to round first, would *add* quantisation error rather than remove
scatter.

### Corrected order — the two items are coupled, not independent

`exact_colsum` is a **variance reduction for a count emitter**, so it can only be evaluated
downstream of one:

1. **NB emission** (with the right library norm — 2,000 for our eval space) is the
   prerequisite, not a parallel option.
2. **`exact_colsum` on top of it** then removes the scatter that emission introduces, and the
   comparison that means something is `NB emission` vs `NB emission + exact_colsum`.

⚠️ And a caution on the target metric: VCC measured `exact_colsum` at **+0.038–0.040 on `nmae`
and consistent `pds` gains**, because those read the group-sum pseudobulk — but **`jac` moved
−0.002…−0.004**. `overlap_at_N` is Wilcoxon/DE-based, in the same family as `jac`. The
mechanism predicts it helps `pearson_delta`/`mae` and does **little or nothing for
`overlap_at_N`**, which is the metric this was meant to push.

### What proceeds unblocked

The loss work is independent of all of this and is being implemented and launched now: a clean
adjmap NB arm, a joint SmoothL1+NB loss, DCOS integration, and the loss-weight sweep with the
optimisation knobs frozen at `s313`.

## ⭐ F-emit1 — inventory: what NB / exact-colsum / calibration evidence ACTUALLY exists

**Question.** Before proposing emission work: has an **adjacency-map** Occam ever been trained
with NB, has it ever been finally scored, and does `exact_colsum` exist here at all?

**Answers, all verified from the runs and the code rather than from memory.**

### 1. No adjmap model has EVER been trained with NB

| | `pert_model_type` | `loss_type` | evidence |
|---|---|---|---|
| F6 loss grid (`a01`–`a25`, 24 runs) | **`gnn`** | 12 × `nb`, 12 × `smooth_l1` | `gnn_config.pert_model_type` read from `a01`'s checkpoint |
| adjmap sweep (`s000`–`s4xx`, incl. our best) | **`adjacency_map`** | `smooth_l1` **held fixed** | `rhj_adjmap_sweep.sbatch:105` |

So **every NB number we have is GNN-era**, and the adjmap sweep pins `loss_type=smooth_l1` as a
standing decision. **NB is untested on adjmap.**

### 2. The NB arms were never finally scored — a launch bug, not a result

All 13 `a*` arms submitted to the split-half panel have **empty split dirs**. Scoring array
`18034289` (2026-09-17) failed every task with:

```
ValueError: calibrate='quantile' needs the calibrated prediction.
            Fit it with occam.vcell.panel.calibrate.fit_control_quantile_map
```

**1 of 27 `a*` arms has any `agg_results.csv`.** So there is **no test-panel number for any NB
model**, GNN or otherwise. The only NB evidence is F6's *validation* metric.

### 3. What F6 actually said (validation metric, GNN, 12 vs 12)

| loss | n | best | worst | mean |
|---|---|---|---|---|
| `smooth_l1` | 12 | 0.3543 | 0.3276 | **0.3400** |
| `nb` | 12 | 0.3019 | 0.2765 | **0.2919** |

12/12 with no range overlap — but ⚠️ **F6 itself records the confound**: at
`cosine_loss_weight 0.5` the `smooth_l1` arms are genuinely half-cosine while the NB arms are
effectively pure-NB, because the two recon terms differ by an order of magnitude. F6 calls it
*"a loss-configuration comparison, not a clean likelihood comparison"*. And per F7 the
within-arm epoch-to-epoch swing is **±0.20**, against a between-arm span of 0.012.

### 4. `exact_colsum` does not exist in this repo

`grep -rln exact_colsum src/ scripts/ docs/` → **nothing**. It lives only in the research tree.
`calibrate.py`'s own header says why: *"the second, count-space scorer and its emission
machinery are gone -- Occam scores with cell-eval v1 only"*. **NB emission went with it.** The
only emission-side knob here is the quantile calibration.

### 5. Calibration, measured on our best arm (`s313`, 5 splits)

| | `pearson_delta` | `mae` ↓ | `overlap_at_N` |
|---|---|---|---|
| `model_nocal` | **0.3415** | **0.0210** | **0.0642** |
| `model_cal` | 0.2732 | 0.0890 | 0.0243 |

Calibration costs **−0.068 `pearson_delta`**, **4.2× worse `mae`**, and **−62% `overlap_at_N`**.
It is not close, and `overlap_at_N` — the metric we would be trying to push — is what it damages
most.


## What the protocol is for

**Overall objective.** Predict a perturbation's effect in a context where that perturbation
was never measured, from a single control cell. The headline is how far the model is from
**experimental reproducibility** on the 611 held-out HepG2 perturbations — not how far it is
from an arbitrary floor.

**The question.** Scored against the same half-A reference, how does the model fed half-B
control cells compare with the half-B perturbed cells themselves?

**What is varied.** Which cells land in half A and which in half B — 5 independent,
batch-stratified splits. **Held fixed:** the 611-perturbation cohort, the 6,983-gene axis,
the checkpoint, the arm set, and the per-perturbation row counts.

**What each outcome means.** A positive gap (reference better) is expected and its size is
the headline. A **negative** gap is a red flag, not a triumph: at median `m` = 22 both the
reference and the experimental prediction are noisy pseudobulks while a low-variance
predictor is attenuated on one side only. The count-stratified gap must **shrink** as `m`
grows; one that does not is noise-dominated whichever sign it has.

Panel definition and its caveats: `data/benchmarks/rhj_fewshot/splithalf/CARD.md`.

---

## F1 — the zero-signal floor, measured

**Status: COMPLETE for both anchors.** Timing probe, job `18021925`, split 0, 2026-09-16.
Neither arm is a model: `ctrlmean` predicts the half-B control mean tiled across every cell,
`ctrlcell` predicts the drawn half-B control cell itself. Both carry no biology by
construction, so whatever they score is what the metric reads from the emission alone.

Configuration: split 0 of 5, cohort 611 perturbations (`86c91f938024b2a2…`), 18,575 rows +
2,488 half-B controls in the prediction file, 18,575 rows + 2,488 half-A controls in the
target, 6,983 genes (`ef64f2ba7ff7816e…`), `--allow-discrete`, `--batch-size 2000`,
`--num-threads 16`, cell-eval v1 profile `full`.

| metric | `ctrlmean` | `ctrlcell` | what it says |
|---|---|---|---|
| `pearson_delta` | **0.0006** | **−0.0003** | both are a readable zero — see the correction below |
| `discrimination_score_l2` | **0.5040** | **0.5059** | chance is 0.5. Neither anchor tells perturbations apart |
| `discrimination_score_l1` / `cosine` | 0.5015 / 0.4970 | 0.5046 / — | same |
| `overlap_at_N` | 0.0072 | **0.0** | see the bracketing note below |
| `de_direction_match` | 0.4883 | 0.4837 | chance is 0.5 |
| `de_spearman_sig` | 0.4048 | *undefined* | no significant gene to rank |
| `de_sig_genes_recall` | 0.3080 | **0.0** | |
| `de_nsig_counts_pred` *(real 111.6)* | **3,026.8** | **0.0** | a 27.1× over-call vs zero calls |
| `mae` / `mae_delta` *(lower better)* | 0.0231 / 0.0231 | 0.0296 / 0.0295 | |
| `mse` / `mse_delta` *(lower better)* | 0.00158 / 0.00157 | 0.00229 / 0.00228 | |
| `pr_auc` / `roc_auc` | 0.0120 / 0.3902 | 0.0188 / 0.5051 | |
| `clustering_agreement` | 0.0011 | 0.0017 | |
| `pearson_edistance` | −0.1202 | 0.6300 | |

**Read every model number against these two rows.** On the H1 panel the `ctrlmean` anchor
*beat* every model arm on `pearson_delta`; here both anchors sit at zero and at chance on
discrimination, which is what a floor should look like and is evidence the protocol's
B-control substitution is doing its job.

### ⭐ The two anchors bracket the DE metrics from opposite sides

This was not designed in and it is the most useful thing the probe produced. Both arms carry
zero signal, and they fail the DE tests in **opposite** directions:

| | within-group variance | genes called significant | why |
|---|---|---|---|
| `ctrlmean` | **zero** (every row identical) | **3,026.8** of 6,983 | Wilcoxon reads near-zero variance as overwhelming evidence — the pseudobulk-variance artifact |
| `ctrlcell` | **natural** (real single cells) | **0.0** | the "perturbed" rows are drawn from the same pool as the controls, so nothing is significant |
| real | natural | 111.6 | |

So a model arm's `de_nsig_counts_pred` has a floor on each side: **0** (real cell-to-cell
variance, no signal) and **3,026.8** (no variance, no signal). A model calling ~112 is
calibrated; one calling thousands is emitting under-dispersed cells, not finding biology.
`overlap_at_N` inherits this: `ctrlcell` scores exactly **0.0** because N on its own side is
zero, which makes it the honest floor for the DE metrics, while `ctrlmean`'s 0.0072 is a
small number obtained by calling almost everything.

`ctrlcell` is therefore the load-bearing floor for **both** families — `pearson_delta`
(−0.0003, pure control-sampling noise at `m` cells) and the DE metrics (exactly 0).

### ⚠️ Correction to a pre-registered prediction

The plan predicted `ctrlmean`'s `pearson_delta` would be **NaN** — "undefined by
construction", because with half-B controls on the prediction side the `ctrlmean` prediction
*is* the half-B control mean, so its delta is exactly zero and `pearsonr` is undefined.

**Measured: 0.000623, not NaN.** The delta is zero only in exact arithmetic; in float32 the
prediction pseudobulk (mean over `m` identical rows) and the control pseudobulk (mean over
2,488 rows) round differently, leaving ~1e-8 of noise, and the correlation of that noise
against the real delta is 0.0006 — a readable zero rather than a hole in the table.

This is the better outcome and nothing needs changing: the NaN branch in
`occam.vcell.panel.protocol` still fires if the value ever is non-finite, and `ctrlcell`
remains the load-bearing floor because its delta is real control-sampling noise at `m`
cells. But the claim "it would silently vanish from the report" was wrong, and the
prediction is recorded here as not-held.

---

## F2 — protocol cost, measured

**Status: COMPLETE.** Job `18021925`, split 0, `COMPLETED` in **30 min 47 s**, **MaxRSS
14.16 GB** on 16 CPU threads, no GPU (the probe's prediction is the identity).

| | measured |
|---|---|
| `ctrlmean` — **cold**, computes and caches the real-side DE | 1,169.4 s (19.5 min) |
| `ctrlcell` — **warm**, reads the real-side DE from the cache | 658.5 s (11.0 min) |
| **real-side DE**, shared by every arm of a split (`cold − warm`) | **510.9 s (8.5 min)** |
| **pred-side DE + all 28 metrics**, per additional arm | **658.5 s (11.0 min)** |

So one split of 5 arms is `510.9 + 5 × 658.5` = **63 min**, and the 5-split array is
**5.3 h of compute in ~1.1 h of wall clock** (plus model prediction and calibration, which
the probe did not measure — budget 1.5 h per array task). The `rhj_splithalf.sbatch` limit
of 12 h is margin, not estimate. 96 GB is generous against 14.16 GB, but a real run adds
the calibration fit over 20,000 control cells.

**The prior estimate was 2.0–2.8 h per split; the measurement is 1.05 h** — the extrapolation
across a 5× change in panel size was pessimistic by ~2.2×. The probe existed because the same
class of extrapolation put a 3 h wall limit on a 3.2 h training job the same day (see
`runs/aborted_3h_limit/WHY.md`). It was worth 31 minutes.

### ⚠️ The probe's own printed decomposition was inverted

The job's stdout reported `real-side DE −510.9 s`. A negative time: the record from that job
predates `scored_order`, so the script fell back to the **requested** arm order
(`ctrlcell,ctrlmean`) rather than the order `build_arms` actually scores in (anchors first,
`ctrlmean` before `ctrlcell`). The numbers above are the corrected assignment, read against
the job log's own `cached real-side DE ->` / `reusing real-side DE for target` lines, which
show unambiguously that `ctrlmean` paid for the DE. `run_split` now records `scored_order`
and the probe refuses to print a negative decomposition.

---

## F3 — the baseline arm needs 420 GB, not 128 GB (measured by killing it)

**Status: FAILED, cause identified, resubmitted.** Job `18022390_26`, arm `a26`
(`run_baselines: true`): **OUT_OF_MEMORY**, exit `0:125`, **MaxRSS 117.19 GB** against
`ReqMem 128 GB`, at 29 min 58 s. No numbers were produced.

Cause is arithmetic, not a leak. `_run_baselines` builds an **in-memory** training dataset
to compute `MeanBaseline`'s per-perturbation deltas — the one path in the vcell stage that
is not streaming — and it builds it on the **encoder's 44,728-gene axis**, not the
benchmark's 6,983:

| axis | expression | bool gene mask |
|---|---|---|
| 6,983 (the benchmark's) | 16.2 GB | 4.1 GB |
| **44,728 (the encoder's)** | **103.8 GB** | 26.0 GB |

580,212 perturbed cells × 44,728 × 4 bytes = 103.8 GB, which accounts for the observed
117.2 GB with overhead. The other 26 arms never reach this path: they stream, and their
steady-state RSS is ~2 GB.

**Resubmitted at `--mem=420G` on vco**, which has an 8 TB group cap. The launcher's
`#SBATCH --mem=128G` is correct for the 26 streaming arms and must not be raised for them —
420 GB × 27 would be 11 TB and would not schedule. The baseline arm is submitted separately
for exactly this reason, and its header now says so.

⚠️ This does not affect the benchmark's own `meanpert` arm. That one is built by
`scripts/build_splithalf_panel.py` from the 1,657 HepG2 training perturbations on the 6,983
axis, precomputed to a single (6,983,) vector in 1.5 min at ~2 GB. The two are different
baselines anyway — see F4.

---

## F4 — `MeanBaseline` on this benchmark is a cross-cell-line TRANSFER baseline

**Status: established by reading the code, not yet measured.** Worth stating before its
number arrives, because the number will be misread otherwise.

`baselines.py:120` decides whether a perturbation was "seen" by checking **every training
cell type at once**:

```python
has_specific = any(pname in self.pert_deltas.get(train_ct, {})
                   for train_ct in self.pert_deltas)
```

All 103 HepG2 validation perturbations **are** trained in Jurkat and RPE1 — that is the
benchmark's design. So `has_specific` is true for all of them and the prediction becomes
*HepG2 control mean + the count-weighted mean of that perturbation's Jurkat and RPE1
deltas*.

That is **not** the global mean-training-delta baseline. It is the non-learned version of
exactly what the model is asked to do, which makes it the sharpest comparator on the whole
benchmark: if the model cannot beat "copy this perturbation's delta from the other two cell
lines", the FiLM machinery is buying nothing. Its log line `MeanBaseline: N specific + M
global` is the check — expect ~103 specific, 0 global.

The global mean-training-delta baseline is the split-half protocol's **`meanpert`** arm, on
the 611 test perturbations, cell-eval scored. The two are not comparable: different
perturbation sets, different metric, different scale.

---

## ⛔ THE COMPARISON SET — decision 2026-09-17

**Every comparison between arms on this benchmark is made on these four cell-eval metrics:**

| metric | direction | chance |
|---|---|---|
| `pearson_delta` | higher better | 0 |
| `mae` | **lower better** | — |
| `overlap_at_N` | higher better | 0 |
| `discrimination_score_l1` | higher better | **0.5** |

Note **l1**, not l2. `discrimination_score_l2` and the other 24 metrics stay in the tables
for completeness but are not used for ranking. Wired into `scripts/paired_report.py` and
`scripts/write_baseline_results.py`, so regenerating either puts these four first.

`mae` is control-**independent** (error against the real pseudobulk); `mae_delta` is
control-dependent. Both are carried, because "MAE" was named without disambiguating and
they answer different questions.

### The table, on the comparison set

All 5 splits unless noted. `±` is `ddof=1` across splits.

| arm | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` *(chance .5)* | n |
|---|---|---|---|---|---|
| `ctrlcell` | −0.0018 ± 0.0031 | 0.0296 | 0.0000 | 0.5038 | 5 |
| `ctrlmean` | 0.0016 ± 0.0028 | 0.0231 | 0.0081 | 0.5014 | 5 |
| `meanpert` | 0.1324 ± 0.0030 | 0.0290 | 0.0108 | 0.5037 | 5 |
| `transfer` | 0.2323 ± 0.0021 | 0.0299 | 0.0351 | **0.6404** | 5 |
| **`experimental`** | **0.3722 ± 0.0010** | 0.0257 | **0.1942** | **0.7821** | 5 |
| **adjmap `s129` uncalibrated** | **0.3348 ± 0.0032** | **0.0210** | 0.0618 | 0.6479 | 5 |
| adjmap `s129` calibrated | 0.2883 ± 0.0008 | 0.0856 | 0.0250 | **0.7528** | 5 |
| gnn `a00` calibrated | 0.0020 | 0.0298 | 0.0000 | 0.5051 | **1** |

Reading on the agreed four:

* **`pearson_delta`** — adjmap uncalibrated 0.3348 beats `transfer` 0.2323 by **+0.1025** and
  sits **0.0374** under `experimental`. Calibration costs 0.0465.
* **`mae`** — adjmap uncalibrated **0.0210 is the best value in the table**, better than
  `experimental`'s 0.0257 and every baseline. Calibration makes it **4× worse** (0.0856),
  the single largest calibration effect anywhere here.
* **`overlap_at_N`** — adjmap 0.0618 beats `transfer` 0.0351 but is **3.1× below**
  `experimental` 0.1942. This is the weakest axis for the model and the one the DE over-call
  (3,263 predicted significant against 111.6 real) directly explains.
* **`discrimination_score_l1`** — adjmap uncalibrated 0.6479 is just above `transfer`
  0.6404; **calibrated 0.7528 is far better** and approaches `experimental` 0.7821.
  `meanpert` sits at chance (0.5037), exactly as its construction requires.

**The two settings win different metrics.** Uncalibrated wins `pearson_delta` and `mae`;
calibrated wins `discrimination_score_l1`. `overlap_at_N` prefers uncalibrated. So on the
agreed set it is **3–1 for uncalibrated**, and no single setting dominates — which is itself
the finding, and the reason both are now scored.

---

## ⭐⭐ F12 — the adjacency-map model BEATS the transfer baseline, and calibration HURTS

**Status: 5 of 5 splits, one checkpoint (`s129`), both calibration settings.** Job
`18035791`, all 10 tasks COMPLETED. This answers the question F11 raised.

`s129` = the best adjacency-map arm so far by validation (**0.3676**), config
`lr 3e-4, decoder_lr 1e-3, cosine 0.25, data_perts_only 0, film_mode full, fh256, fmf1`,
batch 2048, `adjacency_map`.

| metric | **`model_nocal`** | **`model_cal`** | `transfer` | `experimental` | `ctrlcell` |
|---|---|---|---|---|---|
| `pearson_delta` | **0.3348 ± 0.0032** | 0.2883 ± 0.0008 | 0.2332 | **0.3722** | −0.0018 |
| `overlap_at_N` | 0.0618 ± 0.0010 | 0.0250 ± 0.0008 | 0.0351 | 0.1942 | 0.0000 |
| `discrimination_score_l2` | 0.6912 ± 0.0011 | **0.7792 ± 0.0029** | 0.7023 | 0.8398 | 0.5046 |
| `de_direction_match` | 0.6806 ± 0.0045 | 0.6978 ± 0.0060 | 0.6891 | 0.7563 | 0.4832 |
| `de_nsig_counts_pred` *(real 111.6)* | 3,263 ± 11 | 4,239 ± 26 | 238.9 | 118.7 | 0.0 |
| `mae_delta` ↓ | **0.0211 ± 0.0000** | 0.0856 ± 0.0016 | 0.0299 | 0.0258 | 0.0296 |

### The model is not at the floor. It beats the bar.

**`model_nocal` `pearson_delta` 0.3348 against `transfer`'s 0.2332 — a gap of +0.1016**, and
it reaches **90% of replicate agreement** (0.3348 / 0.3722). The paired gap to
`experimental` is only **0.0374**, against a split-to-split std of 0.003.

On `mae_delta` it is **better than the experimental arm** (0.0211 vs 0.0258) — expected, and
not a triumph: a low-variance predictor is attenuated on one side only, which is exactly the
asymmetry F9 warns about. Read the `pearson_delta` and discrimination rows for the claim.

**So F11 was not a statement about the approach.** The identity-learning reading of `a00` was
specific to that run, not to the method.

### ⚠️ Calibration HURTS here, and the two metric families disagree about it

| | `pearson_delta` | `discrimination_score_l2` | `mae_delta` ↓ |
|---|---|---|---|
| uncalibrated | **0.3348** | 0.6912 | **0.0211** |
| calibrated | 0.2883 | **0.7792** | 0.0856 |
| effect of calibration | **−0.0465** | **+0.0880** | **+0.0645 (worse)** |

Control-fit quantile calibration **costs 0.047 `pearson_delta` and quadruples `mae_delta`,
while adding 0.088 to discrimination**. The project's standing note is that calibration is
worth roughly **+0.37** `pearson_delta` on the H1 panel and that "an uncalibrated comparison
of decoder heads measures almost nothing". **That does not transfer to this benchmark** — the
sign is reversed on the headline metric.

The mechanism is consistent with the split-half design: calibration maps the prediction's
per-gene marginals onto the *model-visible controls'* marginals. Here the model-visible
controls are the **half-B** controls, which are also the prediction file's own control
pseudobulk — so the mapping pulls the prediction toward the very reference its delta is
measured against, shrinking the delta. It preserves *ranking* between perturbations (hence
discrimination improves) while compressing magnitude (hence `pearson_delta` and `mae_delta`
degrade).

⚠️ **Consequence for the 27-arm GNN sweep now running:** it was launched with
`--calibrate quantile`, i.e. **calibrated only**. On this evidence that is the worse of the
two settings for the headline metric, so those numbers will understate every arm. They are
still internally comparable, but `model_nocal` should be added.

### What it does and does not establish

* **n = 1 checkpoint.** Five splits agree to ±0.003, so the *measurement* is solid; whether
  0.3348 is typical of adjacency-map arms needs more of them.
* **`a00` vs `s129` differ in more than the perturbation model** — also batch size, `lr`,
  `decoder_lr`, `cosine_loss_weight`, `film_hidden_dim` and `film_mean_free`. The 134 GNN
  tasks in flight will show whether *every* GNN arm sits at the floor; only then is
  "GNN vs adjacency map" the right axis to name.
* **Both model arms over-call DE by 29–38×** (3,263 and 4,239 against 111.6 real), far worse
  than `transfer`'s 239. The emitted cells are under-dispersed, which is the
  pseudobulk-variance artefact this repo documents, and it is why `overlap_at_N` stays at
  0.06 while `pearson_delta` reaches 0.33.

---

## ⛔⛔ F11 — the first model arm scores AT THE FLOOR. val 0.337 → cell-eval 0.002

**Status: n = 1 (arm `a00`, split 0). The remaining 134 tasks are running.** Flagged
immediately because if it holds across arms it is the finding of the whole campaign.

Arm `a00` (`film_mode=full`, `fh0`, `fmf0`, `smooth_l1`), `best_model.pth`, epoch 29,
**validation `pearson_delta` 0.3368**. Scored by cell-eval on split 0 as `model_cal`:

| metric | `model_cal` | `ctrlcell` *(the zero-signal floor)* | `transfer` *(the bar)* | `experimental` |
|---|---|---|---|---|
| `pearson_delta` | **0.0020** | −0.0018 | 0.2332 | 0.3722 |
| `overlap_at_N` | **0.0** | 0.0 | 0.0351 | 0.1942 |
| `discrimination_score_l2` | **0.5063** | 0.5046 | 0.7023 | 0.8398 |
| `de_direction_match` | **0.4839** | 0.4832 | 0.6891 | 0.7563 |
| `de_nsig_counts_pred` | **0.0** | 0.0 | 238.9 | 118.7 |
| `mae_delta` ↓ | **0.0297** | 0.0296 | 0.0299 | 0.0258 |

**The model is indistinguishable from `ctrlcell` on every one of the six metrics.** Chance
is 0.5 for discrimination and direction match, and it sits at 0.506 / 0.484. It calls
**zero** genes differentially expressed, exactly like the arm whose prediction *is* the
control cell.

### What that most likely means

`mae_delta` 0.0297 against `ctrlcell`'s 0.0296, and `de_nsig_counts_pred` 0.0 against 0.0,
say the emitted profile is **almost exactly the control cell it was fed**. The model appears
to have learned close to the identity — it reproduces its input and adds no
perturbation-specific delta that survives to the scored axis.

Quantile calibration cannot rescue that and does not: it maps the prediction's per-gene
marginals onto the controls' marginals, which is a no-op for an output already at the
control distribution.

### Why validation said 0.337 and the benchmark says 0.002

The two are not the same estimand, and the difference is structural rather than a bug:

1. **The validation metric shares one control pseudobulk between the real and the predicted
   side** (it falls back to group-mean with `stream_v2`, F7 / `metric_granularity`). Control
   noise therefore *cancels* in both deltas, and any systematic control-relative offset the
   model does reproduce counts as signal. The benchmark deliberately breaks that: half-A
   controls on the real side, half-B on the predicted side, so nothing cancels.
2. **Validation is 103 perturbations, group means. The benchmark is 611, single cells**, with
   `m ≈ 22` per half.
3. **Validation was read at the peak of a ±0.20 oscillation** (F7), so 0.3368 is the high
   end of a swing, not a converged value.

This project documented the risk in advance — selection and benchmark metrics ranking models
*oppositely* — and this is that risk realised at full strength. **`best_metric` is not a
proxy for this benchmark and must not be used to choose between arms.**

### ⚠️ SUPERSEDED IN SCOPE BY F12

F12 scores the best adjacency-map checkpoint at `pearson_delta` **0.3348**, above the
`transfer` bar. So F11 is a fact about `a00` and not about the approach, and the
identity-learning reading applies to that run only. Whether it applies to every GNN arm is
what the 134 tasks in flight will say.

### What would change this reading

* **134 more tasks.** If every arm sits at the floor, the conclusion is about the approach.
  If some arms separate, then `best_metric` is merely a *bad* proxy rather than an
  anti-correlated one.
* The adjacency-map sweep is a different architecture and a different optimisation regime;
  F11 says nothing about it yet.
* ⚠️ It does **not** implicate the benchmark. The five baselines span 0.000 → 0.372 with
  split-to-split std 0.001–0.002 (F9), so the instrument resolves signal perfectly well
  when signal is present.

---

## ⭐ F9 — the reference ladder, measured by cell-eval on 4 of 5 splits

**Status: 4 of 5 splits, all five reference arms** (split 4 rerunning after an OOM; split
3's `ctrlmean` lost to a transient pdex shared-memory race and is being retried). 611
perturbations, 18,575 rows per arm, 2,488 half-B controls, `ddof=1` across splits.

| arm | `pearson_delta` | `overlap_at_N` | `discrimination_score_l2` | `de_direction_match` | `mae_delta` ↓ |
|---|---|---|---|---|---|
| `ctrlmean` | 0.0010 ± 0.0010 | 0.0077 | 0.5040 | 0.4684 | 0.0231 |
| `ctrlcell` | **−0.0005 ± 0.0013** | **0.0000** | 0.5050 | 0.4832 | 0.0296 |
| `meanpert` | 0.1335 ± 0.0020 | 0.0109 | **0.5060** | 0.5776 | 0.0290 |
| **`transfer`** | **0.2332 ± 0.0010** | 0.0351 | 0.7023 | 0.6891 | 0.0299 |
| **`experimental`** | **0.3720 ± 0.0011** | 0.1949 | 0.8392 | 0.7563 | 0.0258 |

Chance is **0.5** for `discrimination_score_l2` and `de_direction_match`.

### The split-to-split std is 0.001–0.002, and that is the headline

Every arm's spread across independent splits is **one to two thousandths**. The paired
design is doing exactly what the reviewer asked it to: split choice contributes almost
nothing, so a model-vs-reference gap larger than about **0.005** will be resolvable.

Put that beside F7: the *validation* metric swings **±0.20** between consecutive epochs,
while this benchmark's split-to-split noise is **±0.001**. The benchmark is ~200× the
cleaner measurement, and it is the one that should decide anything.

### The ladder reads correctly, top to bottom

* **`ctrlcell` is a true zero** — `pearson_delta` −0.0005, `overlap_at_N` exactly 0,
  discrimination at chance. The floor is a floor.
* **`meanpert` gains delta accuracy but cannot discriminate**: 0.1335 `pearson_delta`, yet
  `discrimination_score_l2` **0.5060**, i.e. chance. Exactly right — it predicts the *same*
  delta for every perturbation, so it captures the average response direction and nothing
  perturbation-specific. This is the arm that makes "the model learned something
  per-perturbation" a testable claim.
* **`transfer` genuinely discriminates**: 0.7023, well above chance, at 0.2332
  `pearson_delta` — **63% of the way to replicate agreement**. This is the bar.
* **`experimental` 0.3720** is replicate agreement at median m = 22, not an upper bound.

### ⚠️ This is a different picture from F5, and both are right

On the *validation* side, `MeanBaseline` scored **0.5567** and beat every model arm. Here
`transfer` scores **0.2332**, below `experimental`. The two are not the same quantity:
`MeanBaseline` compares group means on 103 perturbations with the internal metric, while
`transfer` adds a cross-line delta to a *single* drawn control cell and is scored by
cell-eval on 611 with the A/B control asymmetry. **The F5 headline does not carry over
numerically** — what carries over is the question: can the model beat a plain cross-line
average? On this benchmark that bar is **0.2332 ± 0.0010**.

---

## F5b — the first full-eval launch failed: 135 tasks on one tuple, 1 split on memory

**Status: diagnosed and fixed; relaunched.** Recorded because the failure mode is worth
knowing and because the reference numbers are partial until the rerun lands.

| job | outcome |
|---|---|
| `18033268_[0-3]` reference, splits 0–3 | **COMPLETED**, 38–56 min each |
| `18033268_4` reference, split 4 | **OUT_OF_MEMORY** at 64 GB, 27 min |
| `18033290_[0-134]` model arm, 27 ckpts × 5 splits | **all 135 FAILED** |

### The 135

`VirtualCellTrainer.from_vcell_checkpoint` returns a **tuple** `(trainer, ckpt)`.
`eval_splithalf.load_model` bound it to one name, so every task loaded 898 MB of weights,
printed `Loaded vcell checkpoint: epoch=29`, and then died on
`AttributeError: 'tuple' object has no attribute 'current_epoch'`.

**`scripts/eval_panel.py:102` had the identical bug**, unnoticed — the H1 panel scorer was
broken the same way.

`scripts/check_call_arity.py` does cover `scripts/`, and still could not catch this: it
resolves module-level `def`s, and this is a **classmethod on an imported class**. Extending
it to cross-module attribute calls is real work, so the guard is instead
`tests/test_checkpoint_loader_arity.py`, which asserts the return really is a 2-tuple and
that every call site unpacks two names. Both sites fixed; the loader now also returns the
already-loaded checkpoint so `assert_checkpoint_matches` stops reading 898 MB twice.

### The one OOM

Splits 0–3 fit in 64 GB; split 4 did not, at 27 min. Every split has identical row counts
(18,575 per half, 2,488 controls), so 64 GB was simply marginal and split 4 lost the
coin-flip on peak allocation. Rerun at **128 GB**. The launcher's default is raised to
match, since a 27-minute OOM costs more than the memory does.

---

## Reading order for whoever picks this up

1. `data/benchmarks/rhj_fewshot/splithalf/CARD.md` — the frozen panel, and the two things
   the protocol cannot express.
2. `scripts/eval_splithalf.py --help` — the probe and the real run.
3. `scripts/paired_report.py --help` — the only valid summary, and the three arithmetic
   traps it closes.
4. This file — every measured number, with its configuration and its anchor.
