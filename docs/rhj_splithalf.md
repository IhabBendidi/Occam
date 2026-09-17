# `rhj_fewshot` split-half paired evaluation — findings

**This file owns every number that comes out of the split-half protocol.** Written as it is
read, partial results labelled partial, superseded numbers kept with the correction beside
them.

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
