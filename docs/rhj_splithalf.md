# `rhj_fewshot` split-half paired evaluation — findings

> **Chronological findings log.** Current numbers: `docs/STATUS.md`. Arms, checkpoints and
> scores that this log mentions but that are not in STATUS were moved to `archive/` on
> 2026-09-23 (`archive/MANIFEST.tsv` gives each one's new path); treat them as history.

**This file owns every number that comes out of the split-half protocol.** Written as it is
read, partial results labelled partial, superseded numbers kept with the correction beside
them.
## ⭐⭐⭐ F-sched3 — **COMPLETE. `eta_min` 3e-4 IS a real direction, and it was outside the W grid's band. `Y06` (Q02, 150 ep, `eta_min` 3e-4) = 0.4043 is the first arm above 0.40 and the new campaign best.**

**Objective.** Best model on SEEN perturbations in an UNSEEN context: 611 held-out HepG2
perturbations predicted from one half-B control cell, split-half paired `pearson_delta`,
cell-eval 0.6.0, uncalibrated.

**Question.** F-sched2 found the epochs ladder positive in 8/8 contrasts but decelerating
(30→60 +0.0116…+0.0264, 60→90 +0.0025…+0.0083). Is it live past 90?

**Varied:** `epochs` ∈ {120, 150} × `eta_min` ∈ {1e-5, 1e-4, **3e-4**}. **Held fixed:** the Q02
loss composition exactly as `W23` (smooth_l1 + dcos 0.5 + delta_mag 0.01363 + latent MSE
0.0044), batch 2048, lr 1e-3, decoder_lr 1e-3, seed 42, arm s313 otherwise. Grid and
pre-registration: `scripts/grids/rhj_epochs_ext.tsv`. Jobs `18252962` (train, 6/6 COMPLETED,
1:38–2:06 wall) → `18263829` (eval, 60/60 tasks COMPLETED, no pdex failures).

**Anchors:** `experimental` 0.3722 (mae 0.0257, ovl@N 0.1942, disc_l1 0.7821) · Q02 (30 ep)
0.3783 · `E3` (60 ep) 0.3933 · `W23` (90 ep, 1e-5) 0.3959 · `W22` (90 ep, 1e-6) 0.3985.

### The result: **0.4043**, and every metric except `mae` at its campaign maximum

| arm | epochs | `eta_min` | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|---|
| **`Y06`** | **150** | **3e-4** | 5/5 | **0.4043** ± 0.0019 | 0.0215 | **0.0946** | **0.7288** |
| `Y03` | 120 | **3e-4** | 5/5 | **0.4015** ± 0.0018 | 0.0217 | 0.0919 | 0.7247 |
| `Y05` | 150 | 1e-4 | 5/5 | **0.3983** ± 0.0023 | 0.0217 | 0.0931 | 0.7223 |
| `Y01` | 120 | 1e-5 | 5/5 | **0.3978** ± 0.0028 | 0.0217 | 0.0889 | 0.7141 |
| `Y04` | 150 | 1e-5 | 5/5 | **0.3963** ± 0.0021 | 0.0217 | 0.0913 | 0.7176 |
| `Y02` | 120 | 1e-4 | 5/5 | **0.3960** ± 0.0021 | 0.0223 | 0.0904 | 0.7221 |

`Y06` is +0.0058 on `pearson_delta`, +0.0059 on `ovl@N` and +0.0098 on `disc_l1` over `W22`,
the previous best. `ovl@N` 0.0946 and `disc_l1` 0.7288 are campaign maxima. `mae` 0.0215 ties
the best Q02-family value.

### ⚠️ CORRECTION TO F-sched2 — `eta_min` is a direction after all, ABOVE the band it tested

F-sched2 concluded that `eta_min` "is not a tunable direction — it is ±0.003 of seed-like
variance", from a grid spanning **1e-7 … 1e-4**. That conclusion is correct *inside that band*
and wrong as a general claim, because the band was too low. Extend it one step to **3e-4** and
the ordering becomes consistent at BOTH lengths:

| epochs | 1e-5 | 1e-4 | **3e-4** |
|---|---|---|---|
| 120 | 0.3978 | 0.3960 | **0.4015** |
| 150 | 0.3963 | 0.3983 | **0.4043** |

| contrast | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | sign |
|---|---|---|---|---|---|
| `Y03` (120, 3e-4) − `W22` (90, 1e-6) | **+0.0030** | +0.0002 | **+0.0052** | **+0.0090** | 4/5 |
| `Y04` (150, 1e-5) − `W22` | −0.0021 | +0.0002 | +0.0045 | +0.0020 | 1/5 |

**3e-4 wins at 120 and at 150; the two lower floors swap places between the two lengths.** So
the *within-band* non-monotonicity F-sched2 measured was real noise, and there is ALSO a real
gradient at the top of the range that the band could not see. With lr = 1e-3, 3e-4 means the
learning rate never falls below ~0.3x peak — a materially different regime from the 1e-7…1e-4
arms, all of which decay to ≤0.1x.

### ⚠️ The plateau depends on the FLOOR — so "90 is the plateau" was wrong

| `eta_min` | 90 → 120 | 120 → 150 |
|---|---|---|
| 1e-5 | +0.0019 (4/5) | **−0.0014 (0/5)** ← over-trains |
| 1e-4 | +0.0003 (3/5) | **+0.0023 (4/5)** |
| **3e-4** | — (not run at 90) | **+0.0028 (5/5)** ← still climbing |

At a LOW floor the schedule saturates and then reverses; at a HIGH floor it is still gaining at
150 epochs. This is a genuine interaction, and it retracts the partial reading recorded earlier
in this finding (*"the ladder has essentially flattened… 90 is approximately the plateau"*),
which was computed from the 1e-5 and 1e-4 columns before the 3e-4 column existed. **Neither
knob can be tuned alone** — pre-registered branch 3 of the W grid, which the W grid itself
could not exhibit.

✅ **`ovl@N` and `disc_l1` rise monotonically with length at every floor** (+0.0024…+0.0027 and
+0.0002…+0.0041 for 120→150), even where `pearson_delta` falls. `mae` improves slightly with
length at the two higher floors (−0.0002…−0.0005, 0/5 positive = 5/5 better) and worsens at
1e-5. The metrics do not agree on the optimum; `Y06` happens to lead on three of four.

### What this opens (NOT run — proposal only)

The best arm sits at the EDGE of both axes: the longest schedule and the highest floor tested,
with both margins still positive (+0.0028 for 120→150 at 3e-4, +0.0058 over `W22`). The
untested directions are **180+ epochs at 3e-4** and **a floor above 3e-4** (5e-4 is 0.5x lr;
1e-3 would be no decay at all and is not a candidate). ⚠️ Both are endpoint optima, which
F-loss3 flagged as the signature of a knob whose range is wrong rather than a finished tuning.

## ⛔⛔⛔ F-disp3 — **CORRECTION TO F-disp2's MECHANISM CLAIM. The terms hit a PROXY, not the dispersion. Measured properly, dispersion got WORSE — `radial_spread` 8× worse — and the FDR-significant gene count did not improve at all.**

**Asked directly ("for the runs that didn't collapse, did we improve the number of
FDR-significant genes?"), so measured rather than recalled. The answer is no, and the reason
invalidates a claim I made two findings ago.**

### The FDR-significant gene count did not move

Eval-side diagnostic (`disp_*` in each arm's `metrics.json`, mean over splits, `model_nocal`):

| arm | term | splits | std_pred/std_real | nsig pred | nsig real | nsig RATIO |
|---|---|---|---|---|---|---|
| `W23` | control | — | **0.026** (F-disp1) | ~3,475 | 113 | **30.8** |
| `Z01` | `dcos_conc` 1× | 5 | 0.0446 | 3,847 | 114 | 33.9 |
| `Z02` | `dcos_conc` 5× | 5 | 0.0415 | 4,116 | 114 | 36.2 |
| `Z03` | `radial_spread` 1× | 4 | 0.0381 | 3,447 | 113 | 30.4 |
| `Z04` | `radial_spread` 5× | 5 | 0.0680 | 3,432 | 114 | 30.2 |
| `Z05` | `gene_var` 1× | 5 | 0.0337 | 3,466 | 114 | 30.5 |
| `Z06` | `gene_var` 5× | 5 | 0.0413 | 3,505 | 114 | 30.8 |

**Every arm is still ~30× over-calling significance and still at 3–7% of the real dispersion.**
The best arm on this axis (`Z04`, 0.0680) is 2.6× the control's 0.026 and still 15× short of 1.0.

### ⚠️ THE TRAINING DIAGNOSTIC WAS AN ARTEFACT — and I reported it as verification

F-disp2 states that the training logs "confirm they DID" hold dispersion at 99–102% of real,
and calls it "MECHANISM VERIFIED". **That was wrong.** Re-measured on the trained checkpoints
with real single control cells (`scripts/loss_scale_disp.py`, 128 perturbations × 8 controls),
on the SAME perturbations the terms trained on:

| checkpoint | training log claimed | radial std_pred/std_real | per-gene std_pred/std_real |
|---|---|---|---|
| `W23` control | — | **0.2225** | **0.1079** |
| `Z05` `gene_var` 1× | per-gene **0.9923** | 0.1548 | **0.0961** ↓ |
| `Z03` `radial_spread` 1× | radial **0.9944** | **0.0269** ↓↓ | **0.0265** ↓↓ |

**`radial_spread` made the collapse 8× WORSE than the control** (0.2225 → 0.0269) while its own
training diagnostic read 0.9944.

**The mechanism of the artefact.** Both moment terms measure spread around a **detached EMA of
the model's own per-group mean** (`disp_ema_momentum` 0.99, ~90 updates per group per epoch).
That centre LAGS the model, so the residual `(x − EMA)²/σ²_real` is a mixture of two things:
genuine cell-to-cell spread *and* the model's temporal DRIFT since the EMA was formed.
Minimising `(m2 − 1)²` can be satisfied by the drift term alone — and the optimiser took that
route, collapsing the per-cell dispersion toward zero while keeping the proxy pinned at 1.0.
The design note in `config.py` argued the EMA centre was needed so that a *bias* could not
satisfy the term. It closed that hole and opened a different one.

### What F-disp2's numbers do and do not mean

* ✅ **The COSTS stand.** `Z01`–`Z06` really do lose 0.029–0.209 `pearson_delta` against
  `W23`, 0/5 splits positive, monotone in weight. Those are measurements of these three terms
  as implemented.
* ⛔ **The CONCLUSION does not.** "The distributional hypothesis is falsified" is retracted:
  the arms never restored dispersion, so the experiment did not test what it was built to test.
  What is established is narrower and still useful — **these three proxy formulations cost
  0.03–0.21 and buy nothing**, and one of them actively deepens the collapse.
* ⛔ **"The mechanism worked and the hypothesis was wrong" is retracted.** It was the reverse:
  the hypothesis is untested and the mechanism did not work.

### What would actually test it

The failure is the lagging centre, so the fix is a centre that cannot absorb drift: a **second
forward pass on a different control cell for the same perturbation**, giving an unbiased
within-group variance from the pair `(p₁ − p₂)²/2` with no running statistic at all. That is
the design F-disp1's implementation note rejected as "expensive" (one extra forward on a
subsample) in favour of the EMA. The extra forward is the honest price of measuring dispersion;
the EMA was cheaper and measured something else.

⚠️ **The lesson is about the instrumentation, not the biology.** F-loss1's rule was "a term
that is ON and unmeasured is how DCOS burned three GPUs", and the response here was to
instrument the terms — correctly. But the instrument was **derived from the term's own
objective**, so it could only ever report success. An instrument must be independent of the
thing it validates: the eval-side `disp_*` numbers, which are, disagreed with the training-side
ones by 25× and were sitting in `metrics.json` the whole time.

## ⛔⛔ F-disp2 — ⚠️ **ITS MECHANISM CLAIM IS RETRACTED BY F-disp3 ABOVE — read that first.** **COMPLETE. ALL THREE dispersion terms HURT, 6/6 arms, 0/5 splits positive on every metric.** Matching real per-cell dispersion costs −0.029 to −0.209 `pearson_delta`. The hypothesis is falsified.

**Objective.** Best model on SEEN perturbations in an UNSEEN context: 611 held-out HepG2
perturbations from one half-B control cell, split-half paired, cell-eval 0.6.0, uncalibrated.

**Question.** F-disp1 measured that our per-cell predictions carry 2.6% of the real
within-perturbation dispersion and that all four scored metrics are pseudobulk. Three terms
were built to restore that dispersion, and the training logs confirm they DID (99–102% of real).
Does restoring it change any of the four scores?

**Varied:** which dispersion term is on (one per arm) × weight (1× / 5× parity, both measured
against Q02's own DCOS contribution). **Held fixed:** Q02's loss composition, 90 epochs,
`eta_min` 1e-5, seed 42 — so `W23` is the exactly-matched paired control. Grid:
`scripts/grids/rhj_dispersion.tsv`. Jobs `18266534` (train, 6/6 clean) → `18296324` (eval, `%6`).

### Every arm, against the matched control

| arm | term | weight | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|---|
| **`W23`** | — (control) | — | 5/5 | **0.3959** ± 0.0031 | **0.0216** | **0.0887** | **0.7142** |
| `Z03` | `radial_spread` | 0.512 (1×) | 4/5 ⚠️ | 0.3677 ± 0.0036 | 0.0227 | 0.0717 | 0.6721 |
| `Z05` | `gene_var` | 0.473 (1×) | 5/5 | 0.3673 ± 0.0017 | 0.0223 | 0.0582 | 0.6798 |
| `Z04` | `radial_spread` | 2.56 (5×) | 5/5 | 0.3557 ± 0.0021 | 0.0225 | 0.0695 | 0.6548 |
| `Z06` | `gene_var` | 2.37 (5×) | 4/5 ⚠️ | 0.3453 ± 0.0019 | 0.0242 | 0.0589 | 0.6652 |
| `Z01` | `dcos_conc` | 13.09 (1×) | 5/5 | **0.2024** ± 0.0016 | 0.0318 | 0.0185 | 0.5989 |
| `Z02` | `dcos_conc` | 65.47 (5×) | 5/5 | **0.1868** ± 0.0019 | 0.0374 | 0.0173 | 0.6032 |

### Paired contrasts — 24 of 24 metric-contrasts negative

| arm | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | sign |
|---|---|---|---|---|---|
| `Z03` (radial 1×) | **−0.0287** | +0.0012 | −0.0170 | −0.0420 | 0/4 |
| `Z05` (gene_var 1×) | **−0.0286** | +0.0007 | −0.0305 | −0.0343 | 0/5 |
| `Z04` (radial 5×) | **−0.0402** | +0.0009 | −0.0192 | −0.0594 | 0/5 |
| `Z06` (gene_var 5×) | **−0.0511** | +0.0026 | −0.0298 | −0.0488 | 0/4 |
| `Z01` (dcos_conc 1×) | **−0.1935** | +0.0102 | −0.0702 | −0.1153 | 0/5 |
| `Z02` (dcos_conc 5×) | **−0.2090** | +0.0159 | −0.0714 | −0.1110 | 0/5 |

**Every arm is worse on every metric, on every split** — `mae` is worse in 5/5 (or 4/4) of each
arm's splits, which for an ERROR metric means the sign is the same as the others. And **every
term is monotone in its weight**: 5× is worse than 1× for all three (radial −0.0402 vs −0.0287,
gene_var −0.0511 vs −0.0286, dcos_conc −0.2090 vs −0.1935). A dose-response on three
independent terms is not an optimisation accident; it is the effect.

### What this settles

1. **The distributional hypothesis is FALSIFIED on this benchmark.** F-disp1's metric audit
   predicted "no direct score to win through the distribution channel" and flagged that the
   terms *could* cost `ovl@N`. The measurement is worse than that prediction: they cost every
   metric, including the two that are pure means.
2. **The 2.6% dispersion collapse is not a defect to repair — it is the model's optimal answer
   to a pseudobulk objective.** Given one control cell, the score-maximising prediction is the
   conditional MEAN; per-cell spread around it is variance the metrics only punish. Forcing the
   spread back in is forcing the model off its own optimum, and the size of the loss tracks how
   hard you force it.
3. **`dcos_conc` is in a different class from the other two, exactly as the design predicted.**
   The two moment-matching terms are mean-preserving and cost ~0.03; `dcos_conc` matches a
   FIRST moment against an absolute target and costs ~0.20, dropping the model BELOW the
   `transfer` baseline (0.2323) with `disc_l1` 0.599 a third of the way to the 0.5 chance floor.
   F-disp1 already measured why: a real cell's Δ from the control mean is ~4× the perturbation
   and mostly the cell's own identity, so instructing the model to reproduce that cosine is
   instructing it to inject cell-identity noise into its delta.

✅ **The mechanism worked; the hypothesis was wrong.** Training logs: `Z01`/`Z02` held cos at
0.2181/0.2028 against the real 0.1992; `Z03`/`Z04` held the radial ratio at 0.9944/0.9970;
`Z05`/`Z06` at 0.9923/1.0179 per-gene — against `W23`'s trained collapse of 0.075 / 0.22 /
0.108. There is no version of this result that is a plumbing artefact, which is the whole
reason the terms were instrumented before they were swept.

⚠️ **What this does NOT establish.** These are all four PSEUDOBULK metrics on ONE benchmark. A
single-cell-level metric (per-cell Wasserstein, energy distance, a discriminator) would score
these arms differently, and nothing here says dispersion is worthless for a task that reads it.
It says dispersion is not a lever for `rhj_fewshot`'s four metrics, and that a dispersion term
run against a pseudobulk objective is a direct cost.

✅ **`radial_spread` and `gene_var` at 1× are the cheapest way to buy realism if it is ever
wanted**: −0.029 `pearson_delta` for a dispersion ratio of ~1.0 instead of ~0.1. That is the
price list, not a recommendation.

## ⛔ F-sched4 — **THE SCHEDULE IS SATURATED. 180 epochs buys nothing and a floor above 3e-4 costs at 150. `Y06` (150 ep, `eta_min` 3e-4) = 0.4043 stands as the campaign best.**

**Objective.** Best model on SEEN perturbations in an UNSEEN context: 611 held-out HepG2
perturbations from one half-B control cell, split-half paired, cell-eval 0.6.0, uncalibrated.

**Question.** F-sched3's `Y06` sat at the EDGE of both axes with both margins positive, which
F-loss3 flags as a range problem rather than a finished tuning. So: 180 epochs, and `eta_min`
5e-4 / 7e-4 (0.5x and 0.7x of lr; 1e-3 would be no decay at all).

**Varied:** epochs (150, 180) × `eta_min` (3e-4, 5e-4, 7e-4), minus (150, 3e-4) = `Y06`.
**Held fixed:** the Q02 loss composition, seed 42, arm s313. Jobs `18306151` (train, 5/5
COMPLETED, 2:02–2:24, **zero nan-guard events even at 7e-4**) → `18319056` (eval).

⚠️ 4 of 5 splits on four arms, 2 of 5 on `AA05` — 7 eval tasks lost to the pdex shm race
(F-infra1). The means may move by ~0.001; the conclusion does not depend on that.

| arm | epochs | `eta_min` | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|---|
| **`Y06`** (the bar) | 150 | **3e-4** | 5/5 | **0.4043** | 0.0215 | 0.0946 | 0.7288 |
| `AA03` | **180** | 7e-4 | 4/5 ⚠️ | **0.4045** ± 0.0034 | 0.0217 | 0.0957 | **0.7303** |
| `AA02` | **180** | 5e-4 | 4/5 ⚠️ | 0.4034 ± 0.0018 | 0.0219 | 0.0959 | 0.7290 |
| `AA01` | **180** | 3e-4 | 4/5 ⚠️ | 0.4016 ± 0.0032 | **0.0214** | **0.0960** | 0.7241 |
| `AA05` | 150 | 7e-4 | 2/5 ⚠️ | 0.3930 | 0.0221 | 0.0850 | 0.7162 |
| `AA04` | 150 | 5e-4 | 4/5 ⚠️ | 0.3923 ± 0.0024 | 0.0218 | 0.0897 | 0.7230 |

### Paired against `Y06`

| contrast | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | sign |
|---|---|---|---|---|---|
| `AA03` (180, 7e-4) | **+0.0003** | +0.0002 | +0.0011 | +0.0015 | 2/4 |
| `AA02` (180, 5e-4) | −0.0007 | +0.0003 | +0.0013 | +0.0003 | 1/4 |
| `AA01` (180, 3e-4) | −0.0026 | −0.0001 | +0.0014 | −0.0047 | 0/4 |
| `AA04` (150, 5e-4) | **−0.0118** | +0.0002 | −0.0049 | −0.0057 | 0/4 |
| `AA05` (150, 7e-4) | **−0.0099** | +0.0006 | −0.0080 | −0.0116 | 0/2 |

### Two separate findings, and they resolve the endpoint question

1. **180 epochs is flat.** All three 180-epoch arms land at 0.4016–0.4045 against `Y06`'s
   0.4043, and the best of them is +0.0003 at 2/4 splits — noise. The epochs axis, which paid
   +0.0116…+0.0264 (30→60), +0.0025…+0.0083 (60→90) and +0.0003…+0.0028 (90→150), pays
   **nothing** from 150→180. **That is the plateau, and it is the first time in this campaign
   that lengthening the schedule has not helped.**
2. **A floor above 3e-4 COSTS at 150 epochs**: `AA04` −0.0118 and `AA05` −0.0099, 0/4 and 0/2,
   and they lose on `ovl@N` and `disc_l1` too. At 180 epochs the floor stops mattering
   (0.4016 vs 0.4045 across three decades). So `eta_min` 3e-4 was not the start of a trend —
   it was the optimum, and F-sched3's edge was the peak.

✅ **`Y06` stands: Q02, 150 epochs, `eta_min` 3e-4, 0.4043 / mae 0.0215 / ovl@N 0.0946 /
disc_l1 0.7288.** The schedule axis is closed; `ovl@N` is the only metric still drifting up
(+0.0011…+0.0014 in every 180-epoch arm), and it is the metric furthest from `experimental`
(0.1942), so it is the one to chase — but not with more epochs.

⚠️ **Validation was wrong about the ranking, for the fifth grid running.** `AA03` had the
highest validation score in the campaign (0.4613) and `Y06` had 0.4390 — 0.022 lower — yet they
are tied on test. `AA01`/`AA02` at 0.4511/0.4504 are BELOW `Y06` on test. F-ckpt4's rule holds:
validation selects the epoch (Spearman +0.99) and misranks the arm.

## ⭐⭐⭐ F-disp1 — **our per-cell predictions carry 2.6% of the real within-perturbation dispersion** (568/568 perturbations), which inflates FDR-significant genes **30.8×** — but all four scored metrics are PSEUDOBULK, so a distributional loss has no direct score to win

**Objective.** Predict a perturbation's effect in a cellular context where that perturbation was
never measured, **from a single control cell**. Headline: split-half paired `pearson_delta`,
cell-eval 0.6.0, uncalibrated, 611 held-out HepG2 perturbations on the 6,983-gene axis.

**Question this answers.** Asked directly: does a *distributional* loss make sense — on the
whole prediction, or specifically on the `delta_mag` and `dcos` terms? Before designing one,
two things had to be measured rather than assumed: **(a)** is our predicted single-cell
distribution actually wrong, and **(b)** do any of the four scored metrics read a distribution
at all. Both were measured today, on `W23` (the campaign-best arm), split 00.

**What is varied / held fixed.** Nothing is varied — this is a read-only diagnostic on
artefacts already on disk (`runs/rhj_splithalf/W23__*/split_00/`). 21,063 rows × 6,983 genes,
568 perturbations with ≥10 cells, median 24 cells per perturbation.

### (a) The dispersion is collapsed — by a factor of ~38

| quantity | pred | real | ratio |
|---|---|---|---|
| mean per-gene within-perturbation std | **0.0026** | **0.1009** | **0.026** |
| median per-perturbation median(std_pred / std_real) | — | — | **0.021** |
| mean ‖Δ‖ of the group MEAN | 2.5027 | 2.9230 | 0.856 |
| perturbations where pred std < real std | **568 / 568 = 100%** | | |
| control rows (injected real — sanity check) | 0.1040 | 0.1039 | 1.001 |

The control row check is the control for this measurement: those rows are injected real cells,
and they match to 3 decimals, so the collapse is a property of the **predicted perturbed rows**
and not of how the file is read.

**Interpretation: the model is very nearly a per-perturbation lookup table.** A predicted cell
varies by 2.6% of the biological spread, which means the prediction is almost independent of
*which control cell it was given* — on a benchmark whose stated task is "predict from a single
control cell". The mean effect is, by contrast, roughly right (‖Δ‖ ratio 0.856).

⚠️ **This is what the Q02 loss composition asks for, not an accident.** `dcos` 0.5 pushes every
cell's Δ toward one shared target direction and `delta_mag` 0.01363 pulls every cell's ‖Δ‖ to
the group value, while `smooth_l1` — the only per-cell term — is ~2% of the total in this
direction-dominated regime (F-loss5). Parallel directions + equal magnitudes = a constant Δ.
The 2.6% is the loss working as specified.

### (b) But every scored metric is a pseudobulk metric — read from cell-eval 0.6.0 source

| metric | what it actually consumes | sensitive to dispersion? |
|---|---|---|
| `pearson_delta` | `_generic_evaluation(..., use_delta=True)` → `iter_bulk_arrays` group means | **no** |
| `mae` | same path, `use_delta=False` → group means | **no** |
| `discrimination_score_l1` | `perturbation_effect` on bulk arrays, ranked distances | **no** |
| `overlap_at_N` | `compute_overlap`: `k_eff` = **real**'s significant-gene count; both lists ranked by \|fold_change\| of the means; pred is FDR-filtered before ranking | **only through the pred FDR filter** |

`cell_eval/_types/_de.py:241` — `k_eff = real_genes.size if not k else k` — so N comes from the
ground truth alone; our inflated significance does not set N. And `get_top_genes` ranks by
`ABS_FOLD_CHANGE`, a function of the group means.

### The dispersion collapse DOES distort the DE statistics — measured

| | pred | real | ratio |
|---|---|---|---|
| FDR-significant genes per perturbation (α = 0.05) | **3,475** (49.8% of 6,983) | **113** (1.6%), median 17 | **30.8×** |
| perturbations where pred calls more than real | **100%** | | |

This is the pseudobulk-variance pitfall documented in `CLAUDE.md`, now measured on our own best
arm: near-zero within-group variance makes the Wilcoxon test call half the transcriptome
significant. ⚠️ **But it is currently NOT binding on the score**, because the pred FDR filter
only has to admit N genes and N is at most 113 on average; a 3,475-gene pool always covers it.

### What this means for a distributional loss — the pre-registration, before any arm is run

1. **A distributional term cannot raise any of the four scores *through the distribution
   channel*.** All four read group means. Anyone expecting +`overlap_at_N` from fixing
   dispersion is expecting it from the one channel that is provably not binding.
2. ⚠️ **It could LOWER `overlap_at_N`.** Restoring real dispersion shrinks pred's significant
   pool from ~3,475 toward ~113. For perturbations where real has many significant genes, the
   pred pool could become *smaller* than N, and the missing ranks score as misses.
3. ⚠️ **It cannot lower `mae`/`pearson_delta` either** — those are mean quantities and a
   mean-preserving dispersion term is neutral on them **in expectation**; any movement is a
   second-order optimisation effect, which is exactly the thing that has to be measured rather
   than argued.
4. ✅ **The legitimate reason to run it is as a REGULARISER on the mean path**: forcing the
   output to depend on the input control cell forces the model to use per-cell covariates it
   currently ignores. That is the same class of intervention as VICReg — and F-vicreg3 is the
   warning: VICReg's +0.0060 at 30 epochs went to −0.0016 at 90. **Screen a dispersion term at
   90 epochs directly**, not at 30, or the result will not be believable.
5. ✅ **Report the mechanism, not only the score.** Any such arm must report
   `std_pred / std_real` and the significant-gene ratio alongside the four metrics, so a null
   score can be distinguished from a term that never took effect.

### Implementation note — a set-to-set loss needs a sampler change; moment-matching does not

Measured on `nadig_2024_hepg2.h5ad` (145,473 rows, 2,394 perturbations): rows are **not** sorted
by perturbation (mean run length 1.0), and in a 2,048-row batch there are ~1,169 distinct
perturbations with **median count 1** (max 68, the controls). So MMD / Sinkhorn / any set-to-set
objective has no set to work with under the current sampler and would need grouped batches
(G perturbations × K cells), which is a real change to `stream_v2`'s buffer assembly AND a
confound — grouped batches change the gradient even with the old loss, so such an arm needs its
own control.

**The cheap route needs no sampler change.** With one cell per group per batch, a per-gene
second-moment match is still an unbiased stochastic estimator: penalise
`|(pred_i − μ̂_real)² − σ̂²_real|` per gene, where `μ̂_real` and `σ̂²_real` come from precomputed
per-group tables. The group-MEAN table already exists and is already loaded for `delta_mag`
(7,179 groups × 44,728 genes, 1.28 GB); a variance table mirrors it exactly. The batch even
already carries `group_id` (`stream_v2.py:193`), so the loss side is a few lines. Radial-only
variants are cheaper still: matching the spread of ‖Δ‖ or the concentration of Δ directions
needs **one scalar per group**, not a gene-wide table.

### The real dispersion targets, now measured for every group (`scripts/pert_group_dispersion.py`)

7,179 groups, 627,869 cells, in the model's own log1p space on the 44,728-gene encoder axis,
around the same references the existing terms use (`gvar` around the group mean; radius and
cosine around the LINE CONTROL MEAN, which is what `delta_mag` and `dcos` are defined against).

| source | groups (n≥2) | per-gene std | ‖cell Δ‖ | cos(cell Δ, group Δ) |
|---|---|---|---|---|
| hepg2 | 2,393 / 2,393 | 0.0963 | 11.529 ± 2.285 | **0.2536** ± 0.1165 |
| jurkat | 2,392 / 2,393 | 0.1258 | 13.420 ± 2.753 | **0.1906** ± 0.0939 |
| rpe1 | 2,393 / 2,393 | 0.1104 | 12.505 ± 2.022 | **0.2927** ± 0.1396 |
| pooled (as loaded) | 7,178 / 7,179 | — | 12.485 ± 2.353 | **0.2456** ± 0.1167 |

⚠️ **A real cell's own Δ aligns with its group's mean Δ at only ~0.25.** ‖cell Δ‖ ≈ 12.5 against
a group effect of ≈ 2.9, so a single cell's deviation from the control mean is ~4× the
perturbation itself and mostly its own identity. That is the correct target for a
concentration term — not 1.

### ⚠️ CORRECTION — the model is UNDER-aligned per cell, not over-concentrated

Measured on `W23`'s checkpoint with **real single control cells** as input (128 perturbations ×
8 control cells, 9,612 present genes, `scripts/loss_scale_disp.py`):

| quantity | pred | real | ratio |
|---|---|---|---|
| cos(pred Δ, group Δ), per cell | **0.075** | **0.238** | 0.32 |
| radial std_pred / std_real | — | — | **0.2225** |
| per-gene std_pred / std_real | — | — | **0.1079** |
| `smooth_l1` per element | 0.035431 | | |
| `dcos` = 1 − cos(Δpred, Δtrue) | 0.924964 | | |

**When F-disp1 was proposed, the claim was that `dcos` OVER-concentrates the predicted
directions and that a concentration term would relax it. That is wrong, and this measurement
is why:** per-cell, our alignment is 0.075 against a real 0.238. `dcos` drives the cosine
toward 1 but the model never gets near it, because the per-cell Δ it has to produce is
dominated by the control cell's own identity, which the model must reconstruct through the
decoder and does imperfectly. A concentration term therefore pushes alignment **UP** on this
base and partially duplicates `dcos` rather than opposing it. It is still the right quantity —
calibrating to the real value instead of maximising — but the sign of the current error is the
opposite of what was predicted.

⚠️ **Two dispersion ratios, two spaces.** The scored panel gives 0.026 (0.0026 / 0.1009, over
6,983 panel genes in scoring space); this per-cell measurement gives 0.108 (over 9,612 present
genes of the hepg2 file on the encoder axis, with 8 control inputs). **The panel number is the
one the metrics see**; both say collapsed, and the 4× between them is a space-and-gene-set
difference, not a disagreement about the model.

### ✅ MECHANISM VERIFIED AT THE REAL SCHEDULE — all 6 Z arms trained clean, every one holding dispersion at parity

Job `18266534`, 6/6 COMPLETED (1:17–1:21 wall each), **0 nan-guard events in any arm**. Final
epoch diagnostics, against `W23`'s trained collapse (cos 0.075 · radial 0.2225 · per-gene
0.1079) and the real targets:

| arm | term | weight | final diagnostic | real target | val `pearson_delta` |
|---|---|---|---|---|---|
| `Z01` | `dcos_conc` | 13.09 (1×) | cos **0.2181** | 0.1992 | 0.2227 |
| `Z02` | `dcos_conc` | 65.47 (5×) | cos **0.2028** | 0.1992 | 0.1966 |
| `Z03` | `radial_spread` | 0.512 (1×) | radial ratio **0.9944** | 1.0 | 0.4092 |
| `Z04` | `radial_spread` | 2.56 (5×) | radial ratio **0.9970** | 1.0 | 0.3948 |
| `Z05` | `gene_var` | 0.473 (1×) | per-gene ratio **0.9923** | 1.0 | 0.4279 |
| `Z06` | `gene_var` | 2.37 (5×) | per-gene ratio **1.0179** | 1.0 | 0.3956 |

**Every term did exactly what it was built to do.** The dispersion the base model destroys over
90 epochs (down to 2.6–22% of real) is held at 99–102% of real in the matching arm. So whatever
the test scores say, this is a measurement of the HYPOTHESIS, not of a term that failed to fire
— the F-loss1 trap is closed.

⚠️ **The two `dcos_conc` arms have catastrophic VALIDATION scores** (0.2227 and 0.1966 against
`W23`'s 0.4419 and the other four arms' 0.39–0.43). Validation misranks *arms* (F-ckpt4), so
this is not a test result — but a 0.22 is not a ranking wobble, it is a different regime.
Expect `Z01`/`Z02` to score far below the control: calibrating per-cell alignment to 0.199
appears to fight the pseudobulk objective directly. ⚠️ Pre-registered before the test scores
land, so it can be checked rather than rationalised afterwards.

✅ Cost: 1:17–1:21 per arm, **the same as the control** (`W23` 1:15) — the three terms add no
measurable wall time even with the 1.28 GB gene-variance table and its EMA.

**Eval:** job `18296324`, 30 tasks (6 arms × 5 splits), **`%6`**, nocal only, grid
`configs/rhj_Z_eval.tsv`. `%6` is the measured-safe concurrency: the Y eval ran 60/60 tasks
clean at `%6`, while F-infra1 measured ~50% marginal failures at `%12` and ~30% at `%20`. The
training throttle was moot — all six arms finished before it mattered.

### ✅ PLUMBING VERIFIED — the 2-epoch smoke test passes, and the terms visibly work

All three weights on at 1× parity (`dcos_conc` 13.09, `radial_spread` 0.512, `gene_var` 0.473),
2 epochs, Q02 config, `SMOKE4`:

| | epoch 1 | epoch 2 | target |
|---|---|---|---|
| batches completed | **312 / 312** | **312 / 312** | no drops |
| nan-guard lines | **none** | **none** | none |
| cos(pred Δ, group Δ) | 0.1377 | **0.1865** | REAL 0.1995 |
| radial std_pred/std_real | 0.9984 | 0.9983 | 1.0 |
| per-gene std_pred/std_real | 0.7467 | **0.9735** | 1.0 |
| loss | 1.1924 | 0.6964 | — |
| grad_norm (max) | 3.92 (695) | 0.96 (18.8) | — |

**The terms are not inert and they move the quantity they target**: the direction concentration
climbs 0.138 → 0.187 toward the real 0.200 in one epoch, and the per-gene ratio 0.747 → 0.974
toward 1.0. Epoch-1 loss is 1.19 against the baseline Q02's 0.689, i.e. the three terms add
~0.50 at parity, as designed. ⚠️ `grad_norm` max spikes to 695 on epoch 1 and settles to 18.8
on epoch 2; clipping is on, but this is the thing to watch if a 90-epoch arm diverges.

### Three bugs, and the one piece of reasoning that cost two extra iterations

| attempt | symptom | cause | fix |
|---|---|---|---|
| 1 | non-finite **LOSS**, batch 77 | `0 * NaN` from the 1-of-7,179 group with `n_cells < 2`, masked per ELEMENT | `nan_to_num` the gene table; row-select |
| 2 | non-finite **GRADIENT**, finite loss, batch 77 | `.norm()`'s backward is `0/0` for a zero-delta row | clamped sum-of-squares norm; guarded division |
| 3 | same, batch 77 | the per-group **SCALAR** tables are still NaN for that group, and the mask is applied AFTER the arithmetic | slice rows before every gather; `nan_to_num` all five tables |

⚠️ **The mistake worth remembering is attempt 1's reasoning**, not the bug: "the scalars can
stay NaN because they are consumed by row INDEXING" is false, because the indexing happens
after the subtraction and the division, so autograd still multiplies a zero incoming gradient
by a NaN local one. The pre-existing `dcos` term never hit this precisely because it slices
first (`dp[has]` before the cosine) -- the new terms now copy that pattern exactly.

⚠️ **A finite loss with a non-finite gradient is the failure mode to look for here.** The
nan-guard caught every instance and dropped the batch, so the run looked healthy (1 batch of
312, 0.32%) while one term silently contributed nothing on those steps. Without the
per-epoch `[disp]` print reading `nan`, this would have shipped as "the term did not help".

### The three terms, as implemented

| term | what it matches | target | weight, 1× parity | 5× |
|---|---|---|---|---|
| `dcos_conc_weight` | per-cell cos(Δpred, group Δ) → the group's real mean cosine | `cos_mean` | **13.09** | 65.47 |
| `radial_spread_weight` | batch-mean of ((r − EMA_r)/`rad_std`)² → 1 | `rad_mean`, `rad_std` | **0.512** | 2.56 |
| `gene_var_weight` | batch-mean of ((pred − EMA_μ)/`real_std`)² → 1 | `gvar` | **0.473** | 2.37 |

Parity is against **Q02's own DCOS contribution**, `0.5 × 0.924964 = 0.4625` — the term that
dominates Q02's gradient today — so a 1× arm adds a term of that size.

⚠️ **Two design choices that are the difference between a dispersion term and another
mean-error term**, both forced by a measured fact: a 2,048-row batch holds a **median of ONE
cell per perturbation** (1,169 distinct perturbations per batch, measured on the corpus, which
is not sorted by perturbation — mean run length 1.0).

1. **Batch-level moment matching, not a per-row penalty.** `(x_i − target)²` on a *spread*
   would drive every cell to the same value, i.e. reinforce the collapse. Both moment terms
   are `(m2 − 1)²` on a normalised second moment pooled over the batch.
2. **The centre is the model's OWN per-group mean, tracked as a detached EMA**
   (`disp_ema_momentum` 0.99, warm-started on the real mean, ~90 samples/group/epoch).
   Centring on the *real* mean would let a constant bias satisfy the term with no dispersion
   at all — `tests/test_dispersion_terms.py` pins exactly that property.

A set-to-set objective (MMD, Sinkhorn) is **not** implementable on this sampler for the same
reason: with one cell per group there is no set. It would need grouped batches, which changes
the gradient even with the old loss and so needs its own control arm.

### Plumbing, and what is printed

* `scripts/pert_group_dispersion.py` → `cache/pert_disp/*.npz` (1.28 GB gene-variance table
  per source), keyed and ORDERED like `cache/pert_means/*.npz`.
* `stream_v2.load_group_dispersion` places every row **through the group-mean index**, never by
  glob order, and RAISES if the two tables share no key.
* The table-loading gate now includes all three weights, and **refuses to train** without the
  table rather than matching a variance target of zero — which would be an instruction to
  collapse, the very pathology being fixed. (F-loss1's lesson, applied in advance.)
* Three config layers wired: dataclass, `configs/rhj_fewshot.yaml`, `VCELL_SCHEMA['loss']`.
  `tests/test_dispersion_terms.py` asserts all three for every new key.
* **Training stdout, every epoch:** `[disp] cos(pred delta, group delta)=… vs REAL …
  radial std_pred/std_real=…  per-gene std_pred/std_real=…`, printed outside any
  `if wandb_run:` (these arms run offline).
* **Eval stdout, next to every arm's metrics:** `[disp] <arm>: per-gene within-pert std pred …
  real … RATIO …  |  FDR-significant genes/pert pred … real … RATIO …`, also stored in
  `metrics.json` under `disp_*`. Skippable with `OCCAM_SKIP_DISP_DIAG=1`; costs ~20–40 s/arm.
### ⭐ The collapse is LEARNED, not architectural — measured at epoch 1

The 2-epoch smoke run prints the diagnostics from the very first epoch, and they are nothing
like the trained model's:

| | epoch 1 (smoke) | trained `W23` | real |
|---|---|---|---|
| radial std_pred / std_real | **0.998** | 0.2225 | 1.0 |
| per-gene std_pred / std_real | **0.745** | 0.1079 | 1.0 |
| cos(pred Δ, group Δ) | 0.138 | 0.075 | 0.200–0.238 |

**An almost-untrained model is already close to the right dispersion; 90 epochs of Q02's loss
drive it down by 4–7×.** So the 2.6% collapse F-disp1 measured is not a property of the
architecture, the decoder or the streaming path — it is something the objective TEACHES, which
is consistent with `dcos` + `delta_mag` asking for parallel, equal-magnitude deltas while
`smooth_l1` (the only per-cell term) carries ~2% of the weight.

⚠️ This reframes what the three terms are for: they do not have to CREATE dispersion the model
cannot express, only stop the rest of the loss from removing it. It also predicts that their
effect will grow with schedule length, and that on a 30-epoch base they would look weaker —
the opposite of the VICReg pattern (F-vicreg3), where the gain shrank with length.

### Two bugs the 2-epoch smoke test caught, before any GPU was spent on an arm

Both were invisible to the unit tests and would have produced a plausible-looking null result.

1. **`0 * NaN = NaN` poisoned one batch per epoch.** Exactly ONE group of 7,179 has
   `n_cells < 2`, so its dispersion targets are NaN by design. The gene-variance term masked
   per ELEMENT (`_uu * row_mask`), and a NaN times a zero mask is still NaN, so the whole
   batch's loss went non-finite: `[nan-guard] epoch 1 batch 77: NON-FINITE LOSS — batch
   dropped` (1 of 312, 0.32%, every epoch), and the epoch's own diagnostic printed
   `per-gene std_pred/std_real=nan`. Fixed twice over: `load_group_dispersion` now
   `nan_to_num`s the gene table, and the term ROW-SELECTS instead of multiply-masking (which
   is also cheaper). ⚠️ The nan-guard did its job — it dropped the batch instead of poisoning
   the weights — which is precisely why this would have shipped unnoticed as "a term that
   didn't help".
2. **The EMA warm start made the epoch-1 diagnostic read a bias as a spread.** Starting the
   centre at the REAL group mean printed `radial std_pred/std_real=1.0380` on epoch 1 — which
   looks like "dispersion already matched" and is actually the model's radius sitting ~1 real
   std away from the real radius. That is the exact confound the EMA centre exists to remove.
   Both centres are now initialised LAZILY from the model's own first observation per group
   (momentum 0 on first touch), and a group whose centre has not been set yet is skipped for
   that batch rather than measured against a centre of 0.

* Launcher: 3 new columns (`dcos_conc_weight radial_spread_weight gene_var_weight`) appended
  **after** `base`, so all 22-column grids parse unchanged, plus a field-count guard that
  refuses a row with more columns than the launcher reads — the failure that put
  `1e-7<TAB>M03` inside `eta_min` and was only caught by a tab in a directory name.

## ⭐⭐⭐ F-sched2 — **COMPLETE, 24/24 arms. 90 > 60 > 30 epochs on BOTH bases at EVERY `eta_min` (8/8 paired contrasts, 5/5 splits each). Within 1e-7…1e-4, `eta_min` is ±0.003 of seed-like VARIANCE.** `W22` (Q02, 90 ep, `eta_min` 1e-6) = **0.3985** is the best arm in THIS grid.

⚠️ **SUPERSEDED IN TWO PLACES BY F-sched3 — read that first.** (1) `eta_min` IS a direction *above* this grid's band: 3e-4 beats every floor here at both 120 and 150 epochs. (2) The epochs plateau depends on the floor, so this grid's "90 is best" holds only for floors ≤1e-4. The campaign best is now `Y06` (150 ep, 3e-4) = **0.4043**.

**Objective.** Best model on SEEN perturbations in an UNSEEN context — here, 611 held-out HepG2
perturbations predicted from one half-B control cell, scored by cell-eval 0.6.0 on the paired
split-half protocol (5 splits, uncalibrated `model_nocal`).

**Question this answers.** F-sched1 found that 60 epochs + `eta_min` 1e-5 beat the 30-epoch
default on all three bases (`E3` = 0.3933), but it changed BOTH knobs at once. This grid
separates them: is the gain training LENGTH, or the LR FLOOR?

**Varied:** `epochs` ∈ {30, 60, 90} × `eta_min` ∈ {1e-7, 1e-6, 1e-5, 1e-4}, on two bases
(M03, Q02). **Held fixed:** everything else in each base's loss composition, the cohort, the
axis, the seeded control draw.

**Pre-registered reading (written before the numbers):** gain tracks epochs at every floor →
it is training length and `eta_min` is incidental. Gain tracks `eta_min` at every length → it
is the floor, and F-sched1's extra epochs were wasted compute. Only the (long, high-floor)
corner wins → genuine interaction. 90 no better than 60 → 60 is the plateau.

**Anchors:** `ctrlcell` −0.0018 · `ctrlmean` 0.0016 · `meanpert` 0.1324 · `transfer` 0.2323 ·
`experimental` 0.3722 (`mae` 0.0257) · M03 base (30 ep) 0.3679 · Q02 base (30 ep) 0.3783 ·
`E1` (M03, 60 ep/1e-5) 0.3749 · `E3` (Q02, 60 ep/1e-5) 0.3933.

### The outcome: **epochs, not `eta_min`**

| contrast | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | sign |
|---|---|---|---|---|---|
| M03 30 → 60 ep (`W05`−`W03`, 1e-7 vs 1e-5) | **+0.0108** ± 0.0018 | −0.0005 | +0.0039 | +0.0023 | 4/4 |
| M03 60 → 90 ep (`W11`−`W05`) | **+0.0049** ± 0.0019 | +0.0002 | +0.0050 | +0.0111 | 4/4 |
| M03 30 → 90 ep (`W11`−`W03`, both 1e-5) | **+0.0161** ± 0.0017 | −0.0003 | +0.0089 | +0.0135 | 5/5 |
| M03 30 → 60 ep at 1e-6 (`W06`−`W02`) | **+0.0128** ± 0.0030 | −0.0006 | +0.0035 | +0.0088 | 5/5 |
| Q02 60 → 90 ep (`W23`−`E3`, both 1e-5) | **+0.0025** ± 0.0018 | −0.0001 | +0.0070 | +0.0094 | 5/5 |
| **`eta_min` 1e-5 → 1e-4 at 90 ep, M03** (`W12`−`W11`) | **−0.0001** ± 0.0018 | −0.0002 | −0.0015 | −0.0015 | 3/5 |
| **`eta_min` 1e-5 → 1e-4 at 90 ep, Q02** (`W24`−`W23`) | **−0.0002** ± 0.0027 | +0.0005 | +0.0002 | **+0.0049** | 2/5 |

**Reading: the epochs axis is the only reproducible DIRECTION.** 30 → 90 epochs moves
`pearson_delta` +0.0161 on M03 (5/5) and 60 → 90 moves Q02 +0.0025 (5/5), monotonically and on
every metric. F-sched1's `E3` gain was **training length**, not the floor.

⚠️ The `eta_min` contrasts in the two rows above are ≈0 **at 90 epochs specifically**; across
the filled-in grid the within-row spread is 0.0051–0.0103 with no consistent winner. See
"Reading the surface" below — the honest statement is that `eta_min` cannot be tuned, not that
it is negligible.

⚠️ **CORRECTION to what validation suggested.** Mid-flight I noted the validation composite
ranked `eta_min` 1e-4 ABOVE 1e-5 on both bases (W12 0.4284 > W11 0.4212; W24 0.4430 > W23
0.4419) and flagged it as "never tried in F-sched1". On TEST that ordering does not survive:
both contrasts are −0.0002/−0.0001 with 2–3 of 5 positive. The validation margin was noise.
This is the same validation-misranks-ARMS / selects-EPOCHS-well pattern as F-ckpt4.

✅ **The returns decelerate but have not stopped.** M03: +0.0108 (30→60) then +0.0049 (60→90).
Q02: +0.0150 (30→60, `Q02`→`E3`) then +0.0025 (60→90). 90 epochs is NOT the plateau on either
base, but the next step would buy ~0.002.

✅ **`eta_min` 1e-4 does buy `disc_l1` on Q02** (+0.0049, 5/5) at zero `pearson_delta` cost, so
`W24` is the arm to pick if discrimination is the target. `W24`'s `disc_l1` 0.7190 ties `X02`
for the campaign maximum.

⚠️ **`mae` is flat-to-slightly-worse with length.** M03's 90-epoch arms hold the best `mae` of
any model arm (`W12` 0.0208, `W11` 0.0210 — against `experimental` 0.0257), and the Q02 family
stays at 0.0216–0.0221. Length is not an `mae` lever in either direction.

### Every arm scored so far, ranked (snapshot 2026-09-21, eval array 18249983 COMPLETE)

| arm | base | epochs | `eta_min` | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|---|---|
| `W22` | Q02 | **90** | 1e-6 | 5/5 | **0.3985** ± 0.0025 | 0.0215 | 0.0868 | 0.7156 |
| `W21` | Q02 | **90** | 1e-7 | 5/5 | **0.3981** ± 0.0018 | 0.0216 | 0.0855 | 0.7183 |
| `W23` | Q02 | **90** | 1e-5 | 5/5 | **0.3959** ± 0.0031 | 0.0216 | 0.0887 | 0.7142 |
| `W24` | Q02 | **90** | 1e-4 | 5/5 | **0.3957** ± 0.0022 | 0.0221 | 0.0889 | 0.7190 |
| `W19` = `E3` | Q02 | 60 | 1e-5 | 5/5 | **0.3933** ± 0.0027 | 0.0217 | 0.0817 | 0.7047 |
| `W18` | Q02 | 60 | 1e-6 | 5/5 | **0.3921** ± 0.0030 | 0.0216 | 0.0819 | 0.7089 |
| `W20` | Q02 | 60 | 1e-4 | 5/5 | **0.3921** ± 0.0034 | 0.0216 | 0.0792 | 0.7068 |
| `W17` | Q02 | 60 | 1e-7 | 5/5 | **0.3898** ± 0.0026 | 0.0216 | 0.0811 | 0.7040 |
| `W09` | M03 | **90** | 1e-7 | 5/5 | **0.3842** ± 0.0033 | 0.0209 | 0.0816 | 0.6852 |
| `W11` | M03 | **90** | 1e-5 | 5/5 | **0.3805** ± 0.0023 | 0.0210 | 0.0814 | 0.6852 |
| `W12` | M03 | **90** | 1e-4 | 5/5 | **0.3805** ± 0.0023 | 0.0208 | 0.0799 | 0.6836 |
| `W08` | M03 | 60 | 1e-4 | 4/5 ⚠️ | **0.3800** ± 0.0021 | 0.0208 | 0.0765 | 0.6814 |
| `W16` | Q02 | 30 | 1e-4 | 5/5 | **0.3794** ± 0.0026 | 0.0229 | 0.0795 | 0.7017 |
| `W13` = `Q02` base | Q02 | 30 | 1e-7 | 5/5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 |
| `W06` | M03 | 60 | 1e-6 | 5/5 | **0.3777** ± 0.0020 | 0.0208 | 0.0745 | 0.6750 |
| `W10` | M03 | **90** | 1e-6 | 5/5 | **0.3773** ± 0.0022 | 0.0208 | 0.0812 | 0.6791 |
| `W14` | Q02 | 30 | 1e-6 | 5/5 | **0.3769** ± 0.0017 | 0.0227 | 0.0767 | 0.7015 |
| `W05` | M03 | 60 | 1e-7 | 4/5 ⚠️ | **0.3754** ± 0.0031 | 0.0208 | 0.0763 | 0.6739 |
| `W07` | M03 | 60 | 1e-5 | 5/5 | **0.3749** ± 0.0021 | 0.0210 | 0.0708 | 0.6800 |
| `W01` | M03 | 30 | 1e-7 | 4/5 ⚠️ | **0.3683** ± 0.0018 | 0.0212 | 0.0721 | 0.6715 |
| `W15` | Q02 | 30 | 1e-5 | 4/5 ⚠️ | **0.3679** ± 0.0025 | 0.0231 | 0.0702 | 0.6936 |
| `W02` | M03 | 30 | 1e-6 | 5/5 | **0.3650** ± 0.0031 | 0.0214 | 0.0711 | 0.6663 |
| `W03` | M03 | 30 | 1e-5 | 5/5 | **0.3644** ± 0.0012 | 0.0213 | 0.0725 | 0.6716 |
| `W04` | M03 | 30 | 1e-4 | 5/5 | **0.3622** ± 0.0024 | 0.0214 | 0.0694 | 0.6701 |

✅ **All 24 arms scored.**  ⚠️ Rows below 5/5 are PARTIAL (F-infra1 pdex race); their means may move.

### The full surface, all four metrics (`mae` is an ERROR: LOWER is better)

Cell = mean over the splits scored so far, `(n)` beside it. `—` = not yet scored.

**`pearson_delta` ↑**

| base | epochs | 1e-7 | 1e-6 | 1e-5 | 1e-4 | row spread |
|---|---|---|---|---|---|---|
| M03 | 30 | 0.3683 (4) | 0.3650 (5) | 0.3644 (5) | 0.3622 (5) | 0.0061 |
| M03 | 60 | 0.3754 (4) | 0.3777 (5) | 0.3749 (5) | 0.3800 (4) | 0.0051 |
| M03 | **90** | 0.3842 (5) | 0.3773 (5) | 0.3805 (5) | 0.3805 (5) | 0.0069 |
| Q02 | 30 | 0.3783 (5) | 0.3769 (5) | 0.3679 (4) | 0.3794 (5) | 0.0114 |
| Q02 | 60 | 0.3898 (5) | 0.3921 (5) | 0.3933 (5) | 0.3921 (5) | 0.0035 |
| Q02 | **90** | 0.3981 (5) | 0.3985 (5) | 0.3959 (5) | 0.3957 (5) | 0.0028 |

**`mae` ↓ (ERROR)**

| base | epochs | 1e-7 | 1e-6 | 1e-5 | 1e-4 | row spread |
|---|---|---|---|---|---|---|
| M03 | 30 | 0.0212 (4) | 0.0214 (5) | 0.0213 (5) | 0.0214 (5) | 0.0002 |
| M03 | 60 | 0.0208 (4) | 0.0208 (5) | 0.0210 (5) | 0.0208 (4) | 0.0002 |
| M03 | **90** | 0.0209 (5) | 0.0208 (5) | 0.0210 (5) | 0.0208 (5) | 0.0002 |
| Q02 | 30 | 0.0225 (5) | 0.0227 (5) | 0.0231 (4) | 0.0229 (5) | 0.0006 |
| Q02 | 60 | 0.0216 (5) | 0.0216 (5) | 0.0217 (5) | 0.0216 (5) | 0.0001 |
| Q02 | **90** | 0.0216 (5) | 0.0215 (5) | 0.0216 (5) | 0.0221 (5) | 0.0006 |

**`overlap_at_N` ↑**

| base | epochs | 1e-7 | 1e-6 | 1e-5 | 1e-4 | row spread |
|---|---|---|---|---|---|---|
| M03 | 30 | 0.0721 (4) | 0.0711 (5) | 0.0725 (5) | 0.0694 (5) | 0.0030 |
| M03 | 60 | 0.0763 (4) | 0.0745 (5) | 0.0708 (5) | 0.0765 (4) | 0.0056 |
| M03 | **90** | 0.0816 (5) | 0.0812 (5) | 0.0814 (5) | 0.0799 (5) | 0.0017 |
| Q02 | 30 | 0.0774 (5) | 0.0767 (5) | 0.0702 (4) | 0.0795 (5) | 0.0093 |
| Q02 | 60 | 0.0811 (5) | 0.0819 (5) | 0.0817 (5) | 0.0792 (5) | 0.0026 |
| Q02 | **90** | 0.0855 (5) | 0.0868 (5) | 0.0887 (5) | 0.0889 (5) | 0.0035 |

**`disc_l1` ↑ (chance 0.5)**

| base | epochs | 1e-7 | 1e-6 | 1e-5 | 1e-4 | row spread |
|---|---|---|---|---|---|---|
| M03 | 30 | 0.6715 (4) | 0.6663 (5) | 0.6716 (5) | 0.6701 (5) | 0.0054 |
| M03 | 60 | 0.6739 (4) | 0.6750 (5) | 0.6800 (5) | 0.6814 (4) | 0.0075 |
| M03 | **90** | 0.6852 (5) | 0.6791 (5) | 0.6852 (5) | 0.6836 (5) | 0.0061 |
| Q02 | 30 | 0.7029 (5) | 0.7015 (5) | 0.6936 (4) | 0.7017 (5) | 0.0093 |
| Q02 | 60 | 0.7040 (5) | 0.7089 (5) | 0.7047 (5) | 0.7068 (5) | 0.0049 |
| Q02 | **90** | 0.7183 (5) | 0.7156 (5) | 0.7142 (5) | 0.7190 (5) | 0.0049 |


### The complete Q02 epochs ladder — 8/8 paired contrasts positive, 5/5 splits each

| `eta_min` | 30 → 60 ep | 60 → 90 ep |
|---|---|---|
| 1e-7 | **+0.0116** (5/5) | **+0.0083** (5/5) |
| 1e-6 | **+0.0152** (5/5) | **+0.0063** (5/5) |
| 1e-5 | **+0.0264** (4/4) | **+0.0025** (5/5) |
| 1e-4 | **+0.0127** (5/5) | **+0.0036** (5/5) |

**Every cell is positive at 5/5 (4/4 where a split is missing). There is no `eta_min` at which
lengthening the schedule fails.** 60 → 90 is worth +0.0025…+0.0083, still positive and still
decelerating, so 90 is not the plateau. ⚠️ The 1e-5 column's +0.0264 is inflated by `W15`
being a 4/5 partial — the other three columns bracket the 30 → 60 step at +0.0116…+0.0152.

### ⚠️ The campaign best moved: `W22` (Q02, 90 ep, `eta_min` **1e-6**) = 0.3985

| contrast | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | sign |
|---|---|---|---|---|---|
| `W22` − `W23` (1e-6 vs 1e-5, both 90 ep) | **+0.0026** ± 0.0016 | −0.0001 | −0.0020 | +0.0015 | 5/5 |
| `W21` − `W23` (1e-7 vs 1e-5, both 90 ep) | **+0.0022** ± 0.0015 | +0.0001 | −0.0033 | +0.0042 | 4/5 |
| `W22` − `W19`/`E3` (90 ep vs 60 ep) | **+0.0051** ± 0.0013 | −0.0001 | +0.0050 | +0.0109 | 5/5 |

`W22` 0.3985 and `W21` 0.3981 both beat `W23` 0.3959 at 5/5 and 4/5. ⚠️ **This does NOT make
`eta_min` 1e-6 the right floor** — it makes the point below sharper. The Q02 90-epoch row is
0.3981 / **0.3985** / 0.3959 / 0.3957 (spread 0.0028, best at 1e-6); the Q02 60-epoch row is
0.3898 / 0.3921 / **0.3933** / 0.3921 (spread 0.0035, best at **1e-5**); the M03 90-epoch row
is **0.3842** / 0.3773 / 0.3805 / 0.3805 (spread 0.0069, best at **1e-7**). Three rows, three
different winners, every row non-monotone.

✅ **SECOND DETERMINISM CHECK PASSED.** `W19` is configuration-identical to `E3` and reproduces
it to **+0.0000 ± 0.0000 on all four metrics, 0/5 splits differing** — from a different run
directory, a different job, a different day. Together with `W13` = the `Q02` base (also exact),
that means every contrast in this file is free of run-to-run drift, and the `eta_min` spread
discussed here is genuinely attributable to `eta_min`.

### Reading the surface — **epochs is a DIRECTION, `eta_min` is VARIANCE**

**Down the epochs axis, every column improves, on every metric, on both bases.** M03's
90-epoch row is the best or tied-best cell in `pearson_delta`, `ovl@N` and `disc_l1`; `mae` sits
at 0.0208–0.0210 for the whole 60/90 block against 0.0212–0.0214 at 30 epochs, so even `mae`,
flat in the paired contrasts, prefers the longer schedule to the 30-epoch default.

⚠️ **CORRECTION to the first version of this finding, which said `eta_min` "does nothing".**
With the rows now filled in, the within-row spread on `pearson_delta` is **0.0051–0.0103** —
comparable to the 30→60 epochs gain (+0.0108) and LARGER than the 60→90 gain (+0.0049). What
it is not is a *direction*: the winning value differs in every row (1e-7 on M03 90 ep and Q02
30 ep, 1e-4 on M03 60 ep, 1e-5 on Q02 90 ep by 0.0002) and every row is non-monotone — on the
M03 90-epoch row 1e-6 is the WORST cell (0.3773) while its neighbours 1e-7 and 1e-5 give
0.3842 and 0.3805.

**So the correct statement is: `eta_min` is not tunable, but it is not free either.** It behaves
like a seed — an unpredictable ±0.003 that cannot be optimised on n=1 and can cost up to 0.0069
if the unlucky value is picked. That is the strongest available argument for seed replicates:
the campaign has been reading 0.0025 margins off single runs whose LR-floor nuisance alone
spans more than that.

⚠️ **The best arm of each family is a different `eta_min`.** `W09` (M03, 90 ep, **1e-7**)
0.3842 leads M03 — above `W11`/`W12` (0.3805), which an earlier version of this finding named —
and `W22` (Q02, 90 ep, **1e-6**) 0.3985 leads everything. Q02 still beats M03 at every length
and every floor.

✅ **Two determinism checks passed on the way** — `W13` = the `Q02` base and `W19` = `E3`,
both exact on all four metrics from different run directories. See the section above.


⚠️ **Arm-id map** (so the surface can be cross-checked against `runs/`): `W01`–`W04` = M03 30 ep
at 1e-7/1e-6/1e-5/1e-4, `W05`–`W08` = M03 60 ep, `W09`–`W12` = M03 90 ep, `W13`–`W16` = Q02
30 ep, `W17`–`W20` = Q02 60 ep, `W21`–`W24` = Q02 90 ep.

## ⛔ F-vicreg3 — **VICReg does NOT compound with the longer schedule.** `X01`/`X02` are at or below the no-VICReg arm at the same length; the +0.0060 of F-vicreg2 was something the long schedule already buys.

**Question this answers.** F-vicreg2 found `V09` (variance + covariance together, on `delta_z`)
worth +0.0060 over `Q02` at 30 epochs, and F-sched1 found 60 epochs worth +0.0150 on the same
base. Are those two gains INDEPENDENT (the mechanisms differ → they add), or the SAME gain
reached two ways?

**Varied:** VICReg on/off × schedule length. **Held fixed:** the Q02 loss composition,
`eta_min` 1e-5, cohort, axis, control draw.

**Pre-registered reading (verbatim from the launch grid):** "X01 ~ 0.3996 → the two gains are
independent and both mechanisms are real. X01 ~ 0.393 (i.e. no better than E3) → VICReg's gain
was the same thing the longer schedule already buys. X01 > 0.400 → positive interaction.
X02 > X01 → the epochs effect is still unsaturated even with VICReg on."

### The outcome: **the second branch. The gains are the same gain.**

| arm | what it is | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|
| `W23` | Q02, 90 ep, **no VICReg** | 5/5 | **0.3959** ± 0.0031 | 0.0216 | 0.0887 | 0.7142 |
| `X02` | Q02 + VICReg, 90 ep | 5/5 | **0.3943** ± 0.0021 | 0.0220 | 0.0843 | **0.7189** |
| `E3` | Q02, 60 ep, **no VICReg** | 5/5 | **0.3933** ± 0.0027 | 0.0217 | 0.0817 | 0.7047 |
| `X01` | Q02 + VICReg, 60 ep | 5/5 | **0.3920** ± 0.0034 | 0.0217 | 0.0793 | 0.7079 |
| `V09` | Q02 + VICReg, 30 ep | 5/5 | **0.3843** ± 0.0016 | 0.0225 | 0.0808 | 0.7065 |
| `Q02` | base, 30 ep | 5/5 | **0.3783** ± 0.0027 | 0.0225 | 0.0774 | 0.7029 |

**Naive independent sum predicted 0.3993 at 60 epochs. Measured `X01` = 0.3920.**

| paired contrast | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | sign |
|---|---|---|---|---|---|
| `X01` − `E3` (VICReg at 60 ep) | **−0.0014** ± 0.0018 | +0.0000 | −0.0024 | +0.0032 | 2/5 |
| `X02` − `W23` (VICReg at 90 ep) | **−0.0016** ± 0.0012 | +0.0004 | −0.0045 | **+0.0048** | 0/5 |
| `X02` − `E3` (VICReg **and** +30 ep) | +0.0009 ± 0.0019 | +0.0003 | +0.0025 | **+0.0142** | 4/5 |
| `X02` − `X01` (epochs, VICReg on) | **+0.0023** ± — | +0.0003 | +0.0050 | +0.0110 | 5/5 |

**Reading.** Held at fixed length, VICReg is **negative on `pearson_delta` and `ovl@N`** and
**positive on `disc_l1`** — 0/5 and 5/5 respectively at 90 epochs, i.e. consistent, not noise.
Its 30-epoch gain was a faster approach to the same place the longer schedule reaches on its
own; adding both does not add their margins. `X02` − `E3` looks positive (+0.0009) only because
it bundles the +30 epochs, and `X02` loses to `W23`, the same length without VICReg.

✅ **What VICReg does durably buy is `disc_l1`**: +0.0048 (5/5) at 90 ep, +0.0032 (5/5) at
60 ep, +0.0142 vs `E3`. `X02` 0.7189 and `W24` 0.7190 are the campaign maximum. If the target
is discrimination rather than delta correlation, the variance+covariance terms are still worth
carrying; on the headline metric they are not.

✅ **The epochs effect IS still unsaturated with VICReg on** (`X02` − `X01` = +0.0023 at 5/5,
matching `W23` − `E3` = +0.0025 without it) — consistent with F-sched2. Length and VICReg do
not interact; length just dominates.

### The full 3x2 surface, all four metrics

`delta` = VICReg ON minus OFF at the same length. **The sign flips between 30 epochs and
60/90 on `pearson_delta` and `overlap_at_N`, and does NOT flip on `disc_l1`.**

| metric | 30 ep OFF→ON | 60 ep OFF→ON | 90 ep OFF→ON |
|---|---|---|---|
| `pearson_delta` ↑ | 0.3783 → 0.3843 (**+0.0060**) | 0.3933 → 0.3920 (−0.0014) | 0.3959 → 0.3943 (**−0.0016**) |
| `mae` ↓ (ERROR) | 0.0225 → 0.0225 (+0.0000) | 0.0217 → 0.0217 (+0.0000) | 0.0216 → 0.0220 (+0.0004) |
| `overlap_at_N` ↑ | 0.0774 → 0.0808 (**+0.0034**) | 0.0817 → 0.0793 (−0.0024) | 0.0887 → 0.0843 (**−0.0045**) |
| `disc_l1` ↑ | 0.7029 → 0.7065 (+0.0036) | 0.7047 → 0.7079 (+0.0032) | 0.7142 → 0.7189 (**+0.0048**) |

OFF arms are `Q02` / `E3` / `W23`; ON arms are `V09` / `X01` / `X02`. The paired per-split
contrasts for the 60- and 90-epoch columns are in the table above; the 30-epoch column is
F-vicreg2.

**Reading: VICReg buys `disc_l1` at every length (+0.0032…+0.0048, and it GROWS with length),
and it buys `pearson_delta`/`ovl@N` only at 30 epochs, where it is substituting for training
the model has not done yet.** `mae` is untouched at 30 and 60 and marginally worse at 90.
That is a regulariser behaving exactly like a regulariser: it helps an under-trained model and
gets in the way of a well-trained one, except on the metric its terms actually target —
`disc_l1` rewards separating perturbations from each other, which is what the covariance term
does by construction.

✅ **`X01` is now 5/5.** Its split 4 was lost to the pdex shm race in `18247298` and was
re-scored by backfill job `18252613` (nocal only — the calibrated mode is retired). The
completed mean is **0.3920** ± 0.0034, against 0.3926 ± 0.0036 on the 4 splits, and the
`X01` − `E3` contrast firms up to **−0.0014** ± 0.0018 at **2/5** positive on five paired
splits. **The conclusion is unchanged and no longer rests on a partial arm.** For the record,
`X01` − `W23` (VICReg + 60 ep vs no VICReg + 90 ep) is −0.0039 on `pearson_delta`, −0.0095 on
`ovl@N` and −0.0062 on `disc_l1`, all 0/5 — the long schedule alone beats VICReg plus the
short one on every metric that matters. Its −0.0017 against
`E3` is computed on the 4 common splits and is the weaker of the two VICReg-at-fixed-length
contrasts; the 90-epoch pair (`X02` − `W23`, 5/5 paired, 0/5 positive) carries the conclusion.
A one-task backfill would complete it.

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

## ⛔ F-latent5 — the 0.01× latent arms are WORSE than 0.1× on both bases. **Too little latent supervision is harmful, on M06 as well.**

**Status note.** These arms trained on 2026-09-20 but were never scored: their eval job
`18216433` was CANCELLED by its `afterok` dependency when 10/12 split-reconstruction arms
crashed (F-recon2). The checkpoints survived; only the scoring was lost. Rescored as `18217089`,
10/10 tasks, both arms 5/5.

**Question.** `P11` (`both` at 0.1× parity) was the best combination arm on M03. Does an order
of magnitude lower — 0.01× — do better? M03 weights lmse 0.000157 / lcos 0.000097; M06 lmse
0.000044 / lcos 0.000059.

### Measured — test cell-eval, uncalibrated, 5/5

| arm | base | n | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|
| **M03** (no latent) | M03 | 5/5 | **0.3679 ± 0.0017** | **0.0212** | 0.0722 | **0.6715** |
| `P11` both 0.1× | M03 | 5/5 | 0.3670 ± 0.0023 | **0.0211** | **0.0738** | 0.6653 |
| `P01` MSE 0.1× | M03 | 5/5 | 0.3600 ± 0.0019 | 0.0213 | 0.0693 | 0.6589 |
| **`T01` both 0.01×** | M03 | 5/5 | **0.3595 ± 0.0027** | 0.0214 | 0.0700 | 0.6653 |
| **`Q02`** MSE 1× | M06 | 5/5 | **0.3783 ± 0.0027** | 0.0225 | 0.0774 | 0.7029 |
| `Q01` MSE 0.1× | M06 | 5/5 | 0.3725 ± 0.0024 | 0.0227 | **0.0794** | 0.7008 |
| **M06** (no latent) | M06 | 5/5 | 0.3681 ± 0.0023 | 0.0232 | 0.0703 | **0.7047** |
| **`T02` both 0.01×** | M06 | 5/5 | **0.3617 ± 0.0023** | 0.0235 | 0.0714 | 0.7026 |

`T01 − M03`: `pearson_delta` **−0.0085** · `mae` +0.0002 · `ovl@N` −0.0022 · `disc_l1` −0.0062.
`T02 − M06`: `pearson_delta` **−0.0063** · `mae` +0.0004 · `ovl@N` +0.0011 · `disc_l1` −0.0021.

### Reading

1. **0.01× is worse than 0.1× on both bases**, by 0.0075 (M03: 0.3595 vs 0.3670) and by more
   against M06's better arms. Going lower did not help.
2. **⛔ `T02` is the first latent arm to HURT M06.** Every other M06 latent arm measured
   (Q01–Q08) beat the base; `T02` is −0.0063 below it. So F-latent4's "latent supervision helps
   M06" has a **lower bound**: the term must be large enough to do something. At 0.01× parity it
   is not a weak-but-positive signal, it is a perturbation to the optimisation with no
   compensating benefit.
3. **The M03 dose-response is non-monotone**: 0.01× 0.3595 → 0.1× 0.3670 → parity 0.3606
   (`P02`) → 10× 0.3592 (`P03`). A hump at 0.1×, and every point below the 0.3679 base. On M03
   the best latent weight is still zero.
4. `disc_l1` is below base in both (−0.0062, −0.0021), consistent with F-latent3's finding that
   latent supervision costs discrimination on every M03 arm.

⇒ **Neither 0.01× arm is adoptable.** The useful latent range on M06 is around **parity**
(`Q02` 0.3783, `Q05` 0.3787 at 4/5); below 0.1× the term stops helping and starts hurting.

## ⭐⭐⭐ F-latent4 — the latent MSE **HELPS on M06 and HURTS on M03**. Setup-dependence again, and `Q02` = **0.3783** is the best arm in the campaign.

⚠️ **CORRECTION TO MY OWN STATUS REPORT.** I reported "the M06 half finished — all 11 Q-arms at
5/5". That was false. The shell check wrote `Q$i__*`, which bash parses as the variable `i__`
(underscores are legal in identifiers); it was undefined, so the glob collapsed to `Q*` and
matched `Q01` on every iteration, reporting its 5/5 eleven times. **Only Q01–Q03 are scored.**

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line.
**M03 reference** 0.3679 / 0.0212 / 0.0722 / 0.6715 · **M06 reference** 0.3681 / 0.0232 /
0.0703 / 0.7047 (test, uncalibrated, 5/5). `mae` is an ERROR: LOWER IS BETTER.

### M06 base — latent MSE arms only (Q04–Q11 still scoring)

| arm | what | n | `pearson_delta` | Δ pΔ | `mae` ↓ | Δ mae | `ovl@N` | Δ ovl | `disc_l1` | Δ disc |
|---|---|---|---|---|---|---|---|---|---|---|
| — | **M06 reference** | 5/5 | **0.3681** | — | **0.0232** | — | **0.0703** | — | **0.7047** | — |
| Q01 | MSE 0.1× | 5/5 | 0.3725 ± 0.0024 | **+0.0044** | 0.0227 | **−0.0005** | **0.0794** | **+0.0091** | 0.7008 | −0.0039 |
| **Q02** | **MSE 1× (parity)** | 5/5 | **0.3783 ± 0.0027** | **+0.0102** | **0.0225** | **−0.0007** | 0.0774 | **+0.0071** | 0.7029 | −0.0018 |
| Q03 | MSE 10× | 4/5 ⚠️ | 0.3721 ± 0.0028 | +0.0040 | 0.0229 | −0.0003 | 0.0755 | +0.0052 | 0.6968 | −0.0079 |

**All three MSE arms beat M06 on `pearson_delta`, `mae` AND `overlap_at_N` simultaneously.**
Only `disc_l1` is (marginally) down, −0.0018 to −0.0079.

### The same three arms on M03 — opposite sign

| arm | `pearson_delta` vs base | | arm | `pearson_delta` vs base |
|---|---|---|---|---|
| P01 MSE 0.1× | 0.3600 (**−0.0079**) | | Q01 MSE 0.1× | 0.3725 (**+0.0044**) |
| P02 MSE 1× | 0.3606 (**−0.0073**) | | Q02 MSE 1× | 0.3783 (**+0.0102**) |
| P03 MSE 10× | 0.3592 (**−0.0087**) | | Q03 MSE 10× | 0.3721 (**+0.0040**) |

**Identical intervention, opposite sign, 3/3 on each base.** Not noise: the M03 deficits
(−0.007…−0.009) and M06 gains (+0.004…+0.010) are each several times the ±0.002–0.003 sds.

### Reading

1. **`Q02` at 0.3783 is the best arm this campaign has produced** — above M03 (0.3679), D03
   (0.3686) and M06 (0.3681), and **above `experimental` 0.3722**. ⚠️ `experimental` is replicate
   agreement, not a ceiling: a model can pass it via the two-sided noise asymmetry. It is NOT a
   claim of super-experimental accuracy.
2. **⛔ The third setup-dependent lever, after `delta_mag_weight` (F-loss9).** The pattern is
   now consistent enough to state as a working rule: **a term that repairs a
   direction-dominated loss (D03/M06, where `smooth_l1` is 2.1% of the total) tends to harm a
   balanced one (M03, where it is 51.5%).** `delta_mag_weight` did exactly this; the latent MSE
   does it again. **Stop generalising any loss-term result across bases without testing both.**
3. **The optimum is at parity (1×), interior.** 0.1× and 10× both underperform 1× on every
   metric. That is the fourth time the scale-match point has been the optimum (F-loss7 `dcos`,
   F-latent3's target-invariance, and now this) — the heuristic keeps paying.
4. **`mae` moved, slightly** (0.0232 → 0.0225). Small, but this is the first perturbation-side
   lever to move `mae` at all, against F-recon1's prediction that it could not. ⚠️ It is −0.0007
   against an M03-vs-D03 gap of 0.0109, so it does not overturn F-recon1 — it bounds it.

⚠️ Q03 is 4/5 splits. ⚠️ **Q04–Q11 (all cosine and combination arms) are unscored**, so nothing
is known about the cosine on M06 — and on M03 the cosine behaved differently from the MSE.

## ⛔ F-latent3 — latent-space losses do NOT improve M03. 11/11 arms, no arm better on any metric that matters.

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line. **M03 reference**
(test, uncalibrated, 5/5): `pearson_delta` 0.3679 · `mae` 0.0212 · `overlap_at_N` 0.0722 ·
`disc_l1` 0.6715.

**Question.** Does supervising the latent delta — by MSE (position), cosine (direction), or both
— improve M03, and at what scale? Weights at 0.1× / 1× / 10× M03's own measured parity point
(F-latent2: MSE 0.0157, cos 0.0097, cos(`z_hat`) 0.179).

**Status.** M03 base (P-arms) complete, 11/11 arms scored. **M06 base (Q-arms) barely started**
(Q01 1/5, the rest 0/5) — job `18215244` still running, 5 tasks FAILED to the pdex shm race.

### Measured — test cell-eval, uncalibrated

| arm | what | n | `pearson_delta` | Δ pΔ | `mae` ↓ | Δ mae | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|---|---|
| — | **M03 reference** | 5/5 | **0.3679** | — | **0.0212** | — | **0.0722** | **0.6715** |
| P01 | MSE 0.1× | 5/5 | 0.3600 ± 0.0019 | −0.0079 | 0.0213 | +0.0001 | 0.0693 | 0.6589 |
| P02 | MSE 1× | 5/5 | 0.3606 ± 0.0032 | −0.0073 | 0.0213 | +0.0001 | 0.0710 | 0.6612 |
| P03 | MSE 10× | 4/5 ⚠️ | 0.3592 ± 0.0033 | −0.0087 | 0.0215 | +0.0003 | 0.0687 | 0.6571 |
| P04 | cos(`delta_z`) 0.1× | 4/5 ⚠️ | 0.3551 ± 0.0025 | −0.0128 | 0.0216 | +0.0004 | 0.0673 | 0.6663 |
| P05 | cos(`delta_z`) 1× | 5/5 | 0.3578 ± 0.0015 | −0.0101 | 0.0212 | −0.0000 | 0.0675 | 0.6637 |
| **P06** | cos(`delta_z`) 10× | 4/5 ⚠️ | **0.3692 ± 0.0033** | **+0.0013** | 0.0213 | +0.0001 | 0.0720 | 0.6655 |
| P07 | cos(`z_hat`) 1× | 5/5 | 0.3570 ± 0.0030 | −0.0109 | 0.0214 | +0.0002 | 0.0702 | 0.6547 |
| P08 | both 1× | 5/5 | 0.3640 ± 0.0019 | −0.0039 | 0.0211 | −0.0001 | 0.0660 | 0.6521 |
| P09 | both 10× | 5/5 | 0.3633 ± 0.0022 | −0.0046 | 0.0212 | −0.0000 | 0.0667 | 0.6568 |
| P10 | MSE 10× + cos 1× | 4/5 ⚠️ | 0.3643 ± 0.0033 | −0.0036 | 0.0212 | +0.0000 | **0.0723** | 0.6599 |
| **P11** | both 0.1× | 5/5 | 0.3670 ± 0.0023 | −0.0009 | **0.0211** | −0.0001 | **0.0738** | 0.6653 |

### Reading

1. **No arm improves M03 on `pearson_delta`.** The only positive is P06 at **+0.0013**, which
   sits inside M03's own ±0.0017 sd and is measured on 4/5 splits. A tie, not a gain.
2. **⛔ `disc_l1` is LOWER in all 11 arms** (0.6521–0.6663 vs 0.6715). Latent supervision costs
   discrimination uniformly — the one unambiguous, consistent effect in the table.
3. **`mae` is inert.** Every arm lands in 0.0211–0.0216, a total range of **0.0005**, against a
   0.0109 gap between M03 and D03. **The latent losses do not touch `mae` at all**, which is
   consistent with F-recon1: `mae` is a full-profile error dominated by control reconstruction,
   and every term swept here acts on the perturbation.
4. **Two dose-responses that point OPPOSITE ways**, the most interesting structure here:
   - **cosine alone RISES with weight**: 0.3551 → 0.3578 → 0.3692 (0.1× → 1× → 10×).
   - **cosine+MSE FALLS with weight**: 0.3670 → 0.3640 → 0.3633 (0.1× → 1× → 10×).
   So the cosine wants more weight when alone and less when the MSE is present. The MSE is not
   merely neutral; it changes the sign of the cosine's dose-response.
5. **✅ PRE-REGISTERED OUTCOME CONFIRMED: the weight absorbs the scale, and the cosine TARGET
   does not matter.** P07 (`z_hat`, 1× parity) = 0.3570 and P05 (`delta_z`, 1× parity) = 0.3578
   — a 0.0008 difference, inside the sds — *despite the raw terms differing by 18.4×*
   (0.9733 vs 0.0528). Scale-matching the weight fully neutralised an 18× scale gap. **That
   generalises the F-loss5/F-loss7 scale-match heuristic beyond gene space** and is the most
   transferable result in this sweep.
6. **P11, the user-added `both 0.1×` arm, is the best combination arm**: closest to M03 on
   `pearson_delta` (−0.0009), best `mae` of any arm (0.0211), and **the best `overlap_at_N` in
   the table (0.0738, above M03's 0.0722)**. It was not in my proposed grid.

⇒ **M03 stands. No latent arm replaces it.** The honest summary is that latent supervision is
mildly harmful across the board, with one tie (P06) and one `overlap_at_N` win (P11) that are
within or near noise.

⚠️ Four arms are 4/5 splits (P03, P04, P06, P10) — P06's +0.0013 in particular rests on 4
splits and should not be quoted as a gain. ⚠️ The M06 base is untested; F-loss9 showed
`delta_mag_weight` was setup-dependent, so M06 could behave differently and is not predicted by
this table.

## ⭐⭐⭐ F-ckpt4 — checkpoint sweep COMPLETE on all three bases. **Validation selection is sound (Spearman +0.992…+0.993), but it mis-picks on M06 at a cost of −0.0037.**

Completes F-ckpt2/F-ckpt3, which covered M03 only. 90 of 90 checkpoints scored.

| arm | base | Spearman `val` vs cell-eval | `val` pick | cell-eval argmax | **cost of `val`** | max consecutive swing | 4-metric pick |
|---|---|---|---|---|---|---|---|
| C1 | M03 | **+0.992** | 25 | 25 | **+0.0000** | **0.1744** | 25 |
| C2 | M06 | **+0.992** | **17** | **25** | **−0.0037** | 0.0827 | 25 |
| C3 | Q02 | **+0.993** | 25 | 25 | **+0.0000** | 0.0755 | 25 |

### Reading

1. **Validation selection is sound but not perfect.** Rank correlation is +0.992–0.993 on all
   three bases, and it lands on the exact argmax for M03 and Q02. **On M06 it picks epoch 17
   against cell-eval's 25 and gives up 0.0037** — about a third of the Q02-over-M06 margin
   (+0.0102). So the codebase's warning is not purely about ranking ARMS: validation can
   mis-pick WITHIN an arm too, rarely and cheaply.
2. **All three bases have their cell-eval argmax at epoch 25 of 30**, and all three 4-metric
   composites agree. The best checkpoint is consistently ~5 epochs from the end, which argues 30
   epochs is near a sensible stopping point rather than well past one.
3. **⛔ The extreme oscillation is an M03 property, not a universal one.** M03's largest
   consecutive-epoch swing is **0.1744**; M06's is 0.0827 and Q02's 0.0755 — less than half.
   **F-ckpt1 generalised from M03 and overstated the case.** The oscillation is real everywhere
   but twice as violent on the balanced-loss base. Q02, the best base, is also the most stable.
4. **⇒ Protocol settled: `best_model.pth` is the right checkpoint for all three bases**, with
   the M06 caveat noted. The epochs × `eta_min` grid needs no change.

⚠️ Selection split 0 only; a different split could shift the argmax by an epoch or two.

## ⭐⭐⭐ F-ckpt3 — C1's full 30-epoch curve: **validation and cell-eval agree UNANIMOUSLY. Spearman +0.992, and all four metrics plus the composite pick epoch 25.**

Completes F-ckpt2 (which had 23 of 30 epochs).

| | value |
|---|---|
| **Spearman, `val_pΔ` vs cell-eval `pΔ`** (30 epochs) | **+0.992** |
| validation's pick | **epoch 25** |
| cell-eval `pearson_delta` argmax | **epoch 25** |
| cell-eval `mae` argmin | **epoch 25** |
| cell-eval `overlap_at_N` argmax | **epoch 25** |
| cell-eval `disc_l1` argmax | **epoch 25** |
| 4-metric composite | **epoch 25 (1.000)** |
| **cost of selecting on validation** | **+0.0000** |
| largest consecutive-epoch swing | **0.1744** |
| last 5 epochs, cell-eval `pΔ` | 0.3469 · 0.3591 · 0.3245 · 0.3552 · 0.3210 |

### Reading

1. **Unanimous.** Every one of the four metrics, the composite, and the validation metric select
   the same epoch, and epoch 25's composite is a perfect 1.000 — it is the argmax of all four
   simultaneously. There is no trade to adjudicate here and nothing for a 4-metric selector to
   add over `val_pearson_delta` on this arm.
2. **⇒ `best_model.pth` is the right checkpoint, and the epochs × `eta_min` sweep needs no
   protocol change.** That was the blocker; it is cleared for M03's regime.
3. **The oscillation persists to the end of training** — the last five epochs swing 0.3210 to
   0.3591 (0.0381), and the largest consecutive jump anywhere is 0.1744. Training does not
   settle by epoch 30; it keeps sampling good and bad states.
4. **⇒ This is a strong argument for more epochs that has nothing to do with convergence.**
   If the model keeps oscillating and the selector reliably catches the peaks, then more epochs
   = more peaks sampled = a better selected checkpoint, even with no improvement in the
   underlying trajectory. **F-sched1's gain is real, but "better convergence" and "more peaks
   sampled" remain indistinguishable** — and the epochs × `eta_min` sweep will not separate them
   either, because both mechanisms predict the same monotone gain with epochs. Separating them
   needs the per-epoch curve at 60 and 90 epochs, not just the selected checkpoint.

⚠️ C1 is M03's loss on split 0. C2 (M06) is 20/30 and C3 (Q02) unscored.

## ⭐⭐ F-vicreg2 — FINAL VICReg table, 10/10 arms at 5/5. **Only `V09` (var + cov TOGETHER) beats Q02, at +0.0060. Every other arm is BELOW the base.**

Completes F-vicreg1, which had 8 of 10 arms at 4/5. The ranking is now fully resolved and one
conclusion changed: `V10` drops from +0.0009 to **−0.0010** on its fifth split, so **`V09` is the
sole arm above the base**, not one of two.

**Q02 base:** 0.3783 · `mae` 0.0225 · `ovl@N` 0.0774 · `disc_l1` 0.7029.

| arm | var | cov | γ | target | n | `pearson_delta` | Δ | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|---|---|---|---|
| **`V09`** | **1×** | **10×** | 0.55 | `delta_z` | 5/5 | **0.3843 ± 0.0016** | **+0.0060** | **0.0225** | **0.0808** | **0.7065** |
| `V10` | 1× | 1× | 0.55 | `z_hat` | 5/5 | 0.3773 ± 0.0031 | −0.0010 | 0.0228 | 0.0771 | **0.7113** |
| `V03` | 0 | 10× | — | `delta_z` | 5/5 | 0.3761 ± 0.0019 | −0.0022 | 0.0228 | 0.0746 | 0.6979 |
| `V07` | 1× | 0 | **1.0** | `delta_z` | 5/5 | 0.3754 ± 0.0036 | −0.0029 | 0.0228 | 0.0757 | 0.7050 |
| `V01` | 0 | 0.1× | — | `delta_z` | 5/5 | 0.3748 ± 0.0020 | −0.0035 | 0.0226 | 0.0745 | 0.7042 |
| `V08` | 1× | 1× | 0.55 | `delta_z` | 5/5 | 0.3744 ± 0.0027 | −0.0039 | 0.0229 | 0.0749 | 0.7075 |
| `V05` | 0.1× | 0 | 0.55 | `delta_z` | 5/5 | 0.3743 ± 0.0021 | −0.0040 | 0.0226 | 0.0752 | 0.6986 |
| `V04` | 0 | 1× | — | `z_hat` | 5/5 | 0.3736 ± 0.0029 | −0.0047 | 0.0227 | 0.0778 | 0.6984 |
| `V02` | 0 | 1× | — | `delta_z` | 5/5 | 0.3728 ± 0.0017 | −0.0055 | 0.0230 | 0.0777 | 0.7047 |
| `V06` | 1× | 0 | 0.55 | `delta_z` | 5/5 | 0.3720 ± 0.0024 | −0.0063 | 0.0226 | 0.0758 | 0.7070 |

### Reading

1. **`V09` is the only arm above the base.** +0.0060 `pearson_delta`, `ovl@N` +0.0034, `disc_l1`
   +0.0036, `mae` unchanged at 0.0225 — three metrics up, one held, none worse.
2. **9 of 10 arms are BELOW the base**, spanning −0.0010 to −0.0063. So VICReg is not broadly
   helpful; **one specific combination is, and the rest are mildly harmful.**
3. **The interaction is confirmed and is the whole effect.** var-only best (`V07`) −0.0029;
   cov-only best (`V03`) −0.0022; var+cov at 1×/1× (`V08`) −0.0039 — all negative. Only 1×/10×
   is positive. **This corrects F-latent7's pre-registration, which was mine:** it predicted cov
   would matter and var would not, from the std distribution. Neither alone matters; the ratio
   between them does.
4. **`disc_l1` rises in 8 of 10 arms** (0.6979–0.7113 vs base 0.7029 — 8 above or equal), with
   `V10` highest at 0.7113. So the terms do something real to the representation even where
   `pearson_delta` falls; they just usually cost more than they buy.
5. **γ makes little difference**: `V07` (γ=1.0) −0.0029 vs `V06` (γ=0.55) −0.0063 at the same
   weight, so the paper's γ is mildly better than my std-derived one — the opposite of what I
   argued, and small either way.

⚠️ **The arm spread is 0.0123, which is smaller than the consecutive-epoch swing on every base**
(M03 0.1744, M06 0.0827, Q02 0.0755 — F-ckpt3). **`V09`'s +0.0060 is within the range a single
epoch of training moves this model.** It needs a seed replicate before it is more than a lead,
and that caveat applies to the whole table, not just the winner.

⚠️ Not measured: whether `V09` actually raised the effective rank. F-latent7 found 32.9/512 on
Q02, and that was the motivation for the term — the score moving is not evidence the mechanism
worked. `scripts/loss_scale.py`-style rank probe on `V09`'s checkpoint is the direct check.

## ⭐⭐ F-vicreg1 — **variance AND covariance TOGETHER beat Q02 (+0.0063); either alone does not.** My pre-registration was wrong about which term matters.

⚠️ **Corrects a pre-registration that was mine.** F-latent7 predicted: *"only 0.6% of dims are
dead, so the VARIANCE term has little to fix; the effective rank is 33/512, so the COVARIANCE
term has a real target. Expect `cov_weight` to matter and `var_weight` not to."* Neither term
alone helps. **Their combination does.**

**Q02 base:** `pearson_delta` 0.3783 · `mae` 0.0225 · `ovl@N` 0.0774 · `disc_l1` 0.7029.

| arm | var / cov / γ / target | n | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|
| **`V09`** | **1× / 10× / 0.55 / `delta_z`** | 4/5 ⚠️ | **0.3846 ± 0.0017** | **0.0225** | **0.0809** | **0.7064** |
| `V10` | 1× / 1× / 0.55 / `z_hat` | 2/5 ⚠️ | 0.3792 ± 0.0041 | 0.0228 | 0.0770 | **0.7115** |
| `V03` | 0 / 10× / — / `delta_z` | 5/5 | 0.3761 ± 0.0019 | 0.0228 | 0.0746 | 0.6979 |
| `V07` | 1× / 0 / **1.0** / `delta_z` | 4/5 ⚠️ | 0.3761 ± 0.0037 | 0.0228 | 0.0756 | 0.7052 |
| `V08` | 1× / 1× / 0.55 / `delta_z` | 4/5 ⚠️ | 0.3749 ± 0.0027 | 0.0228 | 0.0751 | 0.7075 |
| `V01` | 0 / 0.1× / — / `delta_z` | 4/5 ⚠️ | 0.3749 ± 0.0023 | 0.0226 | 0.0744 | 0.7042 |
| `V05` | 0.1× / 0 / 0.55 / `delta_z` | 4/5 ⚠️ | 0.3747 ± 0.0023 | 0.0226 | 0.0753 | 0.6982 |
| `V04` | 0 / 1× / — / `z_hat` | 4/5 ⚠️ | 0.3736 ± 0.0033 | 0.0227 | 0.0780 | 0.6982 |
| `V02` | 0 / 1× / — / `delta_z` | 5/5 | 0.3728 ± 0.0017 | 0.0230 | 0.0777 | 0.7047 |
| `V06` | 1× / 0 / 0.55 / `delta_z` | 4/5 ⚠️ | 0.3727 ± 0.0022 | 0.0225 | 0.0762 | 0.7069 |

### Reading

1. **`V09` is the only arm above the base, at +0.0063, and it improves three metrics while
   holding `mae` exactly** (0.0225 = base): `ovl@N` +0.0035, `disc_l1` +0.0035. ⚠️ **4/5 splits.**
2. **Neither term alone works.** Best cov-only is `V03` at 0.3761 (−0.0022); best var-only is
   `V07` at 0.3761 (−0.0022). Both *below* base. Yet var 1× + cov 10× is +0.0063. **That is an
   interaction, not an additive effect** — which is what VICReg's own design claims (variance
   stops dimensions dying, covariance stops them duplicating; either alone leaves the other
   failure mode open).
3. **Weight ratio matters more than either weight.** `V08` (1×/1×) is 0.3749 while `V09`
   (1×/10×) is 0.3846 — a 0.0097 gap from the covariance weight alone, given the same variance
   weight. The 10× covariance point is where it works.
4. **`γ` = 1.0 is not obviously wrong.** `V07` (paper's γ) ties `V03` and beats `V06` (γ=0.55)
   by +0.0034 at the same weight. My reasoning for γ=0.55 from the measured std was sound but
   is not supported by the outcome; both γ values sit within noise of each other.
5. **`z_hat` is competitive**: `V10` at 0.3792 (2/5) has the best `disc_l1` of any arm here
   (0.7115). Too few splits to rank, but it is not the dead end `delta_z`'s dominance elsewhere
   suggested.

⚠️ **Eight of ten arms are 4/5 or worse** (the pdex shm race), and the whole spread across arms
(0.3727–0.3846, i.e. 0.0119) is smaller than the **0.1744** consecutive-epoch swing F-ckpt1/3
measured on this panel. **`V09` needs its fifth split and a seed replicate before it is more
than a lead.**

## ⭐⭐⭐ F-ckpt2 — **validation selection AGREES with cell-eval: Spearman +0.988, and both pick the same epoch.** The selector is fine; the MODEL oscillates.

**This refines F-ckpt1 and corrects part of its framing.** F-ckpt1 measured a 0.175 swing between
consecutive epochs on the test metric and inferred that any single-checkpoint comparison is "partly
a draw from this oscillation". True — but it left open whether `val_pearson_delta` could even see
the oscillation. It can, almost perfectly.

**Question.** Does the checkpoint that `val_pearson_delta` selects agree with what cell-eval
would pick, and what does choosing on validation cost?

### Measured — C1 (M03's loss), 23 of 30 epochs with BOTH scores

| epoch | `val_pΔ` | cell-eval `pΔ` | 4-metric | |
|---|---|---|---|---|
| 1 | 0.1571 | 0.1390 | 0.000 | |
| 4 | 0.2136 | 0.2149 | 0.469 | ↓ dip, seen by BOTH |
| 7 | 0.3396 | 0.3187 | 0.779 | |
| 12 | 0.3544 | 0.3387 | 0.899 | |
| **13** | **0.1482** | **0.1642** | 0.243 | ↓ the 0.175 collapse — **validation sees it** |
| 14 | 0.3340 | 0.3199 | 0.816 | |
| 16 | 0.3748 | 0.3496 | 0.928 | |
| 19 | 0.3824 | 0.3544 | 0.917 | |
| **22** | **0.3886** | **0.3581** | **0.985** | ⭐ **val, 4-metric AND cell-eval `pΔ` all pick this** |
| 23 | 0.3561 | 0.3368 | 0.948 | |

| | value |
|---|---|
| **Spearman, `val_pΔ` vs cell-eval `pΔ`** | **+0.988** |
| Spearman, `val_pΔ` vs 4-metric composite | **+0.921** |
| validation's pick | epoch 22 |
| 4-metric pick | epoch 22 |
| cell-eval `pΔ` argmax | epoch 22 |
| **cost of selecting on validation** | **+0.0000** |

### Reading

1. **✅ `val_pearson_delta` is a GOOD checkpoint selector.** Rank correlation +0.988 with the
   test metric across 23 epochs, and it lands on the exact argmax. Selecting on validation costs
   **nothing** here.
2. **⇒ The codebase's warning is about a DIFFERENT claim, and both can be true.**
   `pipeline.py` records that this criterion "was MEASURED to rank checkpoints wrongly on the
   real panel ... picked one arm and put the eventual bake-off winner last" — that is about
   ranking **ARMS against each other**. This measurement is about ranking **EPOCHS within one
   arm**. Validation does the second well and, per that note, the first badly. I had been
   treating them as one claim.
3. **⇒ F-ckpt1's oscillation is real MODEL variation, faithfully reported.** Validation sees the
   epoch-13 collapse (0.1482 against its neighbours' 0.35) just as cell-eval does. So the
   selector is not noisy — the model genuinely passes through bad states, and the metric
   correctly says so.
4. **⇒ This weakens the "lottery" reading of F-sched1.** More epochs do give more draws, but the
   selector reliably identifies the good draws, so a longer run finding a better state is a REAL
   benefit rather than a selection artifact. E3's +0.0150 is more likely to be genuine than
   F-ckpt1 implied. ⚠️ Not settled: this says selection is sound, not that 60 epochs converges
   better than 30. The 2D epochs × `eta_min` sweep is still the experiment for that.
5. **⇒ The epochs × `eta_min` sweep can use `best_model.pth`.** The protocol question is
   answered for this base: no need to select on cell-eval, and no need to fall back to
   `final_model.pth`.

⚠️ **One base, one split.** C1 is M03's loss, scored on split 0 only. C2 (M06) and C3 (Q02) are
still scoring, and three levers have already behaved differently across those bases (F-loss9,
F-latent4, F-recon3) — so the +0.988 should be confirmed on them before the protocol is treated
as settled for all three.

## ⭐⭐⭐ F-ckpt1 — **consecutive epochs differ by up to 0.17 on the TEST metric.** The F7 oscillation is in the MODEL, not just the validation metric.

**Status: C1 (M03's loss) at 19 of 30 checkpoints; C2/C3 still scoring.** Selection split 0 only.

### Measured — every epoch of one run, scored by cell-eval on split 0

| epoch | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` | 4-metric |
|---|---|---|---|---|---|
| 1 | 0.1390 | 0.0354 | 0.0274 | 0.5028 | 0.000 |
| 3 | 0.2764 | 0.0241 | 0.0396 | 0.6093 | 0.605 |
| 4 | **0.2149** | 0.0289 | 0.0424 | 0.6162 | 0.475 |
| 6 | 0.2933 | 0.0239 | 0.0579 | 0.6400 | 0.791 |
| 7 | 0.3187 | 0.0229 | 0.0495 | 0.6419 | 0.790 |
| 10 | **0.2336** | 0.0261 | 0.0535 | 0.6362 | 0.648 |
| 11 | 0.3172 | 0.0231 | 0.0583 | 0.6545 | 0.860 |
| 12 | **0.3387** | 0.0221 | 0.0592 | 0.6575 | 0.913 |
| **13** | **0.1642** | 0.0333 | 0.0285 | 0.6138 | 0.245 |
| 14 | **0.3199** | 0.0234 | 0.0548 | 0.6505 | 0.828 |
| 16 | 0.3496 | 0.0218 | 0.0598 | 0.6617 | 0.941 |
| 17 | 0.3367 | 0.0226 | 0.0610 | 0.6607 | 0.919 |
| **18** | 0.3442 | 0.0221 | **0.0670** | **0.6653** | **0.981** ⭐ |
| 19 | **0.3544** | **0.0217** | 0.0567 | 0.6625 | 0.931 |

**4-metric pick: epoch 18.** Per-metric argmax: `pearson_delta` and `mae` → epoch 19;
`overlap_at_N` and `disc_l1` → epoch 18. So the composite is a genuine 2–2 split, not a
rubber-stamp of one metric.

### Reading

1. **⛔ Epoch 12 → 13 → 14 goes 0.3387 → 0.1642 → 0.3199.** A **0.175** collapse and recovery
   between consecutive epochs, with epoch 13 landing near the `meanpert` anchor (0.1324). Epoch
   3 → 4 does the same at smaller scale (0.2764 → 0.2149), and 9 → 10 again.
2. **This is NOT split noise.** Across the 5-split arms the between-split sd is 0.002–0.003, so
   a single split sits within ~0.003 of its 5-split mean. **A 0.175 swing is two orders of
   magnitude larger.** The oscillation is real model variation between epochs.
3. **⇒ F7 is worse than recorded.** F7 documented a ±0.20 oscillation in the VALIDATION metric
   and treated it as a selection nuisance. It is present at the same magnitude in the TEST
   metric, i.e. the checkpoints themselves swing — the model passes through genuinely bad states
   mid-training.
4. **⇒ Any single-checkpoint comparison between arms is partly a draw from this oscillation.**
   Every loss-term margin in this campaign — Q02-over-M06 (+0.0102), M03-over-L11 (+0.0264),
   the latent gains (+0.004…+0.010) — is small relative to a 0.175 inter-epoch swing. The
   RANKINGS may survive (all arms got one draw each under the same protocol) but no individual
   margin below ~0.01 should be read as a property of the loss term until replicated.
5. **⇒ F-sched1's confound is live and now plausible in the "lottery" direction.** More epochs
   = more draws against this oscillation, so a 60-epoch arm has a genuinely higher expected
   maximum than a 30-epoch arm at equal model quality. **That does not make E3's 0.3933 wrong,
   but it does mean the gain cannot yet be attributed to better convergence.**
6. **The trend is still rising at epoch 19** (0.3544, the best `pearson_delta` so far), so
   30 epochs is not obviously past the useful point for this arm.

⚠️ C1 is 19/30 and single-split. ⚠️ C2 (M06) and C3 (Q02) are unscored, and F-loss9/F-latent4
showed those bases behave differently, so this does not describe them.

## ⭐⭐⭐ F-sched1 — **60 epochs + `eta_min` 1e-5 improves ALL THREE bases, on nearly every metric. `E3` = 0.3933 is the new campaign best.**

**The largest single gain of the campaign, and the only intervention so far to improve all four
metrics at once.** Every loss-term result in this campaign was measured at a stopping point that
was too early.

**What is varied:** the LR schedule only — epochs 30 → 60, cosine floor `eta_min` 1e-7 → 1e-5.
Every loss term is bit-identical to its base, passed as overrides. **Held fixed:** arm s313
throughout, seed 42.

### Measured — test cell-eval, uncalibrated

| arm | = base | n | `pearson_delta` | Δ | `mae` ↓ | Δ | `ovl@N` | Δ | `disc_l1` | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| **`E3`** | **Q02** | 5/5 | **0.3933 ± 0.0027** | **+0.0150** | **0.0217** | **−0.0008** | **0.0817** | **+0.0043** | 0.7047 | **+0.0018** |
| **`E2`** | **M06** | 5/5 | **0.3867 ± 0.0021** | **+0.0186** | 0.0224 | **−0.0008** | 0.0750 | **+0.0047** | **0.7178** | **+0.0131** |
| **`E1`** | **M03** | 4/5 ⚠️ | **0.3752 ± 0.0022** | **+0.0073** | **0.0210** | **−0.0002** | 0.0707 | −0.0015 | 0.6800 | **+0.0085** |
| — | Q02 base | 5/5 | 0.3783 | — | 0.0225 | — | 0.0774 | — | 0.7029 | — |
| — | M06 base | 5/5 | 0.3681 | — | 0.0232 | — | 0.0703 | — | 0.7047 | — |
| — | M03 base | 5/5 | 0.3679 | — | 0.0212 | — | 0.0722 | — | 0.6715 | — |

### Reading

1. **`E3` at 0.3933 is the campaign best**, +0.0150 over Q02 and **+0.0211 above `experimental`
   0.3722**. ⚠️ `experimental` is replicate agreement, not a ceiling — beatable via the
   two-sided noise asymmetry — so this is NOT a claim of super-experimental accuracy.
2. **`E2` and `E3` improve ALL FOUR metrics simultaneously.** No other intervention in this
   campaign has done that: `dcos` traded `mae` for correlation, the latent losses traded
   `disc_l1`, the split traded `pearson_delta`, `cdel` lost everywhere. A dominant win means
   there is no decision to make — it is simply better.
3. **⇒ Every prior comparison was made at an undertrained stopping point.** This was one of the
   three pre-registered outcomes and it is the one that landed. The loss-term rankings are not
   invalidated — all arms were compared at the same 30 epochs — but their MARGINS are suspect,
   because a schedule worth +0.015 to +0.019 dwarfs most of them (Q02-over-M06 was +0.0102,
   M03-over-L11 +0.0264).
4. **Setup-dependence does NOT appear.** All three bases improve, unlike `delta_mag_weight`,
   the latent losses and `cdel`, each of which flipped or varied between bases. A schedule
   change is evidently a different class of lever from a loss term.
5. **`E2` gains most on `pearson_delta`** (+0.0186) and takes `disc_l1` to **0.7178**, second
   only to D03's 0.7263. **`E1` gains least** (+0.0073) and is the only one to lose a metric
   (`ovl@N` −0.0015). The ordering M06/Q02 > M03 matches the pattern that the
   direction-dominated regime responds more to extra supervision.

⚠️ **`E1` is 4/5 splits.** ⚠️ **Confounded by design, and this matters for what to do next:**
epochs and `eta_min` moved TOGETHER, so this does not say which did the work. 60 epochs also
doubles the pool `best_model.pth` selects over while the validation metric oscillates ±0.20
between consecutive epochs (F7) — so part of the gain may be extra draws at a lucky epoch rather
than a better optimum. **The checkpoint sweep (C1/C2/C3, 30 per-epoch checkpoints each) is the
experiment that separates those two explanations**, and it is already trained.

## ⭐⭐⭐ F-latent7 — **the latent is using ~5% of its dimensions.** Effective rank 32.9 / 512 on Q02, 24.6 / 512 on M03.

**Why measured.** VICReg's variance term is a HINGE, `max(0, γ − std_j)` — exactly zero until a
dimension collapses below γ. With the paper's γ=1 it would be permanently saturated or
permanently inert depending on our scale, and either way teach nothing. γ has to come from the
measured per-dim std, as every weight in this campaign has (F-loss5). The measurement then
answered a larger question than it was asked.

### Measured — 512 real HepG2 perturbations, 512-d latent

| | Q02 | M03 |
|---|---|---|
| per-dim std, mean / median | 0.5978 / 0.5514 | 0.5600 / 0.5270 |
| per-dim std, min / max | 0.0021 / 2.4032 | 0.0018 / 2.5120 |
| std p1 / p5 / p10 / p25 | 0.023 / 0.056 / 0.162 / 0.340 | 0.019 / 0.042 / 0.095 / 0.243 |
| dims below std 0.01 / 0.05 / 0.10 | 0.6% / 4.7% / 7.2% | 0.6% / 6.2% / 10.7% |
| **EFFECTIVE RANK** (cov eigenspectrum) | **32.9 of 512 — 6.4%** | **24.6 of 512 — 4.8%** |
| mean \|off-diagonal covariance\| | 0.0657 | 0.0651 |
| VICReg covariance term (Σ off-diag² / d) | 7.050 | 8.518 |
| VICReg variance term at γ=1 | 0.4346 | 0.4724 |
| VICReg variance term at γ=median | 0.1171 | 0.1374 |

### Reading

1. **⛔ Severe INFORMATIONAL collapse, which no previous diagnostic would have caught.**
   Effective rank is ~5–6% of the latent width. F-latent1 measured `delta_z_ratio` 0.160 with
   98.2% of the movement perturbation-specific and concluded, correctly, that the model has NOT
   collapsed to the identity and does NOT emit one shared shift. **Both remain true.** They are
   statements about the MAGNITUDE and the SHARED-vs-SPECIFIC split of the delta; neither
   constrains how many independent directions the specific part occupies. It occupies ~25–33.
2. **Dimensional collapse is mild; informational collapse is severe.** Only 0.6% of dims are
   genuinely dead (std < 0.01) and ~5–7% are below 0.05 — so the variance term has little to
   fix. The covariance term is the one with a target: the dims are alive but **heavily
   redundant**, which is exactly what effective rank 33 with 512 live-ish dims means.
   ⇒ **Expect `cov_weight` to matter and `var_weight` not to.** Pre-registering that now.
3. **Q02 has a higher effective rank than M03** (32.9 vs 24.6, +34%). Q02 = M06 + latent MSE,
   and it is the better arm (0.3783 vs 0.3679). Consistent with latent supervision buying
   representational spread — and a reason to expect VICReg to help most where latent
   supervision already does, i.e. on M06/Q02 rather than M03 (F-latent4's setup-dependence).
4. This is a plausible mechanism for **F-cdel2**: if the perturbation-specific signal lives in
   ~30 directions, the panel-mean delta is not a separable offset but overlaps the same small
   subspace — so subtracting it removes signal, which is what cdel did on all three bases.

### ⚠️ What this measurement CANNOT distinguish

`delta_z` and `z_hat` come out **bit-identical** in every statistic above. That is an artifact
of the probe, not a finding: it feeds one control profile per cell line (the HepG2 control
mean), so `z_hat = z_ctrl + delta_z` differs from `delta_z` by a CONSTANT, and a constant shifts
neither std nor covariance. In training the control varies per row, so the two genuinely differ
there. **Nothing here says which of the two is the better VICReg target** — that is what the
sweep has to answer.

## ⛔ F-cdel2 — centre-delta is negative on **all three bases**, and `disc_l1` falls on all three. Not bimodal here.

Updates F-cdel1, which had M03 only. `cdel` = `ctrl + (delta − mean_over_perturbations(delta))`.
Every row below is the SAME checkpoint, SAME drawn controls, SAME split, ONE forward pass — the
only difference is the transform.

| base | arm | n | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|
| M03 | nocal | 5/5 | **0.3679 ± 0.0017** | **0.0212** | **0.0722** | **0.6715** |
| M03 | +cdel | 5/5 | 0.2956 ± 0.0029 | 0.0220 | 0.0692 | 0.6504 |
| M06 | nocal | 5/5 | **0.3681 ± 0.0023** | **0.0232** | 0.0703 | **0.7047** |
| M06 | +cdel | 5/5 | 0.3081 ± 0.0020 | 0.0234 | **0.0750** | 0.6886 |
| Q02 | nocal | 5/5 | **0.3783 ± 0.0027** | **0.0225** | 0.0774 | **0.7029** |
| Q02 | +cdel | 4/5 ⚠️ | 0.3144 ± 0.0015 | 0.0228 | **0.0778** | 0.6825 |

| base | Δ `pearson_delta` | Δ `mae` | Δ `ovl@N` | Δ `disc_l1` |
|---|---|---|---|---|
| M03 | **−0.0723** | +0.0008 | −0.0030 | **−0.0211** |
| M06 | **−0.0600** | +0.0002 | **+0.0047** | **−0.0161** |
| Q02 | **−0.0638** | +0.0004 | +0.0004 | **−0.0204** |

**Reading.** `pearson_delta` loses 0.060–0.072 on every base, and **`disc_l1` falls on every
base** — the transform's own pre-registered signature was discrimination UP. It is wrong here,
consistently, not bimodally as on H1 (+0.1394 / −0.3335 across checkpoints there). The only
positive is `ovl@N` on M06 (+0.0047), which is small and not reproduced on M03.

⇒ **The perturbation-common component is load-bearing signal on this panel, not a removable
offset.** Consistent with `meanpert` scoring 0.1324. **cdel is not adoptable on any base.** The
H1 winner was `model_cal_cdel` — CALIBRATED — and calibration is off by user direction, so that
recipe is unreachable here by construction.

## ⭐ F-latent6 — `Q05` completes at 5/5 and **ties `Q02`**. The latent cosine and MSE are interchangeable on M06.

| arm | what | n | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|
| **`Q05`** | latent **cosine** 1× | 5/5 | **0.3786 ± 0.0028** | 0.0226 | 0.0761 | **0.7078** |
| **`Q02`** | latent **MSE** 1× | 5/5 | **0.3783 ± 0.0027** | **0.0225** | **0.0774** | 0.7029 |
| M06 | base | 5/5 | 0.3681 ± 0.0023 | 0.0232 | 0.0703 | 0.7047 |

**0.0003 apart on `pearson_delta`**, an order of magnitude inside their sds. Q05 takes `disc_l1`
(+0.0049), Q02 takes `ovl@N` (+0.0013) and `mae` (−0.0001). **Two different latent losses — one
position, one direction — reach the same place.** Combined with F-latent3's finding that the
cosine TARGET does not matter once the weight is scale-matched, the picture is that M06's
direction-dominated loss benefits from *any* latent supervision at roughly parity weight, and
the specific form is second-order.

## ⛔ F-recon4 — the split-reconstruction grid: **no arm beats its base on `pearson_delta`, on either base.** But `disc_l1` reaches its campaign maximum.

| arm | base | mode / delta target | n | `pearson_delta` | `mae` ↓ | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|---|
| — | M06 | **base** | 5/5 | **0.3681** | 0.0232 | 0.0703 | 0.7047 |
| — | M03 | **base** | 5/5 | **0.3679** | **0.0212** | **0.0722** | 0.6715 |
| `G10` | M06 | `delta_only` | 5/5 | 0.3607 ± 0.0026 | 0.0247 | 0.0691 | **0.7138** ⭐ |
| `G05` | M03 | `delta_only` | 5/5 | 0.3582 ± 0.0025 | 0.0227 | 0.0711 | 0.6979 |
| `G03` | M03 | `cmean` / cell | 5/5 | 0.3575 ± 0.0023 | 0.0224 | 0.0663 | 0.6871 |
| `G04` | M03 | `cmean` / group_mean | 5/5 | 0.3540 ± 0.0022 | 0.0225 | 0.0689 | 0.6931 |
| `G09` | M06 | `cmean` / group_mean | 4/5 ⚠️ | 0.3517 ± 0.0028 | 0.0249 | 0.0601 | 0.7039 |
| `G08` | M06 | `cmean` / cell | 5/5 | 0.3511 ± 0.0027 | 0.0247 | 0.0636 | **0.7127** |
| `G06` | M06 | `raw` / cell | 5/5 | 0.3499 ± 0.0025 | 0.0242 | 0.0653 | **0.7129** |
| `G07` | M06 | `raw` / group_mean | 4/5 ⚠️ | 0.3456 ± 0.0019 | 0.0237 | 0.0634 | 0.7020 |
| `G01` | M03 | `raw` / cell | 5/5 | 0.3424 ± 0.0033 | 0.0248 | 0.0632 | 0.6932 |
| `G02` | M03 | `raw` / group_mean | 5/5 | 0.3282 ± 0.0024 | 0.0240 | 0.0590 | 0.6922 |
| `G11` | M03 | `raw`/cell `w_delta` 3 | 4/5 ⚠️ | 0.3236 ± 0.0033 | 0.0248 | 0.0556 | 0.6778 |
| `G12` | M03 | `raw`/cell `w_delta` 10 | 4/5 ⚠️ | 0.2933 ± 0.0040 | 0.0304 | 0.0577 | 0.6110 |
| `G13` | M03 | `raw`/cell `w_delta` 30 | 2/5 ⚠️ | 0.2706 ± 0.0066 | 0.0371 | 0.0501 | 0.5730 |

**Reading.**
1. **Setup-dependence does NOT appear here** — unusually. The split hurts `pearson_delta` on
   both bases by a similar margin (M06 −0.0074 at best, M03 −0.0097 at best), unlike
   `delta_mag_weight` and the latent losses which flipped sign between them.
2. **`disc_l1` reaches its campaign maximum in the split arms**: `G10` **0.7138**, `G06` 0.7129,
   `G08` 0.7127 — all above every non-split arm including D03's 0.7263… ⚠️ correction: D03 is
   0.7263 and remains the maximum. The split arms are the best on the M06 base specifically.
3. **`delta_only` is the best split mode on BOTH bases** (G10, G05) — dropping `L_ctrl` entirely
   beats every weighting of it. The control-reconstruction term is not merely mis-weighted; it
   is not wanted.
4. **The ratio ladder is monotonically destructive**, and steeply: 0.3424 → 0.3236 → 0.2933 →
   0.2706 as `w_delta` goes 1 → 3 → 10 → 30, with `mae` degrading in lockstep (0.0248 → 0.0371).

⇒ **No split arm is adoptable.** F-recon1's decomposition was real and measured; every
intervention built on it has cost `pearson_delta`.

## ⛔ F-cdel1 — centre-delta on M03: **all four metrics worse**, including `disc_l1`, which its own documentation predicted would RISE

**First cdel measurement ever on this panel.** `cdel` does not exist in the Occam repo; it was
ported for this test from the research tree's `virtual_cell/h1_eval.py::_center_delta`, where it is part
of the H1 shipping recipe (`model_cal_cdel_a1`, F294).

**What it does.** `ctrl + (delta − mean_over_perturbations(delta))` — removes the component of
the predicted effect identical across every perturbation, leaving only what distinguishes them.
Verified before use: shared component removed (‖panel-mean delta‖ 19.62 → 8.4e-07),
perturbation-specific part preserved (max diff 7.75e-07), <5-perturbation guard returns input
unchanged, and the mean is over PERTURBATIONS not rows (an abundant perturbation cannot define
the centre).

**Pre-registered signature**, from the research tree's own note: *"`pds` UP, `nmae`/`mse` DOWN"*
— i.e. discrimination should improve at the cost of magnitude accuracy.

### Measured — M03, test, uncalibrated, 5/5. Same checkpoint, same drawn controls, same split, ONE forward pass

| arm | n | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|---|
| `M03` `model_nocal` | 5/5 | **0.3679 ± 0.0017** | **0.0212** | **0.0722** | **0.6715** |
| `M03` `model_nocal_cdel` | 5/5 | **0.2956 ± 0.0029** | 0.0220 | 0.0692 | 0.6504 |
| **Δ** | | **−0.0723** | **+0.0008** | **−0.0030** | **−0.0211** |

### Reading

1. **All four metrics degrade.** `pearson_delta` loses **0.0723** — the largest single-transform
   drop measured in this campaign, and more than the entire gap between M03 and the previous
   best L11 (0.0264).
2. **⛔ `disc_l1` FALLS (−0.0211), which falsifies the pre-registered signature.** The transform
   exists to strip the perturbation-common component so that only the discriminative part
   remains; discrimination was the one thing it should have bought. It did not.
3. **What that implies about the model.** If removing the panel-mean delta destroys
   discrimination as well as correlation, the shared component is not a constant offset sitting
   on top of perturbation-specific signal — the predictions are closer to *shared-component
   plus noise*, and the part that distinguishes perturbations does not survive on its own. That
   is consistent with `meanpert` scoring 0.1324 on this panel: the common component is a large
   fraction of what the model gets right.
4. **Why H1 differed.** There `model_nocal_cdel` was **+0.1394** on one checkpoint and
   **−0.3335** on others. This is the same bimodality, landing on the negative side. The H1
   winner was `model_cal_cdel` — **calibrated** — and calibration is now off by user direction,
   so the H1 recipe is not reproducible here by construction.

⚠️ **M03 only.** M06 and Q02 are still scoring. Three levers have already shown opposite signs
between M03 and M06 (F-loss9, F-latent4, F-recon3), so this does NOT predict them — and M06/Q02
carry a much larger `dcos` weight, which is the term that makes predictions perturbation-specific
in the first place.

## ⛔ F-recon3 — **F-recon1's PREDICTION IS FALSIFIED.** Splitting the reconstruction makes `mae` WORSE, not better, in every arm. Phase 2 is worse than phase 1.

⚠️ **This overturns a pre-registration that was mine.** F-recon1 measured that `smooth_l1` is
~99.8% control reconstruction on M03 and hypothesised: *"this is why `mae` has been immovable —
`mae` is a full-profile error dominated by the same term, while every lever tried acts on the
perturbation. It predicts that improving control reconstruction moves `mae`."* The split is the
first lever that touches that term directly. **`mae` moved the WRONG WAY in 5/5 arms.**

**M03 base reference:** `pearson_delta` 0.3679 · `mae` 0.0212 · `ovl@N` 0.0722 · `disc_l1` 0.6715.

### Measured — test cell-eval, uncalibrated, job `18217067`

| arm | mode / delta target | n | `pearson_delta` | Δ pΔ | `mae` ↓ | Δ `mae` | `ovl@N` | `disc_l1` |
|---|---|---|---|---|---|---|---|---|
| — | **M03 base (no split)** | 5/5 | **0.3679** | — | **0.0212** | — | **0.0722** | 0.6715 |
| `G05` | `delta_only` / group_mean | 2/5 ⚠️ | 0.3601 ± 0.0016 | −0.0078 | 0.0226 | **+0.0014** | 0.0718 | **0.6985** |
| `G03` | `cmean` / **cell** (phase 1) | 5/5 | 0.3575 ± 0.0023 | −0.0104 | **0.0224** | **+0.0012** | 0.0663 | 0.6871 |
| `G04` | `cmean` / **group_mean** (PHASE 2 PROPER) | 5/5 | 0.3540 ± 0.0022 | −0.0139 | 0.0225 | +0.0013 | 0.0689 | 0.6931 |
| `G01` | `raw` / **cell** (phase 1) | 5/5 | 0.3424 ± 0.0033 | −0.0255 | 0.0248 | +0.0036 | 0.0632 | 0.6932 |
| `G02` | `raw` / **group_mean** (phase 2) | 5/5 | 0.3282 ± 0.0024 | −0.0397 | 0.0240 | +0.0028 | 0.0590 | 0.6922 |

### Reading

1. **⛔ `mae` is worse in every arm**, by +0.0012 to +0.0036. The split gives the control term an
   explicit, separately-weighted gradient and `mae` still degrades. **F-recon1's mechanism was
   right — the decomposition is real and measured — but its PREDICTION was wrong.** Knowing
   which term dominates a loss does not mean re-weighting that term improves the metric it
   dominates.
2. **⛔ Phase 2 is WORSE than phase 1, on both modes**: `raw` 0.3282 vs 0.3424 (−0.0142),
   `cmean` 0.3540 vs 0.3575 (−0.0035). **Denoising the delta target HURTS.** This is the clean
   answer to the phase-2 question and it is negative — the single-cell target's noise was not
   the binding constraint, and removing it costs signal the model was using.
3. **`raw` is the worst mode despite being the most faithful.** Per-cell controls (−0.0255) lose
   to line means (−0.0104). Reconstructing a specific control cell is dominated by that cell's
   own noise, so `L_ctrl` spends its gradient on unlearnable variation — which is exactly why
   its balance was FLAT at 32% in F-recon2 while `cmean`'s decayed. **The flat balance was the
   warning sign, not the feature.**
4. **✅ One thing the split DOES buy: `disc_l1` rises in all five arms** (0.6871–0.6985 vs
   0.6715), +0.016 to +0.027. Separating control reconstruction from delta prediction improves
   perturbation IDENTIFIABILITY even as it costs correlation and magnitude accuracy.
5. The ordering is the reverse of validation, which put `delta_only` and `cmean`/`group_mean`
   top. Validation misranked again — the fourth instance (F-loss7, F-loss9, F-latent4).

⇒ **No split arm is adoptable on M03. The M03 base stands.** ⚠️ G05 is 2/5, the M06 arms
(G06–G10) and every ratio arm are still scoring, and F-loss9/F-latent4 both showed
setup-dependence — so M06 may differ and is NOT predicted by this table.

## ⭐⭐ F-recon2 — three split modes, all verified on the real streaming path. **Each gives a different `L_ctrl`/`L_delta` balance, and only `raw` is stable.**

**Why three.** The streaming path encodes every control cell ONCE into a pool and serves 512-d
embeddings, so there is no per-cell gene-space control. `L_ctrl = recon(D(z_ctrl), ctrl)` compared
44,728 genes to 512 latents and killed 10/12 arms of `18216421`. Three ways to supply a
gene-space reference, run separately at the user's direction.

### Measured — real config (`rhj_fewshot`, streaming, 3 epochs, weights 1:1, UNWEIGHTED terms)

| mode | control reference | ep1 `L_ctrl` / `L_delta` | ep3 | `L_ctrl` share | stable? |
|---|---|---|---|---|---|
| **`raw`** | per-cell control expression (5 GB pool) | 0.004643 / 0.009641 | 0.004457 / 0.009468 | **32.5% → 32.0%** | **yes, flat** |
| **`cmean`** | line control MEAN | 0.001566 / 0.005208 | 0.000352 / 0.005099 | **23.1% → 6.4%** | no, decays 4.4× |
| **`delta_only`** | none; denoised group-mean delta | — / 0.000384 | — / 0.000282 | 0% by construction | n/a |
| *(paired, smoke only)* | paired control cell | 1.847 / 0.139 | 0.265 / 0.166 | 93.0% → 61.5% | no, decays 7× |

### Reading

1. **`raw` is the only mode whose balance holds still** — 32/68 across all three epochs, ratio
   0.47 throughout. Reconstructing a SPECIFIC control cell stays hard, because the cell's own
   noise never becomes learnable. That makes it the only mode where a fixed weight means a fixed
   balance, and therefore **the only one where a ratio sweep does what a ratio sweep is for**.
2. **`cmean` self-balances to the delta.** `L_ctrl` decays 4.4× (0.001566 → 0.000352) while
   `L_delta` is flat, so the delta term rises 77% → 94% of the loss with no weight tuning.
   Decoding `z_ctrl` to the LINE MEAN is an easy target that the model solves early. This is the
   weighting F-recon1 wanted, obtained structurally rather than by a hyperparameter.
3. **The three modes are not variants of one experiment** — they impose 32/68, 6/94 and 0/100.
   Running them separately was the right call; a single "the split" result would have been an
   artifact of whichever reference was chosen.
4. `delta_only`'s `L_delta` is **13× smaller** than `cmean`'s (0.00038 vs 0.0051) because the
   group-mean target removes the single-cell noise the other modes fit.

### ⛔ Three wiring faults found by verifying, each of which would have produced a plausible wrong number

1. **The gate.** `cmean`/`delta_only` read `cmeans`/`gmeans`, loaded only when
   `_needs_group_means` fires — and `recon_split` was not in it. Both terms would have been
   SILENTLY ZERO. That is F-loss1 exactly. Now gated and in the refusal list.
2. **Wrong internal path.** I added raw collection to `_accumulate`, which serves ONLY
   `extra_dense` (the H1 pool). The main per-file scan appends to `pools` at the bottom of its
   loop and never calls it, so raw stayed empty for all three corpora. Caught by the row-count
   assertion, not by inspection. **All three `pools.setdefault` sites are now enumerated and
   covered**: `_accumulate`, the cache-hit branch (which REFUSES, since cached blobs hold
   embeddings only), and the main scan.
3. **An error message that asserted an unverified cause.** The first failure printed "the
   embedding cache was hit", which I had not established and which was false — the real cause
   was (2). Rewritten to report the actual row counts on both sides and name the invariant.

⚠️ `raw` costs a full re-scan of all 28,474 control cells (~75s) because the embedding disk cache
cannot serve dense rows, plus 4.74 GB CPU. `cmean` and `delta_only` are free.

## ⭐⭐⭐ F-recon1 — **`smooth_l1` is ~99.8% control RECONSTRUCTION and ~0.2% perturbation.** The prediction error is inherited autoencoder error, not perturbation error.

**Credit: this came from the user's question** — "pred and tgt don't have the same `c`, because
for pred, `c` would be the *reconstructed* control, which has errors on it too". That is exactly
right, and it overturns the answer I had just given.

**What I got wrong.** I had shown `smooth_l1(pred − c, tgt − c) ≡ smooth_l1(pred, tgt)` and
concluded a delta `smooth_l1` was a re-parameterisation, not a new loss. **That holds only when
the SAME `c` is subtracted from both sides.** The prediction's natural control is the model's own
`D(z_ctrl)`; the target's is the real control `c`. Then the residual is
`(pred − tgt) − (D(z_ctrl) − c)`, which is NOT `pred − tgt`. The identity does not apply, and the
size of `D(z_ctrl) − c` decides how different the two objectives are.

### Measured — 512 real HepG2 perturbations, present-gene mask, group-mean targets

| quantity | M03 | M06 |
|---|---|---|
| `‖D(z_ctrl) − c‖` — control RECONSTRUCTION error | **15.3394** | **10.7215** |
| `‖t − c‖` — true EFFECT size | 2.8838 | 2.8838 |
| **ratio reconstruction / effect** | **5.32×** | **3.72×** |
| `‖pred − t‖` — full-profile error, what `smooth_l1` sees | 15.5781 | 13.0849 |
| residual, delta vs REAL control | 15.5781 *(≡ full, the identity)* | 13.0849 |
| residual, delta vs RECONSTRUCTED control | **3.8779** | **6.0310** |

As loss values:

| term | M03 | M06 |
|---|---|---|
| `smooth_l1(pred, t)` — the CURRENT loss | 0.009473 | 0.005531 |
| `smooth_l1(D(z_ctrl), c)` — control reconstruction ALONE | **0.009453** | 0.002754 |
| `smooth_l1(pred − D(z_ctrl), t − c)` — the DELTA | **0.000509** | 0.001563 |

### Reading

1. **On M03 the reconstruction loss is 0.009473 and the control-reconstruction term alone is
   0.009453 — 99.8% of it.** The perturbation contributes essentially nothing to the gradient
   that `smooth_l1` supplies. The delta term is **0.000509, about 5.4% the size**, and on M06
   28%.
2. **`‖pred − t‖` ≈ `‖D(z_ctrl) − c‖`** (15.58 vs 15.34 on M03). The model's full-profile error
   is almost entirely **inherited autoencoder reconstruction error**, not perturbation error.
   Measured against the reconstructed control instead, the perturbation residual is **3.88**, a
   4× smaller number. **The perturbation head is substantially better than the full-profile
   error makes it look.**
3. ⚠️ **This probably explains why `mae` has been immovable.** Cell-eval's `mae` is a
   full-profile error on the predicted rows, so it is dominated by the same reconstruction term.
   Every lever tried so far — `dcos`, `delta_mag`, the latent losses — acts on the
   PERTURBATION, which is ~0.2–28% of that quantity. **That is a hypothesis, not a measured
   claim**: it predicts that improving control reconstruction moves `mae` and little else, and
   that has not been tested.
4. This is why the F-loss5 picture held: with `dcos` at D03's weight the reconstruction term is
   2.1% of the total loss, and 99.8% of THAT 2.1% is control reconstruction. The perturbation
   was being supervised almost entirely by `dcos`.

### ⇒ The proposal this justifies: SPLIT the reconstruction loss

Replace the single `smooth_l1(pred, tgt)` with two separately-weighted terms, as the user
suggested:

```
L_ctrl  = smooth_l1(D(z_ctrl), c)                 # can the model pass the control through?
L_delta = smooth_l1(pred - D(z_ctrl), tgt - c)    # is the predicted CHANGE right?
```

Equal weighting would give the delta term ~18.6× more relative pull on M03 than it has today.
The two are not redundant and neither is a re-parameterisation of the current loss.

⚠️ **Costs one extra decoder forward per step** (`D(z_ctrl)`), which the current loss does not
need. ~1 extra decoder pass on a 1024-wide MLP — cheap relative to the encoder, but not free.

### ⭐ MEASURED on the REAL training path, single-cell targets — and the ratio MOVES

Split implemented and smoke-verified end to end (`recon_split=true`, both weights 1.0, 3 epochs,
`rhj_smoke`). The two terms, UNWEIGHTED, per epoch:

| epoch | `L_ctrl` | share | `L_delta` | share | **ratio `L_ctrl`/`L_delta`** |
|---|---|---|---|---|---|
| 1 | 1.847306 | 93.0% | 0.138828 | 7.0% | **13.31×** |
| 2 | 1.458650 | 91.1% | 0.141990 | 8.9% | **10.27×** |
| 3 | 0.264932 | 61.5% | 0.165586 | 38.5% | **1.60×** |

**⚠️ THIS BREAKS THE FIXED-WEIGHT SCALE-MATCHING RECIPE.** Every scale-matched weight in this
campaign (`nb_scale`, `dcos` 0.01, `delta_mag` 0.01363, the latent parity weights) was a
CONSTANT chosen from a one-off measurement, and that was sound because those ratios were stable.
Here the ratio moves **8×** in three epochs: `L_ctrl` falls 1.85 → 0.26 as the autoencoder learns
to pass the control through, while `L_delta` is roughly flat (0.139 → 0.166). A weight that
balances the two at epoch 1 leaves the delta term 8× over-weighted by epoch 3.

Two readings, and they imply different designs:
1. **The split self-balances.** Because `L_ctrl` decays and `L_delta` does not, equal weights
   (1.0 / 1.0) automatically hand the delta term more influence as training proceeds — which is
   the schedule one would want. Then no scale-matching is needed and the sweep should be over
   the RATIO of weights, anchored at 1:1.
2. **It needs a schedule.** If the end-state balance is what matters, the weight should be set
   from the CONVERGED ratio, not the initial one.

⚠️ These numbers are from the `rhj_smoke` config — 720 cells/epoch over 3 epochs, tiny — so the
absolute values and the convergence point will differ on the real corpus at 30 epochs. **The
trend (L_ctrl decaying toward L_delta) is the transferable part; the 1.60× endpoint is not.**
The first real arm should be read for its own trajectory before any weight is fixed.

⚠️ The same run reports `delta_z_ratio=0.0002`, i.e. a latent collapsed to the identity. Expected
for 3 epochs on 720 cells; noted so it is not mistaken for a finding.

### ⛔ A REPORTING FAULT FOUND WHILE VERIFYING THIS

The first split smoke ran clean and printed NOTHING, because the `[recon-split]` print — and the
`[latent]` print added earlier the same session — both sat inside `if wandb_run:`. Sweep and
competitor runs set `WANDB_MODE=offline`, so **every latent diagnostic reported as "now printed
to stdout" was in fact invisible in every run since**. Both now print unconditionally, verified
by comparing their indentation against the enclosing `wandb_run` block rather than by eye. This
is the same wandb-only-diagnostics failure F-latent1 recorded, reintroduced by me.

⚠️ **Confound to control for:** at `target_pb_k=1` the target is a single cell whose noise is
3–70× the effect, so `L_delta` would be fitting mostly noise. It should be swept against the
group-mean target, or `L_delta` should use the precomputed group means the way `dcos` and
`delta_mag` already do.

## ⭐⭐ F-latent2 — latent losses wired and verified; **`delta_z`, `z_hat` and `both` are ONE option, not three**, and the scale-match weights are measured

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line. This is infrastructure
for sweeping latent-space losses on the two preferred arms, **M03** (`pearson_delta` 0.3679,
`mae` 0.0212, `ovl@N` 0.0722, `disc_l1` 0.6715) and **M06** (0.3681 / 0.0232 / 0.0703 / 0.7047).

**What was done.** (1) Added `latent_cos_weight` — `1 − cos(delta_z, delta_z_target)`, the
direction-only counterpart of the existing MSE latent losses and the latent analogue of the
gene-space `delta_cos_weight` that produced M03 (F-loss7). (2) DECLARED `latent_loss_type`,
`latent_loss_weight`, `latent_loss_alpha` and `latent_cos_weight` in `configs/rhj_fewshot.yaml`.
(3) Verified the wiring on a real checkpoint with real data. (4) Measured the term scales.

### ⛔ `latent_loss_type` has FOUR names and TWO behaviours

`delta_z`, `z_hat` and `both` (at ANY `alpha`) are **the same loss**:

```
delta_z − delta_z_target = delta_z − (z_target − z_ctrl) = (z_ctrl + delta_z) − z_target
                         = z_hat − z_target
```

The residuals are the same tensor, so the MSEs are the same function. Confirmed numerically:
`MSE(delta_z, delta_z_target)` = 3.0075335503, `MSE(z_hat, z_target)` = 3.0075333118,
`allclose` True, residual 2.4e−07 (float noise). Confirmed again on a trained checkpoint: M03
gives **0.600705 for both**.

⇒ **Sweeping `latent_loss_type` over {`delta_z`, `z_hat`, `both`} would burn three GPUs to
produce one number**, exactly the shape of F-loss1. Only **`projected`** is genuinely different
(it applies `latent_proj_head`). `latent_loss_alpha` is **inert** for the same reason — it
interpolates between two identical quantities.

### Wiring verification — on M03's checkpoint, 256 real HepG2 perturbations

| `latent_loss_type` | `latent_cos_weight` | MSE term | cosine term | verdict |
|---|---|---|---|---|
| `none` | 0.0 | 0.000000 | 0.000000 | ✅ both off |
| `none` | 1.0 | 0.000000 | **0.973322** | ✅ cosine works WITHOUT the MSE |
| `delta_z` | 0.0 | **0.600705** | 0.000000 | ✅ MSE alone |
| `delta_z` | 1.0 | 0.600705 | 0.973322 | ✅ independent |
| `z_hat` | 1.0 | 0.600705 | 0.973322 | ⚠️ identical to `delta_z`, see above |

Each term is zero exactly when its knob is off and non-zero exactly when on, and the cosine
moves the total loss by `w × 0.973322`. **This is the check F-loss1 did not have.**

### Measured scales — what the sweep weights must be built from

`smooth_l1` here is the unmasked gene-space term, used as the common yardstick.

| base | gene `smooth_l1` | latent MSE | latent cos | `latent_loss_weight` for parity | `latent_cos_weight` for parity |
|---|---|---|---|---|---|
| **M03** | 0.009436 | 0.6007 | 0.9733 | **0.0157** | **0.0097** |
| **M06** | 0.005708 | 1.2942 | 0.9654 | **0.0044** | **0.0059** |

### The cosine TARGET matters — and unlike the MSE, the two are genuinely different

Cosine is not translation-invariant, so `cos(delta_z, delta_z_target)` and `cos(z_hat, z_target)`
are NOT the same loss (whereas their MSEs are). Added `latent_cos_target` ∈
{`delta_z`, `z_hat`}, default `delta_z`. Measured:

| base | `1−cos` on `delta_z` | on `z_hat` | ratio |
|---|---|---|---|
| M03 | **0.973322** | 0.052802 | 18.4× |
| M06 | **0.965420** | 0.118391 | 8.2× |

`z_hat` cosine sits at 0.05–0.12, i.e. `cos ≈ 0.9`, compressed by the control baseline every
perturbation shares — the same reason gene-space `cosine_loss_weight` on the absolute profile is
near-useless. It is not fully degenerate, so it is worth ONE arm, but `delta_z` carries far more
signal per unit weight.

### Full wiring matrix — 14/14 OK, both bases, 256 real HepG2 perturbations

Every term is zero exactly when its knob is off, non-zero exactly when on, the two knobs are
independent, and the ACTIVE `latent_cos_target` is the one that reaches the loss (asserted
against the per-target diagnostics, not assumed).

### ⛔ TWO WIRING FAULTS FOUND BY CHECKING, both fixed

1. **The knobs were NOT reachable.** Declaring them in `config.py` and `rhj_fewshot.yaml` was
   not enough: the config loader has its OWN routing table (`src/occam/config/schema.py`) which
   knew only `latent_loss_type`. `latent_loss_weight`, `latent_loss_alpha` and
   `latent_cos_weight` were rejected as unknown keys and the config would not load at all. The
   test suite caught it (3 failures). **This is the third distinct layer at which a loss term
   can be silently unreachable** — dataclass, YAML, loader schema — and F-loss1 was only the
   first of them. Verified by differencing a loaded config with and without `--set`:

   | key | default | after `--set` |
   |---|---|---|
   | `latent_loss_type` | none | delta_z |
   | `latent_loss_weight` | 0.0 | 0.016 |
   | `latent_cos_weight` | 0.0 | 0.01 |
   | `latent_cos_target` | delta_z | z_hat |

2. The CellFlow `emission_note` fix changed a vendored file without refreshing
   `competitors/MANIFEST.json`, the integrity baseline. Refreshed. **220 tests pass.**

Two things to carry into the sweep:
1. **The latent MSE is ~64–227× the gene `smooth_l1`**, and the latent cosine ~103–169×. A
   weight of 1.0 — the CURRENT DEFAULT for `latent_loss_weight` in `config.py` — would make the
   latent term ~99.5% of the loss. Nobody has ever run it, which is fortunate.
2. **M06's latent MSE is 2.15× M03's** (1.2942 vs 0.6007): its latent sits further from the
   encoded target. The parity weights therefore differ between bases by ~3.6×, so **a single
   weight grid cannot serve both** — each base needs its own.

## ⛔ F-loss9 — `delta_mag_weight` on M03's setup is **strictly harmful on all four metrics**. The M06 win was SETUP-driven, not weight-driven.

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line. User's goal: hold M03's
metrics and push `mae` below 0.0212.

**Question (pre-registered, F-loss8).** On M03's setup, does a larger `delta_mag_weight` push
test `mae` below 0.0212 while holding `pearson_delta` ~0.3679? And is M06's 28% `mae` win
weight-driven or setup-driven — `N01 ≈ M06` ⇒ weight, `N01 ≠ M06` ⇒ setup.

**Answer: no, and it is setup-driven.** `N01` carries M06's EXACT weight (0.01363) on M03's
setup and is worse than M03 on **all four** metrics.

**Varied:** `delta_mag_weight` ∈ {0.01363, 0.1363, 0.6815} = {1×, 10×, 50×} M06's.
**Held fixed:** `delta_cos_weight` 0.01 (M03's), arm s313 throughout, seed 42. Jobs `18215120`
(3/3 trained, 30 epochs, no errors) → `18215123`.

### Measured — test cell-eval, uncalibrated

| arm | `delta_mag_weight` | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ |
|---|---|---|---|---|---|---|
| **M03** (reference, mag off) | 0 | 5/5 | **0.3679 ± 0.0017** | **0.0212** | **0.0722** | **0.6715** |
| **N01** | 0.01363 (1×) | 5/5 | 0.3197 ± 0.0037 | 0.0230 | 0.0586 | 0.6264 |
| **N02** | 0.1363 (10×) | 4/5 ⚠️ | 0.2705 ± 0.0043 | 0.0248 | 0.0473 | 0.5726 |
| N03 | 0.6815 (50×) | 0/5 | pending | | | |

`N01 − M03`: `pearson_delta` **−0.0482** · `mae` **+0.0018 (worse)** · `overlap_at_N` **−0.0136**
· `disc_l1` **−0.0451**. Every metric degrades, and monotonically with weight through N02.

### Reading — and the mechanism that makes it coherent

1. **`N01 ≠ M06`, so the answer to the pre-registered question is SETUP.** Same weight
   (0.01363), different host arm: on D03 it cut `mae` 28% (0.0321 → 0.0232) for 0.0005
   `pearson_delta`; on M03 it costs 0.0482 `pearson_delta` and makes `mae` WORSE. The term is
   not a general `mae` lever.
2. **Why, mechanistically.** F-loss5 measured the loss shares. In **D03's regime `smooth_l1` is
   2.1% of the loss** — magnitude is essentially unsupervised, and `mag_loss` adds it back, so
   it helps. In **M03's regime `smooth_l1` is 51.5%** — magnitude is ALREADY supervised, so
   `mag_loss` is redundant, and its ratio-penalty gradient (which grows without bound as the
   ratio departs from 1) just distorts training. **`delta_mag_weight` is a repair for a
   direction-dominated loss, not an improvement to a balanced one.**
3. **⚠️ The magnitude ratio improved while every score got worse.** N01's ratio is **4.683**, the
   best of any arm measured (M03 6.584, D03 5.589, M06 5.872) — and N01 is worse on all four
   metrics. **This kills the magnitude-ratio probe as a selection signal.** It was already
   discredited in the other direction by F-loss8 (M06's ratio said null, test said −28% `mae`).
   It measures what it claims to measure; it simply does not predict test performance. Use it to
   understand a checkpoint, never to choose one.
4. **Validation agreed this time** (M03 0.4031 → N01 0.3378 → N02 0.2982 → N03 0.2740, monotone),
   unlike F-loss7 where it inverted the ordering. The distinction that held: F-loss7's arms
   differed by **0.008** on validation and reversed; these differ by **0.065** and did not. A
   validation gap inside ~0.01 is not evidence of ordering; a gap of 0.05+ has been reliable so
   far. ⚠️ n=2 episodes — a heuristic, not a rule.

⇒ **M03 stands as the preferred arm** (`pearson_delta` 0.3679, `mae` 0.0212, `ovl@N` 0.0722,
`disc_l1` 0.6715), with **M06** the alternative when `disc_l1` matters more than `mae`. The
magnitude term has no role on M03. `mae` below 0.0212 was not achieved by this route.

⚠️ N02 is 4/5 and N03 is still scoring. N03 (50×) is expected to be worse still on the monotone
trend, but that is a projection, not a measurement.

## ⭐⭐⭐ F-loss8 — the complete sweep: **`delta_mag_weight` DOES work on test (`mae` −28%)**, and `M03`/`M06` are a two-point Pareto frontier that both beat D03

⚠️ **SECOND CORRECTION TO F-loss6.** F-loss6 predicted `M06` would be a null, because the
magnitude-ratio probe showed 5.872 vs D03's 5.589 — no improvement. **On the test metric M06
cuts `mae` by 28%.** The probe was misleading, in exactly the way F-loss6 itself flagged as an
untested caveat: `scripts/loss_scale.py` feeds the model control MEANS while training feeds
single CELLS. **The probe's magnitude ratio must not be used to predict `mae` again.**

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line. User's goal for this
sweep: hold D03's metrics and improve `mae` massively.

**Varied:** `delta_cos_weight` ∈ {0.002 … 0.05} (M01–M05), plus M06 = D03's config +
`delta_mag_weight` 0.01363. **Held fixed:** arm s313 optimisation, seed 42, 30 epochs.
Jobs `18186699` (train) → `18187190` (test cell-eval, 30/30 COMPLETED).

### Measured — test cell-eval, UNCALIBRATED, all 5/5 splits

`pearson_delta` ↑ · `mae` **↓ (ERROR)** · `overlap_at_N` ↑ · `disc_l1` ↑ (chance 0.5)

| arm | `dcos` | share | `delta_mag` | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|---|---|---|
| L11 pure smooth_l1 | inert | 0% | 0 | 0.3415 ± 0.0031 | **0.0210** | 0.0642 | 0.6534 |
| M01 | 0.002 | 15.9% | 0 | 0.3457 ± 0.0022 | 0.0211 | 0.0658 | 0.6522 |
| M02 | 0.005 | 32.0% | 0 | 0.3553 ± 0.0030 | 0.0212 | 0.0598 | 0.6624 |
| **M03** ⭐ | **0.01** | 48.5% | 0 | **0.3679 ± 0.0017** | **0.0212** | **0.0722** | 0.6715 |
| M04 | 0.02 | 65.4% | 0 | 0.3642 ± 0.0026 | 0.0217 | 0.0703 | 0.6704 |
| M05 | 0.05 | 82.5% | 0 | 0.3624 ± 0.0021 | 0.0228 | 0.0693 | 0.6921 |
| D02 | 0.25 | 96.3% | 0 | 0.3661 ± 0.0023 | 0.0277 | 0.0568 | 0.7180 |
| D03 | 0.5 | 97.9% | 0 | **0.3686 ± 0.0021** | 0.0321 | 0.0718 | **0.7263** |
| **M06** ⭐ | 0.5 | 97.9% | **0.01363** | **0.3681 ± 0.0023** | **0.0232** | 0.0703 | **0.7047** |

### M06's deltas — the magnitude term in isolation

M06 differs from D03 by ONE setting, `delta_mag_weight` 0 → 0.01363:

| metric | M06 − D03 | reads as |
|---|---|---|
| `pearson_delta` ↑ | **−0.0005** | tie (sd 0.0021–0.0023) |
| `mae` ↓ | **−0.0090 = 28% BETTER** | **the objective, delivered** |
| `overlap_at_N` ↑ | −0.0016 | tie |
| `disc_l1` ↑ | −0.0216 | small real loss (0.7263 → 0.7047) |

**`delta_mag_weight` does what F-loss5 said it should**, on its first use in 541 runs: it buys a
28% `mae` reduction for a 0.0216 `disc_l1` cost, at no cost in `pearson_delta` or
`overlap_at_N`.

### The frontier — three non-dominated arms, and the choice is `mae` vs `disc_l1`

`pearson_delta` for M03 (0.3679), M06 (0.3681) and D03 (0.3686) is a **three-way statistical
tie** — the spread 0.0007 sits inside every arm's sd. So `pearson_delta` does not choose between
them, and the real axis is:

| arm | `mae` ↓ | `disc_l1` ↑ | what it is |
|---|---|---|---|
| **M03** | **0.0212** (best of the three) | 0.6715 (worst) | cheapest error, weakest identity |
| **M06** | 0.0232 | **0.7047** | **the balanced point** |
| D03 | 0.0321 (worst) | **0.7263** (best) | strongest identity, 51% worse error |

**M06 recovers 61% of D03's `disc_l1` advantage over M03 (0.0332 of 0.0548) while giving up only
0.0020 of `mae`.** On a joint reading it is the best single arm in the campaign.

### Reading

1. **`mae` is flat up to ~48% dcos share, then collapses**: 0.0210 → 0.0211 → 0.0212 → 0.0212 →
   0.0217 → 0.0228 → 0.0277 → 0.0321. It was never necessary to pay it; D03 paid 51% extra error
   for `disc_l1` alone.
2. **`disc_l1` rises monotonically with dcos share** (0.6522 → 0.7263) and is the ONLY metric
   that wants a dominant direction term. `delta_mag_weight` is the cheaper way to buy it back.
3. **`pearson_delta` peaks at the scale-match point and is then flat/slightly declining**
   (0.3679 at 48.5%, 0.3642, 0.3624, 0.3661, 0.3686). F-loss5's arithmetic
   (0.008443 / 0.796206 = 0.01060) predicted the peak before the sweep ran.
4. **Calibrated results agree that calibration still loses**, for all 9 arms: best calibrated
   `pearson_delta` is D02's 0.2993 vs 0.3686 uncalibrated, and every calibrated `mae` is
   0.089–0.112 against 0.021–0.032. Extends F-loss3 to the M-series.
   ⚠️ M06 is the WORST calibrated arm on `pearson_delta` (0.2711) despite being near-best
   uncalibrated — calibration and the magnitude term are doing the same job and conflict.

### Every anchor, for reading the above

`ctrlcell` −0.0018 · `ctrlmean` 0.0016 · `meanpert` 0.1324 · `transfer` 0.2323 ·
**best competitor TxPert 0.3463** · `experimental` 0.3722 (`mae` 0.0257, `ovl@N` 0.1942,
`disc_l1` 0.7821).

**M03 and M06 both beat every competitor**, and both now have a `mae` (0.0212 / 0.0232) BETTER
than `experimental`'s 0.0257 while `pearson_delta` sits 0.004 below it.

## ⭐⭐⭐ F-loss7 — **CORRECTION TO F-loss6.** On TEST, the scale-matched arm `M03` holds D03's `pearson_delta` and cuts `mae` by **34%**. Validation ranked these arms BACKWARDS.

⚠️ **THIS OVERTURNS F-loss6's CONCLUSION, WHICH WAS MINE AND WAS WRONG.** F-loss6 read
validation `best_metric` only, saw M03 (0.4031) below D03 (0.4108), and concluded "no hump —
the scale-matched arms are all worse, the direction term wants to dominate." On the test metric
the ordering is reversed and the sweep has a clear optimum at the scale-match point. This is
the documented failure mode of this project — validation and cell-eval ranking models
oppositely — and I walked into it after quoting it as a risk two findings earlier.

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line. The user's stated goal
for this sweep: *keep D03's performance on all metrics and improve `mae` massively*.

**Question.** With `dcos` brought down to `smooth_l1`'s scale, is there a weight holding D03's
`pearson_delta` 0.3686 while `mae` falls from 0.0321 toward L11's 0.0210?

**Answer: yes, at `delta_cos_weight` 0.01 — the exact scale-match point predicted by F-loss5.**

### Measured — test cell-eval, uncalibrated, job `18187190`

`pearson_delta` ↑ · `mae` **↓ (ERROR)** · `overlap_at_N` ↑ · `disc_l1` ↑ (chance 0.5)

| arm | `delta_cos_weight` | dcos share | n | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|---|---|---|
| L11 pure smooth_l1 | inert | 0% | 5/5 | 0.3415 ± 0.0031 | **0.0210** | 0.0642 | 0.6534 |
| M01 | 0.002 | 15.9% | 5/5 | 0.3457 ± 0.0022 | 0.0211 | 0.0658 | 0.6522 |
| M02 | 0.005 | 32.0% | 5/5 | 0.3553 ± 0.0030 | 0.0212 | 0.0598 | 0.6624 |
| **M03** | **0.01 (matched)** | **48.5%** | 5/5 | **0.3679 ± 0.0017** | **0.0212** | **0.0722** | 0.6715 |
| M04 | 0.02 | 65.4% | 4/5 ⚠️ | 0.3646 ± 0.0028 | 0.0217 | 0.0703 | 0.6705 |
| M05 | 0.05 | 82.5% | — | pending | | | |
| D02 | 0.25 | 96.3% | 5/5 | 0.3661 ± 0.0023 | 0.0277 | 0.0568 | 0.7180 |
| D03 | 0.5 | 97.9% | 5/5 | 0.3686 ± 0.0021 | 0.0321 | 0.0718 | **0.7263** |
| M06 = D03 + `delta_mag` | 0.5 | 97.9% | — | pending | | | |

### `M03` vs the two arms it has to beat

| metric | vs **D03** (0.3686 / 0.0321 / 0.0718 / 0.7263) | vs **L11** (0.3415 / 0.0210 / 0.0642 / 0.6534) |
|---|---|---|
| `pearson_delta` ↑ | **−0.0007** — a tie, inside the 0.0017–0.0021 sd | **+0.0264** |
| `mae` ↓ | **−0.0109, a 34% REDUCTION** | +0.0002 — a tie |
| `overlap_at_N` ↑ | **+0.0004** — a tie | **+0.0080** |
| `disc_l1` ↑ | **−0.0548** — the one real loss | **+0.0181** |

**`M03` strictly dominates `L11`** (better or tied on all four). **Against `D03` it converts a
0.0548 `disc_l1` loss into a 34% `mae` win**, holding `pearson_delta` and `overlap_at_N` level.
That is the stated objective, achieved on three of four metrics.

### Reading

1. **The optimum is interior and at the predicted place.** `pearson_delta` rises 0.3415 →
   0.3457 → 0.3553 → **0.3679** → 0.3646 across dcos shares 0% → 15.9% → 32% → 48.5% → 65.4%,
   peaking at the 50/50 scale-match point F-loss5 computed from the raw term magnitudes
   (0.008443 / 0.796206 = 0.01060). The arithmetic predicted the optimum before it was run.
2. **`mae` is flat at ~0.0210–0.0212 for every arm up to 48.5% dcos, then degrades sharply**
   (0.0217 at 65%, 0.0277 at 96%, 0.0321 at 98%). So `mae` is not traded away gradually — it
   survives intact until the direction term starts dominating, and then collapses. There was
   never a reason to pay it.
3. **`disc_l1` is the only quantity that genuinely wants a dominant direction term**
   (0.6534 → 0.6715 → 0.7180 → 0.7263, monotone in dcos share). If `disc_l1` is what a
   submission is scored on, D03 remains the arm; for every other metric M03 is at least as good.
4. **Validation `best_metric` ranked these backwards.** It put D03 (0.4108) above M03 (0.4031)
   and M01 (0.3771) level with L11 (0.3769), while test says M03 ≈ D03 ≫ M02 > M01 > L11.
   ⚠️ **Do not select DCOS arms on `best_metric`.** It tracks `disc_l1`-like direction quality,
   which is exactly the axis that keeps rising with dcos share.

⚠️ **M04 is 4/5 splits** and M05/M06 are still scoring. M06 is the arm that matters next: if
`delta_mag_weight` recovers `mae` while KEEPING D03's `disc_l1` 0.7263, it beats M03 outright.
F-loss6 measured M06's magnitude ratio as 5.872 (no better than D03's 5.589) and predicted a
null — **that prediction now carries the same validation-only weakness that F-loss6 itself
did**, so it should be checked against the test number, not trusted.

## ⛔ F-loss6 — ⚠️ **SUPERSEDED ON `pearson_delta`/`mae` BY F-loss7 — read that first.** The scale-match sweep and the first `delta_mag` run: neither reduces the 5.6× over-shoot, and VALIDATION gets no better

⚠️ **F-loss6's headline conclusion — "no hump, the scale-matched arms are all worse" — is WRONG.** It rested on validation `best_metric`, which ranks these arms BACKWARDS (F-loss7). On the test metric `M03` ties D03 on `pearson_delta` and cuts `mae` 34%. What survives from F-loss6: the magnitude-ratio measurements, and the fact that `delta_mag_weight` did not move the ratio.

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline split-half
paired `pearson_delta`. Anchors: `transfer` 0.2323 · L11 0.3415 (`mae` 0.0210) · D02 0.3661
(`mae` 0.0277) · D03 0.3686 (`mae` 0.0321) · `experimental` 0.3722 (`mae` 0.0257).

**Questions.** (a) With `dcos` brought down to `smooth_l1`'s scale, is there a weight that keeps
D03's direction metrics while recovering `mae`? (b) Does `delta_mag_weight` — never used in any
of 540 prior runs — reduce the measured 5.589× effect-size over-shoot?

**Varied:** `delta_cos_weight` ∈ {0.002, 0.005, 0.01, 0.02, 0.05} (M01–M05, scale-match point
0.01060), and one arm M06 = D03's config unchanged + `delta_mag_weight` 0.01363. **Held fixed:**
arm s313 optimisation throughout, seed 42. Job `18186699`, 6/6 COMPLETED.

**Pre-registered meaning.** A hump in the sweep ⇒ direction and magnitude are jointly learnable
and D03 was mis-scaled. A monotone fall as `dcos` shrinks ⇒ 0.3686 genuinely requires near-total
direction supervision. For M06: ratio → 1 ⇒ the term works and `mae` should follow.

### Validation `best_metric`, and the re-measured magnitude ratio

`MAGNITUDE RATIO` = `‖pred − ctrl‖ / ‖true − ctrl‖`, **1.0 = correct effect size**.

| arm | `delta_cos_weight` | dcos share | `delta_mag_weight` | epoch | val `best_metric` | **magnitude ratio** |
|---|---|---|---|---|---|---|
| L11 (reference) | inert | 0% | 0 | 25 | 0.3769 | 5.703 |
| M01 | 0.002 | 15.9% | 0 | 25 | 0.3771 | **6.170** |
| M02 | 0.005 | 32.0% | 0 | 27 | 0.3974 | — |
| M03 | **0.01 (matched)** | 48.5% | 0 | 25 | 0.4031 | **6.584** |
| M04 | 0.02 | 65.4% | 0 | 25 | 0.3981 | — |
| M05 | 0.05 | 82.5% | 0 | 25 | 0.4026 | — |
| D02 (reference) | 0.25 | 96.3% | 0 | 21 | **0.4100** | 5.513 |
| D03 (reference) | 0.5 | 97.9% | 0 | 13 | **0.4108** | **5.589** |
| **M06** | 0.5 | 97.9% | **0.01363** | 17 | 0.4078 | **5.872** |

### Reading — both hypotheses fail

1. **No hump. The scale-matched arms are all WORSE on validation than the dominant ones.**
   M03 at the exact 50/50 point scores 0.4031 against D03's 0.4108 and D02's 0.4100, and M01 at
   15.9% dcos collapses to 0.3771 — statistically indistinguishable from L11's pure-smooth_l1
   0.3769. Validation rises monotonically with dcos share across the whole range
   (0.3771 → 0.3974 → 0.4031 → ~0.40 → 0.4108). **The direction term wants to dominate**; there
   is no blend that buys `mae` back for free at the validation level.
2. **⛔ `delta_mag_weight` did NOT reduce the over-shoot — it slightly increased it.**
   M06's ratio is **5.872** vs D03's **5.589**, on the same probe with the same inputs. Its
   validation is also marginally below D03 (0.4078 vs 0.4108). The pre-registered "ratio → 1"
   outcome did not happen, at this weight.
3. **Lowering `dcos` makes the over-shoot WORSE, not better** (M03 6.584, M01 6.170 vs D03
   5.589). So the over-shoot is not something the direction term was causing — consistent with
   F-loss5, where L11 with DCOS inert was already at 5.703.

### ⚠️ Why M06's null must NOT yet be read as "`delta_mag_weight` does not work"

The term was definitely LIVE: `_needs_group_means` includes `delta_mag_weight` and **refuses to
start** if the mean table is absent (the F-loss1 guard), the resolved config records
`delta_mag_weight: 0.01363`, and M06's epoch trajectory differs from D03's. So this is not
F-loss1 repeating.

But two things could still explain a null at this weight, and neither is tested:
- **The weight may simply be too small.** It was set so the mag term equals the *dcos* term
  (0.398). Both are then ~44× `smooth_l1`, but `mag_loss` is a *ratio* penalty whose gradient
  w.r.t. the prediction shrinks as the ratio grows, so equal loss VALUE does not mean equal
  pull. A 10–50× larger weight is the untested region.
- **⚠️ PROBE CAVEAT, applies to every ratio in this table.** `scripts/loss_scale.py` feeds the
  model the control MEAN, while training feeds single CELLS. If the model's output magnitude
  responds to input noise, the absolute 5.6–6.6 is probe-dependent. **The cross-checkpoint
  comparison is still valid** — same probe, same inputs, same 512 perturbations — but the
  absolute value should not be quoted as "the model over-shoots by 5.6×" without repeating it
  on single-cell inputs. That has not been done.

**The decisive number is the test `mae` from job `18187190`** (30 tasks, chained, `CALIBRATE=both`),
not validation `best_metric` and not this probe. Recorded as pending, not inferred.

## ⭐⭐ F-comp1 — CellFlow completes the panel at **0.2154**, the ONLY competitor BELOW `transfer`, and it discriminates at near-chance

**Objective.** Predict a perturbation's effect in a cellular context where that perturbation was
never measured, from a single control cell. Headline: split-half paired `pearson_delta` on
`rhj_fewshot`, cell-eval 0.6.0, uncalibrated.

**Question this answers.** Where does the fourth and last competitor port sit, on the same
instrument as the other three and as our own arms — and does it clear the `transfer` bar that
says a model is earning its parameters?

**Varied:** the model. **Held fixed:** the 611-perturbation cohort, the 6,983-gene axis, the
seeded per-split half-B control draw, the emission space, the scoring path.

**Anchors:** `ctrlcell` −0.0018 · `ctrlmean` 0.0016 · `meanpert` 0.1324 · **`transfer` 0.2323** ·
`experimental` 0.3722. `disc_l1` chance = **0.5**.

### Measured — 5/5 splits, uncalibrated

| split | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| 0 | 0.2164 | 0.0250 | 0.0315 | 0.5457 |
| 1 | 0.2214 | 0.0252 | 0.0314 | 0.5407 |
| 2 | 0.2120 | 0.0251 | 0.0324 | 0.5455 |
| 3 | 0.2154 | 0.0251 | 0.0305 | 0.5460 |
| 4 | 0.2118 | 0.0251 | 0.0315 | 0.5399 |
| **mean** | **0.2154 ± 0.0039** | **0.0251** | **0.0315** | **0.5436** |

### The full competitor panel, uncalibrated, all 5/5 splits

| arm | `pearson_delta` | vs `transfer` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|---|
| TxPert (single-cell controls) | **0.3463 ± 0.0035** | +0.1140 | 0.0227 | 0.0582 | 0.6704 |
| PerturbDiff (genept) | 0.2449 ± 0.0038 | +0.0126 | 0.0287 | 0.0734 | 0.6650 |
| STATE | 0.2362 ± 0.0019 | +0.0039 | 0.0291 | **0.0849** | 0.6639 |
| **CellFlow** | **0.2154 ± 0.0039** | **−0.0169** | **0.0251** | 0.0315 | **0.5436** |

### Reading

1. **CellFlow is the only competitor that fails the `transfer` bar**, by −0.0169. `transfer` is
   the same perturbation measured in a *different cell line* — a model below it is not earning
   its parameters on this task.
2. **Its `disc_l1` is 0.5436 against a chance floor of 0.5.** The other three sit at 0.66–0.67.
   `disc_l1` asks whether a prediction can be matched back to WHICH perturbation produced it, so
   CellFlow is close to not distinguishing perturbations at all — while still scoring 0.2154 on
   `pearson_delta`, well above `meanpert` 0.1324. That combination is the signature `meanpert`
   exists to expose, and CellFlow is the first arm in this campaign to show it strongly.
3. **`mae` 0.0251 is the best of any competitor** and better than every DCOS arm of ours. A model
   that predicts something close to a shared average profile will do well on a magnitude error
   and badly on discrimination — consistent with (2). ⚠️ This is why `mae` must never be read
   alone.
4. **Val → test gap −0.029** (validation 0.2449 gene-space after the C3 fix → test 0.2154), the
   same direction and size as the other arms, so the selection metric behaved normally. The
   number is low because the model is, not because selection failed.

⚠️ **The C3 correction is what makes this readable at all.** Before it, CellFlow's selection
metric scored the prediction against a **PCA-reconstructed** ground truth and reported
`pearson_delta` 0.4027 with `overlap_at_N` 0.908. Scored against untouched targets it is 0.2449
validation / 0.2154 test. The inflated number was ~0.19 too high and would have made CellFlow
look like the best arm on the panel.

⚠️ Deliberate deviations are recorded per arm in `docs/rhj_competitors_results.md`; none of
these numbers reproduces the method's published figure, and CellFlow's PCA basis uses scanpy's
`random_state=0`, so all seeds share one representation.

## ⭐⭐⭐ F-loss5 — DCOS is ~100× LARGER than smooth_l1, so `delta_cos_weight` 0.5 means **98% direction, 2% magnitude** — and the model over-shoots the effect by **5.6×**

**Objective.** Best model on SEEN perturbations in an UNSEEN cell line; headline is split-half
paired `pearson_delta`. Anchors: `meanpert` 0.1324 · `transfer` 0.2323 · `experimental` 0.3722 ·
D03 **0.3686** · D02 **0.3661** · L11 0.3415.

**Questions this answers.** (a) What is the actual SCALE of the delta-cosine term relative to
`smooth_l1`, i.e. what does `delta_cos_weight` really buy? (b) Is there a second cosine term on
output counts? (c) Why is `mae` the one metric DCOS makes worse, and what would fix it?

**What is varied:** nothing — this is a measurement on three trained checkpoints. **Held fixed:**
inputs are the REAL HepG2 control means from `cache/pert_means/hepg2.npz` (encoder axis, log1p,
`target_sum` 2000, verified against the checkpoint's own range: ctrl max 3.709, group-mean max
4.105, prediction max 4.070), 512 real perturbations, present-gene mask (9,612 genes).

### (b) — There is exactly ONE cosine term, and it is NOT on counts

`pipeline.py:2106`: `dcos_loss = (1 - cosine_similarity(pred - ctrl_mean, group_mean - ctrl_mean))`,
computed on the **continuous log1p output**, on the **delta**, masked to present genes. No second
cosine on counts exists anywhere in D02 or D03, and none is added. The other optional terms
(`ranking_loss`, `gpl_loss`, `delta_reg`, `latent_loss`) are not cosines on counts either.

There is also a plain `cosine_loss_weight` on the ABSOLUTE profile (`expr_loss = (1-w)*recon +
w*cos`), set to 0.5 in these arms — but that one is the near-useless quantity F-loss5's comment
block warns about: the absolute profile is dominated by the shared baseline, so two predictions
that disagree entirely about the effect still score ~0.99 on it.

### (a) — Measured scale, per ELEMENT over the same present-gene mask

⚠️ Normalisation is the whole finding. Both terms must be per-element means over the same mask;
dividing a batch-summed error by a one-row gene count inflates `smooth_l1` by the batch size and
inverts the conclusion. (I made exactly that error first and reported DCOS as 0.2× — it is ~100×.)

| checkpoint | `smooth_l1` (per element) | `dcos` = 1−cos | **ratio** | share of loss at its OWN weight |
|---|---|---|---|---|
| **D03** (`dcos` 0.5) | 0.008443 | 0.796206 | **94.3×** | **97.9%** |
| **D02** (`dcos` 0.25) | 0.008036 | 0.833635 | **103.7×** | **96.3%** |
| L11 (`dcos` inert) | 0.009635 | 0.927644 | 96.3× | — (term was off) |

**`delta_cos_weight` is not a blend knob — it is an off-switch for `smooth_l1`.** At 0.25 the
reconstruction term is already down to 3.7% of the total; at 0.5 it is 2.1%; at 1.0 it is 1.0%.
The "weight" reads like a half-and-half mix and is nothing of the kind.

⇒ **This fully explains F-loss4's metric pattern.** `pearson_delta` (a correlation) and
`disc_l1` (identity) are direction quantities and improve. `mae` is a MAGNITUDE quantity and
degrades, because the term that supervised magnitude was switched off. It also explains why the
0.25 → 0.5 step changed so little (96.3% → 97.9%): both arms are already in the direction-only
regime, so the sweep from F-loss2 was flat for a structural reason, not a tuning one.

### (c) — The model over-shoots the perturbation effect by 5.6×, and cosine CANNOT see it

`‖pred − ctrl_mean‖ / ‖group_mean − ctrl_mean‖`, where **1.0 = correct size**:

| checkpoint | magnitude ratio | mean prediction | mean group-mean truth | mean control |
|---|---|---|---|---|
| D03 | **5.589** | 0.0646 | 0.0261 | 0.0263 |
| D02 | **5.513** | 0.0689 | 0.0261 | 0.0263 |
| L11 (no DCOS) | **5.703** | 0.0938 | 0.0261 | 0.0263 |

**The over-shoot predates DCOS** — L11, with the term inert, is the WORST at 5.703. So DCOS did
not cause it; DCOS just removed the only term that was pushing back on it. Cosine similarity is
scale-invariant by construction, so a direction-only loss is blind to a 5.6× magnitude error.

### ⇒ The lever for "keep D03's metrics, fix `mae`": `delta_mag_weight`, which EXISTS and has NEVER been used

`pipeline.py:2079` already implements exactly the missing term:

```python
mag_loss = ((‖pred − ctrl_mean‖ / ‖group_mean − ctrl_mean‖) − 1.0)**2
```

It is declared (`config.py:149`), wired into the total loss (`pipeline.py:2176`), gated into the
group-mean loader, and **set to 0.0 in all 540 runs on disk**. It penalises precisely the
quantity measured at 5.6 above, and it is orthogonal to DCOS: cosine fixes the direction, this
fixes the length of the same vector. Both read the same precomputed group means, so it costs no
extra forward pass.

⚠️ `mag_loss` is ~`(5.6−1)² ≈ 21` at the current over-shoot — **an order of magnitude larger
again than `dcos`**, so it needs its own scale-matched weight or it will switch BOTH other terms
off. The same trap this finding is about. A weight near `0.01–0.05` puts it in the same decade as
`dcos`; that is arithmetic from the numbers above, **not a measurement**.

**Not launched.** Proposed only.

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

**✅ RESOLVED 2026-09-19.** All 31 missing scores were re-run (job `18182488`, 21 tasks at
`%5`, then `18190848` for the one that lost the race again). **16/16 arms are now complete on
both calibration modes**, and `docs/rhj_scoreboard.md` has no `PARTIAL` model rows left (the
four that remain are the n=1 emission arms, which are single-split by deferral, not by this bug).

**The partial means were not materially wrong**, which is worth recording against the instinct
to distrust every earlier number: `L07`, the worst-affected uncalibrated arm at 4/5, moved
0.3098 → **0.3094** on completion (−0.0004), and its calibrated mean 0.2619 → **0.2612**. The
across-split sd of ~0.003 was a good guide to the error. The bug was still serious — it was
silent, and it made `D03` unreportable — but no ordering in F-loss3 changed.

**Two fixes shipped so it cannot recur silently:** `run.py` RAISES on an empty cell-eval result
instead of printing "skipping" and exiting 0; and it RETRIES up to 3 times first, since the race
is transient and each attempt draws a fresh segment name. The retry demonstrably works — a
backfill task logged `cell-eval succeeded on attempt 2/3` and completed.

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

---

## DECISION 2026-09-22 — the Q02 / 150 ep / eta_min 3e-4 schedule is now the DEFAULT

Y06's configuration (Q02 loss base, 150 epochs, `eta_min` 3e-4, batch 2048, lr and
decoder_lr 1e-3) is no longer an arm — it is what `configs/rhj_fewshot.yaml` ships. Adopted
by user decision after F-sched3 (this benchmark: **0.4043**, campaign best), F-sched4 (180
epochs flat, +0.0003) and F-nor6 (the same schedule is worth **+0.0840 pΔ** on
`norman_combo`, positive in 43/48 paired cells).

Full recipe, evidence and the procedure for a new dataset: **`docs/default_recipe.md`**.
Pinned by `tests/test_default_recipe.py`.

⚠️ Consequence for the grids in `configs/rhj_*.tsv`: rows that relied on the config's old
defaults (30 epochs, `eta_min` 1e-7, batch 256, decoder_lr 3e-4, all three loss weights 0)
now resolve differently and will hash to a NEW run directory instead of reusing the old one.
The completed runs are unaffected — each keeps its own `resolved_config.yaml`.

## ⏳ RB — Y06 batch-size ladder (LAUNCHED 2026-09-23, running)

**Objective.** Best model on unseen perturbations in HepG2 from one control cell; headline
`pearson_delta`, nocal, 5 splits. Anchors: meanpert 0.1324 · transfer 0.2323 ·
experimental 0.3722 · **Y06 (bs 2048) 0.4043**.

**Question.** Norman improved as batch size fell (NE25 bs 128 0.9003/0.7940 vs NQ02 bs 2048
0.8169/0.7380). Norman gets 27 steps/epoch at bs 2048, RHJ gets 298. Does smaller batch also
help on RHJ, where training is not step-starved?

**Varied:** batch size. **Fixed:** everything else in Y06 (Q02 loss, 150 ep, eta_min 3e-4,
lr 1e-3 not rescaled, seed 42). Grid `scripts/grids/rhj_batch.tsv` (rows byte-identical to
the Y06 seed-replicate row except batch_size).

**Outcomes.** Smaller batch beats Y06 → batch size is a general lever, not Norman step
starvation. Flat/worse → the Norman gain was step starvation; RHJ stays at 2048.

| arm | batch | steps vs Y06 | walltime | job |
|---|---|---|---|---|
| RB512 | 512 | 4× | 24 h | 18365066_0 |
| RB256 | 256 | 8× | 24 h | 18365066_1 |
| RB128 | 128 | 16× | 36 h | 18365067_2 |
| RB064 | 64 | 32× | 48 h | 18365068_3 |
| RB032 | 32 | 64× | 72 h | 18365069_4 |

Y06 took 2h06m at bs 2048. Run dirs `runs/rhj_fewshot_RB<bs>__..._bs<bs>_e150_em3e-4_s42`.
**Scoring is automatic:** `scripts/slurm/rhj_autoeval_daemon.sh` (job 18365082, cpu, 4 days)
submits `rhj_splithalf_models.sbatch --array=0-4%6 CALIBRATE=none` per arm when
`final_model.pth` appears. Results → `runs/rhj_splithalf/RB<bs>__<tag>/`.

## QOS routing (2026-09-23, user rule)

`hooke-predict` first → `vco` when hooke is full (cap 32 H100) → `normal` only for jobs with a
time limit ≤ 4 h, because `normal` is preempted (REQUEUE) by hooke-predict/vco.
Applied by hand to all pending jobs on 2026-09-23 (19 to vco, 3 Norman eval arrays to normal),
then enforced by `scripts/slurm/qos_rebalancer.sh` (job 18365346, cpu, 4 days, polls 5 min,
logs every move to runs/slurm/qos_rebalance_*.out).


## Baselines at control-draw seeds 43–46 (launched 2026-09-23)
`scripts/repro/baselines.sh` now takes `SEED` (the half-B control draw, rng = seed + 1000·split;
moves ctrlcell / meanpert / transfer, not ctrlmean / experimental). Jobs 18365483–86 →
`runs/rhj_splithalf/reference_s4N`. Seed 42 = the published `runs/rhj_splithalf/reference`.

### Retries 2026-09-23 (pdex /dev/shm race, "cell-eval returned no metrics" after 3 attempts)
Resubmitted exactly the failed splits: baselines s43 split 4 (18368323), s44 split 3 (18368324),
s45 split 4 (18368325), s46 splits 1,3,4 (18368326); PerturbDiff genept s44 splits 0,1,3
(18368321), s46 split 1 (18368322); RB512 split 1 (18368327).

## F-infra1 ROOT CAUSE and FIX (2026-09-23)

**Cause:** systemd-logind on the hop nodes runs with `RemoveIPC=true` (live value via
`busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager
RemoveIPC` → `b true`; logind.conf is silent, so the default applies). When a user's last login
session on a node ends, logind deletes every /dev/shm object that user owns. Slurm jobs are not
login sessions (`UsePam=no`), but every `ssh <node>` is — so one monitoring ssh wipes the live pdex
segment (`/psm_*`) of every cell-eval on that node ~10–20 s after logout, and torch DataLoader
`/torch_*` segments too. More jobs per node = more victims per wipe. Reproduced on hop01. Not the
CPython resource-tracker bug (pdex workers are forked and share the parent's tracker).

**Fix (live, default ON):** `scripts/cell_eval_shmsafe.py` runs the same cell-eval CLI with pdex's
SharedMemory replaced by an anonymous MAP_SHARED mmap inherited by the forked workers (no /dev/shm
name to delete); `src/occam/decoder_train/metrics.py::run_cell_eval` invokes it with the jupyter
interpreter resolved from the `cell-eval` executable. Verified: real_de / pred_de / results /
agg_results byte-identical to stock cell-eval (anonymous and file backends, and through
run_cell_eval); stress test deleting /psm_* every 0.5 s: stock fails with the production error,
launcher exits 0 with identical outputs. `OCCAM_CELLEVAL_SHMSAFE=0` restores stock.

**Rules until an admin sets `RemoveIPC=no`:** never `ssh` into a node running our jobs (use
`srun --jobid=<id> --overlap`, which opens no login session); torch DataLoader workers remain
exposed (PerturbDiff relaunched with `PD_WORKERS=0`).

### RB ladder — first result (2026-09-23): RB512 is WORSE than Y06 on pΔ and mae

Seed 42, 5 splits, nocal. `mae` ↓. Anchors: meanpert 0.1324 · transfer 0.2323 · experimental 0.3722.

| arm | batch | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|
| Y06 | 2048 | **0.4043 ± 0.0019** | **0.0215** | 0.0946 | 0.7288 |
| RB512 | 512 | 0.3878 ± 0.0028 | 0.0245 | **0.1006** | **0.7374** |

On RHJ, going from bs 2048 to 512 costs −0.017 pΔ and +0.003 mae, and gains +0.006 overlap and
+0.009 disc_l1 — the opposite of Norman, where smaller batches won on pΔ (F-sweep1). Consistent
with the step-starvation reading (Norman 27 steps/epoch at bs 2048, RHJ 298). n = 1 seed.

Status: RB256 finished training (scoring via the ledger); RB064 epoch 43, RB032 epoch 30.
⚠️ RB128 HUNG at epoch 37 from 06:06 (its dataloader /dev/shm on hop21 was wiped by an ssh logout
at ~05:31 — F-infra1; NF32's earlier hang on the same node has the same cause). Cancelled, run dir
archived, relaunched from scratch: job 18368869 (36 h).

### F-seed-Y06 (2026-09-23 15:30). Y06 across seeds: seed 42 was a favourable draw

nocal, 5 splits per seed, `mae` ↓. Seed 43 hung and is being retrained (job 18385637).

| seed | pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|
| 42 | 0.4043 | 0.0215 | 0.0946 | 0.7288 |
| 44 | 0.3980 | 0.0216 | 0.0946 | 0.7223 |
| 45 | 0.3929 | 0.0226 | 0.0835 | 0.7249 |
| 46 | 0.3859 | 0.0226 | 0.0834 | 0.7179 |
| **mean ± SD (4 seeds)** | **0.3953 ± 0.0077** | 0.0221 | 0.0890 | 0.7235 |

The seed SD of pΔ (0.0077) is 4× the split SD (0.0019) and larger than every gap among the
rhj recipe arms (AA03 0.4044, AA02 0.4035, Y06 s42 0.4043, Y03 0.4015, RB256 0.3978): the single-
seed ranking of those arms is NOT resolved. Y06 stays above TxPert (0.3463) and `experimental`
(0.3721) at every seed. RB256 (bs 256, seed 42): 0.3978 / mae 0.0217 / ovl 0.0989 / dl1 0.7291.

Y06 seeds on the unseen-cell-line panels (pΔ, seeds 42 / 44 / 45 / 46): h1_seen 0.0925 / 0.0930 /
0.0972 / 0.0972; k562_seen 0.1716 / 0.1677 / 0.1770 / 0.1728; k562_seen_h1set 0.1630 / 0.1599 /
0.1637 / 0.1626 — stable across seeds.

⚠️ NF42 and S43_Y06 hung (dataloader stall, no output for ~4 h, timed out at 6 h on hop50 /
hop21) with no ssh from me after 05:31; relaunched unchanged (num_workers is part of the
determinism contract): NF42 18385636, S43_Y06 18385637.

### AA03 seed replicates launched (2026-09-23, user request)
Y06 stays the official rhj run. AA03 (Q02 base, 180 ep, eta_min 7e-4; 0.4044 at seed 42, a tie
with Y06) at seeds 43–46 to see its seed behaviour: `scripts/grids/rhj_seeds_aa03.tsv`, job
18386809; scored on rhj_fewshot by the ledger.

### 2026-09-24: Y06 and AA03 complete at 5 seeds × 5 splits (F-seed-Y06, F-seed-AA03)

rhj_fewshot, nocal, from `data/release/final/scores_by_seed_split.csv`. Per-seed split-mean pΔ:

| arm | s42 | s43 | s44 | s45 | s46 | mean (sd) pΔ ↑ | mae ↓ | overlap_at_N ↑ | disc_l1 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Y06 (published: 150 ep, eta_min 3e-4) | 0.4043 | 0.3976 | 0.3980 | 0.3929 | 0.3859 | **0.3957 (0.0068)** | 0.0220 | 0.0885 | 0.7233 |
| AA03 (180 ep, eta_min 7e-4) | 0.4044 | 0.4028 | 0.3985 | 0.3961 | 0.3954 | **0.3994 (0.0040)** | 0.0222 | 0.0952 | 0.7316 |

AA03 − Y06 = +0.0037 pΔ, paired by seed 5/5 ≥ 0 (differences +0.0001 / +0.0052 / +0.0005 /
+0.0032 / +0.0095); overlap_at_N +0.0067, disc_l1 +0.0083, mae +0.0002 (worse). Y06 stays the
official run (user, 2026-09-23). Y06 OOD at 5 seeds: h1_seen 0.0939 (0.0032), k562_seen 0.1713
(0.0039), k562_seen_h1set 0.1609 (0.0034).

Paired AA03 − Y06 over seeds 42–46 (mean ± sd of the per-seed difference; seeds where AA03 is
higher): pΔ +0.0037 ± 0.0039 (5/5) · mae +0.0003 ± 0.0005 (4/5 higher = worse) · overlap_at_N
+0.0067 ± 0.0056 (5/5) · disc_l1 +0.0083 ± 0.0065 (5/5). AA03 had never been scored on the OOD
panels; its 15 units (h1_seen, k562_seen, k562_seen_h1set × 5 seeds) were registered and
submitted 2026-09-24 at the user's request for the comparison (jobs 18431111–125).
