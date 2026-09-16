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

## Reading order for whoever picks this up

1. `data/benchmarks/rhj_fewshot/splithalf/CARD.md` — the frozen panel, and the two things
   the protocol cannot express.
2. `scripts/eval_splithalf.py --help` — the probe and the real run.
3. `scripts/paired_report.py --help` — the only valid summary, and the three arithmetic
   traps it closes.
4. This file — every measured number, with its configuration and its anchor.
