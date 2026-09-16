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

**Status: PARTIAL.** One arm of the timing probe (job `18021925`, split 0, 2026-09-16). This
is the `ctrlmean` **anchor**, not a model: its prediction is the half-B control mean tiled
across every cell. It carries no biology by construction, so whatever it scores is what the
metric reads from the emission alone.

Configuration: split 0 of 5, cohort 611 perturbations (`86c91f938024b2a2…`), 18,575 rows +
2,488 half-B controls in the prediction file, 18,575 rows + 2,488 half-A controls in the
target, 6,983 genes (`ef64f2ba7ff7816e…`), `--allow-discrete`, `--batch-size 2000`,
`--num-threads 16`, cell-eval v1 profile `full`.

| metric | `ctrlmean` | what it says |
|---|---|---|
| `pearson_delta` | **0.0006** | a readable zero — see the correction below |
| `discrimination_score_l2` | **0.5040** | chance is 0.5. The anchor cannot tell perturbations apart |
| `discrimination_score_l1` / `cosine` | 0.5015 / 0.4970 | same |
| `overlap_at_N` | **0.0072** | 0.7% of the true DE genes |
| `de_direction_match` | **0.4883** | chance is 0.5 |
| `de_spearman_sig` | 0.4048 | |
| `de_sig_genes_recall` | 0.3080 | recovered by calling ~27× too many genes significant |
| `de_nsig_counts_pred` *(real 111.6)* | **3,026.8** | a 27.1× DE over-call, the zero-variance-prediction artifact |
| `mae` / `mae_delta` *(lower better)* | 0.0231 / 0.0231 | |
| `mse` / `mse_delta` *(lower better)* | 0.00158 / 0.00157 | |
| `pr_auc` / `roc_auc` | 0.0120 / 0.3902 | |
| `clustering_agreement` | 0.0011 | |
| `pearson_edistance` | −0.1202 | |

**Read every model number against this row.** On the H1 panel the `ctrlmean` anchor *beat*
every model arm on `pearson_delta`; here it sits at 0.0006 and at chance on discrimination,
which is what a floor should look like and means the protocol's B-control substitution is
doing its job.

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

## F2 — protocol cost

**Status: PENDING.** Job `18021925`, split 0. First arm (`ctrlmean`, cold, includes the
shared real-side DE) completed in ~20 min; the second (`ctrlcell`, warm, reads the real-side
DE from the target-file sha256 cache) is running. The decomposition
(`real DE = cold − warm`) and the 5-arm/5-split extrapolation land here when it exits.

The prior estimate being tested was 2.0–2.8 h per split, extrapolated across a 5× change in
panel size from the decoder-stage runs. That extrapolation is why the probe exists — the
same class of extrapolation put a 3 h wall limit on a 3.2 h training job (see
`runs/aborted_3h_limit/WHY.md`).

---

## Reading order for whoever picks this up

1. `data/benchmarks/rhj_fewshot/splithalf/CARD.md` — the frozen panel, and the two things
   the protocol cannot express.
2. `scripts/eval_splithalf.py --help` — the probe and the real run.
3. `scripts/paired_report.py --help` — the only valid summary, and the three arithmetic
   traps it closes.
4. This file — every measured number, with its configuration and its anchor.
