# `rhj_fewshot` split-half evaluation panel

Built 2026-09-16 by `scripts/build_splithalf_panel.py` v1.0.0. **Frozen** — every number
below is read back from the artifact, not from a plan.

## What this is for

**Overall objective.** Report how far the model is from **experimental reproducibility** on
held-out HepG2 perturbations, rather than how far it is from an arbitrary floor.

**The question this panel enables.** Scored against the same half-A reference, how does the
model fed half-B control cells compare with the half-B perturbed cells themselves?

**What is varied.** Which cells land in half A and which in half B — 5 independent,
batch-stratified splits. **Held fixed:** the cohort, the gene axis, the checkpoint, the arm
set, and the per-perturbation cell counts (`m(p) = floor(n_p/2)` depends only on `n_p`, so
every arm in every split emits the same rows for the same perturbation).

**What each outcome means.** A positive gap (experimental arm better) is the expected
result and its size is the headline. A **negative** gap is not a triumph: at these counts
reference and experimental prediction are *both* noisy pseudobulks while a low-variance
predictor is attenuated on one side only, so a model can beat the "ceiling" from noise
alone. The count-stratified gap must **shrink** as `m` grows; a gap that does not shrink is
noise-dominated whichever sign it has.

## The frozen panel

| | value |
|---|---|
| source | `nadig_2024_hepg2.h5ad`, sha256 `f31a85d8c90f5a2e…`, 145,473 × 9,624 |
| gene axis | **6,983** genes, sha256 `ef64f2ba7ff7816e…` (the 3-way `var.ensembl_gene_id` intersection) |
| normalisation | full-panel row sum over all 9,624 file genes → ×2,000 → `log1p` — matches `stream_v2.load_chunk_sparse`, which takes `rowsum` *before* alignment |
| panel rows | **42,402** (37,426 perturbed + **4,976** controls) |
| cohort | **611 of 621** test perturbations, `min_cells = 10`; sha256 `86c91f938024b2a2…` |
| half size `m` | median **22**, min 5, max 233 |
| strata | `m ≥ 5` **611** · `m ≥ 10` **568** · `m ≥ 25` **278** |
| per split | 18,575 perturbed per half, **2,488 controls per half**, 276 odd cells dropped |
| control-half batch balance | max |A−B| count per batch = **1**, all 5 splits |
| mean-training-perturbation baseline | grand mean over **1,657** HepG2 training perturbations (95,527 cells, **equal weight per perturbation**) |

10 test perturbations are below `min_cells = 10` and are **not scored**: `CHMP3`, `COG3`,
`CPSF2`, `EIF3F`, `HAUS1`, `PRPF38B`, `RFFL`, `RPF2`, `TCP1`, `ZNF676`.

## Why each choice

**`min_cells = 10`, not 4.** At `min_cells = 4` you get `m = 2` and nothing errors — which
is why it is dangerous. A 2-cell pseudobulk has enormous variance, and the variance is
*asymmetric* between arms (the experimental arm is 2 raw cells; a mean-based prediction has
near-zero variance), so the "upper bound" stops being one exactly where it matters most.

**The cohort is frozen once and never intersected again.**
`discrimination_score_l2` is a rank over the cohort and BH-FDR is computed over the whole
flattened DE frame, so changing the membership silently rescales both. It is also
deliberately **not** intersected with the model's `pert2id` — that would make the benchmark
checkpoint-dependent. `protocol.py` refuses up front instead.

**Seeded per (split, perturbation)**, `blake2b(SPLIT_SALT | seed | split | group)`, never
one stream consumed in a loop: adding a perturbation to the cohort must not reshuffle any
other perturbation's halves.

**Batch-stratified halves.** For perturbed cells stratification cannot equalise composition
(median 1 cell per (pert, batch), 53.4% singletons) but it removes the worst case and is
free. For **controls it genuinely works and is mandatory** — 4,976 cells over 56 batches, and
an unstratified split leaves an `A_ctrl_mean − B_ctrl_mean` batch term that enters the real
and the predicted delta **with opposite signs**. Measured: max per-batch imbalance 1 cell.

**The training grand mean is precomputed**, not carried as cells: the 1,657 training
perturbations are needed for nothing else, and hauling their 95,527 cells through the panel
to average them once would have quadrupled its size (164 MB → ~600 MB).

## Two things this panel cannot express

1. **The experimental arm is not an upper bound** at `m` = 22. Call it *"experimental
   replicate agreement at m cells"*, report the count-stratified gap, and report
   `n_splits_arm_better` so "beat the ceiling on 3/5 splits" is visible rather than buried
   in a mean.
2. **`ctrlmean`'s prediction delta is zero by construction.** With B controls on the
   prediction side the `ctrlmean` prediction *is* the B control mean. Predicted NaN;
   **measured 0.000623** (job `18021925`, split 0) — the delta is zero only in exact
   arithmetic, and float32 rounding leaves ~1e-8 of noise to correlate, which is a readable
   zero rather than a hole in the table. Its `mae`, `overlap_at_N` and discrimination
   scores are meaningful throughout (`discrimination_score_l2` 0.5040 against a chance
   value of 0.5). **`ctrlcell` is the load-bearing floor** — its delta is the mean of `m`
   drawn B controls minus the full B control mean, i.e. pure control-sampling noise at `m`
   cells, which is the right floor and scales with `m`. Full table:
   `docs/rhj_splithalf.md` F1.

## Files

| file | contents |
|---|---|
| `panel.npz` | `X` (42,402 × 6,983 float32), `labels`, `batches`, the axis, `source_rows` |
| `cohort.json` | the 611 scored perturbations, cells per perturbation, `half_size`, why each drop |
| `splits.npz` | `{pert,ctrl}_{A,B}_{0..4}` + `dropped_*`, row indices into `panel.npz` |
| `splits_manifest.json` | seed recipe, per-half hashes, batch balance, strata, provenance |
| `train_grand_mean.npy` | (6,983,) float32 — the `meanpert` arm's profile |

Rebuild: `python scripts/build_splithalf_panel.py` (~1.5 min, re-verifies 10 invariants).
