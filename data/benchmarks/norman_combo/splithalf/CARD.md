# `norman_combo` split-half evaluation panel

Built 2026-09-21 by `scripts/norman/norman_build_panel.py` v1.0.0 (`norman_combo/v1.0.0`).
**Frozen** — every number below is read back from the artifact, not from a plan.

## What this is for

**Overall objective.** Predict a perturbation's effect in a context where that exact
perturbation was never measured. `rhj_fewshot` tests an unseen perturbation in a seen cell
line. **This panel tests COMPOSITION:** train on single-gene perturbations, predict the effect
of a PAIR, where both genes were seen individually and the pair never was.

**The question this panel enables.** Scored against the same half-A reference, how does a model
fed half-B control cells compare with (a) the half-B perturbed cells themselves and (b) **the
sum of the two measured single-gene deltas**?

**What is varied.** Which cells land in half A and which in half B — 5 independent,
batch-stratified splits. **Held fixed:** the cohort, the 8,000-gene axis, the 20-pair validation
holdout, and the per-pair cell counts (`m(p) = floor(n_p/2)` depends only on `n_p`).

**What each outcome means.** Our model composes perturbations by **summing their graph
embeddings** (`model.py`, the `max_perts > 1` branch), so `sum_of_singles` is that
architecture's own hypothesis in closed form. **A model at or below it has learned nothing
about interaction.** Above it means something non-additive was learned. `experimental` is the
empirical reference and NOT a ceiling: at these counts both sides are noisy pseudobulks while a
low-variance predictor is attenuated on one side only, so a model can beat it from noise alone.

## The frozen panel

| | value |
|---|---|
| source | `data/eval/pert/norman_2019_k562.h5ad`, sha256 `c6cad008202666ef…`, 91,168 × 33,694, raw CSR int32 counts |
| upstream | GSE133344 (Norman et al. **2019**, *Science* 365:786), `filtered_matrix.mtx.gz` |
| cell QC | Norman's own filter: `good_coverage == True` ∧ `number_of_cells == 1` → 91,168 of 111,445 identified cells |
| gene axis | **8,000** genes, sha256 `6efa91af1af5c8b8…` — in Occam's vocabulary ∧ detected in >8% of QC cells, plus all 102 resolvable perturbation targets force-included |
| normalisation | row sum over all **33,694** file genes → ×2,000 → `log1p`, sum taken BEFORE axis restriction — matches `stream_v2.load_chunk_sparse` |
| panel rows | **38,623** (28,921 perturbed + **9,702** controls) |
| cohort | **111 of 111** eligible double pairs, `min_cells = 40`, 0 excluded |
| validation holdout | **20 double pairs**, stratified on (cell-count tercile × min-degree tercile), seed 42 |
| training set | **105 single-gene conditions**, 47,269 cells (+ the shared controls) |
| half size `m` | median **124**, min 23, max 502 · `m ≥ 20` **111** · `m ≥ 40` **103** · `m ≥ 60` 96 |
| per split | 14,432 perturbed per half, **4,851 controls per half** |
| control-half batch balance | max \|A−B\| count per batch = **1**, all 5 splits |
| baselines | `ctrlcell` · `ctrlmean` · `meanpert` (105 single conditions, equal weight) · **`sum_of_singles`** · `experimental` |

⚠️ **The cell line is K562, and the upstream preprocessed file we were handed says A549.** That
file (`norman_2016_a549_preprocessed.h5ad`) labels all 91,205 of its cells A549; its own counts
disagree — XIST detected in 75.9% of cells (female; A549 is male) and HBZ 28.0 / HBG2 17.2 /
GYPA in 87.9% (erythroid; A549 is lung adenocarcinoma). Norman 2019 is K562. This panel is
built from the GEO primary source and labels it **K562**. Full evidence in
`docs/norman_benchmark.md`.

## Why each choice

**`min_cells = 40`, i.e. `m ≥ 20`, and nothing is excluded.** All 111 test pairs clear it; the
median half is **124 cells**, 5.6× `rhj_fewshot`'s median of 22. A 2-cell pseudobulk has
enormous variance and the variance is *asymmetric* between arms, which is why `rhj_fewshot`
refuses `min_cells = 4`; here the question does not arise.

**The validation holdout is 20 DOUBLE pairs, not held-out singles.** F-ckpt4 measured that our
validation metric selects epochs well and misranks arms; the mitigation is that the validation
quantity must be the same KIND as the test one. A singles validation set would select
checkpoints on single-gene prediction and be blind to composition failure — the whole question.
Stratified on cell count and minimum gene degree in the pair graph, because an unstratified
draw of 20 from 131 can easily take the 20 smallest or 20 hub-gene pairs.

**The 8,000-gene axis force-includes the perturbation targets.** Norman is CRISPR**a**: the
perturbed genes are TFs that are OFF at baseline and get switched ON, so a
baseline-detection filter excludes exactly the genes being perturbed — **41 of 102 resolvable
targets fail the >8% detection rule**, including `FOXO4` (8.1%), `SNAI1` (5.4%), `PRDM1` (2.9%).
A benchmark that cannot see the activated target gene is not measuring the perturbation.
⚠️ Three targets (`C19orf26`, `C3orf72`, `KIAA1804`) are absent from GSE133344's GRCh37-era
symbol list; they remain usable as perturbation LABELS but their own expression is not scorable.

**All 33,694 genes are kept in the source file, not only the 8,000 scored.** The library size
must be the sum over everything the assay measured — exactly as `rhj_fewshot`'s sources keep
8,749–9,624 genes while its panel scores 6,983.

**Controls are shared between the training corpus and the panel** (9,702 cells, 4,851 per half),
as in `rhj_fewshot`, whose panel controls also came from the training file. The held-out
quantity is the PERTURBATION, not the control distribution.

**Seeded per (split, group)**, `blake2b(SPLIT_SALT | seed | split | group)` — the same recipe
and the same reason as `rhj_fewshot`: adding a pair to the cohort must not reshuffle any other
pair's halves.

**Batch-stratified halves, and here they genuinely work on both sides.** Every condition spans
all 8 gemgroups with a median of 31 cells per (condition, batch) and **zero singletons**
(`rhj_fewshot`: median 1, 53.4% singletons). Measured control-half imbalance: max 1 cell per
batch in all 5 splits.

## The `sum_of_singles` baseline is STRONG — read the per-pair numbers, not the mean

Whole-cohort pseudobulk check (`baseline_sanity.json`; NOT the 5-split paired protocol, which is
what the eval reports):

| baseline | pearson(Δ pred, Δ real) mean | median |
|---|---|---|
| **`sum_of_singles`** | **0.8912** | **0.9130** |
| `meanpert` | 0.5892 | 0.6968 |

magnitude ratio ‖Δpred‖/‖Δreal‖ median **1.073**.

**Additivity already explains most double perturbations**, so a model that beats
`sum_of_singles` by a small average margin may be doing nothing on the ~100 additive pairs and
everything on the ~10 non-additive ones. **Report per-pair, and report the non-additive stratum
separately.** The least additive pairs, which is where this benchmark's headroom actually lives:

| pair | `sum_of_singles` r | ‖Δpred‖/‖Δreal‖ | why it is interesting |
|---|---|---|---|
| `BAK1+BCL2L11` | **0.369** | 0.88 | both pro-apoptotic BCL2-family |
| `KIF18B+KIF2C` | 0.549 | 1.12 | redundant kinesins |
| `PLK4+STIL` | 0.665 | 0.92 | same centriole-duplication step |
| `CEBPE+KLF1` | 0.669 | **2.02** | the pair's real effect is HALF the sum — buffering |
| `OSR2+UBASH3B` | 0.707 | 1.75 | buffering |
| `OSR2+PTPN12` | 0.746 | 1.79 | buffering |
| `CEBPE+ZC3HAV1` | 0.758 | **2.42** | strong buffering |
| `C19orf26+TGFBR2` | 0.814 | 0.61 | synergy |

That these are same-pathway and same-family pairs — and that the magnitude ratios go the right
way — is independent evidence the panel's labels, axis and deltas are wired correctly.

## The two additivity strata — 37 / 37, and why they do NOT replace the cohort

`additivity.npz` scores every cohort pair by the **additivity residual**

    residual = ‖Δ_sum_of_singles − Δ_real‖ / ‖Δ_real‖        (0 = perfectly additive)

on the whole-cohort pseudobulk, and cuts it at terciles → **37 / 37 / 37**. `splits_manifest.json`
carries the memberships as named panels `all` · `additive` · `middle` · `nonadditive`.

**The residual, not the correlation.** Additivity fails two ways — wrong direction and wrong
size — and `pearson_r` only sees the first: spearman(r, |mag−1|) = **−0.255**, while
spearman(residual, |mag−1|) = **+0.740**. `CEBPE+ZC3HAV1` has r 0.758 yet the additive
prediction is **2.42× too large**; ranking on r would have filed it as additive.

| stratum | pairs | residual median | `pearson_r` median | ‖Δsum‖/‖Δreal‖ median |
|---|---|---|---|---|
| `additive` (lowest residual) | **37** | 0.373 | 0.938 | 1.03 |
| `middle` | 37 | 0.477 | 0.900 | 0.92 |
| `nonadditive` (highest residual) | **37** | 0.763 | 0.859 | **1.47** |

**The strata are biologically coherent, which is the independent check that they are not noise.**
`nonadditive` collects same-pathway redundancy (`KIF18B+KIF2C`, `PLK4+STIL`, `CDKN1A+CDKN1C`,
`CDKN1B+CDKN1C`, `MAP2K3+MAP2K6`, `BAK1+BCL2L11`) and the erythroid/MAPK synergy hubs
(`CEBPE+KLF1`, `CEBPB+CEBPE`, eight `MAPK1` pairs, six `CEBPE` pairs) — i.e. the epistasis
classes Norman's own paper describes. `additive` is dominated by pairs from unrelated pathways.
A gene is not intrinsically in one stratum: `ETS2+MAPK1` is additive while `IGDCC3+MAPK1` is not.

⚠️⚠️ **SELECTION CAVEAT — the strata are defined by `sum_of_singles`'s own failure.** The
baseline's score PER STRATUM is therefore **not an independent measurement**: it is low on
`nonadditive` by construction. Model-vs-model comparisons across strata are unaffected, and
model-vs-baseline on `all` is unaffected, but **"our model beats sum_of_singles on the
nonadditive stratum" is partly a restatement of how the stratum was drawn.** The honest claim is
a model that beats it on `all` *and* keeps its margin on `additive`, where the baseline is near
its best.

### ⛔ The strata are SUB-PANELS, not a replacement test set

Asked directly, and the answer is no. Using `additive` + `nonadditive` (74 pairs) as the test
set instead of all 111:

1. **It bakes a baseline-derived selection into every future number.** The cohort would be
   defined by where one particular predictor happens to fail — a moving target the moment the
   baseline changes, and not a property of the biology.
2. **It discards the middle tercile**, which is a third of the data and the part where neither
   "additive" nor "non-additive" is a safe description.
3. **It breaks comparability**: 111 is the number every arm, competitor and future paper would
   be read against, and there is no way to recover it from a 74-pair result.

`rhj_fewshot` sets the precedent — it stratifies by half size `m` and reports the strata
*alongside* the full cohort, never instead of it. Same here: **`all` is the headline, the two
strata are the diagnostic**, and they are where a composition claim is actually decided, because
the mean over 111 is dominated by pairs additivity already explains.

## Reading order

1. This file — the frozen panel and why each choice was made.
2. `docs/norman_benchmark.md` — the dataset analysis, the provenance correction, and the
   design proposal this artifact implements.
3. `splits_manifest.json` — the 5 splits, the val holdout, and the measured batch balance.
4. `cohort.json` — the frozen 111 pairs and their half sizes.
