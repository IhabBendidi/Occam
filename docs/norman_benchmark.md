# `norman_combo` — a double-perturbation benchmark, on the `rhj_fewshot` split-half protocol

**Status 2026-09-21: BUILT from the GEO primary source. Nothing trained, nothing uploaded.**

| artifact | path | state |
|---|---|---|
| benchmark source | `data/eval/pert/norman_2019_k562.h5ad` | ✅ 91,168 × 33,694 raw CSR int32, 0.65 GB, sha256 `c6cad008202666ef…` |
| scoring axis | `data/benchmarks/norman_combo/axis.npz` | ✅ **8,000** genes, sha256 `6efa91af1af5c8b8…` |
| frozen panel | `data/benchmarks/norman_combo/splithalf/` | ✅ 111 test pairs · 20 val pairs · 5 splits · `CARD.md` |
| `sum_of_singles` baseline | `…/sum_of_singles.npz` | ✅ built and sanity-checked (pearson 0.891) |
| multi-pert model path | `src/occam/vcell/model.py` | ✅ vectorised, 4 tests |
| scoring wired into `eval_splithalf.py` | — | ⬜ next |
| HuggingFace upload | — | ⬜ blocked on licensing, as the rest of the bundle is |

**Decisions taken by the user 2026-09-21:** rebuild from GEO · **8,000** genes (in Occam's vocab
∧ not mostly-zero) · `val_doubles` (20 pairs) · controls shared between train and panel, as in
`rhj_fewshot` · vectorise the multi-pert path.

⚠️ **The 4,762-gene axis from the preprocessed file is superseded** by the 8,000-gene axis built
here. The old file's 5,045 genes are a strict subset of GEO's 33,694, so the two are nested, but
no number should be compared across them.

## What we are trying to achieve overall

Predict a perturbation's effect in a context where that *exact* perturbation was never measured.
`rhj_fewshot` tests one form of that — an unseen perturbation in a seen cell line. **This
benchmark tests a different and harder form: COMPOSITION.** Train on single-gene
perturbations, then predict the effect of a PAIR of genes perturbed together, where both genes
were seen individually and the pair never was.

## What question this benchmark answers

Our model composes perturbations by **summing their graph embeddings**
(`model.py:450-458`, the `max_perts > 1` path). That is an architectural assumption: a double
perturbation is the sum of its parts in latent space. Norman's double perturbations are the
direct test of it, and the interesting cases are exactly the ones where biology is NOT additive
(synergy, buffering, epistasis).

**What each outcome means.** If double-perturbation prediction lands near the single-gene
level, additive latent composition is sufficient and the interaction module is doing real work.
If it collapses toward the `meanpert`/`transfer` floor while singles score normally, the sum is
the wrong composition operator — and that is a result about the architecture, not about the
data.

## The source, as copied

| | |
|---|---|
| source | `/rxrx/data/valence/biomodality_sc/norman_2016/norman_2016_a549_preprocessed.h5ad` |
| copied to | `data/eval/pert/norman_2016_a549_preprocessed.h5ad` |
| size · sha256 | 2,230,402,078 B · `891bdae71209ee49b14029026c538b05a220385c14ff6ea49f48e255ab919ffe` |
| shape | **91,205 cells × 5,045 genes** |
| `X` | CSR, **already normalised + log1p** by GEARS-style preprocessing (nnz range 0.33–7.69) |
| `layers/counts` | **DENSE float32**, integral, the raw counts — library size median **2,935** UMI (min 485, max 7,920), 403 genes/cell |
| `uns` | `rank_genes_groups_cov_all`, `top_non_dropout_de_20`, `non_dropout_gene_idx` — GEARS pertdata format |
| not copied | `data_pyg/cell_graphs.pkl_batch_0.pkl` (3.7 GB, GEARS cell graphs), `go.csv` (19 MB, GEARS GO graph), `splits/*.pkl` (GEARS simulation splits), `simulation_*_co_expression_network.csv` — none is used by our pipeline, and we build our own splits and graph |

## ✅ PROVENANCE SETTLED — it is Norman **2019**, GSE133344, and the cells are **K562**

The upstream file says `norman_2016` / `A549`. Both are wrong, and three independent lines of
evidence agree:

1. **The cohort is identical.** GSE133344's own `guide_identity` column parses to **105 single
   genes, 131 double pairs, 73 genes in doubles, all 73 with single arms, 8 gemgroups** — the
   same numbers as the preprocessed file, to the unit.
2. **The QC reproduces the cell count.** Norman's own filter (`good_coverage == True` ∧
   `number_of_cells == 1`) gives **91,168** cells against the preprocessed file's **91,205** —
   a 0.04% difference. That file is GSE133344 filtered exactly this way and then reduced to
   5,045 genes.
3. **The expression says K562, not A549.** Measured on the preprocessed file's own counts
   (first 5,000 cells), where K562 is a female erythroleukemia line and A549 a male lung line:

| marker | meaning | mean | frac > 0 |
|---|---|---|---|
| `XIST` | female — X inactivation | 1.82 | **75.9%** |
| `HBZ` | erythroid — K562 | **27.99** | 90.7% |
| `HBG2` | erythroid — K562 | **17.19** | 91.6% |
| `GYPA` | erythroid — K562 | 3.89 | 87.9% |
| `NKX2-1`, `SFTPC` | lung — A549 | not on that axis | — |

Female + haemoglobin + glycophorin is K562. A549 expresses none of it. ⚠️ The mislabelling is
upstream, in `/rxrx/data/valence/biomodality_sc/norman_2016/`, and anyone else using that file
inherits it.

⚠️ **The pasted GEO link was `GSE13344`**, which is an unrelated 2013 series. The Norman 2019
accession is **GSE133344**; that is what was downloaded (`data/external/GSE133344/`).

## Composition, measured

| class | conditions | cells | cells/condition (median, min–max) |
|---|---|---|---|
| control (`non-targeting`) | 1 | **7,353** | — (1,008/928/863/874/821/976/934/949 per batch) |
| **single** (`X+ctrl` / `ctrl+X`) | **152** | **48,407** | 287 (97–997) |
| **double** (`X+Y`) | **131** | **35,445** | 257 (49–1,030) |
| total | 284 | 91,205 | |

- **105 distinct single genes.** 47 of them appear in BOTH orientations (`CEBPE+ctrl` and
  `ctrl+CEBPE`), which is why 152 conditions give 105 genes. Pooled per gene: median 430 cells,
  min 97, and **104 of 105 genes have ≥100 cells**.
- **131 unique unordered double pairs over 73 genes**, no pair present in both orders. Gene
  degree in the pair graph: max 12, median 3, min 1.
- ✅ **All 131 pairs have BOTH constituent genes present as singles.** The composition task is
  therefore clean for every test pair — there is no "unseen gene" confound to separate, unlike
  GEARS' 2/1/0-seen taxonomy.
- **8 batches, 10,606–12,409 cells each.** Every condition spans all 8 batches with median 31
  cells per (condition, batch) and **ZERO singletons**. That is far better than `rhj_fewshot`
  (median 1 cell per (pert, batch), 53.4% singletons) and means batch-stratified halves work
  properly here for perturbed cells, not only for controls.

## Feasibility against our existing stack — four checks, all passed

| check | result |
|---|---|
| gene axis vs our encoder | `var.gene_id` is Ensembl; **4,762 of 5,045 (94.4%)** are in the encoder's 44,728-gene vocabulary. ⚠️ Match on `gene_id`, NOT `gene_name` (3/5,045) — the encoder axis is 99.6% ENSG. |
| perturbation vocabulary | 103/105 single genes and 129/131 pairs are in `pert2id` (18,216 symbols) as-is; with a **2-entry alias map** (`KIAA1804→MAP3K21`, `RHOXF2BB→RHOXF2B`) it is **105/105 genes and 131/131 pairs** |
| double perturbations in the model | **supported with no code change.** `model.forward` takes `List[List[int]]` and the `max_perts > 1` branch sums the per-gene graph embeddings |
| control label | `obs.gene_name == "non-targeting"` — already the label cell-eval expects |

⚠️ **The multi-pert path is a Python double loop** (`for i, perts: for p in perts: pert_z[i] += z_graph[p]`),
i.e. 2×batch_size autograd ops per forward at batch 2048. It is correct but was written as a
fallback, never for a training run. **Vectorise it (one `index_add_` over a flattened index
list) before training anything long**, and measure samples/s against the single-pert fast path
rather than assuming the cost.

⚠️ **`layers/counts` is DENSE.** Our loader reads CSR from `X` and takes the row sum *before*
gene alignment (`stream_v2.load_chunk_sparse`). The prepared benchmark file must therefore be
rewritten with **raw counts as CSR in `X`**; `X` as shipped is already log1p and would be
double-transformed. This is the single most dangerous step in the prep and gets its own test.

⚠️ **Only 594 of the 5,045 genes are on the `rhj_fewshot` 6,983-gene scoring axis.** Scores are
**not comparable between the two benchmarks** in either direction, and no table may put them in
the same column. Different gene axis, different cell line, different task.

## QC and gene selection, in full — every threshold, every count

Reproduce with, in order:
`norman_gene_stats.py` → `norman_pick_axis.py` → `norman_build_h5ad.py` → `norman_build_panel.py`
(all under `scripts/norman/`). Every number below is printed by those scripts.

### Step 1 — CELL QC: Norman's own filter, nothing of our own

From `GSE133344_filtered_cell_identities.csv.gz` (111,445 rows with an assigned guide identity,
against 111,668 barcodes in the matrix — 223 barcodes carry no identity and are dropped):

| filter | kept | dropped | why |
|---|---|---|---|
| has a guide identity | 111,445 | 223 | a cell with no assigned guide has no label |
| `good_coverage == True` | 104,507 | 6,938 | GEO's own coverage flag |
| `number_of_cells == 1` | **91,168** | 13,318 more | multiplets: 11,885 doublets, 1,286 triplets, 147 quadruplets |

**We add NO QC of our own** — no mitochondrial cut, no UMI floor, no gene-count floor. Two
reasons: the upstream flags already encode the authors' judgement about their own assay, and
**this exact filter reproduces the 91,205 cells of the widely-circulated preprocessed file to
within 37 cells (0.04%)**, which is what identified that file's provenance. Adding our own
thresholds would have broken that check and made the cohort incomparable to everyone else's.

⚠️ **`number_of_cells` is not a redundant restatement of `good_coverage`.** The flags overlap
but do not nest: 6,938 cells have `number_of_cells == 0` (exactly the `good_coverage == False`
set) while 13,318 further cells are multiplets *with* good coverage. Filtering on one alone
leaves the other's failures in.

Resulting label composition (`gene_name` in the emitted file):

| class | conditions | cells | cells/condition median (min–max) |
|---|---|---|---|
| control (`non-targeting`) | 1 | **9,702** | 1,213 per gemgroup (1,152–1,283) |
| single (`GENE`) | **105** | 47,269 | 406 (95–1,776) |
| double (`GENE_A+GENE_B`) | **131** | 34,197 | 249 (46–1,057) |

Library size after QC: median **13,361** UMI over all 33,694 genes (p1 4,381, p99 32,692),
median 3,232 detected genes per cell.

⚠️ **Guide-orientation labels are collapsed.** GEO encodes a single perturbation as the guide
pair `{GENE, NegCtrl0}`, and the upstream preprocessed file kept the two orientations as
separate conditions (`GENE+ctrl` and `ctrl+GENE`), which is why it reports 152 single
conditions for 105 genes. We label by the perturbed gene alone: `GENE`. The orientation is an
artifact of which slot the guide occupied, not a different perturbation, and keeping it would
split every single arm's cells in two and halve the `sum_of_singles` baseline's precision.

### Step 2 — GENE QC and the 8,000-gene scoring axis

Statistics are computed on the **91,168 QC cells only** (`--keep-cells`), never on all 111,668
barcodes: a gene's detection rate must describe the cells the benchmark actually scores. They
are computed in the model's own space — `log1p(count / library_size × 2,000)`, the library size
being the row sum over all 33,694 genes — in ONE streaming pass over 362,199,631 nonzeros,
which is exact because the MatrixMarket file is cell-major with ascending gene index (verified
on 3,000,000 entries: 0 out of order), so a cell's library is known by its last entry.

| step | rule | survivors | dropped |
|---|---|---|---|
| 0 | genes in the GEO annotation | 33,694 | — |
| 1 | **in Occam's 44,728-gene Ensembl vocabulary** | **29,187** | 4,507 |
| 2 | **detected in > 8% of the 91,168 QC cells** | **8,279** | 20,908 |
| 3 | **force-include every resolvable perturbation target** | +40 below threshold | — |
| 4 | fill to exactly 8,000 by descending variance of the log1p expression | **8,000** | 279 trimmed |

**Why `> 8%` and not a rounder number.** It is the highest round threshold that still leaves
more than 8,000 candidates, so step 4 — the only arbitrary step — trims just 279 genes (3.4%)
instead of doing the selection. The alternative thresholds, all measured: >1% leaves 12,176,
>2% 11,134, >5% 9,410, >10% only 7,637 (**cannot reach 8,000**).

**Why match on `gene_id` and not `gene_name`.** Occam's vocabulary is 99.6% Ensembl
(`ENSG…`); matching Norman's symbols against it gives **3 of 5,045**, matching Ensembl gives
**94.4%**. The symbol path silently produces an almost-empty axis, and the first version of this
analysis reported exactly that before the cause was found.

⚠️ **Step 3 is the one that decides whether this benchmark measures anything.** Norman is
CRISPR**a**: the perturbed genes are transcription factors that are OFF at baseline and are
switched ON by the perturbation, so a baseline-detection filter excludes precisely the genes
under study. **41 of 102 resolvable targets fail step 2** — `FOXO4` (8.1% detection), `OSR2`
(7.9%), `POU3F2` (7.9%), `ISL2` (7.3%), `FOXA3` (6.2%), `SNAI1` (5.4%), `MEIS1` (4.2%),
`PRDM1` (2.9%) among them. They are force-included, which is a deliberate documented exception,
and the axis file records `is_pert_target` per gene so the exception is auditable.

⚠️ **Three targets cannot be resolved at all**: `C19orf26`, `C3orf72`, `KIAA1804` are absent
from GSE133344's symbol list (a GRCh37-era CellRanger annotation). They remain usable as
perturbation LABELS — Occam's `pert2id` reaches them via a 2-entry alias map
(`KIAA1804→MAP3K21`, `RHOXF2BB→RHOXF2B`) — but their own expression is not scorable. **Their
Ensembl IDs are not guessed**: the one guess attempted resolved `ENSG00000143674` to
`RP5-862P8.2` in this annotation, which is why the script refuses to guess.

Selected axis: detection rate min 0.04% (a forced target), **median 28.2%**, max 100%; 40 genes
below the threshold, all of them forced targets. sha256 `6efa91af1af5c8b8…`.

⚠️ **All 33,694 genes are kept in the emitted `.h5ad`**, not only the 8,000. The library size
must be the sum over everything the assay measured — `stream_v2.load_chunk_sparse` takes the
row sum BEFORE gene alignment, and `rhj_fewshot`'s sources likewise keep 8,749–9,624 genes
while its panel scores 6,983. Writing only the axis would change the normalisation of every
cell.

### Step 3 — the emitted file

`X` is **raw integer counts in CSR**, not the preprocessed `X` of the upstream file, which is
already normalised and log1p'd (nnz range 0.33–7.69) and whose raw counts sit in a DENSE
`layers/counts`. Feeding that file to our loader would log1p an already-log1p'd value —
silently, because the result is still a plausible number. 91,168 × 33,694, 286,342,111
nonzeros, `int32`, 0.65 GB, sha256 `c6cad008202666ef…`.

---

## The two test strata, in full — how `additive` and `nonadditive` are defined

### The statistic: the additivity RESIDUAL, not the correlation

For each cohort pair `p = (A, B)`, on the 8,000-gene axis, in log1p-2,000 space:

```
Δ_real(p)  = mean(cells of p)        − mean(all 9,702 control cells)
Δ_sum(p)   = Δ_single(A) + Δ_single(B)        # each from its own training single arm
residual(p) = ‖Δ_sum(p) − Δ_real(p)‖₂ / ‖Δ_real(p)‖₂        # 0 = perfectly additive
```

Measured over the 111 cohort pairs: residual median **0.477**, min 0.267, max 1.791.

**Why the residual and not `pearson_r`.** Additivity fails two ways — wrong DIRECTION and wrong
SIZE — and a correlation only sees the first:

| | spearman with residual | spearman with `pearson_r` |
|---|---|---|
| \|magnitude ratio − 1\| | **+0.740** | −0.255 |

`CEBPE+ZC3HAV1` has `pearson_r` 0.758, which reads as additive, while `Δ_sum` is **2.42× too
large**. Ranking on `r` files it in the wrong stratum. The residual is a relative L2 error and
captures both modes in one number.

### The cut: residual terciles taken WITHIN effect-size terciles

⚠️ **A plain tercile cut on the residual is partly an EFFECT-SIZE cut, and that had to be
fixed.** The residual is relative, so it inflates when the denominator is small:
spearman(residual, ‖Δ_real‖) = **−0.514**. The raw terciles therefore piled weak-effect pairs
into `nonadditive` — median ‖Δ_real‖ **2.39 there against 3.91** in `additive`. Since effect
size drives all four metrics, a model comparison across those two panels would have been partly
weak-vs-strong-effect rather than additive-vs-non-additive. The tell was in the baselines:
`meanpert`'s `mae` was *lower* on the raw `nonadditive` panel (0.01245 vs 0.01484) purely
because there was less signal there to get wrong.

**The official cut** (`tercile_effect_controlled` in `additivity.npz`):

1. split the 111 pairs into terciles of ‖Δ_real‖ (the effect size);
2. **within each effect tercile**, split by residual into low / middle / high thirds;
3. `additive` = the union of the low thirds, `nonadditive` = the union of the high thirds.

| stratum | pairs | residual median | `pearson_r` median | mag ratio median | ‖Δ_real‖ median |
|---|---|---|---|---|---|
| `additive` | **36** | 0.373 | 0.936 | 1.03 | 3.615 |
| `middle` (not scored) | 39 | 0.463 | 0.903 | 1.01 | 3.363 |
| `nonadditive` | **36** | 0.639 | 0.871 | 1.40 | 3.186 |

spearman(stratum, ‖Δ_real‖) falls from **−0.514 to −0.090**: the confound is gone and the
additivity separation survives (0.373 vs 0.639, against 0.373 vs 0.763 for the raw cut — the
narrower gap is the price of removing the confound, and it is worth paying).

The raw cut is retained in the artifact as `additive_raw` / `middle_raw` / `nonadditive_raw`.

### That the strata are real, not noise — the independent check

They were never asked to be biologically coherent, and they are. `nonadditive` collects
same-pathway REDUNDANCY — `KIF18B+KIF2C` (kinesins), `PLK4+STIL` (one centriole-duplication
step), `CDKN1A+CDKN1C` and `CDKN1B+CDKN1C` (CDK inhibitors), `MAP2K3+MAP2K6` (MAP2Ks),
`BAK1+BCL2L11` (pro-apoptotic BCL2 family) — and the erythroid/MAPK SYNERGY hubs: six `CEBPE`
pairs, `CEBPB+CEBPE`, `CEBPE+KLF1`, and eight `MAPK1` pairs. These are the epistasis classes
Norman's own paper describes. `additive` is dominated by pairs whose genes sit in unrelated
pathways. And membership is a property of the PAIR, not the gene: `ETS2+MAPK1` is additive
while `IGDCC3+MAPK1` is not.

### ⚠️⚠️ The selection caveat, which binds because these ARE the test set

The strata are defined by **`sum_of_singles`'s own residual on the same cells**. Therefore:

* that baseline's score **per stratum** is NOT an independent measurement — it is low on
  `nonadditive` by construction;
* **model-vs-model** comparisons across strata are unaffected, because no model saw the strata;
* **"our model beats the additive baseline on the non-additive pairs" is partly a restatement
  of how the stratum was drawn.** The defensible claim is a margin on `additive`, where the
  baseline is near its best, *and* a larger margin on `nonadditive`.

The strata are frozen in `splits_manifest.json` (`panels`, `scored_panels`) and are identical
for every arm, every split and every competitor — as `rhj_fewshot`'s half-size strata are.

## Proposed design

### The task split

| set | content | conditions | cells |
|---|---|---|---|
| **train** | all single-gene conditions, minus the validation pairs' needs | **152 conditions / 105 genes** | 48,407 |
| **train controls** | `non-targeting` | 1 | 7,353 (shared with the panel — see below) |
| **val** | **20 double pairs**, stratified | 20 | ~5,400 |
| **test** | **111 double pairs** | 111 | ~30,000 |

**Why validate on DOUBLES and not on held-out singles.** F-ckpt4 measured that our validation
metric selects the epoch well but misranks arms; the mitigation is that the validation quantity
must at least be the *same kind* of quantity as the test one. A validation set of singles would
select checkpoints on a task the benchmark is not about (single-gene prediction) and would be
blind to composition failure — the exact thing being measured. Cost: 20 of 131 pairs, 15% of
the cohort.

⚠️ **The alternative is worth stating**: `val_singles` (15 of 152 single conditions) keeps all
131 pairs in the test cohort. It is the right choice ONLY if the 131-pair cohort is judged too
small to give up 20. My recommendation is `val_doubles`; the decision is yours.

**Validation stratification.** Both axes that plausibly predict difficulty: cell count (so val
is not made of the smallest pairs) and minimum gene degree in the pair graph (so val is not all
hub genes). Sample 20 pairs stratified on the cross of (count tercile × degree tercile).

### The split-half protocol — 5 splits, exactly as `rhj_fewshot`

Per split `s ∈ {0..4}`, for each test pair and for the control pool:

- halve the cells into **A** (the reference) and **B**, **batch-stratified** — here that
  actually balances, given 31 cells per (condition, batch) and zero singletons;
- **half-A perturbed** = the scored ground truth;
- **half-B perturbed** = the `experimental` arm, the empirical reference the model is measured
  against (not a ceiling — see the `rhj_fewshot` CARD on why a negative gap is not a triumph);
- the model is fed **half-B control cells** and predicts each pair; its prediction is scored
  against half-A.

Seeding: `blake2b(SALT | seed | split | condition)` per (split, condition), never one stream
consumed in a loop — adding a pair to the cohort must not reshuffle another pair's halves.
(The `rhj_fewshot` protocol's rule, for the same reason.)

**`min_cells = 40`, i.e. half size `m ≥ 20`.** Measured: **all 131 pairs clear m ≥ 20**, 130
clear m ≥ 25, 123 clear m ≥ 40, 111 clear m ≥ 60. So at m ≥ 20 **nothing is excluded** — a
better position than `rhj_fewshot`, which dropped 10 perturbations at `min_cells = 10` and
whose median `m` was 22. Report the count-stratified gap (m ≥ 20 / ≥ 40 / ≥ 60) as that CARD
requires: a gap that does not shrink with `m` is noise-dominated.

**5 splits is feasible** — each split is an independent halving of the same cells, so 5 costs
nothing in data, only in eval time (5 × 131 pairs).

### The scoring axis

**4,762 genes** — the intersection of Norman's `var.gene_id` with the encoder vocabulary —
frozen with a sha256 in the panel, with both `axis_ensembl` and `axis_symbols` carried as
`rhj_fewshot`'s `panel.npz` does. Normalisation to match the loader exactly: row sum over
**all 5,045 file genes** → ×2,000 → `log1p`, the sum taken BEFORE alignment.

### Controls: shared between train and panel, as in `rhj_fewshot`

The 7,353 `non-targeting` cells are used both as the training corpus's controls and as the
panel's control pool (halved A/B per split, 3,676 per half). ⚠️ This is a deliberate choice
copied from `rhj_fewshot`, where the panel's 4,976 controls also came from the training file:
the held-out quantity is the **perturbation**, not the control distribution. If you would
rather hold out controls too, the cost is ~40% of the control pool for the panel and a
training corpus with 4,400 controls instead of 7,353 — say so and I will build it that way.

## Prep steps, in order

1. **Rewrite the benchmark source**: `layers/counts` (dense) → CSR in `X`, drop the GEARS
   `uns`, keep `obs` {`gene_name`, `batch`, `is_control`, `cell_line`, `cell_barcode`} and
   `var` {`gene_id`, `gene_name`}. Emit `norman_combo_a549.h5ad` + sha256. **Test:** round-trip
   10 random cells against the source counts, and assert `X` is integral.
2. **Freeze the cohort and the splits**: `data/benchmarks/norman_combo/` with `cohort.json`,
   `splits.npz`, `panel.npz`, `splits_manifest.json`, `CARD.md` — same artifact set as
   `rhj_fewshot/splithalf/`, built by a sibling of `scripts/build_splithalf_panel.py`.
3. **Vectorise the multi-pert path** and measure samples/s (single-pert fast path is the
   baseline).
4. **Train**: singles-only corpus, the `Y06` recipe (Q02 loss, 150 epochs, `eta_min` 3e-4 —
   F-sched3), and read against the Norman-native anchors, never against `rhj_fewshot` numbers.
5. **Anchors first, model second.** Build `ctrlcell` / `ctrlmean` / `meanpert` / `transfer`
   equivalents for this panel before scoring any model arm. ⚠️ `transfer` has no direct analogue
   here (one cell line); its place is taken by a **`sum_of_singles`** baseline — predict the
   double as the sum of the two measured single-gene deltas. That baseline IS the additivity
   hypothesis stated as a predictor, and it is the number every model arm must beat.
6. **HuggingFace**: add `eval/pert/norman_combo_a549.h5ad` + `benchmarks/norman_combo/` to the
   existing private repo via `scripts/publish_hf.py` (incremental, hardlink-staged). ⚠️
   `scripts/check_upload_gate.py` already refuses PUBLIC release for want of licences; Norman
   2019 is third-party data and adds one more unlicensed dataset, so this stays private until a
   named human resolves licensing.

## The one baseline that matters most

**`sum_of_singles`.** For a test pair (A, B), predict `ctrl_mean + Δ_A + Δ_B`, where each `Δ`
is the measured single-gene delta from the TRAINING half. It costs no model, it is the
additivity hypothesis in closed form, and our architecture composes by summing latent
embeddings — so a model that cannot beat it has learned nothing about interaction. Expect it to
be strong: in the GEARS literature, additive baselines are hard to beat on double perturbations.
It must be in the panel from day one, not added after a model result looks good.


---

## 2026-09-22 — the additivity strata had no script, and now do

**Found by running `scripts/repro/build_benchmark_norman.sh` for the first time.** The
rebuild came out with `panels: []` and `scored_panels: null`: the seven additivity
strata — including `additive` and `nonadditive`, **the two panels every norman_combo
result is reported on** — were produced by an ad-hoc pass during the campaign and were
never committed. A consumer who rebuilt the benchmark got a cohort and splits they could
not score the way we do.

`scripts/norman/norman_stratify_panels.py` now does it, and the recipe was **recovered
from the published artifacts rather than reinvented**. It reproduces them exactly:

| quantity | max abs difference vs published |
|---|---|
| `residual`, `effect_norm`, `magnitude_ratio` | **0.0** |
| `n_cells`, `tercile`, `effect_tercile`, `tercile_effect_controlled` | **0** |
| all 7 panel memberships (36 / 39 / 36 and 37 / 37 / 37) | identical sets |

Verify at any time with `python scripts/norman/norman_stratify_panels.py --verify`.

### The recipe

`d_real` = mean of the pair's panel cells − mean of the panel's controls, on the
8,000-gene axis in the panel's own eval space, over **all** cells (not a half).
`d_sum` from `sum_of_singles.npz`.

| quantity | definition |
|---|---|
| `residual` | ‖d_sum − d_real‖ / ‖d_real‖ — how NON-additive the pair is |
| `effect_norm` | ‖d_real‖ — how LARGE the true effect is |
| `magnitude_ratio` | ‖d_sum‖ / ‖d_real‖ |
| `pearson_r` | corr(d_sum, d_real) |

### ⚠️ The tercile boundary convention is asymmetric and load-bearing

`>=` at the lower edge, `>` at the upper. **Only one of the four combinations reproduces
the published panels**, and the other three give 39/36/36, 39/33/39 or 36/36/39 instead
of 36/39/36.

It matters because of the sizes. With 111 values the 1/3 quantile falls *between*
elements — index (111−1)/3 = 36.67 — so every convention agrees and the raw cut is
37/37/37 either way. Within an effect tercile there are 37 values, index (37−1)/3 = 12.0
lands **exactly on an element**, and the convention decides which side it falls.

### ⚠️ A rebuilt panel cannot score a published checkpoint

`splits_manifest.json:benchmark_sha256` is `sha256(benchmark.json)`, and that file embeds
`git_sha`, `built` and `panel.dir`. A rebuild therefore **cannot** reproduce it even when
every scientific quantity matches — measured: `f458c2d9…` vs the published `8ac7da0d…` on
a rebuild that `verify_benchmark.py` passes completely. `eval_splithalf` refuses on that
field, correctly. Verify with the rebuild; score against the downloaded artifact.

**Open design question, not acted on:** the definition hash could be made reproducible by
hashing a provenance-free projection of `benchmark.json` (dropping `git_sha`, `built`,
`command`, `numpy`, `python`, `panel.dir`). That would let a rebuilt panel score a
published checkpoint — but it changes the published digest, which is recorded in both
configs and in every existing checkpoint, so it is a migration with a decision attached.
Left as-is pending that decision.
