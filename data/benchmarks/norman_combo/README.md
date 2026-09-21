# `norman_combo` — a double-perturbation composition benchmark (K562)

**The task: train on single-gene perturbations, predict the effect of a PAIR** where both genes
were seen individually and the pair never was. 111 test pairs, 5 paired split-halves, an
8,000-gene scoring axis, and five baselines including the one that matters — `sum_of_singles`.

| | |
|---|---|
| source | GSE133344 — Norman et al. **2019**, *Science* 365:786 · K562 CRISPRa Perturb-seq |
| benchmark `.h5ad` | `eval/pert/norman_2019_k562.h5ad` (91,168 × 33,694, raw CSR int32 counts) |
| scoring axis | **8,000** genes, sha256 `6efa91af1af5c8b8…` |
| train / val / test | 105 single conditions / **20** double pairs / **111** double pairs |
| splits | 5, batch-stratified, `blake2b(salt|seed|split|group)` |
| baselines | `ctrlcell` · `ctrlmean` · `meanpert` · **`sum_of_singles`** · `experimental` |

**Read `splithalf/CARD.md` before using any number.** It records the frozen panel, why each
choice was made, the two additivity strata and their selection caveat, and the one thing most
likely to mislead: **`sum_of_singles` already reaches pearson 0.891 on the double deltas**, so
the mean over 111 pairs is dominated by pairs additivity explains, and a composition claim has
to be made on the strata.

⚠️ **The cell line is K562.** The widely-circulated preprocessed file
`norman_2016_a549_preprocessed.h5ad` labels these cells A549 and dates them 2016; both are
wrong. Evidence, from that file's own counts: XIST detected in 75.9% of cells (female; A549 is
male), HBZ 28.0 / HBG2 17.2 / GYPA in 87.9% (erythroid; A549 is lung). This benchmark is built
from the GEO primary source.

## Files

| file | what |
|---|---|
| `axis.npz`, `axis.json` | the 8,000-gene axis, with the detection rate and variance that justified each gene |
| `splithalf/panel.npz` | 38,623 × 8,000 panel matrix, labels, batches, axis, source rows |
| `splithalf/splits.npz` | the 5 batch-stratified half-splits (row indices into the panel) |
| `splithalf/cohort.json` | the frozen 111 pairs and their half sizes |
| `splithalf/splits_manifest.json` | splits, val holdout, named panels (`all`/`additive`/`middle`/`nonadditive`), measured batch balance |
| `splithalf/sum_of_singles.npz` | the additive baseline's per-pair deltas |
| `splithalf/train_grand_mean.npy` | the `meanpert` baseline |
| `splithalf/additivity.npz` | per-pair additivity residual, correlation, magnitude ratio, tercile |
| `splithalf/baseline_sanity.json` | the whole-cohort check of `sum_of_singles` |
| `splithalf/CARD.md` | **the card — read this** |

Built by `scripts/norman/norman_{gene_stats,pick_axis,build_h5ad,build_panel}.py`.
