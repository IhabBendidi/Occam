# `rhj_fewshot` competitors — STATE, TxPert, CellFlow, PerturbDiff

**This file owns every competitor number on this benchmark.** Nothing is scored yet; what
follows is the measured groundwork, so the numbers land on something already verified.

## What this is for

**Objective.** Predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

**The question.** Where does Occam sit against the field, measured on an instrument we
control — same 611 held-out HepG2 perturbations, same 6,983-gene axis, same 5 split-half
splits, same cell-eval, same five baselines?

**What is varied.** The model (4 competitors + Occam) × 5 seeds (42–46). **Held fixed:** the
frozen benchmark artifact, the cohort, the gene axis, the drawn control cells, the emission
space, and the scoring path.

**What each outcome means.** Competitors are scored against the same ladder as Occam:
`ctrlcell` −0.0018 (floor) · `meanpert` 0.1324 · **`transfer` 0.2323 (the bar)** ·
`experimental` 0.3722 (replicate agreement at m = 22, *not* a ceiling). A model below
`transfer` is not earning its parameters; a model above `experimental` is likely exploiting
the two-sided noise asymmetry rather than being better.

⚠️ **Their published numbers are not comparable and are not reproduced here.** `hooke-tx`'s
`replogle_nadig_fewshot` is 2,000 genes / 380 test perturbations / 4 cell lines from a
different preprocessing of the same experiment (their own task config says so). For
orientation only, their inject-real `pearson_delta` was TxPert 0.50787 ± 0.00614, CellFlow
0.44015 ± 0.02057, STATE 0.42574 ± 0.01565, PerturbDiff 0.40306 ± 0.01153, against a
half-split reproducibility ceiling of 0.38600 and a train-perturbation mean of 0.35339.

## Comparison metrics

By decision 2026-09-17, every comparison uses **`pearson_delta`, `mae` ↓, `overlap_at_N`,
`discrimination_score_l1`** — note **l1**, not l2. Chance is **0.5** for
`discrimination_score_l1`.

---

## C1 — the evaluation space, measured and pinned

The highest-risk part of this port, so it was settled by measurement before any code was
copied. All numbers below are from `data/benchmarks/rhj_fewshot/splithalf/panel.npz` and from
the installed cell-eval 0.6.0 source.

### Occam normalises on the FULL gene panel, not on the 6,983-gene axis

```
x_eval = log1p( counts / rowsum_FULL_PANEL * 2000 )     # 6,983-gene axis, axis order
```

| measured on `panel.npz` | value |
|---|---|
| `expm1(X).sum(axis=1)` | median **1731.8**, mean 1719.8, range 1228.0–1931.8 |
| implied axis UMI share | **0.866** |
| value range | [0, **5.9248**] against `log1p(2000)` = 7.6014 |
| zero fraction | **0.5147** |

**It is not 2000.** Had the panel been normalised on the axis sum, that median would be
exactly 2000.0. The 6,983 genes carry 86.6% of each cell's UMI, and the share varies per cell
(hence the 1228–1932 range).

### Why that single fact decides the port

Competitors are handed `target_sum = 10_000` data, because their pipelines assume
pre-normalised input — every arm passes `target_sum=null, log1p=false`, since the hooke-tx
data arrived already `log1p(1e4)`. **Our h5ads are raw counts**, so we do that normalisation
for them. Converting their output back to our space is then **exact**, because the per-cell
denominator cancels:

```
x_eval = log1p( expm1(x_10k) * 0.2 )                    # 2000 / 10000
```

| measured | value |
|---|---|
| max \|round-trip − original\|, float64 | 8.88e-16 |
| max \|round-trip − original\|, **float32** | **0.0 — bit-exact** |
| float32 eps at the max value | 4.77e-07 |

⚠️ **This exactness holds only because both sides use the full-panel denominator.**
Normalising competitors on the axis sum would leave a per-cell factor (median 0.866, and
cell-dependent) that leaves `pearson_delta` untouched — it is scale-invariant per delta
vector — while shifting `mae` and moving the DE calls. `mae` is one of the four comparison
metrics.

### ⛔ What cell-eval will NOT catch

`cell_eval._evaluator._convert_to_normlog`, in full:

```python
if guess_is_lognorm(adata):  return       # "already log-normalized"
if allow_discrete:           return       # our flag
sc.pp.normalize_total(adata)              # to the FILE'S OWN MEDIAN
sc.pp.log1p(adata)
```

and `guess_is_lognorm` (`cell_eval/utils.py:5`) is only *"True if **any** of 500 randomly
chosen cells has a row sum whose fractional part exceeds 0.01"* — it merely asks **"are these
floats?"**. Its own docstring concedes: *"This will not be able to distinguish between
normalized input and log-normalized input."*

Measured, 5 unseeded repeats each:

| input | `guess_is_lognorm` | outcome |
|---|---|---|
| Occam panel (log1p, ts 2000) | True ×5 | left alone ✅ |
| **library-normed LINEAR, no log1p** | **True ×5** | **left alone — silently wrong** |
| raw integer counts | False ×5 | would be renormalised (what `--allow-discrete` blocks) |

`_evaluator.py:273` then hardcodes `pdex_kwargs["is_log1p"] = True` regardless. So a
competitor emitting library-normed **linear** data passes both gates and has its fold changes
computed as though they were log1p. Nothing warns. Worse, `_convert_to_normlog` is called
**separately** for the real and predicted files (`_evaluator.py:167,170`), so the two can
receive different treatment.

### The discriminator cell-eval lacks, and the guard built on it

| | log1p (Occam) | library-norm linear |
|---|---|---|
| max value | **5.01** | 148.59 |
| median `X.sum(1)` | 992.60 | 1730.95 |
| median `expm1(X).sum(1)` | **1731** | 1.59e+18 |

`competitors/shared/emission_space.py` implements `to_eval_space`, `assert_is_log1p` and
`assert_same_space`, which run before **every** cell-eval call — for competitors *and* for
Occam, which is not exempted. Verified to refuse each of:

| failure mode | caught by |
|---|---|
| library-normed linear *(cell-eval accepts it)* | max 148.59 > `log1p(2000)` = 7.6014 |
| log1p at the wrong `target_sum` (10k, unconverted) | library size +400.0% |
| integer re-emission (`expm1`→round→`log1p`) | library size −27.5% |
| a constant matrix (dead emission head) | library size +41.1% |
| negatives / NaN / Inf | explicit counts |
| permuted gene order | var-index comparison |

The integer case is corroboration, not a hypothetical: the repo previously measured that
exact transform costing **0.034 `pearson_delta`**, with library rescaling recovering none of
it.

⚠️ The guard's tolerances are deliberately loose enough to pass a *genuinely different but
valid* model and tight enough to fail a *space* error: ±10% median library size, per-gene
mean correlation ≥ 0.95, zero fraction within ±0.05. They are the same diagnostics that
caught this repo's own emission defects — a `rectified_sigmoid` head leaving 83.6% of genes
constant, and integer emission driving the zero fraction to 0.907 against a target 0.606.

Tests: `tests/test_emission_space.py`, 13 cases. Two of them assert against the real
`panel.npz`; the synthetic fixture is pinned to the real regime on **three** axes — sparsity
(0.7 Poisson over 6,983 genes ⇒ ~0.25/gene, ~0.50 zeros), library size, and the **heavy tail**
(5 genes at high expression ⇒ max ≈ 5, matching the real 5.92). Both the tail and the sparsity
were added after a test passed *vacuously* without them: with all values ≪ 1, `log1p(x) ≈ x`
and linear is genuinely indistinguishable from log1p, so the check that matters was not being
exercised.

---

## Status

| step | state |
|---|---|
| 0 — commit and push | ✅ 26 commits pushed (`0b94934..b567a8b`) |
| 1 — `competitors/` skeleton + gitignore | ✅ rules verified; only `MANIFEST.json` is trackable |
| 5b — emission space | ✅ C1 above, 13 tests |
| 2 — `occam_data.py` reader | next |
| 3 — copy code per arm | pending |
| 4 — wire each dataloader to our data | pending |
| 5 — the prediction bridge | pending |
| 6 — launchers, collector | pending |

Plan: `~/.claude/plans/i-want-to-make-iterative-cray.md`.
Panel definition: `data/benchmarks/rhj_fewshot/splithalf/CARD.md`.
Baselines: `docs/rhj_baseline_results.md`. Protocol findings: `docs/rhj_splithalf.md`.
