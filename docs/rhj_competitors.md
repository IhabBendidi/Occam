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
| **derived** (build 10k from the 2k panel in float64, map back), float64 | 8.88e-16 |
| **derived**, float32 | **0.0 — bit-exact** |
| **independent** (both computed from raw counts in float32) | **4.768e-07 = one float32 ULP** |
| float32 ULP at the max value 5.0 | 4.768e-07 |

⚠️ The independent figure is the one that applies in practice: competitors receive a 10k
matrix computed from counts, not derived from the 2k panel, so each carries its own float32
rounding. One ULP is ~7 orders of magnitude below the ±10% library-size tolerance and cannot
move a metric. An earlier version of this note claimed bit-exactness without distinguishing
the two cases.

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

## C2 — the reader is bit-identical to our own pipeline

`competitors/shared/occam_data.py` reads our three h5ads and the frozen artifact **in
place**. No conversion, no second copy, no format to keep in sync.

**The gate, measured:** at `target_sum=2000.0` the reader reproduces
`splithalf/panel.npz` **bit for bit** — `max |diff| = 0.0` over 42,402 × 6,983 float32, zero
differing values, labels matching. That is what proves competitors see exactly the data the
five baselines were measured on.

It depends on reproducing `build_splithalf_panel.py::_load_axis_restricted` **dtype for
dtype**: float32 values, a float64 row-sum accumulation over the full gene panel, cast back
to float32 before the divide. Any change to that sequence breaks the equivalence while still
"working", so `tests/test_occam_data_reader.py` asserts it.

| measured | value |
|---|---|
| all three lines, shipping `target_sum=10_000` | 656,343 × 6,983 float32, **18.3 GB**, 178 s |
| HepG2 only | 145,473 × 6,983, 48 s |
| `finetune` | **612,122** rows, including all **28,474** controls |
| `validation` | **6,731** rows, exactly the 103 HepG2 val perturbations |
| `test` | **37,490** rows, exactly the 621 HepG2 test perturbations |
| val + test | **44,221** — matches `exclusions.json` exactly |
| Jurkat, RPE1 | **wholly `finetune`** — the positive half of the exclusion |

The name-level split (which TxPert needs) keeps all 724 held-out symbols in `train`,
deliberately: they legitimately train from Jurkat and RPE1, and within HepG2 the three sets
are disjoint, which is what TxPert's straddle check actually requires.

Emitted schema: `gene_pert` / `pert` / `target_gene` / `gene` (controls → `"empty"`, or
`"non-targeting"` on request), `negative_control` / `is_control` (bool), `cell_type` /
`cell_line` (exact casing `HEPG2` / `jurkat` / `RPE1`), `batch_center` (line-prefixed) /
`dataset_batch_num`, `dataset_name`. `var.index` is **symbols**; both `ensembl_gene_id` and
`gene_symbol` are carried. `uns["occam"]` records the axis hash and the space, so a
downstream arm can assert what it was given.

9 tests, including the gate and a refusal on an edited artifact.

---

## C3 — what was copied, and what was not

`scripts/copy_competitor_code.py` copies an explicit per-arm **allowlist** and refuses a
**denylist** even when the allowlist would pull it in, recording every file's sha256 and
upstream path in `competitors/MANIFEST.json`.

| arm | files | size |
|---|---|---|
| PerturbDiff | 111 | 2.83 MB |
| CellFlow | 66 | 1.42 MB |
| STATE | 123 | 1.16 MB |
| TxPert | 42 | 0.96 MB |
| **total** | **347** | **6.6 MB** |

Against **141 GB** upstream, of which ~92 GB is untracked *and* unignored there. No dataset,
cache, checkpoint or venv was copied — the arms read our data through `occam_data.py`.
Upstream commit `18e159145bdd7c59`.

All nine extension points identified during planning are present and verified:
`STATE/state/src/state/tx/data/{__init__,zarr_perturbation}.py`, `STATE/hepg2.toml`,
`TxPert/gspp/data/datamodule.py`, `TxPert/configs/config-txpert-zarr-fewshot.yaml`,
`CellFlow/hooke_bridge/load.py`, `CellFlow/configs/replogle_nadig_fewshot.yaml`,
`PerturbDiff/src/apps/training/training_datamodule_builder.py`, and its manifest TOML.

⚠️ **One upstream claim corrected.** The `PR_*` notes record CellFlow's environment as
unreconstructible because `cellflow/uv.lock` "was gitignored and never committed". The file
is present on disk, and it is now copied — so the arm is reproducible. Also
`.python-version` lives at the CellFlow arm root, not under `cellflow/`.

---

## Status

| step | state |
|---|---|
| 0 — commit and push | ✅ 26 commits pushed (`0b94934..b567a8b`) |
| 1 — `competitors/` skeleton + gitignore | ✅ rules verified; only `MANIFEST.json` is trackable |
| 5b — emission space | ✅ C1 above, 13 tests |
| 2 — `occam_data.py` reader | ✅ C2 above, bit-identical gate passed, 9 tests |
| 3 — copy code per arm | ✅ 347 files, **6.6 MB** from a 141 GB upstream |
| 4 — wire each dataloader to our data | next |
| 5 — the prediction bridge | pending |
| 6 — launchers, collector | pending |

Plan: `~/.claude/plans/i-want-to-make-iterative-cray.md`.
Panel definition: `data/benchmarks/rhj_fewshot/splithalf/CARD.md`.
Baselines: `docs/rhj_baseline_results.md`. Protocol findings: `docs/rhj_splithalf.md`.
