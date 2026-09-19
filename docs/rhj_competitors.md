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

⚠️ **Their published numbers are not comparable and are not reproduced here.** `the upstream benchmark`'s
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
pre-normalised input — every arm passes `target_sum=null, log1p=false`, since the the upstream benchmark
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
`CellFlow/occam_bridge/load.py`, `CellFlow/configs/replogle_nadig_fewshot.yaml`,
`PerturbDiff/src/apps/training/training_datamodule_builder.py`, and its manifest TOML.

⚠️ **One upstream claim corrected.** The `PR_*` notes record CellFlow's environment as
unreconstructible because `cellflow/uv.lock` "was gitignored and never committed". The file
is present on disk, and it is now copied — so the arm is reproducible. Also
`.python-version` lives at the CellFlow arm root, not under `cellflow/`.

---

## C4 — the four environments, and the three things that stopped them building

**Finding: all four arms now have a working interpreter, and TxPert did not need a new one.**

| arm | interpreter | how | notes |
|---|---|---|---|
| **TxPert** | `pyenv` `gspp`, python 3.11.9 | already existed | torch 2.5.1, lightning 2.4.0, torch-geometric 2.6.1, anndata 0.11.3, **cell-eval 0.6.0 / pdex 0.1.24 — OUR versions** |
| **PerturbDiff** | `competitors/PerturbDiff/.venv` | `uv sync`, 4.4 GB | 125 packages, clean first try |
| **STATE** | `competitors/STATE/state/.venv` | `uv sync` after a fix | zarr 3.1.6; needs python ≥3.11, so the existing `state` pyenv (3.10.14, no zarr) could not be reused |
| **CellFlow** | `competitors/CellFlow/cellflow/.venv` | `uv sync` after a fix | zarr 3.4.0; jax + `cellflow-tools`, no existing env had them |

⭐ **TxPert's pre-existing env pins cell-eval 0.6.0 / pdex 0.1.24 — exactly ours.** That is
better for comparability than a fresh build would have been: its *in-training validation*
metric is then computed by the same scorer as the reported number, so its checkpoint selection
and our measurement cannot disagree because of a version. The other three select on their own
pinned cell-eval (CellFlow 0.7.2-era, STATE 0.6.8); that is a training-internal choice and is
stated rather than hidden. **Every reported number comes from our 0.6.0 path regardless.**

### The three build failures, all of them ours, all cheap to fix

1. **`OSError: Readme file does not exist: README.md` — twice.** `STATE/state/pyproject.toml`
   and `CellFlow/cellflow/pyproject.toml` both declare `readme = "README.md"`, resolved
   relative to the package directory. The copy allowlist had neither, so hatchling failed the
   *editable* build. Fixed by adding `state/README.md`, `cellflow/README.md` and
   `cellflow/LICENSE` (the latter for the PolyForm-Noncommercial declaration) to
   `scripts/copy_competitor_code.py`. Tracked tree is still **1.17 MB** (STATE) and
   **1.43 MB** (CellFlow).
2. **A transient `404` on `fsspec-2026.7.0-py3-none-any.whl`.** Three `uv sync` runs in
   parallel; CellFlow's got the 404 while PerturbDiff pulled **the byte-identical URL and
   hash** successfully in the same minute. Not a pin problem — retried serially and it
   resolved. ⚠️ Do not "fix" this by loosening a pin; the lock is what makes the arm
   reproducible.
3. **`srun: Invalid generic resource (gres) specification`.** The gate exited **FAILED in 2 s**
   (job `18042983`) because it wrapped its python in `srun` on the `cpu` partition. **No
   launcher in this repo uses `srun`** — the batch step is the task. Removed from all six new
   launchers.

And one that was not a build failure at all:

4. **`/tmp` is NODE-LOCAL.** The gate's first submission (job `18042891`) pointed at a script
   in the session scratchpad under `/tmp` and failed in 2 s with
   `gate.sh: No such file or directory`. The compute node cannot see the login node's `/tmp`.
   **Launchers live in the repo, on GPFS** — now `scripts/slurm/rhj_competitor_gate.sbatch`
   plus a reproducible `scripts/export_gate_ctrlcell.py`.

## C5 — the launchers, and the footgun that is now a test

**Finding: the `#`-inside-a-continuation footgun is linted, and it found a real omission.**

| leg | script | resources | array over |
|---|---|---|---|
| train | `competitor_train_txpert.sbatch` | h100, 12 cpu, 200 G, 16 h | seed 42–46 |
| train | `competitor_train_state.sbatch` | 1 gpu, 20 cpu, 320 G, 24 h | seed 42–46 |
| train | `competitor_train_cellflow.sbatch` | 1 gpu, 12 cpu, 256 G, 12 h | seed 42–46 |
| train | `competitor_train_perturbdiff.sbatch` | h100, 12 cpu, 128 G, 20 h | seed 42–46 |
| export (Phase A) | `competitor_export.sbatch` | 1 gpu, 8 cpu, 128 G, 2 h | seed; loops all 5 splits |
| score (Phase B) | `competitor_score.sbatch` | cpu, 16 cpu, 64 G, 2 h | split 0–4 |

Resources and every hyperparameter are transcribed from
`the upstream benchmark/reproducibility/<arm>__replogle_nadig_fewshot/launch.*.sh`, with the **time and
memory raised where our axis is 3.5× wider** (6,983 vs 2,000 genes) and nothing else touched.

`tests/test_competitor_manifest.py` enforces four things about them:

* **no `#` inside a backslash-continued command.** The naive version of this check flags every
  usage-example block in every header, because those lines are comments *and* end in `\`. The
  real rule is narrower: a continuation only counts if it STARTED from a non-comment line.
  Upstream, the un-narrow version of this bug dropped 29 of 32 Hydra overrides on STATE, which
  fell back to `name=debug`, auto-resumed an unrelated checkpoint, and **exited 0**.
* **no arm's own scorer is wired into a launcher.** They clip to `[0,14]`, skip
  `pearson_edistance` and `clustering_agreement`, label controls `"empty"`, and run cell-eval
  0.7.2 / pdex 0.2.2–0.3.0. Reference only.
* **each header states objective, question, varied/fixed, and pre-registered outcomes**
  (RULE 0c) **and quotes the `meanpert` / `transfer` / `experimental` anchors.** This caught
  three of the four launchers missing `experimental` 0.3722 — the test is not decorative.
* **no payload is tracked, and the tracked tree stays under 40 MB.**

## C6 — STATE reads our data (step 4b)

**Finding: STATE trains on `rhj_fewshot` through a third `source_format`, and the reason it
needed one is the single most load-bearing detail in the port.**

Measured on a 4,000-cell-per-line subsample, 2026-09-17:

| | rows | note |
|---|---|---|
| train | 10,826 | HepG2 finetune 2,826 + jurkat 4,000 + RPE1 4,000 |
| validation | 330 | 182 HepG2 perturbed rows + the controls that join them |
| test | 1,140 | 992 HepG2 perturbed rows + their controls |
| matrix | `4000 x 6983` float32, max 6.97 | axis order, `['DPM1','GCLC','NIPAL3']` |
| cell types | `HEPG2`, `jurkat`, `RPE1` | exact casing, as its configs match on |
| batch | `(64, 6983)` `pert_cell_emb` | full training batch flows |

Controls joining a split is **their** semantics, not ours: a cell type's controls enter a
split only when that type has a perturbed cell there (`cell_load` behaviour, kept).

### Why a third format and not their existing `anndata` one

Their `anndata` branch normalises, when asked, inside `_transform_expression` — which divides
by `x.sum()` **after gene selection**, i.e. the row sum over 6,983 genes. Our whole
emission-space argument depends on the denominator being the row sum over each file's **FULL**
gene panel.

⚠️ The two differ by a per-cell factor (the axis carries ~86.6% of the library, median ratio
≈1.15). A per-cell factor leaves `pearson_delta` untouched — it is scale-invariant per delta
vector — but it **shifts `mae` and moves the DE calls**, and `mae` is one of our four
comparison metrics. It would also break the exact 10,000 → 2,000 bridge map, which is only
exact because both spaces share a denominator.

So normalisation happens **once**, in `occam_data`, on the full panel, and the datamodule
**refuses to start** with `target_sum` or `log1p` set in occam mode rather than letting a
second transform through.

### What was changed, and what was not

| file | change |
|---|---|
| `tx/data/occam_source.py` | NEW. One `_Source` per cell line, read via `occam_data`. |
| `tx/data/zarr_perturbation.py` | `source_format="occam"` branch in `_Source`, `_ensure_open`, `_load_source`, `_read_split`; 5 new kwargs; the refusal above; `__getstate__` drops the matrix. |
| `configs/data/occam_h5ad.yaml` | NEW. Every training value transcribed unchanged from the official recipe. |
| ~~`occam_rhj_fewshot.toml`~~ | **WITHDRAWN (C23).** It named our three h5ads in `[datasets]`, but the benchmark directory already is the source manifest, and having both meant two owners for one list. Replaced by `ZarrExperimentConfig.from_occam()`, which derives the dataset names from `bench.files`. The datamodule now refuses a TOML and a bench dir together. |

Two things deliberately NOT reused:

* **`occam_split_indices` is not called per line.** It asserts over the *concatenated* panel
  (validation must cover every val perturbation; every control must be in finetune), and
  those are false for one line alone — jurkat contributes no validation rows at all. Reusing
  it would mean weakening its assertions, which is the opposite of their purpose. Per-line
  invariants are asserted separately in `occam_fold_for_line`.
* **`selected_ensembl_gene_ids_path` is refused, not ignored.** The matrix already carries
  exactly the frozen 6,983 genes in axis order, so a second subset could only reorder or
  shrink it.

`num_workers: 0`, matching upstream's own `${NUM_WORKERS:-0}`: an occam source holds its
matrix in RAM and is dropped in `__getstate__`, so a worker is never sent several GB through a
pickle — it raises with the reason instead.

## C7 — ⛔ THE BRIDGE GATE PASSES, to four decimal places

**Finding: the two-phase bridge is inert. It adds no signal and no bias.**

Job `18043091`, 9 min 21 s on the `cpu` partition, 2026-09-17. The "competitor" was the
IDENTITY function on its own control input, so the exported matrix IS the `ctrlcell` baseline
— routed the long way round: `load_panel_split` → eval space → **10,000** → the arm →
**2,000** → `assert_same_space` → `pred.npy` → `scripts/eval_competitor.py` →
`protocol.run_split` → cell-eval 0.6.0.

Compared against **split 0's own** `ctrlcell` row, not the 5-split mean:

| metric | direct `ctrlcell`, split 0 | through the bridge | difference |
|---|---|---|---|
| `pearson_delta` | −0.00031878 | **−0.0003** | **0** to 4 dp |
| `mae` ↓ | 0.029577 | **0.0296** | **0** to 4 dp |
| `overlap_at_N` | 0.0 | **0.0000** | **0** |
| `discrimination_score_l1` | 0.50456 | **0.5046** | **0** to 4 dp |

⚠️ Read against **split 0**, not the all-5-split `ctrlcell` mean (−0.0018 / 0.0296 / 0.0000 /
0.5038). The split-to-split std on `pearson_delta` is 0.001–0.003, so the 5-split mean and any
single split differ by more than this gate's own resolution. Comparing to the mean would have
made an exact pass look like a 0.0015 discrepancy.

**What this rules out.** The double conversion (eval → 10,000 on the way in, 10,000 → 2,000 on
the way out) is exact: if the analytic map were wrong by even a per-cell factor, `mae` would
have moved — it is the metric that is *not* scale-invariant, and it did not move. The control
draw, the row order, the cohort enforcement and the `pred_ctrl = ctrl_B` asymmetry all
survived the file round-trip.

**Competitor training is unblocked** by the plan's own condition: *"⛔ This is the gate — no
competitor training launches before it passes."*

Reproduce with:

```bash
PYTHONPATH=src:competitors/shared python scripts/export_gate_ctrlcell.py --split 0
sbatch --qos=hooke-predict --wckey=hooke-predict scripts/slurm/rhj_competitor_gate.sbatch
```

## C8 — CellFlow reads our data (step 4c)

**Finding: CellFlow trains on `rhj_fewshot` through a `load.py` sibling, and its model block
is byte-identical to the official recipe.**

Measured on the FULL benchmark, 2026-09-17 (its own venv, no subsampling):

| | value |
|---|---|
| train | **612,122 × 6,983** float32, max 8.36 |
| per source | HepG2 finetune **101,252** · jurkat **262,956** · RPE1 **247,914** |
| train perturbations | 2,394 · **28,474 controls** |
| validation | **6,731** cells, **103** perturbations, **0 controls** |
| var index | `['ENSG00000000419', 'ENSG00000001084', ...]`, symbols alongside |
| `gene_pert` | `[('genetic', 'KIAA1143', '')]` — the tuple shape their schema parses |
| vocabulary | **2,381** tokens |
| axis train == val | True |

`pca`, `model` and `train` blocks verified **identical** to
`configs/replogle_nadig_fewshot.yaml` by parsing both, not by reading the diff.

### The three things that needed deciding

1. **The perturbation token is a gene SYMBOL, not an Ensembl id.** `load.py`'s docstring says
   Ensembl because their dumps carry Ensembl; the code one-hot encodes an opaque string.
   `rhj_fewshot` identifies perturbations by symbol throughout — splits, exclusions, the
   611-perturbation cohort, cell-eval's own `target_gene` — so the symbol IS the identity, and
   a perturbation target need not even be on the gene axis.
2. **The gene axis is Ensembl in `var_names`,** matching their convention, while our
   evaluation h5ads use symbols so cell-eval can exclude each target gene by name. Safe
   because the bridge is **positional** — `assert_axis_order` refuses any other order, since a
   reordered axis would silently relabel every gene while every metric still looked fine.
3. **The vocabulary is read from the benchmark's split lists, not from the loaded cells.**
   ⚠️ It fixes the model's condition-input width and cannot grow after training: a test
   perturbation with no column makes `cf.predict` raise `Representation ... not found` and
   produce NO predictions. Upstream hit exactly this on 8 of their 380 test perturbations.

⚠️ `pca.n_comps: 256` is a **PCA rank, not a gene count**, and stays 256 while the axis
widens. It is the dimensionality the flow is learned in; the axis change affects what PCA is
fitted *on*.

`train_cellflow.py` **refuses** if both `occam_bench_dir` and `split_path` are set — the
benchmark owns the split, and a second split file could only disagree with it silently.

## C9 — PerturbDiff reads our data (step 4d)

**Finding: PerturbDiff trains on `rhj_fewshot` through a third store, and the surface is
larger than the other three arms because of its cache/store abstraction.**

| file | change |
|---|---|
| `src/data/occam_bridge.py` | NEW. `OccamMetadataCache`, `OccamStore`, `get_selected_gene_vars_occam`, `retrieve_counts_occam`, `split_index_occam` — duck-typed twins of the five zarr equivalents. |
| `src/data/metadata_cache.py` | `get_occam_cache`, keyed by path like `get_zarr_cache`. |
| `src/data/dataset/dataset_core.py` | `source_format == "occam"` in the store constructor and both format-specific methods. |
| `src/data/data_module/data_module{,_setup}.py` | `OccamPerturbationDataModule` + `occam_setup` / `occam_setup_dataset`. |
| `src/apps/training/training_datamodule_builder.py` | `data_name: OccamH5adFinetune`. |
| `configs/{data,path,cov_encoding}/occam*.yaml`, `configs/manifests/occam_rhj_fewshot.toml` | NEW. |

### Four details that would each have been silent

* **`pert_key_type: symbol`, not `ensembl`.** Their zarr path derives an Ensembl perturbation
  string and then translates Ensembl → symbol for the GenePT lookup. Our `pert_dict` keys are
  already symbols, so that translation must not run — with `ensembl` set, every lookup misses
  and `max_zero_embedding_frac: 0.5` aborts the run. Loud, but the fix belongs in the config.
* **THREE width values must move together:** `data.pad_length`, `model.input_dim` and
  `model.hidden_num[0]`. `data.embed_shape` interpolates `model.input_dim` and the collate
  **asserts** `embed_shape == pad_length`, so a half-applied change fails at the first batch
  rather than training on a silently wrong width. The launcher overrides all three.
* **`selected_gene_file` is refused, not ignored.** A silently ignored gene panel is how an arm
  ends up scored on a different axis than it trained on.
* **The split is computed live and nothing is cached or written.** `split_index_occam` matches
  `split_index_json`'s semantics exactly — HepG2 splits three ways over PERTURBED rows, jurkat
  and RPE1 contribute every row, and **all controls are added to every non-empty split** — but
  reads the benchmark instead of a JSON, so there is no stale-index surface at all. It
  **raises** if HepG2's validation or test split comes out empty, because an arm with no
  validation split silently keeps its final weights instead of selecting a checkpoint.

### Measured on the FULL benchmark, 2026-09-17

| | value |
|---|---|
| datasets | `['HEPG2', 'jurkat', 'RPE1']` |
| vocabularies | 2,395 perturbations · 3 cell types · 167 batches |
| train | HepG2 **96,276** · jurkat **250,943** · RPE1 **236,429** |
| validation | HepG2 **6,731** · jurkat 0 · RPE1 0 |
| test | HepG2 **37,490** · jurkat 0 · RPE1 0 |
| gene panel | 6,983, `['ENSG00000000419', ...]`, symbols `['DPM1', 'GCLC', 'NIPAL3']` |
| one retrieved row | `(1, 6983)`, max 5.57 |

⭐ **`validation` is 6,731 here and 6,731 in CellFlow (C8) — two independently written readers
agreeing on the same split, to the cell.** That is the cross-check that makes both credible.

ⓘ The train counts are PERTURBED cells only: `split_out_control` moves the controls into the
basal pool and mutates `*_num_cell` in place, which is their design and applies to the zarr
path identically. HepG2 96,276 + its 4,976 controls = 101,252, which is exactly CellFlow's
HepG2 finetune count.

One matrix per cell line lives in a process-global cache and is **never pickled**, the same
discipline `zarr_store.py` documents: three `H5adSentenceDataset`s are built over the same
files, and HepG2 alone is 145,473 × 6,983 float32 ≈ 4.1 GB.

## C10 — ⛔ RE-COPYING SILENTLY REVERTED THE ENTIRE PORT, and the tests passed

**Finding: `scripts/copy_competitor_code.py` overwrote all NINE adapted files with the
pristine upstream versions in one command, and `tests/test_competitor_manifest.py` then
PASSED.** Caught by reading `grep -c occam` on the four wired files, not by a test.

MEASURED 2026-09-17. The mechanism is simple and was entirely predictable in hindsight: the
allowlist copies upstream *trees* (`gspp`, `src`, `train_cellflow.py`, ...), and nine of the
files inside them are files we deliberately adapted. A plain re-run to refresh the manifest
copied over every one.

**Why the sha256 test could not catch it.** That test asserts "every recorded file matches its
upstream sha256, EXCEPT the ones in `ADAPTED`". After the revert, the adapted files matched
upstream again — which is indistinguishable from a file that was never adapted. A reverted
port and a clean port look identical to a hash comparison.

**The fix, in three parts:**

1. **`PROTECTED` in `scripts/copy_competitor_code.py`** — the nine paths, per arm. They are
   skipped, and recorded in `MANIFEST.json` with `"adapted": true` and `"sha256": null`, so the
   manifest never claims they match upstream when they deliberately do not.
2. **The skip is printed**, `🔒 N adapted file(s) PRESERVED`, so a re-copy says what it did.
3. **`test_every_adapted_file_is_protected_from_being_re_copied`** — walks the copied tree,
   finds every file whose name is in `ADAPTED`, and asserts that any of them which *also exist
   upstream* are in `PROTECTED`. New files (`occam_source.py`, `occam_bridge.py`,
   `occam_bridge/occam.py`, `export_occam.py`) need no guard: no allowlisted upstream tree
   contains them, so a copy cannot touch them.

Restored from `git checkout --`, which is the only reason this cost minutes rather than the
whole port. **All nine wirings verified present after a second re-copy.**

## C11 — the four Phase-A adapters, and the one deviation all four share

**Finding: every arm conditions on ONE drawn control cell per predicted row, which is OUR
protocol and differs from all four of their own conventions.**

| arm | their control convention | what the adapter does |
|---|---|---|
| **TxPert** | `match_cntr`/`avg_cntr`: a control drawn from, or averaged over, the cell's (cell_type, batch) pool | `_PairedDataset` overrides `_get_control_data` ONLY, returning row `idx`'s own control |
| **STATE** | a control drawn from the (cell_type, batch) pool | batch dict built to `collate_fn`'s contract with `ctrl_cell_emb` = the drawn cells |
| **CellFlow** | `predict` over a control *population* per condition | rows GROUPED BY perturbation, each predicted from its own drawn control cell |
| **PerturbDiff** | `mapping_cells`: a control drawn at RANDOM from the (cell_type, batch) group | `self_condition["cont_emb"]` = the drawn cells |

**Why this is the right choice, not a convenience.** The paired split-half design is what buys
the **0.001–0.003** split-to-split std on `pearson_delta`. If a competitor were fed averaged or
randomly drawn controls while `ctrlcell`, `ctrlmean`, `meanpert`, `transfer` and our own model
were all fed the panel's drawn ones, the comparison would stop being paired and the anchors
would stop being anchors.

### Emission spaces, declared per arm

| arm | head | declared to `export_split` |
|---|---|---|
| TxPert | gene space | `from_target_sum=10_000, from_log1p=True` |
| STATE | `output_space: gene` | `from_target_sum=10_000, from_log1p=True` |
| CellFlow | 256-component PCA decode, `clip_min=0.0` | `from_target_sum=10_000, from_log1p=True` |
| PerturbDiff | gene space through `output_fn: relu` | `from_target_sum=10_000, from_log1p=True` |

⚠️ A relu head is **not** a guarantee of log1p shape, which is why every declaration is
explicit and checked by `assert_same_space` rather than inferred. And ⚠️ **CellFlow's
`clip_max` is left unset on purpose**: their own scorer clips to `[0, 14]`, ours does not, and
clipping only the competitor would be an unrecorded asymmetry.

### Checkpoint resolution refuses rather than guesses

Each adapter resolves the checkpoint the arm selected **on validation** — TxPert's
`best_ckpt_path.txt`, STATE's `best_celleval.ckpt`, PerturbDiff's single `save_top_k=1` file,
CellFlow's `run_meta.json` selection block — and **exits** if it is absent rather than falling
back to the final epoch, which would be silently incomparable to every other arm. PerturbDiff's
also refuses if it finds MORE than one checkpoint, because `save_top_k=1` only prunes files it
tracked itself and a smoke checkpoint once sat beside a real one.

### ⚠️ ALL FOUR ARE UNEXERCISED

No competitor has trained yet, so no checkpoint exists and not one of these adapters has run.
The plan's own rule applies and is not optional: **one arm, one seed, one split end to end
before any 4×5 sweep.** The last two 135-task launches in this campaign died on code that had
never executed once on the real path.

## ⭐ C12 — TxPert TRAINS ON OUR DATA. The port works end to end.

**Finding: TxPert's first real execution against `rhj_fewshot` reached training with no code
change beyond the port.** Job `18067027`, `hop39`, 2026-09-18.

What it loaded, straight from its own log — every line is our benchmark, not theirs:

```
occam: rhj_fewshot: 6,983 genes · val 103 · test 621 · 724 (file, pert) exclusions
                    · lines ['HEPG2', 'RPE1', 'jurkat']
  nadig_2024_hepg2.h5ad:      145,473 x 6,983 ( 4,976 controls)
  nadig_2024_jurkat.h5ad:     262,956 x 6,983 (12,013 controls)
  replogle_2022_rpe1_e.h5ad:  247,914 x 6,983 (11,485 controls)
occam: combined 656,343 cells x 6,983 genes | 2,394 perturbations | X max 8.36
occam: split names -> train 2,394, validation 103, test 621
adata_output_dim : 6983
graph_dict keys: dict_keys(['go', 'string'])
kg_list: ['go', 'string']
zarr splits: 2395 train (global), 103 val, 621 test (both hepg2 only)
```

Every deliberate decision is visible and correct:

| decision | evidence in the log |
|---|---|
| our data, read in place | our three h5ad filenames, 656,343 cells |
| our gene axis | `6,983` throughout, and `adata_output_dim : 6983` |
| our splits | `103 val / 621 test`, `both hepg2 only` |
| our exclusions | `724 (file, pert) exclusions` |
| GO + STRING only | `kg_list: ['go', 'string']` — pxmap/txmap gone |
| normalisation is ours-for-them | `X max 8.36`, i.e. log1p space, not counts |

⭐ **Data loading took 5 min 25 s** (02:39:12 → 02:44:38) for 656k cells over three h5ads, and
the whole path from launch to the Lightning trainer took **~6 min**.

ⓘ **The perturbation vocabulary self-extended by 20 symbols** absent from TxPert's shipped
`gears_gene_set.csv` (9,854 → 9,874), including `ALG1L`, `C14orf178`, `NEDD8-MDP1` and
`CCDC144NL` — the same symbols our own split had to drop for having no graph node. That is the
`_extend_pert_vocabulary` path working as designed: it keeps the cohort intact rather than
silently dropping perturbations, which is what `scripts/eval_competitor.py` would have refused
on anyway.

⚠️ **This validates the DATALOADER, not the adapter.** `competitors/TxPert/export_occam.py` has
still never run — it executes in the export leg (`18067028`), which is `afterok`-chained behind
training. That adapter remains the least-tested part of the port.

### Provenance: it sat in a queue for 19 hours first

Submitted to `vco` at 2026-09-17T15:11 and never started — `QOSGrpGRES`, because another user
held **32 of `vco`'s h100s** for the whole period. Cancelled and resubmitted on
`hooke-predict` (13 running, 0 pending), where it started in **15 seconds**.

⚠️ Worth remembering when a competitor arm looks stalled: check `squeue -h -o "%q %T %u %b"`
for who owns the QOS's GRES cap before assuming the job is broken. Nineteen hours produced no
log file at all, which reads exactly like a failure and was not one.

## C13 — PerturbDiff launched, both perturbation encodings

**Two training runs, `hooke-predict`, started immediately 2026-09-18.** The only axis varied is
the perturbation encoding; everything else is byte-identical between them.

| arm | encoding | train | export | score | run name |
|---|---|---|---|---|---|
| `perturbdiff_genept` | `genept` | `18072184` | `18072185` | `18072186` | `pd_occam_pert_celltype_lr0.002_genept_cs16_s42` |
| `perturbdiff_onehot` | `onehot` | `18072187` | `18072188` | `18072189` | `pd_occam_pert_celltype_lr0.002_onehot_cs16_s42` |

Verified from each run's own resolved config:

| setting | value | source |
|---|---|---|
| `use_cell_set` | **16** | the FEWSHOT recipe (`PD_CELL_SET=16`), not the 32 of their general Replogle setting |
| `cell_set_group_by` | **`pert_celltype`** | the fewshot recipe; NOT `pert_celltype_batch` |
| `input_dim` / `hidden_num` / `pad_length` / `embed_shape` | **6983** / `[6983,512]` / 6983 / 6983 | all four moved together, as the collate asserts |
| `source_format` / `occam_bench_dir` | `occam` / our `rhj_fewshot` | our h5ads, read in place |
| `pert_key_type` | `symbol` | `rhj_fewshot` identifies perturbations by symbol |
| `normalize_counts` | `false` | already log1p(1e4) on the FULL panel row sum |
| lr / micro_batch / steps / validations | 0.002 / 128 / 80,000 / 5 at every 16,000 | transcribed unchanged |
| checkpoint monitor | `validation/celleval/selection_score`, max, `save_top_k=1` | theirs |

⚠️ **`use_cell_set: 32` was dead in the config file** — the launcher had always passed 16, so the
config's value never applied. A value in a config that never takes effect is worse than no
value, so the config now says 16 and cites the recipe. Same for `cell_set_group_by`, which said
`pert_celltype_batch` while the launcher passed `pert_celltype`.

### ⚠️ Why BOTH encodings: genept does not cover our panel

`genept` is what their fewshot launcher passes, and its pickle was built for their **2,000-gene**
panel. Measured coverage of our symbols:

| set | covered by genept | zero embedding |
|---|---|---|
| **the frozen 611-perturbation cohort (what is scored)** | 547 (**89.5%**) | **64 (10.5%)** |
| `val` (103) | 94 (91.3%) | 9 (8.7%) |
| `test_unseen_pert` (621) | 552 (88.9%) | 69 (11.1%) |
| `train` (2,381) | 2,021 (84.9%) | 360 (15.1%) |

The pickle holds **2,024 keys** — their vocabulary, not ours. It clears
`max_zero_embedding_frac: 0.5` comfortably (0.151), so the run proceeds — but **64 of the 611
scored perturbations would share a single all-zero representation**, i.e. the model cannot tell
them apart, for a reason that is an artifact of their asset rather than of the model.

`onehot` is full-coverage over all 2,381 symbols and is the **default** in their own
`cov_encoding` config, whose comment reads *"what the benchmark run uses"*. Their published
0.40306 came from the launcher, i.e. genept. Running both measures what the encoding is worth
on our panel instead of guessing, and the genept arm stays comparable to their number.

### ⚠️ A collision the training leg hid

The train leg separates variants by run name (`PD_RUN_NAME` carries the encoding), but the
**export and score legs keyed only on arm + seed**, so both encodings would have written to
`runs/competitors/perturbdiff/s42/` and `compet_perturbdiff_s42` — the second silently
overwriting the first. Added `ARM_SUFFIX` to both legs (`perturbdiff_genept` /
`perturbdiff_onehot`); it names directories only, and the adapter and venv still resolve from
the base arm name.

`export_occam.py` needed the same treatment: its checkpoint glob `pd_occam_*_s{seed}` matches
both encodings, and its own >1-checkpoint guard would then have refused both. It takes `--penc`
now, which narrows the glob and is also forwarded to `cov_encoding.replogle_gene_encoding` so
the sampling config rebuilds the right covariate encoder.

## ⭐ C14 — the genept pickle was TRUNCATED, not incapable. Rebuilt at 100% cohort coverage.

**Finding: PerturbDiff's genept handicap was an artifact of the asset, and it is removable
without inventing anything.**

`replogle_gene_emb_dict_perturbation_emb_dict.pkl` — the pickle their fewshot recipe passes —
holds **2,024 keys**, their 2,000-HVG panel's vocabulary. The **full GenePT v2 release**, on
disk at `GenePT/genept_embeddings/extracted/GenePT_emebdding_v2/`, holds **133,736 keys at the
same 3,072 dimensions**.

⛔ **The check that licenses the rebuild**, measured 2026-09-18:

| check | result |
|---|---|
| their keys present in the full release | **2,009 / 2,024** |
| max abs value difference, 300 shared keys | **0.000e+00** — byte-identical |

Same embedding model, same dimensionality, identical values. So the rebuild **invents nothing**;
it stops truncating an asset that already covers our genes. Per the project's *"not biology is
not the same as cheating"* rule this is using an asset at its real coverage, which makes the arm
a more faithful representation of the model. `scripts/build_genept_for_occam.py` **refuses** if
the values ever disagree, because then it would be changing the arm rather than un-truncating
it.

### The union, not a replacement — and that is what closes the gap

⚠️ Their pickle has **15 keys the full release lacks** (`AC118549.1`, `C12orf45`, `C7orf26`,
`C9orf16`, `FAM102B`, …) — aliases they added. The output is `full ∪ theirs`, their value
winning on a shared key. Those 15 are exactly what takes the cohort from 98.7% to **100%**:

| set | theirs (2,024 keys) | **rebuilt (2,376 keys)** |
|---|---|---|
| **the 611 scored perturbations** | 547 (89.5%), **64 zero** | **611 (100.0%), 0 zero** |
| all 2,381 symbols | 2,021 (84.9%), 360 zero | **2,376 (99.8%), 5 zero** |

The 5 that remain (`C1orf109`, `CCDC130`, `DDN`, `FRG2`, `TDGF1`) are **train-only** and absent
from both sources; they keep a zero embedding.

⚠️ **This makes the genept arm NOT comparable to PerturbDiff's published 0.40306**, which was
measured with the truncated pickle on their own panel. That caveat travels with the number.

Artifact: `data/assets/genept_occam_rhj_fewshot.pkl` (65.7 MB) plus a `.json` manifest recording
both source sha256s, the value check, the merge rule and the residual 5.

### Relaunched

| arm | encoding | train | export | score |
|---|---|---|---|---|
| `perturbdiff_genept` | genept, **rebuilt pickle** | **`18072825`** | `18072826` | `18072827` |
| `perturbdiff_onehot` | onehot | `18072187` | `18072188` | `18072189` |

The first genept chain (`18072184`) was cancelled ~20 min into 80,000 steps and its checkpoint
directory removed — cheap, and it was training against 64 zeroed perturbations. The onehot arm
was left running untouched: it has no coverage gap, so nothing about it changed.

ⓘ **The comparison is now cleaner than originally planned.** genept-vs-onehot was going to
confound "which encoding is better" with "genept is missing 10% of the cohort". Both arms now
have full cohort coverage, so the contrast measures the encoding.

## C15 — the upstream benchmark's name is gone, and comparability to it is no longer a goal

**Decision 2026-09-18, the user's words: *"we never care about being comparable to hooke-tx. actually remove even mentions of the word hooke from the codebase"*.** **537 references removed** across the four arms, our scripts, our docs and the launcher
headers. Two classes are deliberately KEPT, and `tests/test_competitor_manifest.py` now
asserts nothing else comes back:

| kept | why |
|---|---|
| the Slurm **QOS and wckey** (87) | a cluster resource, not our naming. Every launcher submits to it and four chains were running on it. | | ~~the copier~~ | **deleted 2026-09-18.** Its `UPSTREAM` constant and denylist globs were the last strings naming that tree, and a tool that copies from it is the opposite of independence. |

⚠️ **The sweep broke both of those once each**, which is why they are now shielded and
commented: `UPSTREAM` was rewritten so the copier could not find upstream, and the denylist's
its denylist glob was rewritten, which silently stopped denying anything and let a tree of reference files return on the next copy.

### What was deleted rather than renamed

~200 of the references lived in upstream **reference material for a benchmark we no longer
compare against**: their launchers, their zarr manifests, their design docs, their old configs
and the tests that exercised them. None was reachable from an occam config. Deleting beat
renaming, and the tracked tree fell from **6.6 MB to 5.8 MB**.

### The rename, where identifiers actually moved

| before | after | |---|---|
| the CellFlow bridge package | **`occam_bridge/`** | | its two dump loaders | `load_dump_anndata` / `load_dump_sources` | | the scorer PYTHONPATH arg | `scorer_pythonpath` | | the wandb project | **`Occam-Competitors`** | ### ⭐ CellFlow's selection now scores through OUR cell-eval

Its callback used to shell out to a sibling repo's `celleval_artifacts.py` in that repo's own
venv — which pinned **cell-eval 0.7.2 / pdex 0.2.2–0.3.0**, not the **0.6.0 / pdex 0.1.24**
every reported number comes from. Two problems, both now gone:

* the arm depended on another repo existing at a hard-coded path, and
* its in-training selection metric was computed by a **different scorer** than its reported one.

`cellflow_selection.py` now calls `occam.vcell.panel.celleval_v1.run_celleval_v1` in a
subprocess. Still a subprocess because jax and cell-eval do not co-exist in one environment —
the same constraint that makes the prediction bridge two-phase. The config carries no scorer
paths at all now, and it **raises** if cell-eval returns no `pearson_delta` /
`discrimination_score_l1`, because then no checkpoint could be selected.

### The copier is DELETED

`PROTECTED` could not express *"this whole tree is ours now"*, because the allowlist copied
**directories** (`src`, `configs`, `gspp`). Re-copying therefore restored the originals — and
undid the rename — **twice in one afternoon**. It was first made to refuse, then **deleted**:
a tool whose purpose is to copy from that tree cannot coexist with independence from it.

`scripts/refresh_competitor_manifest.py` does the one job that remains — updating the
integrity baseline after a deliberate edit — and reads nothing outside this repo.

### Manifest schema 1 → 2: integrity, not upstream equality

Schema 1 recorded each file's **upstream** sha256 and a test asserted every file still matched
except an `ADAPTED` allowlist. That became meaningless once the port diverged from upstream
across ~70 files. The hashes are now the port's **own** baseline, which is what the check is
actually for: catching a vendored file modified without the record being refreshed. It caught
exactly that, twice, during this change.

## ⛔ C16 — PerturbDiff died 4 minutes into training. Not the rename: a dataset-name contract.

**Both arms FAILED** (`18072825`, `18072187`, exit 1 after ~4 min) at
`src/common/utils.py::get_short_dsname` → **`assert 0`**.

That function maps a dataset name to a family by substring — `tahoe` / `pbmc` / `replogle` /
`cellxgene` — and asserts otherwise. `_compute_loss` calls it on **every batch**. My
`occam_setup` named datasets after the cell line (`HEPG2`, `jurkat`, `RPE1`), which matches
none of them.

⭐ **Their own zarr path documents this exact trap and solves it the same way I now do** — and
it also calls `get_short_dsname(name)` at setup *"to fail loudly here rather than deep inside
the model."* My port skipped that one line, which is the whole difference between failing in
4 minutes of GPU time and failing at second zero.

**Fix:** datasets are named **`replogle_nadig_<line>`**. That is the honest prefix — these ARE
the Replogle/Nadig sources — so the family the loss branches on is the one it would have picked
for the same data as a zarr dump. The setup-time check is now in place too.

`export_occam.py` resolves the dataset by **suffix** rather than hardcoding either form, since
the dataset name is no longer the cell line.

Verified before relaunch: `['replogle_nadig_HEPG2', 'replogle_nadig_jurkat',
'replogle_nadig_RPE1']` → `['replogle', 'replogle', 'replogle']`.

| arm | encoding | train | export | score |
|---|---|---|---|---|
| `perturbdiff_genept` | genept, rebuilt pickle | **`18075222`** | `18075223` | `18075224` |
| `perturbdiff_onehot` | onehot | **`18075225`** | `18075226` | `18075227` |

Both training, finite gradients (total norm 12.40 / 6.15), **0.145 s/step** → 80,000 steps ≈
**3.2 h** plus five validation cell-eval passes. The two differ in the expected place:
genept trains `class_embedding.emb.weight`, onehot `class_embedding.null_emb.weight`.

## ⭐ C31 — TxPert control-regime A/B: mean leads on VALIDATION, but that comparison is confounded

**Question.** TxPert's published config trains on the `(cell_type, batch)` MEAN control while
our benchmark scores every arm from ONE drawn half-B control cell. Does training on the mean or
on single cells produce the better model *for our protocol*?

**What is varied / held fixed.** One axis: `datamodule.match_cntr/avg_cntr`. Everything else
identical, including `omit_cntr: true` on both. Both arms are **inferenced on single cells**.

### Validation curves (5 passes, 103 HepG2 perturbations)

| pass | single-cell controls | mean controls |
|---|---|---|
| 1 | 0.3416 | 0.3375 |
| 2 | 0.3644 | 0.3859 |
| 3 | 0.3744 | 0.3913 |
| 4 | 0.3781 | 0.3900 |
| 5 (final, selected) | **0.3844** | **0.3988** |

Mean-control leads by **+0.0144**, and both rise monotonically to their final pass.

### ⚠️ THIS DOES NOT YET ANSWER THE QUESTION, and must not be quoted as if it does

**The validation runs under each arm's OWN training regime.** `match_cntr`/`avg_cntr` are handed
to every dataset the datamodule builds (`datamodule.py:1029-1030` train, `1073-1074` val), so
the mean arm is validated on **mean controls** — a denoised input the single arm never sees, and
one our test protocol never provides. The mean arm is being marked on an easier paper.

So the +0.0144 is the sum of two things that this comparison cannot separate: whether
mean-control training produces a better model, and whether mean-control *validation* is simply
an easier measurement.

**The decisive comparison is the TEST panel**, where both arms are exported identically — one
drawn half-B control cell per predicted row. If the ordering reverses there, the validation gap
was the measurement; if it survives, mean-control training genuinely helps even when the model
is later asked to work from a single cell.

⚠️ And the standing fairness constraint still applies regardless of which wins: the panel gives
every arm a single control cell, so **whichever trains better, inference stays single-cell**.
Adopting mean-control *inference* would hand TxPert a denoised basal no other arm receives.

## ⭐⭐ C30 — PerturbDiff 0.2449, and CellFlow's PCA-truth inflation MEASURED at ~0.15

### PerturbDiff (genept) — FINAL, 5/5 splits, `model_nocal`

Checkpoint `epoch=12-step=64000`, resolved from the `.best_ckpt_path` marker (B1), sampler
seeded per split (B2), per-cell batch covariate (M1), whole split in one `predict` call (M2).

| split | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| 0 | 0.2472 | 0.0288 | 0.0738 | 0.6680 |
| 1 | 0.2486 | 0.0286 | 0.0712 | 0.6634 |
| 2 | 0.2445 | 0.0286 | 0.0730 | 0.6638 |
| 3 | 0.2455 | 0.0286 | 0.0729 | 0.6680 |
| 4 | 0.2388 | 0.0287 | 0.0759 | 0.6617 |
| **mean** | **0.2449 ± 0.0038** | **0.0287 ± 0.0001** | **0.0734 ± 0.0017** | **0.6650 ± 0.0029** |

**Clears `transfer` 0.2323**, and currently the best competitor. Still below ours (0.3415) and
`experimental` (0.3722).

### ⭐ CellFlow: the PCA-reconstructed truth was inflating `pearson_delta` by ~0.15

C3 predicted the old selection metric was optimistic because it decoded the TRUTH through the
rank-256 basis. The retrain scores the same quantity the reported metric does — raw gene-space
half-A cells — on all 103 conditions. Same model family, same data, same scorer:

| validation pass | OLD: PCA-reconstructed truth, 60 of 103 conditions | **NEW: raw gene-space truth, 103 of 103** |
|---|---|---|
| 1 | 0.3114 | **0.2485** |
| 2 | 0.3617 | **0.2453** |
| 3 | **0.4027** | **0.2448** |
| 4 | 0.2703 | 0.2310 |

**The inflation is ~0.15 `pearson_delta` at the peak** — larger than the entire gap between
`meanpert` and `transfer`. Anything selected or reported on the old metric was reading a
number that does not exist on the real target.

⚠️ **And the SHAPE changed, not just the level.** The old curve rose 0.3114 → 0.3617 → 0.4027
and looked like a model still improving. The new one is **flat-to-declining** (0.2485 → 0.2453
→ 0.2448 → 0.2310), i.e. it peaks at the first validation. The old rise was partly C4: each
pass scored a DIFFERENT random 60 of 103, so the "improvement" mixed real change with a
changing panel. Two defects that individually look like tuning noise together manufactured a
training curve.

### Standings so far

| | `pearson_delta` | note |
|---|---|---|
| `meanpert` | 0.1324 | ignores perturbation identity |
| `transfer` | **0.2323** | the bar |
| STATE | 0.2362 ± 0.0019 | |
| **PerturbDiff genept** | **0.2449 ± 0.0038** | best competitor |
| **Occam (ours)** | **0.3415** | |
| `experimental` | 0.3722 | replicate agreement, not a ceiling |
| ~~TxPert~~ | ~~0.3077~~ | withdrawn (C28), retraining |

⚠️ TxPert's withdrawn 0.3077 was the only competitor number above ours-minus-a-bit, and it was
wrong in a direction we have not yet measured. Its two retrains (single- vs mean-control) will
say whether it genuinely sits above the others.

## ⭐⭐ C29 — STATE: 0.1306 → 0.2373. The three export fixes were worth +0.107, no retraining.

**Objective / anchors** as C28. **Question:** did the S1-S3 export fixes move STATE off the
`meanpert` baseline, or was the grouping never the cause?

**Answer: they moved it past `transfer`.** Same checkpoint (`best_celleval.ckpt`, step 63,999),
same panel, same scorer — only the export changed.

**FINAL: all 5 splits, `model_nocal`.** ± is the across-split sd.

| metric (direction) | DEFECTIVE (mixed sentences) | **FIXED** | change |
|---|---|---|---|
| `pearson_delta` ↑ | 0.1306 ± 0.0050 | **0.2362 ± 0.0019** | **+0.1056** |
| `mae` ↓ | 0.0337 ± 0.0001 | **0.0291 ± 0.0001** | −0.0046 |
| `overlap_at_N` ↑ | 0.0306 ± 0.0023 | **0.0849 ± 0.0008** | **×2.8** |
| `discrimination_score_l1` ↑ (chance 0.5) | 0.5448 ± 0.0050 | **0.6639 ± 0.0030** | +0.1191 |

Per split: 0.2373 / 0.2388 / 0.2357 / 0.2350 / 0.2340 — spread 0.0048, and the sd HALVED
(0.0050 → 0.0019), which is itself evidence: a contaminated sentence layout added variance
that depended on which perturbations happened to share a batch.

**Read against the anchors:** it was sitting ON `meanpert` **0.1324** — the baseline that
ignores perturbation identity — and now clears `transfer` **0.2323**, the same perturbation
measured in a different cell line. Still below ours (0.3415) and `experimental` (0.3722).

**Every one of the three fixes was export-only**, so this cost a 15-minute re-export rather
than a 6.5-hour retrain: the checkpoint was always fine, we were asking it the wrong question.

### Does STATE need a RETRAIN? No — four independent checks say the training was correct

Asked directly, so verified rather than asserted:

1. **`padded` defaults to True in training.** `training_step(self, batch, batch_idx,
   padded=True)` (`state_transition.py:543`), and grepping every call site shows nothing in the
   training path overrides it — the only `padded=False` call is `_predict.py:154`, a loss probe
   in the predict CLI. Lightning itself calls `training_step(batch, batch_idx)`.
2. **Training sentences are pure and exactly `S` long.** `_create_sentences` groups by
   `(cell_type, perturbation)`; `_create_batches` pads any short sentence to exactly
   `cell_sentence_len` when `not test`. So the concatenated batch is aligned to 64-cell
   boundaries and `reshape(-1, 64, dim)` recovers precisely those sentences.
3. **The batch covariate was per-cell at training**: `batch_name = self.batch[i]`
   (`zarr_perturbation.py:689`), indexed by the row. That is exactly what S3 restored in the
   export — the constant was ours, never theirs.
4. **The in-training validation was already faithful** (`batch_size=1` **with**
   `padded=False`), which is why the same checkpoint reported 0.2049 there throughout.

**The decisive evidence is the result itself.** Changing ONLY the export moved `pearson_delta`
+0.1056. No export change can create signal that is not already in the weights, so the
checkpoint demonstrably carried it. A retrain would cost 6.5 h to reproduce a checkpoint we
have already shown to be sound.

⚠️ **A separate question, still open:** whether STATE's *recipe* is optimal. Its hyperparameters
could not be verified against an official reference — the cited
`reproducibility/state__replogle_nadig_fewshot` does not exist in this checkout — and
`batch_encoder=True` is non-default in every in-repo upstream config. That is a caveat to
report beside the number, not a reason to retrain: it would be re-running an unverifiable
recipe against itself.

⚠️ **Note for the val→test gap model.** STATE's test (0.2373) is now HIGHER than its own
validation (0.2049), where our own arms run ~0.035 *lower* on test than on validation. Not
alarming — different panels (103 vs 611 perturbations) and a different control pairing — but it
means the "val − 0.035" rule does not transfer across arms, and a competitor's validation
number should not be used to predict its test number.

⚠️ **This also retires my earlier caution honestly**: I reported that the new split-0
prediction differed from the defective one by only ~1% on aggregate statistics (between-
perturbation variance 9.049e-04 vs 8.947e-04) and said the score was the real test. It was —
a 1% shift in those summary statistics corresponded to +0.107 `pearson_delta`. **Aggregate
variance statistics are a poor proxy for a DE-based metric**; do not use them to judge whether
a fix landed.

## ⭐⭐ C28 — PORT FIDELITY AUDIT: three of four arms had score-changing deviations

**Objective.** Unchanged: best model on SEEN perturbations in an UNSEEN context (HepG2
`rhj_fewshot`), headline cell-eval `pearson_delta` on the 611-perturbation cohort. Anchors:
`ctrlcell` −0.0018 · `ctrlmean` 0.0016 · **`meanpert` 0.1324** · `transfer` 0.2323 ·
`experimental` 0.3722 · ours 0.3415.

**Question.** C27 showed our export fed STATE mixed-perturbation cell sentences — a defect in
the GROUPING of inputs, which no content check could see. Is that an isolated slip, or a class?

**What is varied / held fixed.** Nothing varied. Each port was read against its OWN pipeline
(native train/predict entry points and official configs), by three independent audits, with
file:line evidence demanded on both sides.

**What each outcome means.** One slip → fix and move on. A class → every arm's number is
suspect until audited, including ones already written down.

**It is a class. Three of four arms were affected, and one of them was TxPert, whose 0.3077
was already recorded as a result (C22).**

### ⛔ STATE — the C27 fix was necessary but NOT SUFFICIENT

| # | finding | measured |
|---|---|---|
| **S1** | `predict_step(..., padded=False)` does not mean "unpadded" — it means **"this whole batch IS one sentence"** (`state_transition.py:409-415`, `reshape(1, -1, dim)`). Our export submitted 4 pure sentences per call, so they were flattened into **one 256-cell sentence spanning ~8-9 perturbations**. `NoRoPE` backbone → no length error, no warning. | `mean\|Δ\|/mean\|pred\|` **0.52**; per-perturbation prediction correlated **0.12-0.25** with the faithful one |
| **S2** | We padded a 30-cell sentence to 64. **STATE's inference does not pad at all** (`_infer.py:810`; the test sampler skips the fill: `if len(sentence) < cell_sentence_len and not self.test`). My C27 `np.resize` tiling repeated cells 0-3 three times and 4-29 twice. | **7.2%** change; exact 2x duplication is attention-invariant (**0.0007%**), so the whole 7.2% was the non-uniform remainder |
| **S3** | The batch covariate was **one constant** (`sorted(...)[0]` = `HEPG2__hepg2-nad-1`, 762 of 42,402 rows) broadcast to all 18,575 rows, against a **trained** `batch_encoder` (`batch_dim=167`) added to every token. A real 30-cell sentence spans ~22 batches. | **0.37** relative change; perturbation-specific delta correlated **0.73-0.86** |

**Both of STATE's own inference paths submit exactly one sentence** — `_infer.py:810,835`
("All tokens in this batch share the same perturbation") and `_predict.py:400`
(`data_module.batch_size = 1`) — and so does our in-training validation, which is why the same
checkpoint scored **0.2049** there and **0.1306** on export.

✅ **STATE's TRAINING was correct**, so no retrain: `training_step` defaults to `padded=True`,
the sampler pads every training sentence to exactly `S`, and the validation pairs
`batch_size=1` **with** `padded=False`. Fixed: one sentence per forward, no padding (0% overhead,
was 131%), per-cell batch (56 distinct, median 19 per sentence), whole split in one `predict`
call, plus a purity assertion — the old guard checked row COUNT, which a sentence-collapse
preserves.

### ⛔ TxPert — 0.3077 IS WITHDRAWN

| # | finding |
|---|---|
| **T1** | Every exported row omitted the `-1` token. Theirs passes `[pert_id, -1]` (`datamodule_hybrid.py:443`) because `data_utils.py:267-271` rewrites a single-gene perturbation as `"SYMBOL+ctrl"`; with `omit_cntr: false` the model adds `z_p[pert] + z_p[-1]` in training, validation **and** their own export. |
| **T2** | The basal input was one control cell, but the checkpoint was trained AND selected with `match_cntr/avg_cntr: true` — the `(cell_type, batch)` **MEAN** control. The model had never seen a single-cell basal. Consistent with validation `pearson_delta` 0.387 vs exported 0.3077. |
| **T4** | The frozen manifest of all five reported splits recorded `"kg_list": []` — the adapter read `cfg.datamodule.kg_list`, which does not exist, so `.get(..., [])` silently claimed the arm ran with **no knowledge graphs**. |

**Decisions (user):** `omit_cntr: true` on **both** sides — the `-1` is the unused second slot
of a combinatorial format and our cohort is single-gene throughout — and TxPert is **retrained
on single-cell controls** so that training, inference and the benchmark protocol all agree.
**Rationale: fairness.** Every other arm predicts from one drawn half-B control; giving TxPert a
denoised batch-mean basal would make its number incomparable. **BatchNorm deliberately
untouched**, same reason. Both are deviations from TxPert's published config and are recorded
as such.

⚠️ An **A/B is running** to measure T2 rather than assume it: two arms identical except for the
training control regime (`single` vs `mean`), both inferenced on single cells.

### ⛔ CellFlow — the export had NEVER run, and selection was inflated

| # | finding |
|---|---|
| **C1** | `predict` requires `obs[control_key].all()` to be **True** (`_cellflow.py:619-622`); we passed all `False`. Those rows *are* controls — the OT source population; `False` belongs on the covariate frame, which is what their own `predict_cellflow.py:215-222,316` does. |
| **C2** | The prediction block carried no `uns`, so `rep_dict['gene_embeddings']` raised `KeyError` (`_datamanager.py:245,401-403`) — not the "Representation not found" our guard told the reader to expect. |
| **C3** | The selection metric decoded the **TRUTH** through the rank-256 PCA basis, deleting everything the basis cannot express. It recorded `pearson_delta` 0.4027 / `overlap_at_N` 0.908 on the same split where TxPert, scored against untouched targets, gets 0.387. The module's own premise — "the selection metric has to be the same quantity as the reported metric" — held for the code and failed for the quantity. |
| **C4** | `n_conditions_on_log_iteration: 60` against **103** conditions, re-drawn each pass from one rng, so the five selection scores that chose the checkpoint were **not comparable to each other**. The apparent rise 0.4089 → 0.4614 → 0.5165 mixes improvement with a changing panel. |
| **C5** | The one-hot vocabulary was built **twice**, independently for train and validation, so widths could diverge. It worked only because the control-injection path overwrote the validation `uns` — which itself only fires because this split has zero controls. |
| **C6** | `abs(hash((label, n)))` is `PYTHONHASHSEED`-randomised, so the subsample was not "deterministic per (condition, n)" as its comment claimed. |

**C3 + C4 determine WHICH checkpoint was kept and cannot be fixed after the fact**, so CellFlow
is **retrained**. Verified live in the new run: `one shared vocabulary: 2395 tokens, 2395
distinct perturbations`; `valid conditions=103 n_log=103 n_end=103`; `gene-space truth: 103
perturbations, 6,731 cells (controls excluded)`.

✅ **Cleared:** no set axis (inference is `jax.vmap` per cell; minibatch OT is training-only and
each batch already holds one perturbation), and — checked at the user's request — **the PCA
basis is fit on TRAINING DATA ONLY** (`centered_pca(adata_train)`, then
`project_pca(ref_adata=adata_train)`; `adata_train` never receives validation rows). ⚠️ The
basis uses `random_state=0`, so it does not vary with `model.seed`.

### ⛔ PerturbDiff — two blockers, no retrain needed

| # | finding |
|---|---|
| **B1** | Checkpoint resolution matched **nothing**: `save_last: True` puts `last.ckpt` beside the selected file (tripping the `>1` guard), the launcher appends `TAG_SUFFIX` *after* `_s${SEED}` so no glob variant matched the real dir, and `--penc` is never passed. The natural workaround would have scored **final-step weights selection never chose** — which their own eval script explicitly refuses. Now reads the `.best_ckpt_path` marker the training leg already writes. |
| **B2** | The export never seeded torch. With `eta: 0.0` the DDIM prediction is a deterministic function of one `th.randn` draw, so two runs of the same checkpoint disagreed and `--seed` did not control it. Theirs: `pl.seed_everything(cfg.optimization.seed)`. Now seeded per split (`42 + 1000*split`, visible in the log as `Seed set to 42 / 1042 / 2042`). |
| **M1** | Same constant-batch defect as STATE's S3. Fixed per-cell; ⚠️ the faithful index is the **perturbed** cell's batch (`batches[a_rows]`), not the drawn control's — `get_covarates` runs on the perturbed row. |
| **M2** | `export_split` chunked at 512 rows while the set partition lived inside `predict`, so ~36 boundaries cut a perturbation in two. Fixed with `batch_size=len(split)`. |

✅ Every training and sampling hyperparameter matches the reference recipe (verified against the
real `pd_fewshot_*` run logs), and EMA is handled correctly — the checkpoint holds the averaged
weights validation scored.

### ⚠️ Corrections to claims in code I wrote myself

- **`is_padded_list` is never seen by the model.** It is used only post-hoc to drop rows
  (`sampling_generation.py:188`), and attention runs across the whole set with **no key-padding
  mask** (`cross_dit_blocks.py:73-75`) — so filler rows **do** influence real ones. My docstring
  claimed it was "the mask the model already consumes for exactly this purpose".
- **`cov_mapped_pert` is consumed by nothing** (grep finds only the producer). My comment called
  it "the control token for the conditioning branch"; there is no such branch.
- **`assert_same_space` / `man["space_check"]`** were removed in C18, and two tests still
  referenced them — they only surfaced now, three test runs deep.
- A **shared TxPert data cache** would have had two concurrent arms writing one `combined.h5ad`
  (both logged "adata not cached" within 60 ms); each arm now has a private cache.

### ⚠️ Three false provenance claims

The recipe files cited as authoritative for STATE, TxPert and PerturbDiff
(`reproducibility/{state,txpert,perturbdiff}__replogle_nadig_fewshot/...`) **do not exist in
this checkout**. All three sets of hyperparameters were confirmed against run logs instead.
TxPert's config additionally records that three of its model hyperparameters came from a
217-run sweep that **maximised a TEST metric** — a selection-on-test concern that belongs
beside its number.

### The general lesson

Every check in place was about the **CONTENT** of input rows — gene axis order, ID space,
library size, control pairing, one-hot lookup by name. **None was about their GROUPING, their
COVARIATES, or the batch layout they were submitted in.** All three produce a well-formed,
well-calibrated matrix that scores at a baseline. Any future arm needs assertions on the
*shape of the request*, not only on the values in it.

## ⛔ C27 — STATE's 0.1306 is OUR BUG, not STATE: we fed it mixed-perturbation cell sentences

**Objective.** Unchanged: best model on SEEN perturbations in an UNSEEN context (HepG2
`rhj_fewshot`), headline cell-eval `pearson_delta` on the 611-perturbation test cohort.
Anchors: `ctrlcell` −0.0018 · `ctrlmean` 0.0016 · **`meanpert` 0.1324** · `transfer` 0.2323 ·
`experimental` 0.3722 · ours 0.3415 · TxPert 0.3077.

**Question this answers.** STATE's test score came out at `meanpert`. Is STATE unable to
generalise to these perturbations, or is our port destroying its signal?

**What is varied / held fixed.** Nothing varied — one checkpoint, five splits, then a read of
how the export builds model inputs versus how training builds them.

**What each outcome means.** Real → report STATE as at-chance-for-identity on this task.
Port bug → the number is meaningless and must not be published as STATE's.

### The measured number (DEFECTIVE — do not cite as STATE's performance)

`compet_state_s42`, 5 splits, `model_nocal`, from `best_celleval.ckpt` (step 63,999):

| split | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |
|---|---|---|---|---|
| 0 | 0.1314 | 0.0337 | 0.0288 | 0.5405 |
| 1 | 0.1381 | 0.0338 | 0.0324 | 0.5463 |
| 2 | 0.1310 | 0.0336 | 0.0336 | 0.5520 |
| 3 | 0.1253 | 0.0338 | 0.0281 | 0.5395 |
| 4 | 0.1270 | 0.0337 | 0.0302 | 0.5455 |
| **mean** | **0.1306** | **0.0337** | **0.0306** | **0.5448** |
| sd | 0.0050 | 0.0001 | 0.0023 | 0.0050 |

**0.1306 against `meanpert` 0.1324** — indistinguishable from the baseline that predicts the
average perturbation response and ignores perturbation identity. `disc_l1` 0.5448 is barely
above its 0.5 chance level. Meanwhile the SAME checkpoint scored `pearson_delta` **0.2049** on
its own validation pass. A model does not lose 0.074 of `pearson_delta` between two held-out
panels of the same design; something differs in how it was asked.

### The cause: cell sentences are built differently at export than at training

STATE does not predict per cell. It predicts per **cell sentence** — `cell_set_len=64` — and
the backbone is a `LlamaBidirectionalModel`, i.e. **bidirectional attention across the whole
sentence**, so every cell's prediction depends on its 63 companions.

`SentenceBatchSampler._create_sentences` (`zarr_perturbation.py`) builds them like this:

```python
keys = np.ravel_multi_index((ct, pert), ...)   # group by (cell_type, PERTURBATION)
...
for i in range(0, len(group), self.cell_sentence_len):
    sent = group[i : i + self.cell_sentence_len]
```

**Every training sentence contains exactly ONE perturbation.** A short final sentence is padded
by sampling that same sentence's own cells with replacement (`_create_batches`).

Our export adapter instead padded the WHOLE matrix to a multiple of 64 and reshaped
sequentially. Measured on split 0 (18,575 rows, 611 perturbations, 30.4 cells each):

| distinct perturbations per 64-cell sentence | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| sentences | **39** | 43 | 101 | 71 | 34 | 2 |

**Only 39 of 290 sentences were pure.** The median sentence mixed **3** perturbations, some
mixed 6. So for ~87% of the panel, STATE was asked to predict each cell while attending over
cells belonging to two to five *other* perturbations — a regime it never saw in training.

That is precisely the shape that collapses to `meanpert`: the predictions still vary by
perturbation (measured between-perturbation variance of group means 8.95e-4, and two arbitrary
perturbations differ by 0.029 against a mean |pred| of 0.141, so the one-hot IS reaching the
model), but the variation is contaminated by whatever else shared the sentence, so it carries
little perturbation-specific information.

⚠️ **Why the earlier checks all passed.** The one-hot is looked up by NAME
(`dm.pert_onehot_map[p]`), so no ordering bug was possible and I verified exactly that. The
emission diagnostics were excellent — library ratio 1.0011–1.0027 across all five splits,
gene-mean correlation 0.9970. **Every check I had was about the CONTENT of the input rows, and
none was about their GROUPING.** For a set-based model the grouping IS part of the input.

### The fix

Build sentences per perturbation, matching `_create_sentences`:

1. group panel rows by perturbation label;
2. chunk each group into sentences of ≤ 64;
3. pad a short sentence from **that perturbation's own cells** (deterministically, by tiling
   rather than the sampler's `rng.choice`, since at predict time the padding is discarded and
   only its effect through attention matters);
4. scatter each sentence's real predictions back to their original row positions.

### Verification of the fix, before spending a GPU

Replaying the new plan on split 0's real labels:

| | before | after |
|---|---|---|
| sentences | 290 | 671 |
| **PURE (1 perturbation)** | **39** | **671 of 671** |
| rows covered exactly once | yes | yes (18,575 of 18,575) |

Padding overhead is 131% — most perturbations have ~30 cells and pad to 64 — so the export
costs ~2.3x more compute. That is the price of matching training, and training pads the same
way.

### ⛔ THE SAME DEFECT WAS PRESENT IN PERTURBDIFF — caught before its first export

PerturbDiff is also set-based: it samples `(b, S, g)` with `S = data.use_cell_set = 16`, and
training groups sets by `data.cell_set_group_by=pert_celltype`, i.e. one perturbation per set.
Its export adapter reshaped sequentially in exactly the same way. Measured on split 0:

| distinct perturbations per 16-cell set | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| sets | 625 | 498 | 36 | 1 |

**53.9% of sets were pure, and 46.1% of the panel's cells (8,560 of 18,575) sat in an impure
set.** Less severe than STATE's 87% because S is 16 against ~30 cells per perturbation, but
the same error. Fixed the same way, with one improvement available here: PerturbDiff has an
`is_padded_list` mask the model already consumes, so filler slots are FLAGGED rather than
silently tiled.

**TxPert and CellFlow are per-cell and unaffected** — TxPert predicts row by row and CellFlow
predicts per condition from a PCA latent, so neither has a set axis. **TxPert's 0.3077
stands.**

⚠️ **The general lesson, and why this is worth a finding.** Every check in place was about the
CONTENT of the input rows — gene axis order, ID space, library size, control pairing, one-hot
lookup by name. **None was about their GROUPING.** For a set-based model the grouping IS part
of the input, and getting it wrong yields a plausible, well-calibrated matrix that scores at a
baseline. Any future arm with a set axis needs a purity check on the emitted plan, not only on
the values.

## ⭐⭐ C26 — the max-value refusal would have discarded a GOOD arm over 1 value in 4,357,392

**Objective.** Unchanged: best model on SEEN perturbations in an UNSEEN context (HepG2
`rhj_fewshot`), headline metric cell-eval `pearson_delta` on the 611-perturbation cohort.
Anchors: floor −0.0018 · `meanpert` 0.1324 · `transfer` 0.2323 · `experimental` 0.3722 ·
ours 0.3415 · TxPert 0.3077.

**Question this answers.** STATE's step-32,000 validation was refused with `max value 17.29
exceeds log1p threshold of 15.0`. Is that a preprocessing/units bug, or a real model output —
and is refusing such a checkpoint the right behaviour?

**What is varied / held fixed.** Nothing varied. The transform chain was audited line by line
and the checkpoint's actual output distribution measured on the real validation split.

**What each outcome means.** A units bug → every STATE number so far is invalid and the port
needs fixing. A genuine unbounded-head excursion → the numbers stand, and the question becomes
a policy one about what to do with out-of-range predictions.

### The chain is CLEAN — it is not a units bug

| stage | what happens | verdict |
|---|---|---|
| input | `log1p(counts / rowsum_FULL_PANEL × 10,000)` in `occam_data` | ✅ measured `rowsum(expm1) ≈ 8,625` on targets — full-panel denominator preserved |
| STATE config | `target_sum: null`, `log1p: false` | ✅ the datamodule **refuses** anything else, so no double transform is possible |
| model output | `state_transition.py:516` → `out_pred = self.relu(out_pred)` | ⚠️ non-negative, **unbounded above** |
| validation scoring | no transform whatsoever; cell-eval told `is_log1p=True` | ✅ no stray `expm1` or renormalisation |
| export | `log1p(expm1(x) × 0.2)`, declared `from_target_sum=10_000` | ✅ exact — the per-cell denominator cancels |

So 17.29 is a **real model output**, not a scaling mistake. A ReLU gives `≥ 0`, which is right
for log1p counts, but nothing caps it at the physical maximum `log1p(10,000) = 9.2104` — the
value a single gene would take if it held the *entire* library.

### What STATE actually predicts (best_celleval.ckpt, epoch 319, 624 validation cells)

| | min | p50 | p99 | p99.9 | p99.99 | **max** | rowsum(expm1) | zero-frac |
|---|---|---|---|---|---|---|---|---|
| **REAL (target)** | 0.0000 | 0.0000 | 2.9994 | 4.4488 | 5.3532 | 6.4344 | 8,625 | 0.5160 |
| **PRED** | 0.0000 | 0.0000 | 2.9988 | 4.4866 | 5.3794 | **9.3583** | 8,640 | 0.5196 |

**The prediction tracks the target to three decimal places all the way to p99.99.** Library
size is 0.2% off (8,640 vs 8,625) and the zero fraction is within 0.004. This is a
well-calibrated emission by every measure that describes the bulk of the matrix.

```
train-space ceiling log1p(10000) = 9.2104
PRED values above it: 1 of 4,357,392  (0.000023%)  in 1 of 624 cells
eval-space max would be 7.7492   (old guard refused above 7.6014)
VERDICT under the old rule: EXPORT WOULD BE REFUSED
```

**One element out of 4.36 million would have discarded the entire arm.** Not degraded its
score — produced *no number at all*, indistinguishable at the top level from a port that does
not work.

### Decision (the user's, 2026-09-18): score high-max checkpoints, do not skip them

> *"actually for state, I want to evaluate even checkpoints with high values. Remove this rule
> of not using checkpoints with high max"*

Two enforcement points existed, and both are removed:

| where | was | now |
|---|---|---|
| `competitors/shared/emission_space.py::assert_representable` | `raise` when `max(X) > log1p(target_sum)` | prints a loud one-line warning and scores; `space_diagnostics` still records `looks_log1p=False` + a note in the manifest |
| `MetricsEvaluator(...)` in STATE's `tx/eval/celleval.py` **and** PerturbDiff's `sampling_eval_celleval.py` | cell-eval's own `guess_is_lognorm` raised at a hard threshold of 15.0 | `allow_discrete=True` disables that check (kwarg confirmed present in cell-eval 0.7.2 in both venvs) |

⚠️ **THE TWO CHECKS NEVER AGREED, AND OURS WAS THE STRICTER ONE.** This is the substance of
the finding, not a side note:

| check | threshold | where it comes from |
|---|---|---|
| cell-eval `guess_is_lognorm` | **fixed 15.0**, independent of `target_sum` | `cell_eval/utils.py`, `max_threshold: float = 15.0` |
| our `assert_representable` | **`log1p(target_sum)` = 7.6014** in eval space (9.2104 in train space) | written as "arithmetically impossible for a library-normalised matrix" |

So a checkpoint that **passed** validation could still be refused at export — and that is
exactly what happened to the BEST one. `best_celleval.ckpt` scored `selection_score` 0.4313 at
validation with no complaint (its eval-space max 7.7492 is less than half of cell-eval's 15.0),
yet our guard refused it for exceeding 7.6014.

**A successfully validated checkpoint does not have a scoring problem.** Our guard was the
thing out of step: it applied a rule about what is possible for *data* to a *prediction*, at a
threshold roughly half of what the scorer itself uses. Removing it aligns the export with the
scorer that produces every reported number.

A second, smaller inconsistency went the other way and is also gone: the reported-number path
already passed `allow_discrete=True` (`celleval_v1.run_celleval_v1`) while the two arms'
validation scorers did not, so validation applied a check that reporting skipped.

**What is still caught.** The failure mode the ceiling originally guarded — raw counts passed
off as log1p — is caught by the library-size factor band (`ROWSUM_FACTOR_BAND = (0.25, 4.0)`),
which raw counts miss by orders of magnitude. Negatives, non-finites, wrong width and
out-of-order gene axes all still raise.

**What is given up, once:** a prediction above `log1p(target_sum)` is impossible for real data,
so an arm carrying many such values will have inflated `nmae`/`mse`. That is now visible in the
manifest instead of fatal — the number is reported with the caveat attached rather than
withheld.

### ⭐ CONFIRMED ON THE REAL TEST COHORT — the arm exports, and it is well calibrated

STATE finished (`18097897`, COMPLETED, 6 h 22 m) and its export ran under the new rule.
Split 0, 18,575 rows x 6,983 genes, from `best_celleval.ckpt`:

```
raw emission : range [0.0000, 12.5607] · zero 0.5433 · median expm1 rowsum 8,666.2
eval space   : range [0.0000, 10.9513] · zero 0.5433 · median expm1 rowsum 1,733.2
[emission] ⚠️  max 10.9513 exceeds log1p(2000) = 7.6014. Scoring it anyway.
representable ✓   diagnostics: rowsum ratio 1.0019 · gene-mean corr 0.9970 · zero 0.5433 (target 0.5170)
for reference, OUR OWN arm: rowsum ratio 0.9399 · gene-mean corr 0.9998 · zero 0.0000
```

**Under the old rule this arm produced nothing at all** — eval-space max 10.95 against a
7.6014 ceiling. Under the new one it exports, with the excursion recorded in the manifest.

**And the prediction is good by every measure that describes the matrix rather than its single
largest element:**

| diagnostic | STATE | our own best arm | target |
|---|---|---|---|
| library-size ratio vs target | **1.0019** | 0.9399 | 1.0 |
| gene-mean correlation | 0.9970 | **0.9998** | 1.0 |
| zero fraction | **0.5433** | 0.0000 | 0.5170 |

STATE's library size is within **0.2%** of the target where ours is 6% low, and its zero
fraction is close to the target's where ours is zero. Refusing it over the maximum would have
discarded the better-calibrated emission of the two.

⚠️ The caveat stands and is the reason the warning prints: values above `log1p(target_sum)` are
impossible for real data, so read this arm's `nmae` and `mse` knowing a few elements are out of
range. `pearson_delta` and `discrimination_score_l1` are rank/correlation quantities and are
far less exposed to it.

### Consequence for the running arm

STATE `18097897` is still running under the **old** code, so its step-32,000 validation stays
lost (3 of 4 landed: `selection_score` 0.4071 / 0.4041 / **0.4313**). It is **not** restarted —
the export leg picks up the new guard, so the arm can be exported and scored regardless, and a
restart would discard 6 h of training to recover one validation point out of five.

## ⭐⭐ C25 — THE REAL ROOT CAUSES, found by reading the eval path instead of patching it

**Objective.** Unchanged: best model on SEEN perturbations in an UNSEEN context (HepG2
`rhj_fewshot`), headline metric cell-eval `pearson_delta` on the 611-perturbation test cohort.
Anchors: floor −0.0018 · `meanpert` 0.1324 · `transfer` 0.2323 · `experimental` 0.3722 ·
ours 0.3415 · TxPert 0.3077.

**Question this answers.** PerturbDiff and CellFlow had each failed validation repeatedly, and
each failure had been met with one more branch in one more function. What is actually wrong?

**What is varied / held fixed.** Nothing varied. This was a read of both evaluation paths end
to end, then a fix where the design is wrong rather than where the exception surfaced.

**What each outcome means.** One coherent defect per arm → fixable once, verifiable before
spending a GPU. A pile of independent defects → the ports are unsound and we say so.

Both turned out to be **one design defect each**, and neither was where the traceback pointed.

---

### PerturbDiff: the occam path was wired into the wrong half of a two-design function

`generate_samples` has **two** designs for "get the control baseline, build the obs frame":

| `source_format` | controls | obs | dispatches on a dataset-family string? |
|---|---|---|---|
| `"zarr"` | `build_zarr_controls` | `build_obs_data_zarr` | **no** — a purpose-built pair |
| anything else | `load_ctrl_adata` → `build_x_ctrl` | `build_obs_data` | **yes** — `replogle`/`pbmc`/`tahoe` substrings of `cfg.data.data_name` |

Our data was routed into the **second**. Every function in that chain therefore had to be
taught about a fourth family separately, and each omission surfaced as a crash ~45 min into a
run, one function further along than the last:

| # | crashed in | how found |
|---|---|---|
| 1 | `load_ctrl_adata` | died at the first validation |
| 2 | `build_obs_data` | died at the first validation, again |
| 3 | the `len(np_mask) > 2000` HVG-restriction block | **never reached** — it reads `ctrl_adata.var.highly_variable`, a column only upstream's control h5ads carry, and `len(np_mask)` is 6,983 on our axis, so it would have entered and raised |

**The fix is the zarr pattern, not a third branch.** `build_occam_controls` +
`build_obs_data_occam` are now siblings of the zarr pair, selected once at the top of
`generate_samples` in a three-way branch. The occam branches previously bolted into
`load_ctrl_adata` and `build_obs_data` are **deleted**: two definitions of where our controls
come from can drift, and the unreachable one is the one nobody checks. The HVG block is now
scoped to upstream's families, with the assumption it rested on asserted rather than assumed
(`np_mask.all()`).

Verified through the real chain — `load_selected_genes` → `build_occam_controls` →
`build_obs_data_occam` → AnnData → `run_cell_eval` → `read_agg_metrics`, the exact sequence
the callback runs:

```
X_ctrl (4976, 6983)   obs keys ['batch','celltype','pert']   ctrl labels ['empty']
np_mask: len 6983  all True: True          <- crash #3 confirmed latent
obs cols ['target_gene','cell_type','batch_center']   controls carry 'empty': OK
selection_score = mean(pearson_delta, discrimination_score_l1) = 0.25718   PUBLISHABLE
```

⚠️ That 0.257 is a stand-in matrix, not a model. The point is only that a score EXISTS.

---

### CellFlow: the selection wrote a control label its own scorer does not recognise

The run did **all** its work — trained, fit the PCA basis, predicted, saved `CellFlow.pkl` —
then reported `cell-eval returned no ['pearson_delta','discrimination_score_l1'] ... Got []`.
Three layers had to be peeled:

1. **A red herring, ruled out.** `adata_pred.h5ad` is written by anndata **0.13.3**; my first
   reproduction read it with the SCORING DRIVER's interpreter (anndata **0.9.2**), which raised on
   `obs/_index` (a `nullable-string-array` group). **That is not the live failure** — cell-eval
   actually runs under the `jupyter` interpreter (anndata 0.11.4), which reads the file fine.
   Fixing what my own reproduction showed would have fixed a non-problem.
2. **The diagnostic was being discarded.** `run_cell_eval` prints cell-eval's stderr tail to
   its own stdout and **returns `{}` rather than raising**, so the child exits **0** with an
   empty metrics file. `_score_with_occam_celleval` surfaced the child's output only on a
   non-zero exit. **That one condition cost three wrong diagnoses.**
3. **The actual cause**, once the child was run by hand:

```
ValueError: Control perturbation (non-targeting) not found in real AnnData:
 ['ALG11+HEPG2' 'ANAPC1+HEPG2' ... 'ZW10+HEPG2' 'empty']
```

Two defects visible in that one line:

| | was | should be | why |
|---|---|---|---|
| control label | `empty` | **`non-targeting`** | `occam.vcell.panel.celleval_v1` hardcodes `CONTROL = "non-targeting"` and takes no override. Every other emitter — `export_predictions.py`, `panel_io`, `splithalf`, `predict`, `calibrate` — relabels at this same boundary. Only the selection did not. |
| perturbation label | `ALG11+HEPG2` | **`ALG11`** | `_condition_label` joined CellFlow's `(pert, cell_type)` condition tuple with `+`. Nothing else uses that format, and the object is single-cell-type by construction, so the suffix carried no information. |

**A third defect in the same function:** the predicted block had 4,976 cells per condition and
the real block ~50 — one predicted cell per OT *source* cell, the source being the whole
injected control pool. That emitted **303,536 predicted rows against 9,056 real ones, an
8.5 GB h5ad per validation pass**, at a per-condition `n` different from the TEST protocol's
(where each perturbation gets one predicted cell per *drawn* control cell, matched to the
target's own count). **Selection must optimise the quantity that is reported**, so counts are
now matched per condition by a deterministic draw — estimator-variance symmetry, not
manufactured signal, since the predicted cells are exchangeable draws from one predictive
distribution.

**A fourth:** `run_meta.json` indexed `data_cfg["data_dir"]` and `["split_path"]`
unconditionally; neither exists on the occam path (the benchmark dir is both). The run
finished all its work and then died describing itself.

### CellFlow verification, on the real artifacts of a real run

The saved 8.5 GB pair was relabelled exactly as the fixed code does, then scored through our
own scorer:

```
as written: pred (303536, 6983)  real (9056, 6983)
after fix : pred (  9056, 6983)  real (9056, 6983)   perts 61   controls 4976/4976
            labels ['ALG11','ANAPC1','ASCC3', ...]
cell-eval done: 28 metrics parsed
  pearson_delta 0.2595 · discrimination_score_l1 0.4997 · mae 0.0331 · overlap_at_N 0.8942
  selection_score = 0.37959
```

⚠️ **A 20-ITERATION model on a 61-condition subsample. NOT a result.** It says the machinery
works and nothing about CellFlow's quality.

---

### ⚠️ SEPARATE FINDING — the arms do not all validate with the SAME cell-eval

Found while tracing the above.

| environment | python | cell-eval | pdex | anndata |
|---|---|---|---|---|
| **`jupyter` — every REPORTED number** | 3.10 | **0.6.0** | **0.1.24** | 0.11.4 |
| CellFlow *validation* (shells out to ours) | 3.13 | **0.6.0** | 0.1.24 | 0.13.3 |
| STATE *validation* (its own venv) | 3.11 | **0.7.2** | 0.2.2 | 0.12.16 |
| PerturbDiff *validation* (its own venv) | 3.11 | **0.7.2** | 0.3.0 | 0.12.19 |

**Test scoring is uniform at 0.6.0 for every arm** — that goes through
`scripts/eval_competitor.py` and is unaffected, which is why TxPert's 0.3077 stands. But STATE
and PerturbDiff *select* checkpoints with **0.7.2 / pdex 0.2.2–0.3.0**, a different scorer from
the one their reported number comes from. CellFlow does not — it shells out to ours, which is
the reason that path is two-phase at all.

**What it costs.** Selection is a *ranking* signal, so a version difference matters only if it
reorders checkpoints; it is not a validity problem for any reported number. It IS a deviation
from the requirement that the competitors use our cell-eval throughout, and 2 of 4 arms
currently violate it.

**Options, with costs — not decided here:**

| option | cost | what it buys |
|---|---|---|
| leave it | none | nothing; the deviation stands, recorded here |
| make STATE/PerturbDiff shell out to ours, as CellFlow does | one adapter each, plus a subprocess per validation (~2–4 min) | all four arms selected by the scorer they are reported by |
| pin 0.6.0 into their venvs | dependency-conflict risk on py3.11 | same, without the subprocess |

## ⭐ C24 — STATE trains and validates correctly. PerturbDiff died on the FOURTH sibling of one bug.

**Objective.** Unchanged: best model on SEEN perturbations in an UNSEEN context, HepG2
`rhj_fewshot`, headline metric cell-eval `pearson_delta` on the 611-perturbation test cohort.
The competitors exist to say whether Occam's **0.3415** is good absolutely or only relatively.

**Question this answers.** After the C23 fixes, do STATE and PerturbDiff actually train AND
validate AND checkpoint on our data, over hours rather than seconds?

**What is varied / held fixed.** Nothing varied; this is the correctness pass at full scale.
Both arms ran their real launcher, unmodified, on the full benchmark.

**What each outcome means.** Validations landing with a rising selection score → the arm is
real and will produce a comparable number. Validations silently swallowed → the arm trains for
hours and yields nothing, which is the failure this campaign keeps hitting.

### STATE — ✅ HEALTHY, and improving

`val/celleval/*` on the 103 HepG2 validation perturbations, from the run's own `metrics.csv`.
`selection_score = mean(pearson_delta, discrimination_score_l1)`, the checkpoint monitor.
**These are VALIDATION numbers on 103 perturbations — not the 611-perturbation test cohort, so
they are NOT comparable to TxPert's 0.3077 or ours 0.3415.** Direction is what matters here.

| step | `pearson_delta` | `disc_l1` | `mae` ↓ | `overlap_at_N` | **`selection_score`** |
|---|---|---|---|---|---|
| 16,000 | 0.1867 | 0.6274 | 0.0688 | 0.0690 | 0.4071 |
| 32,000 | — | — | — | — | **REFUSED, see below** |
| 48,000 | 0.1770 | 0.6313 | 0.0653 | 0.0758 | 0.4041 |
| 64,000 | **0.2049** | **0.6576** | **0.0633** | **0.0832** | **0.4313** |
| 80,000 | *pending* | | | | |

Job `18097897`, RUNNING, 5 h 23 m at the time of reading, one validation left.
**Every metric except `pearson_delta` moves monotonically the right way**, and the best
checkpoint so far is the latest — so 80,000 steps was not obviously too many.

⚠️ **The step-32,000 validation was REFUSED by cell-eval, not swallowed by a bug:**

```
ValueError: Invalid scale: max value 17.29 exceeds log1p threshold of 15.0.
```

The model emitted a value of 17.29 in log1p space — `expm1(17.29) ≈ 3.2e7` counts in one gene —
and cell-eval will not score a matrix that far outside log1p range. **This is the right
outcome and needs no fix:** our own export guard (`assert_representable`) refuses anything above
`log1p(target_sum)`, so a checkpoint whose predictions exceed the representable range could not
be exported or scored anyway. The callback's `except: continue` then did its job — the run
carried on and the next two validations were clean. One transient excursion out of four, on an
unbounded-output model, is training noise rather than a defect.

### PerturbDiff — ⛔ FAILED, and the cause is the same bug a fourth time

Both arms (`18098600` genept, `18098601` onehot) generated samples correctly — **overall
`r2_metric` 0.9991**, 6,731 cells × 6,983 genes, full validation-split coverage, controls
loaded — and then died one call later:

```
ValueError: Unsupported data_name for obs building: OccamH5adFinetune.
            Expected one of replogle/pbmc/tahoe100m.
  sampling_generation_helpers.py:363  build_obs_data
```

**This is the same defect as C23, for the fourth time in two arms.** The pattern: a function
dispatches on a dataset-family string, our data is a fourth family, and each such function has
to be taught about it separately. Instances so far:

| # | where | found by |
|---|---|---|
| 1 | `ZarrPerturbationDataModule.__init__` — no config in occam mode | pre-flight (C23) |
| 2 | `get_datamodule` — demanded a TOML unconditionally | the real run, 21 s in |
| 3 | `STATE/export_occam.py` — passed a TOML *and* a bench dir | reading the call chain |
| 4 | `build_obs_data` — no occam branch | the real run, **45 min in** |

⚠️ **Why the earlier "control adata VERDICT: OK" did not catch it.** That check verified
`load_ctrl_adata` — the function immediately BEFORE the one that failed. Verifying a
neighbour is not verifying the path. **Third time this exact mistake has been made** (C20's
proxy checks, C23's four-proxy pre-flight, and now this), so the sweep was done differently:
`grep` for *every* dispatch on a dataset-family string across `src/apps/sampling/`,
`src/data/` and `src/apps/training/` before writing any fix.

### The fix, and why it is shaped this way

| file | change |
|---|---|
| `sampling_generation_helpers.py` | `build_obs_data` gets an occam branch, keyed on `source_format == "occam"` — the same key `load_ctrl_adata` already uses, so the two cannot disagree |
| " | the occam branch uses **config-driven** column names (`cfg.data.pert_col` / `cell_line_key` / `perturbseq_batch_col`), copying `build_obs_data_zarr`'s discipline. The replogle branch hardcodes `"gene"`, which would emit a column `run_cell_eval` is never told to read |
| " | the occam control frame now also carries the **batch** column, with a loud raise if `occam_data` did not emit it — a missing third column previously surfaced one call later |
| " | both fall-through messages now name `source_format='occam'`, so the next instance of this bug is self-describing |

Control rows keep `"empty"`, our derived control sentinel, in the perturbation column, because
that is how cell-eval locates the baseline.

**Verification is through the real chain this time**, not a neighbour: `load_selected_genes` →
`load_ctrl_adata` → `build_x_ctrl` → `build_obs_data` → `anndata` construction →
`run_cell_eval`, ending in actual cell-eval metrics.

## ⛔ C23 — STATE's occam mode never built a dataset config. Pre-flight caught it; a relaunch would not have.

**Objective.** The standing goal is the best model on SEEN perturbations in an UNSEEN
context — here, the `rhj_fewshot` HepG2 few-shot panel — and the competitors exist to say
whether Occam's number is good in absolute terms or only relative to baselines. A competitor
that crashes at construction contributes nothing, so every arm gets a pre-flight before it
gets a GPU.

**Question this answers.** Are STATE, CellFlow and PerturbDiff actually runnable end to end
on our data after the last round of fixes, or is there another defect between the config and
the first batch?

**What is varied / held fixed.** Nothing is varied — this is a correctness pass. The three
arms were each driven with the *exact* override list their launcher uses, on our frozen
benchmark, with only the RAM caps and step budgets reduced.

**What each outcome means.** Clean → launch. A crash → a bug that would otherwise have
surfaced as a dead Slurm job, which is the failure mode that has cost this port the most time
(C10, C17, C20, C21).

### The defect

`ZarrPerturbationDataModule.__init__` called `ZarrExperimentConfig.from_toml(toml_config_path)`
**unconditionally**, while `configs/data/occam_h5ad.yaml` correctly sets
`toml_config_path: null` — the benchmark directory is the manifest in occam mode, there is no
TOML. Result:

```
TypeError: expected str, bytes or os.PathLike object, not NoneType
  zarr_perturbation.py:886  self.config = ZarrExperimentConfig.from_toml(toml_config_path)
  zarr_perturbation.py:265  with open(path, "rb") as f:
```

**Every other occam hook was already right.** `_Source` has its occam branch, `_read_split`
has its occam branch, `occam_source.py` has the reader, the fold builder and the per-line
invariants. The one missing piece was the object that names the sources — so
`config.get_all_datasets()` had nothing to iterate and `setup()` would have reported
`0 / 0 / 0` even if construction had survived.

⚠️ **Why the earlier "4/4 STATE pre-flight passed" did not catch this.** That pass checked the
Hydra composition, the venv, the gene axis and the import graph — four proxies, all upstream
of construction. The real run then died at 21 s on a *different* defect (the copier's
`DENY_DIRS` had eaten `configs/wandb/default.yaml`), never reaching line 886. Fixing the first
error simply uncovered the second. **Same shape as C20:** a check that passes on a proxy is
not a check.

### The fix

| file | change |
|---|---|
| `occam_source.py` | `occam_datasets(bench_dir)` → the `[datasets]` names, read from `bench.files`, each validated through `occam_line_from_path` up front |
| `zarr_perturbation.py` | `ZarrExperimentConfig.from_occam(bench_dir)`: one dataset per line, all marked `training="train"`, **no** `[splits]` (the split is derived live per line) |
| `zarr_perturbation.py` | `__init__` branches on `source_format`; refuses a TOML **and** a bench dir together — two source lists cannot both be authoritative |
| `zarr_perturbation.py` | `validate()` skips the path-existence warning in occam mode (a `[datasets]` value is a cell-line NAME there, not a path) |

### Pre-flight result, STATE

Real construction, real `setup()`, one real batch, 6,000 cells/line cap:

```
__init__ OK  datasets=['HEPG2', 'jurkat', 'RPE1']
  [occam-state] HEPG2:  finetune 4,216 (incl. 221 controls) · validation 268 · test 1,516
  [occam-state] jurkat: finetune 6,000 (incl. 282 controls) · validation 0 · test 0
  [occam-state] RPE1:   finetune 6,000 (incl. 270 controls) · validation 0 · test 0
train 2,549 batches · val 43 · test 504
pert_cell_emb (128, 6983)  ctrl_cell_emb (128, 6983)  pert_emb (128, 2329)
              cell_type_onehot (128, 3)  batch (128, 167)
axis identical & in order: True        gene axis 6,983, first3 ['DPM1','GCLC','NIPAL3']
```

**The asymmetry is the `rhj_fewshot` design showing up correctly**: HepG2 splits three ways,
Jurkat and RPE1 are wholly training data, because a perturbation held out of HepG2 still
trains from the other two lines. That is what makes the task few-shot-in-a-new-context rather
than never-seen-at-all.

⚠️ `rowsum(expm1) ≈ 8,727`, not 10,000, and that is **correct**: normalisation happens on the
FULL source panel before the 6,983-gene subset, which is exactly what makes the
10,000 → 2,000 emission map exact. A row summing to 10,000 here would mean the wrong
denominator and a broken `mae`.

### Pre-flight result, CellFlow

Real 20-iteration run with a validation at 10:

```
[vocab] 2381 perturbation tokens from rhj_fewshot
  [occam-data] splits: finetune 612,122 (incl. 28,474 controls) · validation 6,731 · test 37,490
  source HEPG2 n_obs=101,252 · jurkat 262,956 · RPE1 247,914
  validation: 6,731 cells from HEPG2 (103 perturbations)
  injected 4,976 train controls into validation (cell-eval split has none)
  loaded n_obs=612,122 n_vars=6,983  wall_s=227.6
  pca n_comps=256 train=612,122 valid=11,707
```

The control-injection line is the fallback added pre-launch: the cell-eval validation split
contains **zero** controls by construction, and CellFlow's `ctrl_x` needs some, so TRAINING
controls are used. Without it the run raises before step 1.

## ⭐⭐ C22 — FIRST COMPLETE COMPETITOR RESULT: TxPert 0.3077 on 5/5 splits. Occam leads by 0.0338.

**All 5 splits, uncalibrated (`model_nocal`), same 611-perturbation cohort, same drawn half-B
controls, same cell-eval 0.6.0.** ± is the across-split sd.

| metric | **TxPert (selected ep49)** | TxPert ep20 interim | **ours `s313`** |
|---|---|---|---|
| `pearson_delta` | 0.3077 ± 0.0032 | 0.3038 ± 0.0037 | **0.3415 ± 0.0031** |
| `mae` ↓ | 0.0228 ± 0.0000 | 0.0231 ± 0.0000 | **0.0210 ± 0.0000** |
| `overlap_at_N` | 0.0522 ± 0.0004 | 0.0486 ± 0.0006 | **0.0642 ± 0.0010** |
| `discrimination_score_l1` | 0.6458 ± 0.0008 | 0.6450 ± 0.0009 | **0.6534 ± 0.0007** |

**Occam wins all four metrics.** Paired over all 5 splits:

| comparison | mean | sd | wins |
|---|---|---|---|
| ep49 − ours | **−0.0338** | 0.0032 | **0/5** |
| ep20 − ours | −0.0377 | 0.0032 | 0/5 |
| **ep49 − ep20** | **+0.0039** | 0.0010 | **5/5** |

⭐ **The gap is 10x its own spread** (−0.0338 against sd 0.0032) and 0/5 splits, so it is a
real margin, not noise. And TxPert is a genuine competitor, not a broken port: **0.3077 clears
the `transfer` bar 0.2323 by +0.075** and sits below `experimental` 0.3722, which is where a
real model should land.

⭐ **Validation selection earned its keep, just barely.** ep49 beats ep20 by **+0.0039** with
sd 0.0010 and **5/5 splits** -- a 4x signal-to-spread margin on a difference the validation
curve itself could not resolve (`selection_score` 0.54327 at ep20 vs 0.54428 at ep50, a 0.001
gap). Selecting on validation picked up something the metric's own noise hid. It also wins on
all four metrics, not just the selection ones.

⚠️ `mae` has sd **0.0000** for every arm, to four decimals. It does not vary across splits
here, so it carries no information about split noise -- only about the arm.

### Caveats that travel with the number

1. **GO + STRING only**, to match Occam's own `graph_types`. Their published 0.50787 used four
   KGs, so this is NOT their configuration and not comparable to it.
2. **One seed (42).** Upstream measured ±0.006 seed noise for TxPert on their panel; our 0.0338
   gap is 5x that, but a single seed is a single seed.
3. **Uncalibrated, by design** -- their model emits gene space directly, our comparison column
   is `model_nocal`, and calibration is a control-fit map we apply to our own decoder.

### ⚠️ Split 4 needs more than 64 GB

Both chains hit `OUT_OF_MEMORY` on split 4 at the score leg's 64 GB default; MaxRSS on the
retry was **65.6 GB** and **69.6 GB**. Splits 0-3 fit. Resubmitted at 128 GB and both completed
in ~14.5 min.

`scripts/slurm/competitor_score.sbatch` asks for 64 GB, which is enough for 4 of 5 splits --
the worst possible amount, because it fails late and only sometimes. **Raised to 128 GB.**

## ⭐ C21 — ALL FOUR ARMS LAUNCHED. Pre-flighting found 9 bugs and burned 0 jobs.

**Decision 2026-09-18, after PerturbDiff cost four 45-minute runs: pre-flight every arm's
validation path on REAL objects before submitting.** The user asked for it explicitly. It is
now the rule for this benchmark.

| arm | bugs found BEFORE launch | jobs burned finding them |
|---|---|---|
| PerturbDiff | 0 -- found by launching | **4 x 45 min** |
| TxPert | 0 -- found by launching | **5 short export runs** |
| **STATE** | pre-flight passed 4/4 first try | **0** |
| **CellFlow** | **5 found** | **0** |

### Why every arm needs this: they all swallow their own validation errors

`ValidationCellEvalCallback` in STATE, the callback in PerturbDiff, and CellFlow's
`SelectionCallback` all wrap themselves in `except Exception: log; continue`. So a defect in
that path does NOT fail where it happens -- it surfaces much later as a checkpoint failure on a
missing metric, in a different component. On PerturbDiff that turned four distinct root causes
into four identical-looking `MisconfigurationException`s, 45 minutes apart.

### STATE, 4/4 first try

```
1. get_var_dims()['gene_names']  6983 names, axis-identical=True  ['DPM1','GCLC','NIPAL3']
2. val_dataloader_for_scoring()  OK, pert_cell_emb (64, 6983)
3. get_control_pert()            'empty'
4. control rows in a val batch   64/64   (inject_real_controls RAISES at 0)
```

⚠️ #1 is checked by IDENTITY with the frozen axis, not by length -- the mistake that cost
PerturbDiff its third run. If `gene_names` were absent, `var` is `None` and cell-eval cannot
exclude each perturbation's own target gene by symbol: a **plausible but wrong** score, which
is worse than a crash.

ⓘ #4 came back 64/64 controls -- the first sentence happens to be all-control. Harmless
because `max_batches` is unset so the callback walks the whole loader, and validation holds
6,731 cells over 103 perturbations. It would bite if anyone set `max_batches` small.

### CellFlow, 5 bugs, 3 from reading the path and 2 from the pre-flight

| # | bug | consequence if launched |
|---|---|---|
| 1 | selection still called the sibling repo's scorer; the config no longer supplied its paths | **KeyError at startup** |
| 2 | `genes=adata_train.var_names` -> **Ensembl**, and those become the var index cell-eval scores | **silently inflated** selection scores, no error |
| 3 | `ctrl_x` read from the VALIDATION split, which has **0 controls** by construction | `SelectionCallback` **raises** before step 1 |
| 4 | `.to_numpy()` on a bool column is read-only, so my own `&=` fallback raised | crash at startup -- **a bug in my own fix** |
| 5 | `requests` is in CellFlow's OPTIONAL `embedding` extra, but `preprocessing/__init__` imports `_gene_emb` EAGERLY | `centered_pca` unreachable -- the core PCA path |

⚠️ **#1 means an earlier claim in C15 was FALSE.** I recorded that CellFlow's selection scores
through our cell-eval; the rewrite had been reverted by a copier run while the config kept my
comments saying otherwise. Redone, and the copier is now deleted so it cannot recur.

⭐ **#2 is the one worth remembering.** The pre-flight printed both, side by side, which is what
made it obvious:

```
genes[:3]      ['DPM1', 'GCLC', 'NIPAL3']
var_names[:3]  ['ENSG00000000419', 'ENSG00000001084', 'ENSG00000001461']
```

Same class as STATE's #1 and PerturbDiff's mask: **two 6,983-long lists in different ID
spaces.** Three arms, three instances, one shape of bug.

**#3 is structural to `rhj_fewshot`,** not a CellFlow defect: all 4,976 HepG2 controls are in
finetune by construction -- they are the paired basal pool -- so validation is 6,731 perturbed
cells and nothing else. It now falls back to the TRAINING controls of the same cell line, which
the pre-flight confirms finds exactly **4,976**, the benchmark's own count. That is the same
`injctrl` convention our own arms use.

**#5 was fixed by making the import LAZY**, not by installing the extra: that would have pulled
~2.5 GB of torch into a jax environment for two Ensembl REST calls this arm never makes, and
the internal registry needs credentials this shell does not have.

ⓘ Also noted: `uv pip install` in `competitors/CellFlow/cellflow` silently targets a leaked
`VIRTUAL_ENV` (a `biomodal` pyenv), not `.venv`. Use
`env -u VIRTUAL_ENV uv pip install --python .venv/bin/python`.

### The four chains

| arm | train | export | score | resources |
|---|---|---|---|---|
| TxPert | ✅ done, `best-ep49` | `18089777` | `18089778` | h100 / 200 G |
| STATE | `18091685` | `18091686` | `18091687` | 320 G / 24 h |
| PerturbDiff genept | `18091494` | `18091495` | `18091496` | h100 / 128 G |
| PerturbDiff onehot | `18091497` | `18091498` | `18091499` | h100 / 128 G |
| CellFlow | `18094421` | `18094422` | `18094423` | 256 G / 12 h |

Plus TxPert's epoch-20 interim (`18088687` / `18088689`), whose split 0 already scored
**0.3014** (C19).

## ⛔ C20 — PerturbDiff failed its first validation FOUR times. The lesson is about verification.

Four runs, ~45 min each, all dying at the same place: the first validation cell-eval at global
step 16,000. **Training was healthy every single time** -- 0.143-0.187 s/step, losses falling
(`training_loss 2.78`, `validation_loss 2.580`), gradients finite (total norm 0.762). Only the
validation callback failed.

| attempt | jobs | cause |
|---|---|---|
| 1 | `18072825` `18072187` | `get_short_dsname` asserts on a dataset name it cannot map to a family; ours were cell lines |
| 2 | `18080122` `18080125` | `load_selected_genes` called `Path(None)` -- our axis needs no panel file |
| 3 | — | (my axis guard compared `ens` to a `selected` I had just changed to `sym`, so it fired on the 2nd dataset) |
| 4 | `18089025` `18089028` | the gene mask selected **zero** columns -> `r2_score` on `shape=(0,)` |

### The real defect was my verification, not the code

`generate_samples` masks with `np.isin(batch["col_genes"][0], load_selected_genes(cfg))`, and
`getitem_impl` fills `col_genes` from **`dataset.selected_genes`** (`dataset_core.py:227`) --
*not* from `gene_vars`, which is the attribute that looks like it should. I fixed `gene_vars`.

Each attempt I added a check, and each check tested a PROXY that passed while the real thing
was broken:

| attempt | what I asserted | why it passed anyway |
|---|---|---|
| 2 | the panel file loads | it did |
| 3 | the panel has 6,983 entries | it did -- **length is not agreement** |
| 4 | `read_occam_line`'s symbols match the panel | they did -- wrong object |
| ✅ | **`dataset.selected_genes` vs the panel, on a real batch** | this is the comparison the model makes |

⭐ **The fix that actually worked was a 9-minute interactive run**, not a check: build the
datamodule, pull a real validation batch, print `np.isin(col_genes, panel).sum()`.

```
col_genes[:3]  ['DPM1', 'GCLC', 'NIPAL3']
panel[:3]      ['DPM1', 'GCLC', 'NIPAL3']
mask.sum()     6983 / 6983          VERDICT OK
pert_emb       (1, 16, 6983)        batch x cell_set 16 x genes
```

It also caught my own broken axis guard on the first try -- a bug that would have cost a fifth
45-minute run.

⚠️ **WHY THIS WAS SO EXPENSIVE, and the one structural thing to remember.** The validation
callback CATCHES its own exceptions, logs *"not publishing a score"*, and lets training
continue. `ModelCheckpoint(monitor="validation/celleval/selection_score")` then fails on the
missing metric. So every defect in that path surfaces **45 minutes later, as a different
error, in a different component**. Four separate root causes all presented as the same
`MisconfigurationException`.

The setup probe in `occam_setup_dataset` now compares the two objects the model compares, after
the datasets exist. Anything else in that path should be checked the same way: interactively,
against the real attribute, before a job is submitted.

Relaunched: genept `18091494`, onehot `18091497`.

## ⭐ C19 — FIRST COMPETITOR NUMBER: TxPert 0.3014 on split 0, from an epoch-20 interim

**The port produces a score.** Job `18088688`, 8 min 46 s, split 0 only.

| metric | TxPert ep20 | `s313` (ours, 5 splits) | `transfer` | `experimental` |
|---|---|---|---|---|
| `pearson_delta` | **0.3014** | **0.3415** | 0.2323 | 0.3722 |
| `mae` ↓ | 0.0231 | **0.0210** | 0.0299 | 0.0257 |
| `overlap_at_N` | 0.0491 | **0.0642** | 0.0351 | 0.1942 |
| `discrimination_score_l1` | 0.6460 | **0.6534** | 0.6404 | 0.7821 |

**TxPert clears the `transfer` bar (0.2323) by +0.069 and sits 0.040 below our best arm** on
`pearson_delta`, on the same 611-perturbation cohort, the same drawn half-B controls, the same
cell-eval 0.6.0, uncalibrated both sides. It is ahead of `transfer` on all four metrics.

⚠️ **THIS IS A FLOOR ON TXPERT, NOT ITS RESULT**, for three reasons that all point the same way:

1. **Epoch 20 of 50.** An interim checkpoint, taken so the adapter could be exercised before
   training finished. Its own validation `selection_score` was still climbing when it was
   taken: **0.51023 (ep10) -> 0.54327 (ep20)**.
2. **One split of five.** Split-to-split spread on this panel is ~0.003, so the ordering is
   unlikely to move much, but a single split is not the claim.
3. **GO + STRING only**, to match Occam's own prior. Their published 0.50787 used four KGs.

### The emission diagnostics, which is what the guard is for now

```
representable ✓   rowsum ratio 0.8701 · gene-mean corr 0.9994 · zero 0.0001
```

Against our own arm's **0.9399 / 0.9998 / 0.0000**. TxPert sits in the same space, slightly
lower-calibrated on total expression — and `mae` reports exactly that: **0.0231 vs our 0.0210**.
That is the diagnostic doing its job as a *number to read* rather than a reason to refuse, which
is the whole point of C18.

ⓘ Note the ranking is consistent across all four metrics here, unlike the `s313`/`s469`
comparison. When an arm is 0.04 behind rather than 0.002, the metrics agree.

## C17 — the TxPert adapter's first execution: five bugs, all from `compose()` vs `@hydra.main`

**Ran on an INTERIM checkpoint (epoch 20, `selection_score` 0.54327) so the adapter could be
exercised before training finishes.** `export_occam.py` had never executed; C11 flagged it as
the least-tested part of the port, and it was.

| # | bug | cause |
|---|---|---|
| 1 | `ImportError: cannot import name 'HybridDataModuleWrapper'` | it is in `gspp.data.datamodule_hybrid`, and `GSPGraph` in `gspp.data.graphmodule` — I guessed both module paths instead of reading `main.py` |
| 2 | `ConfigAttributeError: Key 'added_data_args' is not in struct` | `compose()` returns a **struct-mode** config; `modify_config` *writes* that key |
| 3 | `InterpolationResolutionError: HydraConfig was not set` | `${hydra:runtime.cwd}` is used by `datamodule.cache_dir` and the graph paths |
| 4 | (latent) wrong `runtime.cwd` | training runs from the TxPert dir, the adapter from the repo root — different cache and graph paths, so the export would read a graph the checkpoint never trained on |
| 5 | `AttributeError: 'numpy.ndarray' object has no attribute 'toarray'` | `_to_tensor` calls `.toarray()` unconditionally, so X must be **sparse** |
| 6 | (latent) `KeyError` on `obs[BATCH]` | `BATCH` is `"batch"`, `BATCH_RAW` is `"batch_num"` — **different constants**, and the datamodule's own loader writes the latter while the dataset reads the former |

⭐ **1–4 are one root cause: `compose()` does not reproduce what `@hydra.main` sets up.** Worth
knowing for the other three adapters, which all use the same pattern. 5 and 6 are contract
mismatches with `XcelltypePerturbDataset` that only an execution could surface.

### ⚠️ A process failure of mine, recorded because it cost more than the bugs

I found 1–3 through **SLURM round trips** — submit, wait ~5 min, read one traceback, fix, repeat
— on code I had already flagged as never having run. Worse, when the third fix worked, the job
went **silent for 11 minutes** during the GO+STRING graph build (which logs nothing, and shows
0% GPU because it is CPU-bound), and I read that silence as failure and **cancelled two healthy
jobs**. The worker was at 110% CPU with 51 GB resident the whole time.

The right loop, which the user asked for and which found bugs 5 and 6 immediately, is to run
the adapter **interactively** on an idle GPU and read errors directly. ~9 min per attempt, all
of it the 656k-cell data load, and no queue in between.

## ⛔ C18 — the emission guard would have REJECTED OUR OWN BEST ARM. Split into hard and soft.

**Finding: `assert_same_space`'s zero-fraction test is invalid for comparing a PREDICTION to a
TARGET, and enforcing it on competitors alone was an asymmetry that rejected them for a
property our own model has more extremely.**

It refused TxPert's epoch-20 export. Before accepting that, I measured our own arm on the same
split -- the check that should have been in C1 from the start:

| | zero fraction | median library | max |
|---|---|---|---|
| the real target | **0.5170** | 1,729.9 | 5.925 |
| **OUR `s313` prediction** (scores **0.3415**) | **0.0000** | 1,626.0 (**-6.0%**) | 4.015 |
| TxPert epoch-20 | 0.0001 | 1,505.1 (**-13.0%**) | 4.326 |

⭐ **Our own best arm fails the enforced check by a 0.517 margin against a 0.05 tolerance** --
worse than the competitor it rejected. A conditional-mean regressor essentially never emits an
exact zero; the target is sparse single-cell counts. TxPert's emission is in *the same space as
our own model's*, which is exactly what the guard was supposed to establish.

Our arms never meet this function -- they go straight through `protocol.run_split` -- so the
asymmetry was invisible until a competitor hit it.

### What is now HARD, and what is REPORTED

| check | now | why |
|---|---|---|
| negatives / NaN / Inf | **refuse** | a library-normalised matrix cannot be negative |
| gene axis identity + order | **refuse** | `discrimination_score_l1` excludes each gene BY SYMBOL |
| per-gene mean correlation >= 0.95 | **refuse** | catches wrong units or a scrambled axis (ours: **0.9998**) |
| zero fraction **higher** than target | **refuse** | the integer-re-emission signature (measured 0.907 vs 0.606) -- still fatal |
| zero fraction **lower** than target | **report** | expected of any conditional mean; ours is 0.0000 |
| library size **above** target +10% | **refuse** | no benign explanation at that magnitude |
| library size **below** target, to -25% | **report** | a conditional mean shrinks toward the centre; `mae` should MEASURE this, not refuse it |

⚠️ The direction matters and is the whole fix: **too many zeros is an artifact, too few is a
model.** The original test took an absolute difference and so could not tell them apart.

Verified after the change: our own `s313` prediction returns `same space`, rowsum ratio
**0.9399**, gene-mean corr **0.9998**.

### TxPert's own emission, now measured

* **Negatives**, log1p range down to **-0.3274**. Its head is unconstrained. Clipped at 0 in
  its adapter, and the clip is tiny and now printed every run: **289-422 of 3,575,296 values
  per batch, 0.005-0.012%**, min -0.29.
* Library size **-13.0%** against the target, versus our own **-6.0%**. Both under-shoot; the
  gap between them is a real difference in model behaviour that `mae` will report.

## Status

| step | state |
|---|---|
| 0 — commit and push | ✅ 26 commits pushed (`0b94934..b567a8b`) |
| 1 — `competitors/` skeleton + gitignore | ✅ rules verified; only `MANIFEST.json` is trackable |
| 5b — emission space | ✅ C1 above, 13 tests |
| 2 — `occam_data.py` reader | ✅ C2 above, bit-identical gate passed, 9 tests |
| 3 — copy code per arm | ✅ 347 files, **6.6 MB** from a 141 GB upstream |
| 4a — TxPert reads our data (GO + STRING only) | ✅ verified on real data, `12,000 x 6,983` csr |
| 4b — STATE reads our data | ✅ C6 above, smoke-tested end to end |
| 4c — CellFlow reads our data | ✅ C8 above, smoke-tested on the FULL benchmark |
| 4d — PerturbDiff reads our data | ✅ C9 above, smoke-tested on the FULL benchmark; validation count matches CellFlow's exactly |
| 5b — the four Phase-A adapters | ✅ written (C11); ⚠️ **still UNEXERCISED** — TxPert's runs in the export leg `18067028`, chained behind training |
| first end-to-end arm | ✅ **TxPert TRAINING** on our data (C12), job `18067027` on hooke-predict |
| 5 — the prediction bridge | ✅ Phase A + Phase B written, control draw verified identical to `predict_panel`; **gate PASSED** (C7) |
| 6 — launchers, collector | ✅ C5 above; `scripts/collect_competitors.py` writes `docs/rhj_competitors_results.md` |
| 7 — tests | ✅ `test_competitor_bridge.py` (10) + `test_competitor_manifest.py` (7) |

Plan: `~/.claude/plans/i-want-to-make-iterative-cray.md`.
Panel definition: `data/benchmarks/rhj_fewshot/splithalf/CARD.md`.
Baselines: `docs/rhj_baseline_results.md`. Protocol findings: `docs/rhj_splithalf.md`.
