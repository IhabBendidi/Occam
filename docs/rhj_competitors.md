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
| `occam_rhj_fewshot.toml` | NEW. `[datasets]` names our three h5ads. **No `[splits]` table** — the fold is computed live from the benchmark, so it cannot drift from `splits.json`. |

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
