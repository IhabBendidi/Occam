# Occam

> **Current state — benchmarks, published checkpoints, defaults, where every number lives:
> [`docs/STATUS.md`](docs/STATUS.md).** Anything not listed there is not current. Stale
> checkpoints, results and benchmarks are in [`archive/`](archive/README.md) and must not be used.

A masked autoencoder for single-cell transcriptomes, and a perturbation-prediction model
built on it. The encoder uses **all genes** in the input rather than a highly-variable
subset, corrupts them with **swap noise** rather than zeros, and jointly predicts which
genes were corrupted and reconstructs the expression. A separate high-capacity decoder is
trained on the frozen encoder. A **virtual-cell** head on top predicts what a perturbation
does to a cell.

## What do you want to do?

| I want to… | go to |
|---|---|
| **use a checkpoint** — embed cells, reconstruct, predict a perturbation | [1. Install](#1-install) → [2. Get the weights](#2-get-the-weights-and-the-data) → [3. Use it](#3-use-it-the-python-api) |
| **reproduce the data and splits** exactly as we use them | [4. Rebuild the benchmarks](#4-rebuild-the-benchmarks-data--splits) |
| **reproduce a training run** | [5. Train](#5-train) |
| **evaluate a checkpoint**, ours or yours | [6. Evaluate](#6-evaluate) |
| **see every number we report** | [7. The results](#7-the-results) |

Every path in this repo is **relative** to the repo root, so there is nothing to configure
beyond putting the downloads in place.

---

## 1. Install

Python **3.8+**, a CUDA GPU for training (CPU is fine for inference on a few thousand
cells).

```bash
git clone git@github.com:IhabBendidi/Occam.git && cd Occam
python -m venv .venv && source .venv/bin/activate     # or conda / pyenv
pip install -e ".[eval]"
cp configs/paths.example.yaml configs/paths.yaml       # relative defaults; no edit needed
```

`".[eval]"` adds the evaluation extras. Plain `pip install -e .` is enough to *use* a
checkpoint.

**Scoring needs cell-eval 0.6.0 in its own environment**, because it pins versions that
conflict with training. Every script that scores shells out to it:

```bash
python -m venv .venv-celleval && .venv-celleval/bin/pip install 'cell-eval==0.6.0'
export OCCAM_CELLEVAL_PYTHON="$PWD/.venv-celleval/bin/python"   # or set PYENV_VERSION
```

Check the install:

```bash
python scripts/verify_checkpoint.py      # 28 checks on the released weights
python scripts/_run_tests.py             # the full suite, ~4 min, no GPU needed
```

The suite runs under a minimal in-repo runner (`scripts/_run_tests.py`) so it works
without pytest; `pytest tests/` also works if you have it.

---

## 2. Get the weights and the data

Everything is in **one private HuggingFace repo**, `bendidiihab/Occam` (21.2 GB, 334
files). Download only what you need:

```bash
hf auth login                      # a read token for the private repo

# the released encoder + decoder (0.93 GB) -- needed for everything
hf download bendidiihab/Occam --include 'checkpoints/*.pth' 'checkpoints/*.json' \
    'checkpoints/*.md' --local-dir /tmp/dl && mv /tmp/dl/checkpoints/* checkpoints/release/

# the per-gene GenePT + ESM2 tables the ALL-sources perturbation model reads (0.39 GB)
hf download bendidiihab/Occam --include 'assets/*' --local-dir /tmp/dl && \
    mkdir -p data/assets && mv /tmp/dl/assets/* data/assets/

# the two perturbation-prediction checkpoints (2.23 GB each)
hf download bendidiihab/Occam --include 'checkpoints/vcell/*' --local-dir /tmp/dl
```

| folder | size | needed for |
|---|---|---|
| `checkpoints/` | 5.35 GB | the released encoder+decoder **and** both vcell checkpoints + their cards |
| `eval/` | 6.25 GB | evaluation data — 7 datasets incl. `norman_2019_k562.h5ad` |
| `benchmarks/` | see HF | the frozen `rhj_fewshot`, `norman_combo`, `h1_seen`, `k562_seen`, `k562_seen_h1set` benchmarks (the last three with their raw local copies) |
| `graphs/` | 0.40 GB | the perturbation graph — **required to train or to rebuild a benchmark** |
| `assets/` | 0.39 GB | per-gene GenePT + ESM2 tables — **required by the ALL-sources default** and the embedding arms (`data/assets/README.md`) |
| `metrics/` | 22 MB | every cell-eval metric, per arm, per split (also in git) |
| `metrics_all/` | 38 MB | the same for all 422 eval roots |
| `repro/` | — | the scripts in `scripts/repro/`, for convenience |

Destinations: `checkpoints/*` → `checkpoints/release/` · `eval/pert|scib` →
`data/eval/pert|scib/` · `benchmarks/*` → `data/benchmarks/*` · `graphs/*` →
`data/graphs/` · `assets/*` → `data/assets/`. (The DiverseRNA pretraining corpus was removed from the Hub on 2026-09-24;
it is local only, `data/diverserna/`.)

**The repo is private and must stay private.** Neither dataset's licensing is resolved:
the pretraining corpus has 7 of 9 constituent blocks unattributed and the two identified
(GTEx, TCGA) carry redistribution conditions, and none of the evaluation datasets has a
recorded licence. `scripts/check_upload_gate.py` gates *public* release and currently
refuses, which is the correct state. See [docs/hf_release.md](docs/hf_release.md).

---

## 3. Use it: the Python API

`occam.api` is the **stable surface** — three loaders, three objects. Everything else in
the package is training and benchmarking machinery whose signatures move with the
campaign; this does not. Full reference: [docs/api.md](docs/api.md).

It handles the two things that are easy to get silently wrong: **gene alignment** (the
checkpoint's vocabulary is the truth, not your column order) and **preprocessing**
(library-normalise to the checkpoint's own `target_sum`, then log1p, honouring its skip
flags).

### Embed cells

```python
import anndata as ad
from occam.api import load_encoder

enc = load_encoder("checkpoints/release/occam_small_encoder_decoder.pth")
cells = ad.read_h5ad("my_cells.h5ad")             # RAW COUNTS
z = enc.encode(cells, gene_col="gene_id", verbose=True)    # (n_cells, 512)
```

`gene_col` names the `.var` column holding gene ids; omit it to use `.var_names`. The
released checkpoints use **Ensembl** ids. `verbose=True` prints what alignment did:

```
[occam] 12 cell(s) · 29,187/33,694 input genes matched · 4,507 dropped ·
        15,541/44,728 vocabulary genes zero-filled (34.7%) ·
        normalize=True (target_sum 2000) log1p=True
```

### Decode with the simple decoder

```python
from occam.api import load_decoder

dec = load_decoder("checkpoints/release/occam_small_encoder_decoder.pth")
x_hat = dec.decode(z)                                    # (n_cells, 44728), LOG1P
x_hat = dec.reconstruct(cells, gene_col="gene_id")       # encode then decode
counts = dec.reconstruct(cells, gene_col="gene_id", counts=True)   # back to counts
```

### Predict a perturbation with a virtual cell

Works on the two published checkpoints **and on any you train yourself** — nothing here is
specialised to ours.

```python
from occam.api import load_virtual_cell

vc = load_virtual_cell("checkpoints/vcell/norman_combo_NQ02/best_model.pth")
vc.perturbations[:5]        # the vocabulary it knows (18,216 genes)
vc.compose_modes            # the 12 multi-perturbation composition modes

pred = vc.predict(controls, ["FOXA1", "CBL+CNN1"], gene_col="gene_id")
# AnnData, one row per (control cell x perturbation); obs['target_gene'] is the label

effect = vc.delta(controls, ["CBL+CNN1"], gene_col="gene_id")   # (1, n_genes)
```

- `"A+B"` is a **pair**, composed with `mode=` (default `decode_seq_mean`, the adopted
  default — [docs/default_recipe.md](docs/default_recipe.md)). `mode` has no effect on a
  single perturbation: every reconstruction-based mode is algebraically identical there.
- `predict` returns a **profile**; `delta` returns the **effect**. `delta(baseline=...)`
  chooses what is subtracted — `"reconstructed"` (default, cancels the decoder's own
  error) or `"real"` (relative to the measured control). The two are not comparable; read
  each against its own family.

### From the shell

```bash
occam-embed       --model_dir checkpoints/release --input_data cells.h5ad --output emb
occam-reconstruct --model_dir checkpoints/release --input_data cells.h5ad --output recon.h5ad
occam-benchmark   --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-scib        --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-verify

# perturbation prediction from your own control cells
python scripts/repro/occam_predict.py \
    --checkpoint checkpoints/vcell/norman_combo_NQ02/best_model.pth \
    --controls my_controls.h5ad --gene-col gene_id \
    --perts FOXA1 "CBL+CNN1" --out predictions.h5ad
```

---

## 4. Rebuild the benchmarks (data + splits)

Two benchmarks, both **frozen artifacts** — the axis, the cohort, the seeded splits and
the baselines are what make one person's number comparable to another's.

| | `rhj_fewshot` | `norman_combo` |
|---|---|---|
| question | unseen perturbation, seen cell-line family | unseen PAIR, from singles only |
| data | HepG2 + Jurkat + RPE1 | K562 (Norman 2019, rebuilt from GEO) |
| axis | 6,983 genes | 8,000 genes |
| test | 611 held-out HepG2 perturbations | 111 pairs, in two effect-controlled strata |

### What each benchmark tests

The three `rhj_fewshot`-family benchmarks score the SAME checkpoints (trained on HepG2 +
Jurkat + RPE1) and differ in what is new at test time:

| benchmark | perturbation | assay | cell line | perturbations scored | built by |
|---|---|---|---|---|---|
| `rhj_fewshot` | **few-shot**: withheld from HepG2, seen in Jurkat/RPE1 | seen | seen (HepG2) | 611 | `build_benchmark_rhj.sh` |
| `k562_seen` | seen | seen (Replogle 2022 CRISPRi screen, the RPE1 file's family) | **unseen** (K562) | 528 RHJ test perts | `build_benchmark_k562.sh` |
| `k562_seen_h1set` | seen | seen | **unseen** (K562) | the 80 shared with `h1_seen` | `build_benchmark_k562.sh` |
| `h1_seen` | seen | **unseen** (VCC 2025 H1, different protocol and depth) | **unseen** (H1 hESC) | 80 | `build_benchmark_h1.sh` |

- `k562_seen` uses the same perturbations as the `rhj_fewshot` test set (528 of its 611), so
  RHJ → K562 changes only the cell line.
- `k562_seen_h1set` and `h1_seen` use the SAME 80 perturbations
  (`data/benchmarks/shared/h1_k562_perts.json`), so K562 → H1 changes only the assay.
  Cohort-ranked metrics (`discrimination_score_l1`, `overlap_at_N`) are only comparable
  between panels scored on identical sets.
- None of them trains anything. Input and scoring genes are the `rhj_fewshot` axis genes the
  panel measures (6,640 for K562, 6,535 for H1), identical for every arm. Competitors with a
  cell-type label are given `HEPG2`. Scored with `scripts/repro/ood_seen.sh`
  (`--ood-context` in `scripts/eval_splithalf.py`).

**You do not need to rebuild them** — downloading `benchmarks/` gives you the exact
artifacts. Rebuild to *verify* them, or to build from a different source revision:

```bash
OUT=/tmp/check_rhj    bash scripts/repro/build_benchmark_rhj.sh      # ~25 min, ~120 GB RAM
OUT=/tmp/check_norman bash scripts/repro/build_benchmark_norman.sh   # ~15 min, ~60 GB RAM
OUT=/tmp/check_h1     bash scripts/repro/build_benchmark_h1.sh       # ~10 min, from the HF H1 copy
OUT=/tmp/check_k562   bash scripts/repro/build_benchmark_k562.sh     # ~10 min, from the HF K562 copy
```

Each rebuilds the axis, cohort, splits, the 5 seeded split-halves and (Norman) the
additivity strata, then runs `scripts/repro/verify_benchmark.py`, which compares **every
scientific quantity** against the published artifact — the axis digest, the builder's own
per-file content digests, the perturbation counts, the input `.h5ad` checksums, the cohort
digest and all seven panel memberships pair by pair. It prints *REPRODUCED* or exits 3
naming each difference.

**Both reproduce exactly** (measured 2026-09-22).

⚠️ The check is **semantic, not a byte hash**, and it has to be: `benchmark.json` embeds
`git_sha`, the build date and the output path, so hashing the file reports a mismatch on a
rebuild that is bit-identical in content. `axis.npz` is a zip and carries per-entry
timestamps.

⚠️ Both scripts **refuse to write over a frozen benchmark**. Build into a scratch `OUT`,
or `FORCE=1` to replace it deliberately.

✅ **A rebuilt panel can score a published checkpoint** — no flag needed. See
[6. Evaluate](#6-evaluate).

---

## 5. Train

### The perturbation model (virtual cell)

The config **is** the recipe — no `--set` overrides:

```bash
sbatch scripts/repro/rhj_train.sh        # ~4.5 h, 1x H100
sbatch scripts/repro/norman_train.sh     # ~3 h,   1x H100
```

or directly:

```bash
python -m occam.train --config configs/rhj_fewshot.yaml --stages vcell \
    --run-suffix my_run
```

Both configs ship the **default recipe** adopted 2026-09-22: Q02 loss base, **150
epochs**, `eta_min` **3e-4**, batch 2048, lr and decoder_lr 1e-3. The schedule is worth
more than any loss term measured (+0.026 `pearson_delta` on `rhj_fewshot`, and +0.0840 on
`norman_combo`), so **compare a new idea against this, not against the old 30-epoch
number**. Full evidence and the procedure for a new dataset:
[docs/default_recipe.md](docs/default_recipe.md).

⚠️ **The three loss weights are scale-matched per corpus and must not be copied.** RHJ
uses dcos 0.5 / mag 0.01363 / latent-MSE 0.0044; Norman uses 0.32977 / 0.0400866 /
0.00198826 for the *same* base. `tests/test_default_recipe.py` fails if the two ever match.

⚠️ `best_model.pth` is selected on **validation** `pearson_delta`. Validation picks the
epoch well but **misranks arms** (finding F-ckpt4, held on 5 consecutive grids) — do not
use a validation number to compare two configurations. `final_model.pth` is written only
at the very end, so its presence is the completion marker.

### Finetuning the encoder on your own data (label-free)

Start from the **stage-2 MAE** (`checkpoints/reference/occam_small_stage2_finetune.pth`,
HF `checkpoints/stage2/`), not the released file: only stage 2 keeps the MAE's own
decoder and mask predictor. Its encoder is byte-identical to the released one.

```bash
python scripts/finetune/make_finetune_init.py \
    --parent checkpoints/reference/occam_small_stage2_finetune.pth \
    --out runs/finetune_init/occam_small_stage2_mask0.1.pth
python scripts/finetune/densify_h5ad.py --src mine.h5ad --dst mine_dense.h5ad
python -m occam.train --config configs/finetune_ssl.yaml --set data.train=mine_dense.h5ad
```

It is the stage-2 recipe with 5 epochs instead of 100. The first step is needed: without
it, masking is silently off. See [docs/finetuning.md](docs/finetuning.md).

### The MAE itself

#### The published MAE recipe

> **The released decoder is the softplus + NB one as of 2026-09-16.** It beat
> the previous `rectified_sigmoid` + smooth-L1 decoder by **+0.3747 calibrated
> `pearson_delta`** (0.1985 → 0.5732) on `nadig_2024_jurkat`, with the DE
> over-call factor down from 14.8× to 6.9×. Stages 1–2 come from
> `configs/occam_small.yaml`; **stage 3 comes from
> `configs/decoder_softplus_nb.yaml`**. The superseded decoder is kept at
> `checkpoints/legacy/` — not published — because the golden fixtures pin it and
> it is the port-equivalence evidence. See
> [docs/decoder_retrain_jurkat.md](docs/decoder_retrain_jurkat.md).

`configs/occam_small.yaml` **is** the recipe. It was generated from the released
checkpoints' own embedded configs, not transcribed from run logs, because two of
the three stages were misdocumented in the tree this repo replaces.

| stage | loss | lr | epochs | result |
|---|---|---|---|---|
| 1 pretrain | `nb_softplus`, MLW ramp 0.7 → 0.9 | 4e-4 | 100 | val 0.8786 |
| 2 finetune | `poisson`, MLW ramp 0.7 → 0.0, gradual unmasking | 6e-4 | 100 | val 0.05102 |
| 3 decoder | `nb` (learned per-gene dispersion), `softplus` head, frozen encoder, 2048×4 | **5e-5** | 200 | val 0.05306 (NB) @ ep 124 |

`small` = latent 512, encoder `44728 → 2048 → 1024 → 512`, 209,355,632 params in
the full MAE. The released unified encoder+decoder is 222,736,249 (encoder
94,279,864 + decoder 128,456,385).

The superseded stage 3 (`smooth_l1` β 2.0, `rectified_sigmoid`, 2048×**5**, lr
3e-6, val 0.0011786 @ ep 185, 231,673,723 params unified) is the stage-3 block of
`configs/occam_small.yaml` and the file in `checkpoints/legacy/`.

⚠️ Stage 1 and the superseded stage 3 differ from what was previously published.
Two run directories were written by colliding SLURM array tasks, so the
documented recipe did not produce those weights.
**[docs/reproducing_chain_a.md](docs/reproducing_chain_a.md) has the evidence.**
It is worth reading before you trust any number about this model.


#### Training the MAE, stage by stage

⚠️ `configs/occam_small.yaml` stage 3 trains the **superseded** decoder. To
reproduce the released one, run stages 1–2 from it and stage 3 from
`configs/decoder_softplus_nb.yaml`
(`sbatch scripts/slurm/retrain_decoder.sbatch`).

```bash
python -m occam.train --config configs/occam_small.yaml              # all three stages
python -m occam.train --config configs/occam_small.yaml --dry-run    # plan only
python -m occam.train --config configs/occam_small.yaml --stages decoder
python -m occam.train --config configs/occam_small.yaml --resume force
python -m occam.train --config configs/occam_small.yaml \
    --stages decoder --set stages.decoder.optim.lr=1.0e-05
```

Stages chain through checkpoint files on disk and record what they produced in
`runs/<name>/ledger.json`. A stage is reused only if it completed, its
hyperparameters are unchanged, its parent checkpoint is unchanged, and its
artifacts still exist. Otherwise it **blocks with a field-level diff** rather
than silently reusing weights trained under different settings.

`--set` may only change keys the config already declares, so a run is always
reproducible from a config file rather than from shell history.

Resources: ~6 h + ~9 h + ~12 h on one H100 over 1.43 M cells. To try the
plumbing first:

```bash
python scripts/make_smoke_corpus.py --out /tmp/occam_smoke.h5ad
python -m occam.train --config configs/occam_small_smoke.yaml \
    --set data.train=/tmp/occam_smoke.h5ad          # ~1 min, CPU
```

### The released decoder: softplus + NB

`configs/decoder_softplus_nb.yaml` retrains stage 3 with a softplus head and a
negative-binomial loss on the frozen stage-2 encoder. It was run (job
`18006463`), its epoch-124 checkpoint is the released decoder, and on
`nadig_2024_jurkat` it scores **0.5732** calibrated `pearson_delta` against the
superseded decoder's 0.1985 ([docs/decoder_retrain_jurkat.md](docs/decoder_retrain_jurkat.md)).

`sbatch scripts/slurm/retrain_decoder.sbatch` reproduces it. Read the config
header first — **calibration, not the loss, is the dominant term** (~0.37
pearson_delta against ~0.02), so read decoder-only scores calibrated.


### Building a panel from your own corpus

Build a panel from any perturbation corpus, then score a checkpoint on it:

```bash
python scripts/build_panel.py --source corpus.h5ad --out-dir data/panels/mine \
    --context-col cell_line --context K562 --cells-per-pert 400 --n-controls 18400
```

Controls are drawn once at 2× and split in half: one pool scores, the other is
the only one the model may see. Every cell carries its source provenance, and
the loader refuses a panel whose pools it cannot prove disjoint.

**Read the anchors.** The evaluation always scores two zero-signal arms
alongside the model — `ctrlmean` (the control mean) and `ctrlcell` (the drawn
control cell). They cannot be disabled and the report writer refuses to emit a
table without them. On this evaluation the zero-signal arm scores *higher* than
every model arm on `pearson_delta`; the model wins on `overlap_at_N`. Which
metric an anchor wins is a property of the panel, so the report states it per
run rather than asserting it in general.

See [docs/evaluation.md](docs/evaluation.md).


## 6. Evaluate

Everything is scored under **one protocol**: the paired split-half. For each perturbation
its cells are split in two — half A is the reference the metrics are computed against,
half B is the target, and the model is fed **half-B controls**, so it never sees a cell
that appears on the scoring side. 5 seeded splits, cell-eval 0.6.0, `--calibrate none`
(`nocal`) throughout.

```bash
# our checkpoint (defaults to the published one; pass a path for yours)
sbatch scripts/repro/rhj_eval.sh
sbatch scripts/repro/norman_eval.sh                       # compose=decode_seq_mean
sbatch scripts/repro/norman_eval.sh path/to/best_model.pth

# the baselines -- the scale every score is read against
sbatch scripts/repro/baselines.sh
BENCHMARK=norman_combo sbatch scripts/repro/baselines.sh

# a competitor, train -> export -> score chained
ARM=txpert bash scripts/repro/competitors_rhj.sh          # also state|cellflow|perturbdiff
```

Then collect **all 35** cell-eval metrics, per arm, per split, per panel:

```bash
python scripts/export_split_metrics.py                    # -> data/release/metrics/
```

### Scoring a published checkpoint on a panel you built

This works with no flag. `benchmark_sha256` is `sha256(benchmark.json)`, which embeds
`git_sha`, the build date and the output path, so a verified rebuild can never reproduce
it. Rather than refuse, the guard checks the property that digest stands in for, which is
**stronger** than the digest:

1. the **scoring axis** is identical (`gene_axis_sha256` is a content hash), and the
   benchmark `id`, `n_axis_genes` and `axis_scope` agree; **and**
2. **every perturbation the panel will score is held out of that checkpoint** — absent
   from its `split_info.train`, present in its `benchmark.excluded_perts`.

A different axis fails (1); a different split fails (2). The acceptance is printed, never
silent. `--require-exact-benchmark` demands byte equality instead.

Measured: the published Norman checkpoint on a rebuilt panel gives `pearson_delta` 0.7782
/ `overlap_at_N` 0.3065 / `mae` 0.0137 on split 0 — **identical to the published run** on
all three.

### Reading a score

⚠️ **A score is meaningless without its anchors**, and `pearson_delta` is a rank *within
its own panel* — a `norman_combo` number and an `rhj_fewshot` number are not comparable.

`rhj_fewshot`, `pearson_delta`, mean of 5 splits:

| arm | score |
|---|---|
| `ctrlcell` / `ctrlmean` (zero-signal floors) | −0.0018 / 0.0016 |
| `meanpert` (predict one constant delta) | 0.1324 |
| `transfer` (same perturbation, other cell line) | 0.2323 |
| **Occam `Y06`** | **0.4043** ± 0.0019 |
| `experimental` (replicate agreement) | 0.3722 |

`norman_combo`, `pearson_delta` (additive / non-additive panel):

| arm | score |
|---|---|
| `meanpert` | 0.4634 / 0.4016 |
| **Occam `NQ02` × `decode_seq_mean`** | **0.8169 / 0.7380** |
| **`sum_of_singles`** ⭐ the bar | **0.8824 / 0.7757** |
| `experimental` | 0.9165 / 0.8377 |

⚠️ `experimental` is replicate agreement at the panel's own cell counts, **not an upper
bound** — a model predicting the group mean can beat a noisy empirical half, and on
`rhj_fewshot` ours does.

⚠️ On `norman_combo`, `sum_of_singles` (`Δ_A + Δ_B`) **is this architecture's hypothesis
in closed form**. Our model does not beat it on `pearson_delta`; it does beat it on `mae`
and `overlap_at_N`.

⚠️ Do **not** rank on `overlap_at_N` on `rhj_fewshot`: a zero-signal predictor beats a
real model on it.

---

## 7. The results

| file | what is in it |
|---|---|
| [docs/default_recipe.md](docs/default_recipe.md) | the adopted recipe, its evidence, and how to start a new dataset |
| [docs/norman_results.md](docs/norman_results.md) | the 72-cell composition matrix (6 arms × 12 modes × 2 panels) and six aggregations |
| [docs/rhj_scoreboard.md](docs/rhj_scoreboard.md) | every `rhj_fewshot` arm, ours and the competitors' |
| [docs/rhj_splithalf.md](docs/rhj_splithalf.md) | the findings log — what was tried, what held, what was withdrawn |
| [docs/competitors_on_doubles.md](docs/competitors_on_doubles.md) | how each competitor handles doubles in its own paper |
| [docs/norman_benchmark.md](docs/norman_benchmark.md) | the Norman QC, the provenance correction, and the two test strata |
| `data/release/metrics/` | **all 35 metrics** for our published arms, every baseline and every competitor, per split and panel |

Superseded numbers stay in the docs beside their correction, never deleted — the history
of what was believed when is itself evidence, and three findings have been withdrawn.

---

## Gotchas that will bite you

1. **`decoder_num_blocks` is 4** for the released decoder (the code default).
   Only the superseded `checkpoints/legacy/` decoder is 5 blocks; it will not
   load at 4.
2. **Stage 2's `mask_probability: 0.0` is inert.** Under `gradual` unmasking the
   ramp starts from the *parent checkpoint's* value (0.1), so masking is live
   for most of the finetune.
3. **Stage 1 → 2 loads `strict=False` by design.** The `nb_softplus` parent
   carries a per-gene dispersion tensor the `poisson` child does not. The config
   declares the expectation so it is checked rather than ignored.
4. **`target_sum` is 2000, not 1e4.** It comes from the checkpoint's
   `dataset_info` in preference to its config, which is why.
5. **A unified checkpoint's `model_state_dict` is the decoder**, not the model.
   Use `occam.inference.loader.load_encoder_wrapper`.
6. **Output activation is only recorded in newer files.** The released file
   records `softplus` in `decoder_config`; the legacy file and older research
   checkpoints record nothing. A softplus decoder loaded under the
   `rectified_sigmoid` default loads cleanly and emits ~99% exact zeros, so pass
   it explicitly for those.
7. **Trust checkpoints, not sidecar JSON.** See
   [docs/reproducing_chain_a.md](docs/reproducing_chain_a.md).


## What is not here

* **The XCell benchmark modes**, and the four dataset-specific prediction
  writers that went with them. The panel builder is corpus-agnostic instead.
* **The second, count-space scorer.** Occam scores with cell-eval v1 only.
* **In-training panel-based checkpoint selection.** cell-eval v1 takes ~60 min
  per panel, too slow to run every few epochs, so selection is validation
  `pearson_delta` and the panel score is an end-of-training measurement. That is
  a real limitation, not a preference.
* **A previously published headline figure of 0.5818.** The tree this replaces
  records that measuring the exact quantity it claims returns chance, and asks
  for resolution before it is quoted again. It is retired, and the test suite
  fails if it reappears.


---

## Layout

```
configs/       occam_small.yaml (MAE recipe) · rhj_fewshot.yaml · norman_combo.yaml
src/occam/     api.py <- the public surface
               mae/ decoder_train/ vcell/ evaluation/ inference/ config/ pipeline/ cli/
scripts/repro/ the reproduction path: build, train, eval, baselines, competitors, predict
scripts/       panel building, dataset export, verification, metric export, porting
competitors/   the four ported methods (STATE, TxPert, CellFlow, PerturbDiff)
data/          eval/ benchmarks/ graphs/ diverserna/ release/metrics/
docs/          the recipe, the findings, the benchmarks, the release
golden/        numerical fixtures; the release must reproduce them
tests/         ~270 tests; scripts/_run_tests.py runs them without pytest
```

## Citation

Pending. See [MODEL_CARD.md](MODEL_CARD.md) and
[data/diverserna/DATASET_CARD.md](data/diverserna/DATASET_CARD.md); the pretraining corpus
is **not** cleared for redistribution.
