# Evaluation

Two independent things are called "evaluation" here and they answer different
questions:

| | what it measures | where |
|---|---|---|
| **Encoder benchmark** | whether the embedding separates perturbations and integrates batches | `occam-benchmark`, `occam-scib` |
| **Panel evaluation** | whether the model predicts a perturbation's effect in a held-out context | `occam.vcell.panel` |

The second is the one the publication is about, and the one with the traps.

---

## Panel evaluation

Single cell in, single cell out. One predicted cell per real perturbed cell,
each generated from **one real control cell**. Nothing is averaged anywhere
except inside the deliberately zero-signal `ctrlmean` anchor, where averaging is
the definition of the baseline.

Scored with **cell-eval v1** (`--profile full`). The count-space scorer that the
research tree also used is deliberately absent, along with its emission
machinery.

### The panel

```bash
python scripts/build_panel.py \
    --source corpus.h5ad \
    --out-dir data/panels/mine \
    --pert-col target_gene --control-label non-targeting \
    --context-col cell_line --context K562 \
    --cells-per-pert 400 --n-controls 18400 \
    --thin-to-h5ad reference_depth.h5ad
```

Emits:

```
panel_real.h5ad             perturbed cells + the SCORING control pool
panel_controls_model.h5ad   a DISJOINT pool, the only controls the model sees
genes.csv  perts.csv  control_split.json  panel_manifest.json  seen_split.json
```

**Controls are drawn once at 2× and then halved.** Drawing the two pools
separately, or thinning them with different random streams, would make a leaked
cell undetectable. Depth thinning therefore happens once, before the split.

Half the available controls is a hard ceiling; the builder refuses rather than
quietly shrinking a pool.

⚠️ **Never build the model-visible pool as a subset of the scoring panel.** Then
every control-derived arm is scored against cells it was built from and the
result looks good for the wrong reason. `load_panel` asserts disjointness in
three tiers — source provenance, then row hashing, then refuse — and never falls
back to comparing obs indices. (The check this replaces did exactly that, between
files written by different scripts with different index prefixes, so it could not
fail.)

### Reading the numbers

Every run scores two zero-signal arms alongside the model:

| arm | what it emits | carries biology? |
|---|---|---|
| `ctrlmean` | the model-visible control mean, for every perturbation | no |
| `ctrlcell` | the drawn control cell, unchanged | no |

**They are mandatory.** `build_arms` raises on an empty anchor list and
`write_report` raises if a panel has none. That is not defensive coding:

> Across 41 (run, arm) pairs measured in the research tree, the model beat the
> `ctrlmean` anchor on `pearson_delta` **0 times out of 41**, on
> `discrimination_score_l2` 9/41, and on `overlap_at_N` **41/41**.

So on this panel the field's usual headline metric is won by a predictor with no
signal in it, and the metric the model does win is the one the research tree's
own docs warn against — because *there*, on a different panel, the anchor won
`overlap_at_N` instead.

The lesson is not "distrust `overlap_at_N`". It is that **which metric an anchor
wins is a property of the panel**, so establish it per panel from the anchor arms
and never inherit the claim. The generated report states it per run.

Reported metrics, with direction:

| metric | direction | note |
|---|---|---|
| `pearson_delta` | higher better | the convention — and what the anchor wins here |
| `discrimination_score_l2` | higher better | second convention metric |
| `overlap_at_N` | higher better | the one the model wins; do not quietly drop it |
| `de_direction_match` | higher better | direction only; the shrink sweep's control |
| `mse_delta`, `mae_delta` | **lower better** | errors; collapse under shrink |

`clustering_agreement` and `pearson_edistance` are panel-level, reported for
`all` only.

### Panels

`all` is the headline. `seen` and `unseen` split it by whether a perturbation
was in training, and are computed by **re-aggregating the `all` run's
per-perturbation results** rather than re-running cell-eval.

That is sound for cell-eval v1 and was not for the scorer it replaces: that one
scaled each metric against a per-panel baseline and replicate ceiling, so a
sub-panel needed its own reference build. v1 metrics are per perturbation and
then averaged, so a sub-panel result *is* a restriction of the full run. It turns
3 panels × N arms × ~60 min into 1 panel × N arms.

Requesting `seen`/`unseen` without the training perturbation set is an error, not
a silent empty panel.

### Arms

Beyond the anchors:

* `model_nocal` — raw output.
* `model_cal` — after control-fit per-gene quantile calibration, fitted on the
  **model-visible controls only**. On the decoder-only panels this is worth about
  0.37 `pearson_delta`, against about 0.02 for the loss/head choice. An
  uncalibrated comparison of heads measures almost nothing.
* `_aN` suffixes — delta shrink, `pred = ctrl + α(pred − ctrl)` per cell against
  the control that cell was generated from. Worth sweeping rather than picking:
  predictions are directionally right but far too large, so the magnitude
  metrics collapse under shrink while the scale-invariant ones do not, and the
  sweep separates those two failures.

### Cost, and what that forces

~60 min per panel at 16 threads for 255 perturbations × 400 cells. Real-side
differential expression is the dominant term and is identical across arms within
a panel, so it is computed once and reused — keyed on the target file's content
so a rebuilt panel cannot silently reuse a stale one.

That cost is why **checkpoint selection uses validation `pearson_delta`, not the
panel score.** The research tree measured that criterion ranking checkpoints
wrongly on the real panel. So: do not read the selected checkpoint as "best on
the panel". Score the final checkpoint too and compare.

### cell-eval

cell-eval **0.6.0**. Resolution order:

1. `OCCAM_CELLEVAL_CMD` — explicit, wins over everything
2. `celleval_bin` plus `PYENV_VERSION=<celleval_pyenv>` — this cluster's path
3. `celleval_bin` on `PATH`, no environment mutation (`celleval_pyenv: ""`)

The pyenv step is *this machine's* resolution, not a requirement. The runner
checks `--version` once and **raises** on a mismatch, rather than returning
empty metrics ten arms later.

---

## Encoder benchmark

```bash
occam-benchmark --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-scib      --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
```

Six tasks across three perturbation datasets — gene-relationship recall against
five BMDB sources, perturbation consistency, iLISI, KNN and linear separability,
linear invertibility — plus scIB batch integration on pancreas and PBMC.

Datasets come from the evaluation bundle in `data/eval/`, located through
`configs/paths.yaml` with **relative** paths resolved against the repo root —
so the same config works on any machine. BMDB ships inside the package.

The bundle is 6 datasets and 5.6 GB, repacked from 1,031 GB of source files
(HDF5 fragmentation plus dense storage of sparse data). All of `X`, `obs` and
`var` is carried through, Ensembl IDs included; only `obsm` is dropped, which on
the perturbation files was 28 arrays of precomputed baseline embeddings from
other models. `data/eval/manifest.json` records shape, non-zero count, column
lists, checksum and source path per dataset.

⚠️ **No benchmark numbers are quoted for the released encoder.** The previously
published figures belong to a run that claimed the same experiment directory but
whose weights were never used downstream. Re-measure.

## Decoder-stage evaluation

Reconstruction Pearson/Spearman/MAE plus, optionally, cell-eval on the
per-cell reconstruction of each eval file.

⚠️ `combined_score` — which picks `best_decoder.pth` — includes Spearman over
zero-filled padding genes, and that term collapses for any head that does not
emit exact zeros. The collapse tracks the zero-fill fraction almost perfectly:
TREK at 1.4% padding loses 0.69 Spearman; nadig at 80.2% loses 90.46; Pearson and
MAE barely move. **The released decoder's own epoch-185 selection was made on
this criterion.** Set `eval_present_genes_only: true`, as
`configs/decoder_softplus_nb.yaml` does.

### The BMDB benchmark needs its gene IDs translated

The BMDB gene-relationship databases are keyed by gene **symbols** —
`CORUM.txt` starts `SUB1,GTF3C3` — while the released vocabulary is 44,560
**Ensembl IDs**. Without a translation between them the benchmark matches
almost nothing and reports it as a successful run.

Measured on the released decoder, same weights, same code, the only difference
being whether the map is present:

| | sources resolved | valid pairs (`All`) | mean recall 0.05–0.95 |
|---|---|---|---|
| no map | 2 | **1** | 0.0000 |
| with map | 6 | **142,856** | **0.5467** |

Per source, with the map (percentile-rank recall, higher is better):

| source | valid pairs | recall 0.05–0.95 | recall 0.1–0.9 |
|---|---|---|---|
| CORUM | 37,627 | **0.7680** | 0.8251 |
| HuMAP | 58,423 | 0.6317 | 0.7044 |
| StringDB | 57,928 | 0.5959 | 0.6665 |
| Reactome | 18,420 | 0.4055 | 0.5054 |
| SIGNOR | 11,591 | 0.3323 | 0.4348 |

The map ships as `assets/gene_ids/ensembl_to_symbol.tsv` (43,825 of 44,728
entries, 98.0%), built by `scripts/build_gene_id_map.py`, which records its
source and coverage in `PROVENANCE.json`. It is inside the wheel, so this works
on a bare `pip install`.

Two properties worth knowing:

- **The direction is detected from the data**, by counting how many names match
  the map's Ensembl keys versus its symbol values — not by pattern-matching
  `ENSG`. A checkpoint trained on symbols (an HVG panel) is used directly.
- **The translation is confined to this benchmark.** A decoder's output rows are
  indexed by the vocabulary it was trained on, so `to_symbols` returns the kept
  row indices and the weight matrix is subset with the same ones. No training,
  inference or reconstruction path ever sees a rewritten vocabulary.

⚠️ This was broken in the port, not in the research tree: the original read its
mapping from an absolute path on one cluster, and the absolute-path sweep
removed it with a comment claiming the vocabulary should be used as-is. That
comment was wrong for symbol-keyed databases.

### Scoring a decoder checkpoint that already exists

`occam-celleval` is the eval-only path: the same measurement the pipeline makes
at the end of training, for a checkpoint that already exists. It calls
`DecoderTrainer.run_cell_eval_on_file`, so a released checkpoint and a freshly
trained one are comparable without an "is it the same code?" caveat.

```bash
occam-celleval \
    --checkpoint checkpoints/release/occam_small_encoder_decoder.pth \
    --eval-data data/eval/pert/nadig_2024_jurkat.h5ad \
    --output-dir runs/celleval/released_jurkat \
    --pert-col gene_name --calibrate quantile --modes injctrl \
    --output-activation rectified_sigmoid \
    --num-threads 32 --timeout 36000
```

Or on the cluster, where the defaults already score the release:

```bash
sbatch scripts/slurm/celleval_decoder.sbatch
CKPT=runs/.../best_decoder.pth ACT=softplus sbatch scripts/slurm/celleval_decoder.sbatch
```

Four things this command refuses to guess:

| | |
|---|---|
| **geometry** | read from the checkpoint's `decoder_config`, never from a flag, so it cannot disagree with the weights |
| **output activation** | required explicitly when the checkpoint does not record one. A softplus decoder loads `strict=True` under `rectified_sigmoid` and emits ~99% exact zeros — a dead model that looks healthy. `occam-verify` determines it from a forward pass; pass what it reports |
| **the perturbation column** | `--pert-col` has no safe default across datasets, and a wrong one makes cell-eval *skip* the file with a printed line rather than fail. nadig uses `gene_name`, not `gene` |
| **calibration** | off unless asked for, and printed when on |

⚠️ **Per-gene quantile calibration is the dominant term in this measurement** —
0.23 → 0.60 `pearson_delta` on nadig, 0.21 → 0.81 on TREK, against ~0.02 for the
head/loss choice. It is fit on the eval file's *control* cells, so it cannot
manufacture perturbation signal, only remove the decoder's per-gene shift. State
it next to any number it produced; `metrics.json` records it either way.

Two modes, both against the same real target: `injctrl` (control rows replaced
by the real preprocessed controls — the virtual-cell convention and the harder
number) and `reconctrl` (the decoder's own reconstructed controls, inflated
because its bias cancels in the delta: 0.87 vs 0.59 on replogle_test). Each mode
is a full cell-eval run, so `--modes injctrl` is the default.

**Reusing the ground-truth DE.** cell-eval spends most of its time computing
differential expression for the real data, and that depends only on the target
file — not on the decoder, the calibration or the mode. `--de-real-cache DIR`
stores it keyed by the **target file's own sha256**, so every arm scored against
the same target shares one. Measured on jurkat: it removes a **56-minute** step
from a ~2:50 run. Net wall-clock saving can be smaller — a decoder whose output
has no exact-zero floor produces denser predictions, and its own
prediction-side DE runs longer.

Keying on content rather than on a panel name is what makes the reuse safe. A
target built from different data, different transforms or a different gene
subset hashes differently and misses the cache; there is no path by which one
arm is silently scored against another arm's ground truth.

**The checkpoint outranks the flag.** If a checkpoint records an
`output_activation` and `--output-activation` disagrees with it, the command
refuses. That is not redundant with the earlier check: a launcher carrying
`ACT=rectified_sigmoid` as a default, pointed at a softplus checkpoint, would
otherwise score ~99% exact zeros and report it as a result. It happened here,
and the refusal caught it.

A worked comparison of two decoders on jurkat is in
[decoder_retrain_jurkat.md](decoder_retrain_jurkat.md).
