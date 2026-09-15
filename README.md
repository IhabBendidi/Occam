# Occam

A masked autoencoder for single-cell transcriptomes. It uses **all genes** in
the input rather than a highly-variable subset, corrupts them with **swap
noise** rather than zeros, and jointly predicts which genes were corrupted and
reconstructs the expression. A separate high-capacity decoder is trained on the
frozen encoder.

Two things live here:

1. **One end-to-end pipeline** — pretrain → finetune → decoder, driven by a
   single config file, in one command.
2. **One single-cell perturbation evaluation** — single cell in, single cell
   out, scored with cell-eval v1, with the control cells the model sees strictly
   disjoint from the control cells it is scored against.

```bash
pip install -e ".[eval]"
cp configs/paths.example.yaml configs/paths.yaml   # relative defaults; no edit needed
./scripts/bootstrap_assets.sh                      # copies checkpoints + BMDB
python scripts/verify_checkpoint.py                # 25 checks on the release
```

Evaluation data goes in `data/eval/` — 6 datasets, 5.6 GB, repacked from 1,031 GB
of source files. The shipped config uses **relative** paths resolved against the
repo root, so downloading the bundle into place is the whole of the setup. See
[data/eval/DATASET_CARD.md](data/eval/DATASET_CARD.md).

## Contents

| | |
|---|---|
| [Get the weights and the data](#get-the-weights-and-the-data) | one private HuggingFace repo |
| [Quickstart](#quickstart) | embed and reconstruct with the released weights |
| [The published recipe](#the-published-recipe) | what produced the released model |
| [Train it yourself](#train-it-yourself) | the three-stage pipeline |
| [Evaluation](#evaluation) | the panel, and how to read its numbers |
| [Gotchas](#gotchas-that-will-bite-you) | read this before changing anything |
| [What is not here](#what-is-not-here) | and where it went |

## Get the weights and the data

Everything is in **one private HuggingFace repo**, `bendidiihab/Occam`. Every
path in this repo is relative, so downloading the folders to the destinations
below is the entire setup — there is nothing to configure afterwards.

```bash
hf download bendidiihab/Occam --include 'checkpoints/*' --local-dir /tmp/occam_dl
mv /tmp/occam_dl/checkpoints/occam_small_encoder_decoder.pth checkpoints/release/

hf download bendidiihab/Occam --include 'eval/*'       --local-dir /tmp/occam_dl
mv /tmp/occam_dl/eval/pert/* data/eval/pert/ && mv /tmp/occam_dl/eval/scib/* data/eval/scib/

hf download bendidiihab/Occam --include 'diverserna/*' --local-dir /tmp/occam_dl
mv /tmp/occam_dl/diverserna/* data/diverserna/
```

| what | size | needed for |
|---|---|---|
| `checkpoints/` | 0.93 GB | everything |
| `eval/` | 5.6 GB | `occam-benchmark`, `occam-scib` |
| `diverserna/` | ~8.8 GB | retraining from scratch only |

**The repo is private, and must stay private.** Neither dataset has had its
licensing resolved: the pretraining corpus has 7 of 9 constituent blocks
unattributed and the two identified (GTEx, TCGA) carry redistribution
conditions, and none of the six evaluation datasets has a recorded licence.
`scripts/check_upload_gate.py` gates *public* release and currently refuses,
which is the correct state. `scripts/publish_hf.py` refuses to upload to a repo
that is not private.

See [docs/hf_release.md](docs/hf_release.md).

## Quickstart

```python
import torch
from occam.inference.loader import load_encoder_wrapper

w = load_encoder_wrapper("checkpoints/release/occam_small_encoder_decoder.pth")
# x: (n_cells, 44728), library-normalised to 2000 then log1p,
#    columns in w._gene_names order
with torch.no_grad():
    embeddings = w.model.feature(x)      # (n_cells, 512)
```

From the shell:

```bash
occam-embed       --model_dir checkpoints/release --input_data cells.h5ad --output emb
occam-reconstruct --model_dir checkpoints/release --input_data cells.h5ad --output recon.h5ad
occam-benchmark   --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-scib        --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-speed       --checkpoint checkpoints/release/occam_small_encoder_decoder.pth
occam-verify
occam-celleval    --checkpoint checkpoints/release/occam_small_encoder_decoder.pth \
                  --eval-data data/eval/pert/nadig_2024_jurkat.h5ad \
                  --output-dir runs/celleval/released --pert-col gene_name \
                  --calibrate quantile --output-activation rectified_sigmoid
```

## The published recipe

`configs/occam_small.yaml` **is** the recipe. It was generated from the released
checkpoints' own embedded configs, not transcribed from run logs, because two of
the three stages were misdocumented in the tree this repo replaces.

| stage | loss | lr | epochs | result |
|---|---|---|---|---|
| 1 pretrain | `nb_softplus`, MLW ramp 0.7 → 0.9 | 4e-4 | 100 | val 0.8786 |
| 2 finetune | `poisson`, MLW ramp 0.7 → 0.0, gradual unmasking | 6e-4 | 100 | val 0.05102 |
| 3 decoder | `smooth_l1` β 2.0, frozen encoder, 2048×5 | **3e-6** | 200 | val 0.0011786 @ ep 185 |

`small` = latent 512, encoder `44728 → 2048 → 1024 → 512`, 209,355,632 params in
the full MAE. The released unified encoder+decoder is 231,673,723.

⚠️ Two of these numbers differ from what was previously published. Two run
directories were written by colliding SLURM array tasks, so the documented
recipe does not produce the released weights.
**[docs/reproducing_chain_a.md](docs/reproducing_chain_a.md) has the evidence.**
It is worth reading before you trust any number about this model.

## Train it yourself

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

### A better decoder, prepared but not run

`configs/decoder_softplus_nb.yaml` retrains stage 3 with a softplus head and a
negative-binomial loss. On the decoder-only panels that moves nadig
`pearson_delta` from 0.2285 to 0.6009 and cuts predicted significant genes from
3,361 to 1,572 against a ground truth of 262.

Nothing in this repo submits it. `sbatch scripts/slurm/retrain_decoder.sbatch`
when you want it. Read the config header first — **calibration, not the loss, is
the dominant term** (~0.37 pearson_delta against ~0.02), so an uncalibrated run
of this arm is no better than the released decoder.

## Evaluation

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

## Gotchas that will bite you

1. **`decoder_num_blocks` must be 5** for the released decoder. The code default
   is 4 and the state dict will not load.
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
6. **Output activation is not recorded in the released file.** A softplus
   decoder loaded under the `rectified_sigmoid` default loads cleanly and emits
   ~99% exact zeros. Checkpoints written today record it.
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

## Layout

```
configs/     occam_small.yaml is the published recipe; paths.yaml is machine-local
src/occam/   mae/ decoder_train/ vcell/ evaluation/ inference/ config/ pipeline/ cli/
scripts/     bootstrap, verification, panel building, dataset export, porting tools
data/        diverserna/ (dataset card + export) and panels/
golden/      numerical fixtures; the release must reproduce them
docs/        architecture, training, evaluation, checkpoints, provenance, hf_release
```

## Citation

Pending. See [MODEL_CARD.md](MODEL_CARD.md) and
[data/diverserna/DATASET_CARD.md](data/diverserna/DATASET_CARD.md); the
pretraining corpus is **not** cleared for redistribution.
