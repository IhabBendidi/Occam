# Download models and data

[Back to README](../README.md)

The [Occam Hugging Face repository](https://huggingface.co/bendidiihab/Occam)
contains the models and benchmark data. Install download support with
`pip install -e ".[hub]"`, then run the commands below from your Occam checkout.

The downloader selects individual files, never the whole repository. `--dry-run` prints
remote and local paths without connecting. `--revision COMMIT` selects a specific Hub revision;
otherwise the command resolves `main` to one commit for the entire download.
Downloads use the Hugging Face cache and are resumable. Files are hard-linked into the
checkout where possible to avoid a second copy; different files already at a destination
are not overwritten.

## Encoder–decoder

Use this model to **embed cells**, **reconstruct expression**, or **initialize a new
virtual-cell predictor**. It has not yet learned a perturbation predictor.

[Download file on Hugging Face](https://huggingface.co/bendidiihab/Occam/blob/main/checkpoints/occam_small_encoder_decoder.pth)

```bash
python scripts/download_release.py encoder-decoder
```

Local checkpoint: `checkpoints/release/occam_small_encoder_decoder.pth`.
Load it with `occam.api.load_encoder` or `occam.api.load_decoder`.
For perturbation training, use it as `stages.vcell.init_from` in your configuration.

## Virtual-cell models

These checkpoints include the encoder, decoder, and trained perturbation predictor.
**You do not need to download the encoder–decoder or knowledge tables separately for inference.**

| Model | Trained for | Hugging Face folder |
|---|---|---|
| `rhj` | Single-gene knockdown prediction, trained on HepG2, Jurkat, and RPE1 | [RHJ checkpoints](https://huggingface.co/bendidiihab/Occam/tree/main/checkpoints/occam_seeds/rhj_fewshot/all_sources) |
| `norman` | Double-perturbation prediction from single-gene activation data in K562 | [Norman checkpoints](https://huggingface.co/bendidiihab/Occam/tree/main/checkpoints/vcell/norman_combo_NF32ALLA) |

```bash
python scripts/download_release.py virtual-cell --benchmark rhj --seed 42
python scripts/download_release.py virtual-cell --benchmark norman --seed 42
```

The two local paths are:

```text
checkpoints/vcell/rhj/seed_42/best_model.pth
checkpoints/vcell/norman/seed_42/best_model.pth
```

Download only the model you need. Change `--seed` to 43, 44, 45, or 46 for another training
run; reproducing the reported mean and SD requires all five. The RHJ release corresponds
to `ABL_N_ALL` in the result records; Norman corresponds to `NF32ALLA`.
[Competitor models](https://huggingface.co/bendidiihab/Occam-competitor-checkpoints)
have their own repository and are not downloaded by these commands.

## Data

[Expression datasets](https://huggingface.co/bendidiihab/Occam/tree/main/eval) ·
[Benchmark files](https://huggingface.co/bendidiihab/Occam/tree/main/benchmarks)

```bash
python scripts/download_release.py data --benchmark rhj
```

Choose one dataset group:

| `--benchmark` | Downloads | Local destination |
|---|---|---|
| `rhj` | HepG2, Jurkat, and RPE1 expression files; RHJ evaluation matrix | `data/eval/pert/`, `data/benchmarks/rhj_fewshot/` |
| `norman` | Norman K562 expression file; double-perturbation evaluation matrix | `data/eval/pert/`, `data/benchmarks/norman_combo/` |
| `representation` | Three perturbation datasets, Pancreas, and PBMC68k | `data/eval/pert/`, `data/eval/scib/` |
| `h1` | H1 expression file and evaluation matrix | `data/benchmarks/h1_seen/` |
| `k562` | Replogle K562 expression file and evaluation matrix | `data/benchmarks/k562_seen/` |
| `k562-h1` | K562 expression file and the panel matching H1's target genes | `data/benchmarks/k562_seen/`, `data/benchmarks/k562_seen_h1set/` |

The small benchmark definitions and splits are already tracked in Git. These downloads
supply the large binary files. The DiverseRNA pretraining corpus is **not hosted in this
Hugging Face repository**; see its source in the [TxFM paper](https://arxiv.org/abs/2605.31562).
You can train representations directly on your own `.h5ad` file instead.

## Knowledge sources for training a virtual-cell model

```bash
python scripts/download_release.py knowledge
```

This downloads the GO and STRING graph files to `data/graphs/` and the GenePT and ESM2
embedding tables, with their checksum sidecars, to `data/assets/`.
They are needed to train the perturbation predictor, not to use a trained checkpoint.
See [virtual-cell training](virtual-cell-training.md) for the complete workflow.
