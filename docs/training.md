# Train an encoder–decoder on your own data

[Back to README](../README.md) · [Virtual-cell training](virtual-cell-training.md)

Use this workflow when you want to train cell representations from scratch. For a new
perturbation predictor initialized from the released encoder–decoder, start with the
[virtual-cell guide](virtual-cell-training.md) instead.

## Prepare your data

Provide an AnnData `.h5ad` file, or a folder of files, with cells as rows and genes as columns:

- `.X`: non-negative raw counts. The training code normalizes each cell to a total of 2,000
  and applies `log1p` internally. Do not supply already log-transformed counts.
- `.var_names`, or a named `.var` column: unique gene identifiers. Use Ensembl IDs if you
  want to combine the model with the released data and checkpoints.
- No cell-type or perturbation labels are required.

The default `backed` loader accepts dense or sparse `.X` and reads expression from disk.
For a large **dense** corpus or a folder of files, `--data-mode optim` uses chunked reads.

## Train with one command

After [installation](../README.md#install):

```bash
python scripts/train_representation.py \
    --data my_cells.h5ad \
    --gene-col gene_id \
    --output runs/my_encoder
```

Omit `--gene-col` to use `.var_names`. The command automatically runs denoising pretraining,
gradual unmasking, and expression-decoder training with the encoder frozen. It uses the
current softplus decoder with negative-binomial reconstruction, rather than the older
decoder in `configs/occam_small.yaml`. The complete defaults are in
[`configs/representation.yaml`](../configs/representation.yaml).

No benchmark downloads or Weights & Biases login are required. Benchmark callbacks are
turned off; the final decoder checkpoint is selected by validation reconstruction loss.
These are defaults for training on your data, not a claim to reproduce the released weights
on a different corpus. The internal unmasking phase retains the released Poisson objective.

Useful options:

```bash
# Inspect the plan without training or writing files.
python scripts/train_representation.py --data my_cells.h5ad \
    --output runs/my_encoder --dry-run

# Use a separate validation set and change training duration and batch size.
python scripts/train_representation.py --data train.h5ad --validation val.h5ad \
    --gene-col gene_id --epochs 100 100 200 --batch-size 256 \
    --output runs/my_encoder
```

Without `--validation`, 10% of the data is reserved for validation (`--val-fraction` changes
this). The default loader splits cells; `optim` splits chunks. Keep held-out evaluation
contexts and datasets out of both training and validation files.

## Train on a folder of AnnData files

You can train across multiple datasets without concatenating them into one AnnData object:

```bash
python scripts/train_representation.py \
    --data my_datasets/ \
    --data-mode optim \
    --gene-col gene_id \
    --output runs/my_encoder
```

The loader finds `.h5ad` files recursively, including subfolders, and reads them in chunks.
Each file must contain raw counts in a **dense `.X` matrix**; sparse `.X` is currently
supported only by the single-file `backed` and `in_memory` loaders.

Use the same gene-identifier convention and column name in every file, or omit `--gene-col`
to use `.var_names` throughout. Files can have different gene sets and column orders.
Training uses the gene list from the file with the most genes, aligns other files by gene
identifier, fills missing genes with zeros, and drops genes outside that list.

By default, validation holds out chunks across the folder. To hold out an entire dataset,
place its file **outside the training folder** and pass it with `--validation val.h5ad`.

## Use the result

```python
import anndata as ad
from occam.api import load_encoder, load_decoder

checkpoint = "runs/my_encoder/decoder/checkpoints/best_decoder.pth"
cells = ad.read_h5ad("new_cells.h5ad")
embeddings = load_encoder(checkpoint).encode(cells, gene_col="gene_id")
reconstructed = load_decoder(checkpoint).reconstruct(cells, gene_col="gene_id")
```

`best_decoder.pth` contains **both the encoder and decoder**. Pass the same path to
`stages.vcell.init_from` when [training a virtual-cell model](virtual-cell-training.md).
The final-epoch checkpoint is saved beside it as `final_decoder.pth`.

## Resume or change settings

Rerun the same command to reuse completed stages. If settings changed, choose a new output
directory, or use `--resume force` to archive the old stages and retrain. This does not
promise to resume an interrupted stage at its last minibatch.

Your full configuration is saved as `runs/my_encoder/training.yaml`. For settings beyond
the command-line options, edit a copy and run:

```bash
python -m occam.train --config path/to/training.yaml
```

[Training internals](training-internals.md) describes the stage handoffs and checkpoint
records. [Encoder fine-tuning](finetuning.md) covers adapting a full MAE checkpoint instead
of training from scratch; that requires the MAE training checkpoint, not the inference
encoder–decoder.

## Small CPU test

```bash
python scripts/make_smoke_corpus.py --out /tmp/occam_smoke.h5ad --cells 128 --genes 32
python scripts/train_representation.py --data /tmp/occam_smoke.h5ad \
    --gene-col gene_symbol --output runs/representation_smoke --smoke
```

`--smoke` runs two epochs per stage with smaller layers on CPU. It checks that the pipeline
works; use the regular settings for a scientific training run.
