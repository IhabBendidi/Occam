# Train a virtual-cell model on your own data

[Back to README](../README.md) · [Representation training](training.md)

A virtual-cell model predicts gene expression after a perturbation, given an unperturbed
control cell and the target gene or genes. Training needs measured control and perturbed
cells. You can start from the released encoder–decoder or one you trained yourself.

## 1. Get the starting model and knowledge sources

```bash
python scripts/download_release.py encoder-decoder
python scripts/download_release.py knowledge
```

Skip the first command if you already have your own encoder–decoder. Install the `graph`
extra (`pip install -e ".[graph]"`) for the graph and Parquet readers.

## 2. Prepare an AnnData file

Use one `.h5ad` file, or several files with the same metadata columns:

| Location | Contents |
|---|---|
| `.X` | Non-negative raw expression counts, dense or sparse CSR |
| `.var["gene_id"]` | Unique expression-gene identifiers matching the encoder vocabulary; Ensembl IDs for the released model |
| `.obs["target_gene"]` | Perturbed gene **symbols**, such as `FOXA1`; use `CBL+CNN1` for a double perturbation |
| `.obs["is_control"]` | Boolean control indicator |
| `.obs["cell_line"]` | Biological context for each cell; controls must be available in each context |
| `.obs["batch"]` | Experimental batch identifier |

Label controls `non-targeting` in `target_gene`. Keep expression-gene identifiers and
perturbation symbols distinct: the encoder aligns expression by gene ID, while the knowledge
sources identify perturbations by gene symbol. Unrecognized perturbation symbols cannot
be trained, so check the retained perturbations printed by the loader.

Training samples an unperturbed control from the same context as the perturbed target.
The cells do not need to be paired experimentally. Normalization and `log1p` are applied
internally; do not apply them twice.

## 3. Configure and train

```bash
cp configs/virtual_cell.yaml configs/my_virtual_cell.yaml
```

Edit these fields in the copy:

| Field | Set it to |
|---|---|
| `name` | A unique run name |
| `data.train` and `stages.vcell.data.paths` | Your expression file; `paths` accepts a list of files |
| `stages.vcell.init_from` | `checkpoints/release/occam_small_encoder_decoder.pth`, or your `best_decoder.pth` |
| `stages.vcell.data.gene_col`, `pert_col`, `batch_col` | Your metadata column names |
| `stages.vcell.optim.batch_size` | A batch size that fits your GPU |

```bash
python -m occam.train --config configs/my_virtual_cell.yaml --dry-run
python -m occam.train --config configs/my_virtual_cell.yaml
```

The template freezes the encoder and trains the perturbation MLP, FiLM module, and decoder.
It combines GO, STRING, GenePT, and ESM2 inputs and uses expression reconstruction plus a
cosine loss on perturbation deltas. Tune learning rate, batch size, and
`stages.vcell.loss.delta_cos_weight` on validation data; the template is a starting point
for a new dataset, not a dataset-independent optimum.

The template supports single perturbations and `GENE_A+GENE_B` double perturbations.
It is not an automatic reproduction of the paper's few-shot or double-perturbation splits.

## 4. Choose the generalization setting

By default, the template splits **perturbation labels**, not individual cells: approximately
75% training, 5% validation, and 20% testing. All cells with the same perturbation label
remain together. Settings live under `stages.vcell.split`; the selected labels are saved
in `runs/<name>/vcell/splits.json`.

- To hold out a whole cell line, set `stages.vcell.split.holdout_cell_line` to its exact
  `.obs["cell_line"]` value. It becomes a separate test context.
- To use your own partition, set `stages.vcell.split.json` to a JSON file with `train`,
  `val`, `test_unseen_pert`, `test_unseen_cell`, `cell_lines_train`, and `cell_lines_test`.
  The perturbation fields contain lists of labels; the last two contain lists of contexts.
  Keep training, validation, and test labels disjoint within each evaluation context.

Checkpoints are selected by validation Pearson delta, measuring predicted effects relative
to controls. Use the test partition only for the final evaluation.
The paper's frozen benchmark splits are described in the [benchmarking guide](benchmarking.md).

## 5. Predict with your trained model

```python
import anndata as ad
from occam.api import load_virtual_cell

model = load_virtual_cell("runs/my_virtual_cell/vcell/checkpoints/best_model.pth")
controls = ad.read_h5ad("my_controls.h5ad")
predictions = model.predict(controls, ["FOXA1"], gene_col="gene_id")
predictions.write_h5ad("predictions.h5ad")
```

For a double perturbation, pass a label such as `CBL+CNN1`. Both gene symbols must be in
`model.perturbations`. See the [API reference](api.md) for reconstruction and delta outputs.
