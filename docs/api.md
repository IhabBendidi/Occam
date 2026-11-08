# `occam.api` — the stable public surface

## What this is and why it is separate

Everything else in `occam` is training and benchmarking machinery whose signatures move
with the campaign. **`occam.api` is the surface for someone who just wants to use a
checkpoint**, and it is deliberately small: three loaders, three objects, one helper.

It is a façade — it adds no modelling. It wraps `occam.inference.loader`,
`occam.decoder.external` and `occam.vcell.pipeline.VirtualCellTrainer`, and it exists to
take care of the two things that are easy to get **silently** wrong.

| the footgun | what the API does |
|---|---|
| **Gene alignment.** Your file's column order is not the model's. | The checkpoint's ordered `gene_names` is the truth. Unknown genes are dropped, missing ones zero-filled, and `prepare` **refuses** above 99% zero-fill — because a wrong gene column gives a near-empty input that predicts happily. |
| **Preprocessing.** A raw-count encoder fed log1p data fails silently. | Library-normalise to the checkpoint's own `target_sum`, then log1p, honouring the `skip_normalize` / `skip_log1p` flags it records. |

Pinned by `tests/test_api.py` (13 tests). The surface names are asserted, so renaming one
breaks the suite rather than a downstream user.

## The objects

```python
from occam.api import load_encoder, load_decoder, load_virtual_cell
```

| loader | returns | for |
|---|---|---|
| `load_encoder(path)` | `OccamEncoder` | cells → 512-d embeddings |
| `load_decoder(path, with_encoder=True)` | `OccamDecoder` | embeddings → log1p expression, and the full round trip |
| `load_virtual_cell(path)` | `OccamVirtualCell` | control cell + perturbation → predicted profile |

All three accept a **file or a directory** (a directory is searched for
`best_model.pth`, `occam_small_encoder_decoder.pth`, `final_model.pth`,
`best_decoder.pth`, in that order), and all three fall back to CPU with a printed notice
rather than dying on a machine with no GPU.

### `OccamEncoder`

```python
enc = load_encoder("checkpoints/release/occam_small_encoder_decoder.pth")
z = enc.encode(adata, gene_col="gene_id", verbose=True)     # (n_cells, 512)
enc.gene_names        # the ordered vocabulary
enc.target_sum        # 2000.0
```

### `OccamDecoder`

```python
dec = load_decoder("checkpoints/release/occam_small_encoder_decoder.pth")
x = dec.decode(z)                                  # (n_cells, n_genes) LOG1P
x = dec.reconstruct(adata, gene_col="gene_id")     # encode then decode
c = dec.reconstruct(adata, gene_col="gene_id", counts=True)   # inverted to counts
```

⚠️ The decoder's **own** recorded vocabulary and `target_sum` win over the encoder's. A
unified checkpoint can carry a decoder trained on a narrower axis, and feeding it the
encoder's axis would be wrong.

### `OccamVirtualCell`

```python
vc = load_virtual_cell("checkpoints/vcell/norman_combo_NF32ALLA/best_model.pth")
vc.perturbations          # 18,216 labels it knows
vc.compose_modes          # the 12 composition modes
vc.resolve("CBL+CNN1")    # [id_CBL, id_CNN1], or None

pred   = vc.predict(controls, ["FOXA1", "CBL+CNN1"], gene_col="gene_id")
effect = vc.delta(controls, ["CBL+CNN1"], gene_col="gene_id")
```

`predict` returns an AnnData with one row per (control cell × perturbation);
`obs['target_gene']` is the label and `uns['occam']` records the mode, the space and the
checkpoint's epoch. `delta` returns `(n_perturbations, n_genes)`, averaged over the
control cells.

**Any vcell checkpoint works** — the two published ones and anything from
`python -m occam.train --config <cfg> --stages vcell`. Nothing is specialised to ours.

#### `mode=` — only matters for pairs

`"A+B"` is a pair. `mode` (default **`decode_seq_mean`**) chooses how the two are
composed; it is a **no-op on a single perturbation**, where every reconstruction-based
mode is algebraically identical. That property is what licenses training on singles and
evaluating under every mode without retraining
(`tests/test_composition_modes.py`). The full 72-cell comparison is in
[norman_results.md](norman_results.md); the default's justification is in
[default_recipe.md](default_recipe.md).

#### `baseline=` — it changes the number materially

`delta(baseline="reconstructed")` (default) subtracts the model's own unperturbed
prediction, so the decoder's reconstruction error cancels. `baseline="real"` subtracts the
real preprocessed controls, so the delta is relative to the measured baseline.

⚠️ On `rhj_fewshot`, finding F-recon1 measured the decoder's own control reconstruction
error at **~5× the effect size** — which is exactly the quantity this choice includes or
removes. The two are different conventions (`recon` vs `injctrl`), not two estimates of
one thing: read each against its own family, never mixed.

## `GeneSpace` — the piece worth understanding

```python
from occam.api import GeneSpace
gs = GeneSpace.from_checkpoint(ckpt)            # or GeneSpace(genes, target_sum, ...)
X, library_sizes, report = gs.prepare(adata, gene_col="gene_id")
print(report.describe())
```

A prediction is only meaningful if the input sits in the space the model was trained in,
and that space is defined by the **checkpoint**, never by the file you are holding.
`prepare` returns the aligned, preprocessed matrix, the library sizes, and an
`AlignmentReport` (`n_matched`, `n_input_dropped`, `n_zero_filled`,
`zero_fill_fraction`, …).

⚠️ The library size is computed over the **matched** columns and **before** the scatter.
Normalising afterwards would divide by a sum taken over zero-filled columns — a silent
factor on every value.

**The exact pipeline, in order** (measured 2026-09-23 on the released checkpoint):

1. Read `.X` as **raw counts**. Pick the gene column: the vocabulary is 44,728 ids, of
   which 44,560 are Ensembl `ENSG…` (the checkpoint calls the column `gene_symbol`, but
   it holds Ensembl ids).
2. Keep the input genes that are in the vocabulary. Drop the rest.
3. Library size = the row sum over the **kept** genes.
4. `x / library × 2000` (the checkpoint's `target_sum`, not 1e4), then `log1p`.
5. Put the values into the vocabulary's column order. Vocabulary genes missing from the
   input are **0**.
6. `encoder(x)` → the latent: 512-d for `small` (`|z|` about 55), 768-d for `base`
   (`|z|` about 35). No dropout at eval, and no normalisation of the output. The last
   layer is LayerNorm → Mish, so values are ≥ −0.309.

The benchmark suite (`benchmark_runner`, compact mode) and the vcell data path instead
take the library size over **all** file genes (step 3 before step 2). On the eval files
the kept genes hold ≥ 99.8% of counts, so the two agree: embedding cosine is 1.000000 on
`pbmc_12k` and `nadig_2024_hepg2`, and ≥ 0.99998 on `pancreas_scib`, where the kept genes
hold 97%.

⚠️ A bare array passed to `encode()` / `prepare()` is still normalised and log1p'd. It
must be **raw counts** already in vocabulary order. Passing preprocessed values
transforms them twice.

Silent errors, measured as the cosine of each cell's embedding against the correct one
on `pbmc_12k`. Both look healthy, which is why they are dangerous:

| mistake | median cosine | min cosine |
|---|---|---|
| input already log1p-normalised | 0.9883 | 0.9707 |
| `target_sum` 1e4 instead of 2000 | 0.9955 | 0.9838 |

Cosine near 0.99 is **not** "nearly right" here: the embeddings share a large common
component, so cosine is compressed near 1.

Escape hatches, both deliberate:

- `max_zero_fill=0` allows an arbitrarily narrow panel. A 2,000-HVG file against a
  44,728-gene vocabulary is legitimately 95.5% zero-filled; a wrong gene column is ~100%,
  which is why the default threshold is 0.99 rather than something tidier.
- `symbol_map="path/to/map.csv"` (columns `gene_symbol`, `ensembl_gene_id`) lets a
  symbol-keyed file reach an Ensembl vocabulary.

## `to_counts`

```python
from occam.api import to_counts
counts = to_counts(x_log1p, library_sizes)
```

⚠️ Lossy, and **not** the inverse of a round trip: the model predicts in log1p space, so
this recovers counts *consistent with the prediction*, not the counts you started from.

## What it refuses, and why

Each of these is a failure that is otherwise silent:

| situation | what happens |
|---|---|
| no input gene is in the vocabulary | `ValueError` naming `gene_col` and the symbol map |
| > `max_zero_fill` of the vocabulary zero-filled | `ValueError` with the likely cause |
| two input genes map to one vocabulary column | `ValueError` — the scatter would be ambiguous |
| a duplicated vocabulary in the checkpoint | `ValueError` at construction |
| a bare array whose width is not the vocabulary | `ValueError` — pass an AnnData to be aligned |
| an unresolvable perturbation label | `KeyError` naming the unresolved **components** of `"A+B"` |
| a plain MAE checkpoint passed to `load_decoder` | `ValueError` — it has no external decoder |
| a non-vcell checkpoint passed to `load_virtual_cell` | `ValueError` naming the flag it lacks |
| a checkpoint that predates `output_activation` | a printed warning; pass it explicitly. A softplus decoder rebuilt as `rectified_sigmoid` is ~99% exact zeros **and still loads cleanly** |

## Output space

`decode`, `reconstruct` and `predict` return **log1p** values at the checkpoint's
`target_sum` — the space the models were trained in and the space every reported metric is
computed in. Use `counts=True` or `to_counts()` when you need counts.
