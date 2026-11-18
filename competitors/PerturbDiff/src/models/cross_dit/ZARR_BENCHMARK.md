# `src/models/cross_dit/` — the backbone

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Role |
|---|---|
| `cross_dit_main.py` | `Cross_DiT` module: embedders, conditioning, blocks, heads |
| `cross_dit_blocks.py` | `MM_DiTBlock` / `Cross_DiTBlock` (uses `timm` `Attention`/`Mlp`) |
| `cross_dit_component.py` | `TimestepEmbedder`, `GeneEmbedding`, `ClassEmbedding`, final layers |
| `cross_dit_init.py` | weight initialization |

**Zarr impact: none.** This folder is pure tensor math. It is documented here because two of its
config-driven behaviours interact with how the zarr datasets are *named* and *shaped*.

---

## Shapes

`_setup_dims` (`cross_dit_main.py:72-80`):

```python
self.hidden_size = model_cfg.hidden_num[1]     # 512
self.input_size  = model_cfg.hidden_num[0]     # 2000  (gene dim)
self.output_size = self.input_size
self.input_size *= 2                           # x is concatenated with self-conditioning
```

So `model.hidden_num[0]` must equal `model.input_dim` must equal `data.pad_length` must equal
`data.embed_shape` (asserted in `../../data/dataset/dataset_core.py:44`) — **2000** for this
benchmark, which matches the zarr panel width (`X.zarr` is `(n_cells, 2000)`) and the 2000-line
Ensembl gene list.

Input tensors are `[B, S, G]` with `S = data.use_cell_set` — the cell-set axis produced by
`collate_fn` (`../../data/dataset/dataset_core.py:90-99`). This is the direct analogue of STATE's
`cell_sentence_len` / `model.kwargs.cell_set_len=64`.

## The dataset-name coupling

`_build_x_embedders` (85-100):

```python
if model_cfg.separate_embedder:
    for ds_name in model_cfg.dataset_dict:
        if model_cfg.separate_embedder == "by_name":
            ds_name = get_short_dsname(ds_name)
        x_embedder[ds_name] = nn.Linear(self.input_size, self.hidden_size)
```

and `forward` (228-247) looks up `self.x_embedder[get_short_dsname(self_condition["ds_name"][0])]`.

`get_short_dsname` (`../../common/utils.py:34-45`) maps on substrings — `tahoe`, `pbmc`,
`replogle`, `cellxgene` — and `assert 0` on anything else.

**Consequence for the zarr manifest:** dataset keys must contain one of those substrings.
`ihabxcell_replogle_nadig_hepg2` → `"replogle"` ✅. A key named `hepg2` would crash at model
construction. Also note that all four cell lines collapse to the single bucket `"replogle"`, so
`separate_embedder=by_name` gives them **one shared** embedder — which is probably what you want
for a cross-cell-line benchmark, but it is worth stating explicitly. If you want per-cell-line
embedders, extend `get_short_dsname` rather than renaming the datasets.

`forward` also assumes **one dataset per batch** (it takes `ds_name[0]`), which
`collate_fn`'s assertion at `dataset_core.py:102` guarantees because cell sets are built within a
single `(pert, cell_type, batch)` group.

## Conditioning path

```
t            -> TimestepEmbedder                          -> [B, hidden]
cov_emb      -> ClassEmbedding(hidden_num[0], hidden)     -> [B, hidden]     (from CovEncoder)
gene_emb     -> GeneEmbedding(dim, gene_embedding_type)   -> per-gene tokens (only if use_gene_embedding)
c = t + class_emb   (optionally through SiLU when model.use_class_silu)
x, x_control -> blocks -> final_layer / control_final_layer -> ReLU
```

`use_gene_embedding: False` in `configs/model/base.yaml`, so `gene_emb` is `None` for the default
config and the symbol-keyed gene-embedding pickles are not loaded
(`../lightning/lightning_module.py:54-63`). Keep it off for the first zarr run — that removes
another gene-symbol dependency.

The output activation is ReLU (`cross_dit_main.py:281-282`), i.e. predictions are non-negative,
which is consistent with `log1p`-scale expression and with the cell-eval `[0, 14]` clip applied on
the STATE side.
