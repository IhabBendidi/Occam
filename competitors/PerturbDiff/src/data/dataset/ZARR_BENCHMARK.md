# `src/data/dataset/` — Dataset, IO and grouping

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md) and
[../ZARR_BENCHMARK.md](../ZARR_BENCHMARK.md).

| File | Role |
|---|---|
| `dataset_core.py` | `H5adSentenceDataset` + `collate_fn` + covariate lookup + control pairing |
| `dataset_io.py` | the only place that reads expression values off disk |
| `dataset_grouping.py` | control split, cell-set grouping (+ content-hashed cache), gene selection |

> **Implemented.** `dataset_io_zarr.py` exists, `H5adSentenceDataset` dispatches on
> `data.source_format`, and `dataset_grouping.py` gained `data.cell_set_group_by` plus a
> content-hashed, atomically-written cache. Sections marked "Built" describe what shipped; the
> analysis around them is kept for rationale.

---

## 1. `dataset_core.py`

### `H5adSentenceDataset` (253-375)

A *virtual* dataset over many files. Indexing is done by the **sampler**, which yields
`(ds_name, local_idx, is_padded)` triplets — `__getitem__` receives that tuple, not an int
(`getitem_impl`, 196-250). `__len__` is the summed cell count (315-316).

Construction order (286-324) matters:

1. `H5Store(dataset_path_map, max_open=data_args.max_open_files)` — line 306
2. `split_out_control()` — separates control vs perturbed indices
3. `register_mapping_indices()` — builds the pools the sampler and pairing use
4. `_get_selected_gene_vars()` — resolves gene names / masks per dataset

### `getitem_impl` (196-250)

Per item: fetch the perturbed row, pick a paired control row via `_mapping_cells`, fetch that too,
attach the gene list and the three covariate codes. Returns a 9-tuple.

### `mapping_cells` (121-159)

`assert dataset.data_args.mapping_strategy == "random"`. Picks a control uniformly from the
`(cell_type_code, batch_code)` pool; if that exact pool is empty it relaxes **cell type first**,
then batch (135-149). This is PerturbDiff's analogue of STATE's
`basal_mapping_strategy` (`zarr_perturbation.py:471-481`) — STATE's default `"random"` pools by
cell type only, `"batch"` pools by `(batch, cell_type)` with a cell-type fallback. **PerturbDiff's
default is effectively STATE's `"batch"` mode**; note this when comparing results.

### `get_covarates` (162-193)

Maps local categorical codes to the **global** dicts:

* pert: `meta_cache.pert_dict[cache.pert_categories[code]]`, falling back to `-1` on any exception
  (166-171) — a silent failure mode worth logging during bring-up
* cell type: hard `assert 0` on failure (178-182)
* batch: **`ds_name + "_" + cache.batch_categories[code]`** (186) — the global namespacing that
  `sampling_generation_helpers.py:111` and `data_module_setup.py:140` must agree with

### `collate_fn` (36-118)

* asserts `pad_length == embed_shape` (44) — both must equal 2000 for this benchmark
* divides by `normalize_counts` if truthy (80-82)
* reshapes to `[B/S, S, G]` where `S = use_cell_set` (90-99)
* asserts every cell in a set shares one `ds_name` and one gene list (101-103)

Output keys consumed by the model: `pert_emb`, `cont_emb`, `col_genes`, `cov_pert`,
`cov_mapped_pert`, `cov_celltype`, `cov_batch`, `is_padded_list`, `ds_name`.

### Built — the zarr dispatch

Only two methods are format-specific:

```python
def _retrieve_counts(self, h5f, ds_name, local_idx):   # 350-352
def _get_selected_gene_vars(self):                     # 334-336
```

Both now dispatch on `self.source_format` (read from `data_args` in `__init__`, which also picks
`ZarrStore` vs `H5Store`). Both zarr implementations live in `dataset_io_zarr.py` — gene selection
was *not* split into a separate `dataset_grouping_zarr` module, since it is the same concern as the
row read: mapping `X` columns onto the panel.

Leave `getitem_impl`, `collate_fn`, `mapping_cells` and `get_covarates` untouched — they only
need the cache attributes, which `ZarrMetadataCache` provides.

---

## 2. `dataset_io.py` — replace wholesale for zarr

### `retrieve_counts` (122-160)

```python
if embed_key.startswith("X_") and selected_gene_mask[ds].sum() == 2000:  # line 133
    counts = row_from_obsm(...)      # obsm/X_hvg
elif embed_key == "X":
    counts = row_from_X(...)         # /X CSR or dense
else:
    raise NotImplementedError
```

then masks to the dataset's shared genes and **reorders columns into `selected_genes` order**,
zero-filling missing genes (144-160).

Line 133's `sum() == 2000` coupling is fragile: the branch chosen depends on how many selected
genes happen to be present in the file.

### `row_from_obsm` (10-69) / `row_from_X` (72-119)

Handle dense arrays, AnnData CSR groups (`data`/`indices`/`indptr`), and `read_direct` fast paths.
None of this survives the move to zarr.

### Built — `dataset_io_zarr.py`

```python
def retrieve_counts_zarr(dataset, handle, ds_name, local_idx) -> torch.Tensor:
    x = handle[local_idx]                                   # ram/npy/zarr row
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x[dataset.gene_index_map[ds_name]]                  # var.parquet -> selected order
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)   # keep the [1, G] shape
```

* Return shape must stay `[1, G]` — `collate_fn` does `batch_counts.append(counts[0])` (74-75).
* Build `gene_index_map` **once** at setup from `var.parquet.ensembl_gene_id` (mirrors
  `_Source.__init__`, `zarr_perturbation.py:260-275`), instead of per-row dict lookups.
* Do **not** apply target_sum/log1p: the hooke-tx `X.zarr` is already `log1p(1e4)`-normalized, so
  the STATE run uses `target_sum=null, log1p=false`. Keep `normalize_counts: false`.
* Raise loudly if a selected gene is missing from `var` (STATE errors at
  `zarr_perturbation.py:266-271`); PerturbDiff's current behavior of silently zero-filling
  (`dataset_io.py:156-158`) would quietly change the benchmark.

---

## 3. `dataset_grouping.py`

### `split_out_control` (12-48)

Finds the control code by matching `control_type[ds_name]` against `cache.pert_categories`
(line 24, `assert len(target_code) == 1`), then partitions `data_indices[ds]` into
`{"control": …, "perturb": …}`. With `keep_control_cell=True`, controls also stay in `perturb`
(32-35). For zarr the control category is `"empty"` — set `control_pert: empty` in the data config
and this works unchanged.

### `register_mapping_indices` (115-207)

* `grouped_data_indices`: control rows grouped by `(cell_type, batch)` → used by `mapping_cells`
* `grouped_pert_data_indices`: perturbed rows grouped by `(pert, cell_type, batch)` → consumed by
  the sampler to build cell sets

The three-key grouping (81-113) is an **O(n_pert × n_celltype × n_batch) triple loop** with a
boolean mask per cell. For hepg2 that's ~2058 perts × 1 cell type × ~60 batches ≈ 1.2×10⁵ mask
passes over 96k rows — slow but survivable; across four cell lines it is not. Results are pickled
to `indices_cache_dir`, so it only pays once.

Measured since: the triple loop takes ~2.6 s for hepg2 and ~10-30 s across the four cell lines —
slower than a `ravel_multi_index` group-by would be, but not the bottleneck, so it was left alone
rather than risk changing grouping results for a few seconds.

The cache filename now carries a **content hash** of the split indices, grouping mode,
`keep_control_cell` and control type, and writes are atomic (temp file + `os.replace`). Stale reuse
is therefore structurally impossible, and concurrent launches cannot read a half-written pickle —
both of which used to be real hazards, since the old `{ds_name}_{stage}` key was invariant to all
of those and `keep_control_cell` changes the *index space* the cached positions refer to.

### `get_selected_gene_vars` (209-272)

Loads the selected-gene **pickle** (222-226) and reads gene names from the h5 file, trying
`/var/_index` → `/var/gene_name_index` → `/var/ensembl_id` → `/var/gene_name` →
`/var/gene_short_name` (233-244). Builds `selected_gene_mask` per dataset by `np.isin`.

### Built — gene selection (lives in `dataset_io_zarr.py`, not a separate module)

```python
def get_selected_gene_vars_zarr(dataset):
    selected = [l.strip() for l in open(cfg.selected_gene_file) if l.strip()]  # .txt of Ensembl ids
    for ds in dataset._names:
        var = pl.read_parquet(Path(dataset.dataset_path_map[ds]) / "var.parquet")
        ens = var["ensembl_gene_id"].to_list()
        pos = {g: i for i, g in enumerate(ens)}
        missing = [g for g in selected if g not in pos]
        if missing: raise ValueError(...)
        dataset.gene_index_map[ds] = np.array([pos[g] for g in selected], np.int64)
        dataset.gene_vars[ds] = np.array(var["gene_symbol"].to_list())[dataset.gene_index_map[ds]]
        dataset.selected_gene_mask[ds] = np.ones(len(selected), bool)
```

Two consumers of `selected_genes` care about the *identifier type*:

* `col_genes` in the batch is used to look up **gene-name embeddings**
  (`lightning_module.py:218`, keyed by symbol) — so populate `gene_vars` with `gene_symbol`.
* `apps/sampling/sampling_generation.py:148, 200-215` uses `col_genes` and the selected-gene list
  to align the output adata's `var.index`.

Keep the selected-gene *file* in Ensembl ids (identical to STATE's `--selected_ensembl_gene_ids_path`)
and translate to symbols through `var.parquet` — that keeps the gene panel provably the same across
the two models while preserving symbol-keyed embeddings.

Also relax `sampling_generation_helpers.load_selected_genes`, which currently `pickle.load`s and
asserts `len == 2000` (48-59); the Ensembl list is a plain `.txt` (2000 lines, verified).
