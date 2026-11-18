# `src/models/` — covariate encoding, EMA, timestep sampling

Companion to [../../ZARR_BENCHMARK.md](../../ZARR_BENCHMARK.md). Subfolders:
[cross_dit/](cross_dit/ZARR_BENCHMARK.md), [diffusion/](diffusion/ZARR_BENCHMARK.md),
[lightning/](lightning/ZARR_BENCHMARK.md).

| File | Role | Zarr impact |
|---|---|---|
| `covariate_encoding.py` | `CovEncoder`: pert + cell-type + batch → conditioning vector | ✅ Ensembl→symbol translation + guards (Phase 4) |
| `resampling.py` | `UniformSampler` timestep sampler | none |
| `weight_averaging_callback.py` | vendored Lightning `WeightAveraging` (EMA base class) | none |
| `architectures.py` | present but defines no top-level symbols | none |

---

## `covariate_encoding.py` — `CovEncoder`

`__init__` (22-34) sums three branch widths and projects the concatenation to
`cov_cfg.output_dim` (= `model.hidden_num[0]`, i.e. 2000).

`forward(pert_input, celltype_input, batch_input)` (271-307) is called from
`PlModel._encode_covariates` with `batch["cov_pert"]`, `batch["cov_celltype"]`,
`batch["cov_batch"]` — plain integer code tensors. **The encoder never sees strings**, so the
data-format change is invisible here *except* where pretrained embedding tables are keyed by name.

### Branch by branch

**Perturbation** (`_init_generic_perturbation_encoder`, 39-78)

* `pert_encoding=onehot` → `nn.Embedding(num_pert + 1, hidden_dim)`; forward does
  `pert_encoder(pert_input + 1)` (line 296), reserving index 0 for "no perturbation".
  ✅ **Works with zarr as-is** — codes are just integers.
* `pert_encoding=non` → forces index 0 (line 292); used for unconditional pretraining.
* `pert_encoding=esm2` → loads `pert_embedding_path`, and hard-codes
  `emb.weight[79] += emb.weight.mean()` as the PBMC control slot (lines 70-73). ❌ PBMC-only.

**Tahoe drug branch** (`_init_tahoe_drug_encoder`, 83-145) — `drug_encoding=chemberta_cls`
`eval()`s the pert name as a Python literal (`eval(drugname_drugconc)[0]`, line 100). ❌ Would
crash on Ensembl-id pert strings. Keep `drug_encoding=onehot`.

**Replogle gene branch** (`_init_replogle_gene_encoder`, 150-205) — `replogle_gene_encoding=genept`
loads `replogle_gene_emb_dict_perturbation_emb_dict.pkl` and looks up **`cov_cfg.pert_dict` keys**,
treating the literal string `"non-targeting"` as the control (lines 173-177). Missing keys fall
back to zero vectors with a warning (178-190).

> ✅ **Solved (Phase 4).** With zarr, pert keys are **Ensembl ids** (`ENSG00000163807`) and the
> control is `"empty"`, so every lookup used to miss → an all-zero table and a silent quality
> collapse (a `logger.warning`, not an error). Now:
> `cov_encoding.pert_key_type=ensembl` translates through
> [`src/common/gene_id_map.py`](../common/ZARR_BENCHMARK.md) and `control_pert_key` names the
> control. Measured against the real genept pickle: **1980/2023 resolved, 2.1% zero-filled**.
> `var.parquet` alone would *not* have worked — it covers the measured genes, which overlap the
> perturbed set by only 20%.
>
> The encoder now refuses to build when more than `max_zero_embedding_frac` (default 50%) of
> perturbations lack an embedding, and warns when `control_pert_key` is absent from `pert_dict`
> (which would leave the control row zeroed instead of mean-filled).

**Cell type** (`_init_celltype_encoder`, 210-242)

* `onehot` → `nn.Embedding(num_celltype, hidden_dim)`. ✅ safe default.
* `llm` → loads `celltype_embedding_path` (`meta_data/new_all_emb.pkl`, 737 keys, dim 3072).
  ✅ Upstream keys are **lowercase** (`hepg2`) while the zarr obs carries `HepG2`, so matching is
  now case-insensitive with the substitution logged; all four benchmark cell lines resolve. A
  genuinely missing cell type raises a message naming it, not a bare `KeyError`.

**Batch** (`_init_batch_encoder`, 247-258) → `nn.Embedding(num_batch, hidden_dim)` when
`batch_encoding=onehot`, else no branch. ✅ works with the `f"{ds}_{batch_center}"` namespaced
vocabulary. Cardinality with all four cell lines is a few hundred — fine.

### Covariate configs

`configs/cov_encoding/upstream_zarr.yaml` defaults to all-one-hot, which needs no external pickle:

```
cov_encoding.pert_encoding=onehot        cov_encoding.celltype_encoding=onehot
cov_encoding.drug_encoding=onehot        cov_encoding.batch_encoding=onehot
cov_encoding.replogle_gene_encoding=onehot
```

`genept` and `llm` both work now (assets under `path.tmp_dir`, fetched by a targeted ~75 MB
download). The benchmark runs use `genept`, which is upstream's own Replogle choice; one-hot is
the closer analogue of STATE's setup (plain one-hot perturbations + `batch_encoder=True`), so it
is worth keeping as an ablation.

---

## `resampling.py` / `weight_averaging_callback.py`

`UniformSampler` is selected by `model.schedule_sampler: "uniform"`
(`lightning/lightning_factories.py:97`). `weight_averaging_callback.py` is a vendored copy of
Lightning's `WeightAveraging` — subclassed as `EMAWeightAveraging` in
`lightning/lightning_factories.py:19`, driven by `lightning.ema.{decay,update_steps}`.

Neither is affected by the data format. Keep `ema.decay=0.99`, `ema.update_steps=10` to match the
README's shared training block (and STATE's own EMA-free setup differs here — worth noting when
comparing).

> Aside: the `from lightning.pytorch.callbacks import WeightAveraging` on line 65 of
> `weight_averaging_callback.py` is inside a **docstring example**, not real code. The module
> imports `pytorch_lightning`, and the `lightning` package is deliberately not a dependency.
