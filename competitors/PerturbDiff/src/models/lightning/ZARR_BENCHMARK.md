# `src/models/lightning/` — the LightningModule

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Role |
|---|---|
| `lightning_module.py` | `PlModel`: training/validation steps, loss assembly, optimizers |
| `lightning_factories.py` | `model_init_fn`, `create_diffusion`, `get_optimizer`, `create_named_schedule_sampler`, `EMAWeightAveraging` |

`PlModel` is the hydra target in `configs/lightning/base.yaml`
(`_target_: src.models.lightning.lightning_module.PlModel`) and the class loaded at sampling time
(`../../apps/sampling/sampling_setup.py:92`).

---

## `PlModel.__init__` (29-80)

Builds `CovEncoder`, optionally loads gene-name embedding pickles (54-63, only when
`model_cfg.use_gene_embedding` — **off by default**), builds the backbone via `model_init_fn`,
the diffusion process, the timestep sampler, and the geomloss energy loss. Ends with
`self.save_hyperparameters()` (line 80) — which is why `load_from_checkpoint` can restore
`cov_encoding_cfg` / `model_cfg` / `optimizer_cfg` from the checkpoint itself
(`sampling_setup.py:81-96`).

## The batch contract

`_compute_loss` (200-274) consumes exactly the keys `collate_fn` produces:

```
batch["pert_emb"]        [B, S, G]  perturbed expression      (the diffusion target)
batch["cont_emb"]        [B, S, G]  paired control expression (conditioning)
batch["cov_pert"]        [B, S]     integer codes
batch["cov_celltype"]    [B, S]
batch["cov_batch"]       [B, S]
batch["col_genes"]       list[list[str]]  gene names per set  (only used for gene embeddings)
batch["ds_name"]         list[str]
batch["is_padded_list"]  [B, S]
```

Any zarr data path must emit this dict unchanged. `_encode_covariates` (194-198) just calls
`CovEncoder(cov_pert, cov_celltype, cov_batch)`.

## The `ds_name` coupling (lines 247-255)

```python
keys = set(get_short_dsname(x) for x in self.model.model_cfg.dataset_dict)
name_arr = np.array([get_short_dsname(x) for x in cond["ds_name"]])
for ds in keys:
    return_dict[f"dataset_loss_mse1_{ds}"] = ...
```

Per-dataset loss logging goes through `get_short_dsname` again — so the same naming constraint as
in [../cross_dit/ZARR_BENCHMARK.md](../cross_dit/ZARR_BENCHMARK.md) applies: every zarr dataset
name must contain `tahoe`/`pbmc`/`replogle`/`cellxgene`, and all four ihabxcell cell lines will
report under one `replogle` key.

## Validation / checkpointing

`validation_step` (138-167) logs `validation_{split}_loss` per dataloader index, where `split =
all_split_names[dataloader_idx]` = `["validation", "test"]`. `on_validation_epoch_end` (169-184)
emits `validation_{split}_loss_epoch`.

`configs/lightning/base.yaml` monitors **`validation_validation_loss_epoch`**. With the benchmark
split's empty validation fold that key never appears, so nothing is ever "best" — see
[../../data/data_module/ZARR_BENCHMARK.md](../../data/data_module/ZARR_BENCHMARK.md) §3.5.

Resolved two ways: `run_pipeline` validates the **final** checkpoint rather than `"best"`
(root plan §7.3), and the benchmark runs pass
`lightning.callbacks.checkpoint.monitor=null lightning.callbacks.checkpoint.save_top_k=1`.
The consequence is that you evaluate the last model, not a score-selected one — which matches how
STATE is invoked. To select by score instead, point the monitor at a metric the run emits
(e.g. `validation_test_loss_epoch`).

## `lightning_factories.py`

* `EMAWeightAveraging` (19-33): subclasses the vendored `WeightAveraging` with
  `get_ema_avg_fn(decay)`, updating every `update_steps`.
* `model_init_fn` (36-50): dispatches on `model_cfg.model_type` → `Cross_DiT`.
* `create_diffusion` (52-77): see [../diffusion/ZARR_BENCHMARK.md](../diffusion/ZARR_BENCHMARK.md).
* `get_optimizer` (79-95): builds the optimizer from `configs/optimization/optimizer/adam.yaml`.
* `create_named_schedule_sampler` (97): `"uniform"` → `UniformSampler`.

`configure_optimizers` (276-303) splits decay/no-decay parameters using
`transformers.trainer_pt_utils.get_parameter_names` and attaches the cosine scheduler
(`configs/optimization/scheduler/cosine.yaml`) at **step** interval.

**Zarr impact: none in this folder** beyond the checkpoint-monitor choice above.
