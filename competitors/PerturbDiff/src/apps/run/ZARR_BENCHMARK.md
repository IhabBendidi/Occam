# `src/apps/run/` — the two entrypoints

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Entrypoint |
|---|---|
| `rawdata_diffusion_training.py` | `@hydra.main(config_path="../../../configs", config_name="rawdata_diffusion_training")` |
| `rawdata_diffusion_sampling.py` | `@hydra.main(config_path="../../../configs", config_name="rawdata_diffusion_sampling")` |

Both prepend the repo root to `sys.path` (lines 11-12) so `from src.…` resolves regardless of the
working directory — but the hydra `config_path` is **relative to the file**, so you can launch
from anywhere. `common/paths.py` however reads `configs/path/trixie_path.yaml` **relative to the
CWD** (`src/common/paths.py:10-19`), so in practice run from the repo root.

---

## `rawdata_diffusion_training.py` (56 lines)

```
OmegaConf.resolve(cfg)
setup_loggings(cfg)                      # fills cfg.save_dir_path
pl.seed_everything(cfg.optimization.seed)
trainer     = setup_trainer(cfg)
datamodule  = build_datamodule(cfg, logger)      # <- calls datamodule.setup()
populate_covariate_cfg(cfg, datamodule)          # <- covariate cardinalities into cfg
model       = build_model(cfg, logger, datamodule)
model       = maybe_load_and_patch_checkpoint_model(cfg, model, logger)
maybe_reinitialize_from_scratch(cfg, model, logger)
run_pipeline(trainer, model, datamodule, cfg, logger)
```

The ordering is load-bearing: `build_model` needs `cov_encoding.num_pert` etc., which only exist
after the datamodule has enumerated the categories. Any zarr datamodule must therefore populate
`pert_dict` / `cell_type_dict` / `batch_dict` inside `setup()`, not lazily.

`trainer` is built **before** the datamodule, so a data-loading failure surfaces after DDP has
already initialized.

---

## `rawdata_diffusion_sampling.py` (74 lines)

```
datamodule = build_sampling_datamodule(cfg, logger)   # setup() only
populate_covariate_cfg(cfg, datamodule)
model      = load_sampling_model(cfg, logger, datamodule)   # PlModel.load_from_checkpoint
device     = setup_device(cfg, logger)
datamodule.setup_dataset()                            # <- explicit, unlike training
generate_samples(model, model.diffusion, cfg, device, logger, datamodule, pca_for_decode=None)
```

Note `datamodule.setup_dataset()` is called explicitly here (line 58) whereas in training Lightning
calls it via `trainer.fit`. A zarr datamodule must be safe to `setup()` then `setup_dataset()`
without a trainer.

`generate_samples` returns `(truths, samples, trajectories, decoded)` and the entrypoint only logs
summary statistics — the adatas are written inside `generate_samples` via `save_adata`.

---

## Zarr changes here

**None required in these two files.** They are pure orchestration. Everything is driven by
`cfg.data.data_name`, which the two builders dispatch on:

* `src/apps/training/training_datamodule_builder.py:13-39`
* `src/apps/sampling/sampling_setup.py:9-57`

**Done (Phase 5):** cell-eval runs inside `generate_samples` when
`sampling.run_cell_eval=true`, so one command produces both the adatas and
`<CellType>_results.csv`, mirroring `../../../../STATE/state/predict_state_zarr.sh`.
`sampling.cell_eval_profile` maps to cell-eval's `profile`.
