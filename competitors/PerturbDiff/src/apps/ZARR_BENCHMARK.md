# `src/apps/` — entrypoints and orchestration

Companion to [../../ZARR_BENCHMARK.md](../../ZARR_BENCHMARK.md).

```
apps/
├── run/        two hydra @main entrypoints        -> run/ZARR_BENCHMARK.md
├── training/   datamodule/model builders, trainer -> training/ZARR_BENCHMARK.md
└── sampling/   generation + output writing        -> sampling/ZARR_BENCHMARK.md
```

## The two paths, side by side

| | Training | Sampling / eval |
|---|---|---|
| entrypoint | `run/rawdata_diffusion_training.py` | `run/rawdata_diffusion_sampling.py` |
| hydra config | `configs/rawdata_diffusion_training.yaml` | `configs/rawdata_diffusion_sampling.yaml` |
| datamodule build | `training/training_datamodule_builder.build_datamodule` | `sampling/sampling_setup.build_sampling_datamodule` |
| model | `training/training_model_builder.build_model` (fresh or patched ckpt) | `sampling/sampling_setup.load_sampling_model` (`PlModel.load_from_checkpoint`) |
| loop | `training/training_pipeline.run_pipeline` → `trainer.fit` | `sampling/sampling_generation.generate_samples` |
| output | Lightning checkpoints under `path.diffusion.save_dir/run_name` | `diffusion_predict_*.h5ad` / `diffusion_true_*.h5ad` |

Both call an identically-named `populate_covariate_cfg` that copies the datamodule's covariate
dictionaries into the config before the model is constructed
(`training_datamodule_builder.py:41-50`, `sampling_setup.py:60-69`). The two copies must stay in
sync — they set `cov_encoding.num_pert/num_celltype/num_batch`, the three dicts, and
`model.dataset_dict`.

## STATE equivalence

| STATE | PerturbDiff |
|---|---|
| `state tx train` | `python ./src/apps/run/rawdata_diffusion_training.py` |
| `state tx predict` | `python ./src/apps/run/rawdata_diffusion_sampling.py` |
| `output_dir/name/config.yaml` + `data_module.torch` | *(no equivalent)* config is re-derived from the hydra CLI at sampling time |
| `eval_<ckpt>/{adata_pred,adata_real}.h5ad` + `<ct>_results.csv` | `samples/diffusion_{predict,true}_<ts>.h5ad`, **no metric CSVs** |

Two structural differences to close for a fair benchmark:

1. **No saved data-module state.** STATE pickles the fitted datamodule (`data_module.torch`) so
   prediction re-uses the exact same one-hot maps and splits. PerturbDiff rebuilds the datamodule
   from the config at sampling time, so *the sampling CLI must repeat the training CLI's data
   overrides exactly* or the covariate dicts shift and the checkpoint's embedding tables no longer
   line up. Consider dumping `pert_dict`/`cell_type_dict`/`batch_dict` next to the checkpoint and
   asserting equality on load.
2. **No cell-eval.** PerturbDiff reports R² and energy-MMD only
   (`sampling/sampling_generation.py:148-178`). Adding a cell-eval step is Phase 5 of the root
   plan.
