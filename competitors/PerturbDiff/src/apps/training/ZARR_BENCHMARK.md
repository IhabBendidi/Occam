# `src/apps/training/` — builders, trainer, fit loop

Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Role | Zarr impact |
|---|---|---|
| `training_datamodule_builder.py` | `data_name` → datamodule class; covariate cfg population | ✅ `IhabxcellZarrFinetune` registered |
| `training_model_builder.py` | model construction + checkpoint patching + re-init | none |
| `training_model_checkpoint.py` | checkpoint loading / layer replacement / unpickling shims | none (relevant for finetuning) |
| `training_runtime.py` | `pl.Trainer` assembly | none |
| `training_runtime_helpers.py` | strategy, callbacks, profiler | none |
| `training_pipeline.py` | `trainer.fit` / `trainer.validate` | one caveat |
| `training_model_compare.py` | optional representation comparison | none |

---

## 1. `training_datamodule_builder.py` — the dispatch (done)

```python
def build_datamodule(cfg, logger):                       # lines 5-39
    if   cfg.data.data_name in ["PBMCFinetune"]:                       PBMCPerturbationDataModule
    elif cfg.data.data_name in ["Tahoe100mFinetune","ReplogleFinetune"]: Tahoe100mPerturbationDataModule
    elif cfg.data.data_name in ["Tahoe100mPBMCReplogleCellxGenePretrain"]: PerturbationPretrainingDataModule
    else:
        assert 0                                          # line 35 — bare assert, no message
    datamodule.replace_pert_dict = cfg.cov_encoding.get("replace_pert_dict", False)
    datamodule.setup()
```

**Done:** `IhabxcellZarrFinetune` → `ZarrPerturbationDataModule`, and the bare `assert 0` is now
a `ValueError` listing the known names. Note the class must also be re-exported from
`src/data/data_module/__init__.py`, because the builders import the *package*, not the module.

`micro_batch_size` comes from `cfg.optimization.micro_batch_size` here; the sampling builder uses
`cfg.sampling.batch_size` for some branches and `cfg.optimization.micro_batch_size` for others
(`../sampling/sampling_setup.py:17-54`) — an inconsistency to be aware of when matching batch
shapes between train and eval.

### `populate_covariate_cfg` (41-50)

```python
cfg.cov_encoding.num_pert     = len(datamodule.pert_dict)
cfg.cov_encoding.num_celltype = len(datamodule.cell_type_dict)
cfg.cov_encoding.num_batch    = len(datamodule.batch_dict)
cfg.cov_encoding.{pert,cell_type,batch}_dict = …
cfg.model.dataset_dict        = datamodule.original_dataset_name_list
```

`original_dataset_name_list` feeds the per-dataset embedder dict in `Cross_DiT._build_x_embedders`
(`../../models/cross_dit/cross_dit_main.py:85-100`) and the per-dataset loss keys
(`../../models/lightning/lightning_module.py:247-255`), both via `get_short_dsname`. Make sure the
zarr datamodule sets it (the base `setup()` does, at
`../../data/data_module/data_module_setup.py:110, 147, 488`).

---

## 2. `training_model_builder.py` / `training_model_checkpoint.py`

`build_model` (12-38) instantiates `PlModel` with `cov_encoding_cfg`, `model_cfg`,
`optimizer_cfg`, `trainer_cfg`, `all_split_names`.

`maybe_load_and_patch_checkpoint_model` (`training_model_checkpoint.py:131`) implements the
finetuning path the README uses: load `model.model_weight_ckpt_path`, optionally swap the
2k/12626-gene input layers (`model.replace_2kgene_layer`, `model.replace_1w2gene_layer`),
re-initialize AdaLN, etc. `allow_omegaconf_checkpoint_unpickling` (22) registers safe globals so
OmegaConf objects inside checkpoints survive `torch.load`.

**Zarr impact: none.** But relevant to the benchmark design: if you want PerturbDiff's *pretrained*
checkpoint rather than from-scratch training, the pretrained pert/celltype vocabularies will not
match the zarr ones (Ensembl ids vs symbols, `"empty"` vs `"non-targeting"`). Use
`cov_encoding.replace_pert_dict=true` and `model.separate_embedder=by_name` as the README's
finetune block does, and expect the perturbation embedding table to be reinitialized.

---

## 3. `training_runtime.py` + `training_runtime_helpers.py`

`setup_trainer` (19-43) instantiates the logger from `cfg.lightning.logger`, resolves the strategy,
builds callbacks, and expands **all of `cfg.trainer` as `pl.Trainer(**kwargs)`** — so every key in
`configs/trainer/base.yaml` must be a valid Trainer argument.

`build_callbacks` (37) instantiates `cfg.lightning.callbacks.*` plus the EMA weight-averaging
callback when `lightning.ema.decay > 0`.

For the benchmark run, mirror STATE's cadence (`train_state_zarr.sh`):

```
trainer.max_steps=80000
lightning.callbacks.checkpoint.every_n_train_steps=2000
lightning.ema.decay=0.99  lightning.ema.update_steps=10
```

and use `$NO_WANDB` from the README to swap in `DummyLogger`.

---

## 4. `training_pipeline.py`

```python
trainer.fit(model, datamodule, ckpt_path=cfg.model.ckpt_path, weights_only=False)
trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)          # the final checkpoint
trainer.validate(model, datamodule, ckpt_path=cfg.save_dir_path, weights_only=False)
```

The final validation runs on **the final checkpoint** (`cfg.save_dir_path`, i.e.
`<trainer.default_root_dir>/<run_name>` — a *file*, written by the `save_checkpoint` line above),
not on `ModelCheckpoint`'s `"best"`.

That was a deliberate change: `"best"` requires the monitor
(`lightning.callbacks.checkpoint.monitor`, default `validation_validation_loss_epoch`) to have
fired, and fold 0 of the benchmark split has `validation: []`, so it never does — the old code
died at the very end of an 80k-step run. See the root plan §7.3.

Consequences to keep in mind:

* You evaluate the **last** model, not the best-scoring one. For a from-scratch 80k-step run that
  is the normal choice and matches STATE (`predict_state_zarr.sh` passes an explicit checkpoint).
* To select by score instead, set the monitor to a metric the run actually emits
  (e.g. `validation_test_loss_epoch`) and pass that checkpoint to the sampler explicitly.
* The same `cfg.save_dir_path` file is what you feed to the eval entrypoint:
  `model_checkpoint_path=<default_root_dir>/<run_name>` — verified loadable by
  `load_sampling_model`.

`weights_only=False` on both calls is required under torch ≥ 2.6; see the root plan §7.2.
