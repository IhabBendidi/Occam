# Reproducing the released chain

This document is the provenance record. It is the one place in the repo allowed
to name the research tree Occam replaces, and the one place the old-to-new
symbol map lives.

**Read this before quoting any number about the released weights.** Two of the
three training stages were written by colliding SLURM array tasks, so the
recipe the original documentation gives does not produce the released
checkpoint. Everything below was verified from the checkpoints' own embedded
configs, which are the only primary records.

---

## The short version

| stage | what the old doc said | what the weights say |
|---|---|---|
| 1 pretrain | `poisson`, MLW ramp 0.0 → 0.7 | **`nb_softplus`, MLW ramp 0.7 → 0.9** |
| 2 finetune | lr 6e-4, variant B, gradual unmasking | *(unchanged — this stage is clean)* |
| 3 decoder | lr 1e-5, 200 epochs | **lr 3e-6, 200 epochs** |

`configs/occam_small.yaml` records the right-hand column, because that is what
produced `checkpoints/release/occam_small_encoder_decoder.pth`.

---

## Collision 1 — stage 1 is `nb_softplus`

The research tree allocated run directories as
`exp_<NNN>_<type>_<timestamp>` from a shared `outputs/registry.json`. The `NNN`
counter races when two jobs start in the same second, and two different wandb
runs both claim `exp_045_training_2026-04-28_10-22-41`:

| source | reconstruction loss | MLW schedule | run name |
|---|---|---|---|
| the wandb run whose notes point at that directory | `poisson` | 0.0 → 0.7 | `…mlw_poisson_s0.0_e0.7_cd0.0` |
| **the directory on disk** — `metadata.json` (written at creation, 10:22:41) **and** `checkpoints/final_model.pth`'s embedded `config` | **`nb_softplus`** | **0.7 → 0.9** | **`mlw_nb_softplus_s0.7_e0.9_cd0.0`** |

Everything else agrees — lr 4e-4, weight decay 1.81890136e-09, warmup 15,
`mask_probability` 0.1, `mask_loss_weight` 0.7, `masked_data_weight` 0.75,
`small`, 100 epochs, `target_sum` 2000, 44,728 genes — which is exactly why the
swap went unnoticed for four months.

### Three independent lines of evidence that the weights are the `nb_softplus` run's

1. **The embedded config.** Weights and their config come out of one
   `torch.save`, and that config says `nb_softplus` with
   `wandb_run_name = mlw_nb_softplus_s0.7_e0.9_cd0.0`.
2. **Parameter counts.** Stage 1 has 22 tensors / 209,445,088 parameters; stage
   2 has 21 / 209,400,360. The difference is exactly 44,728 — one per-gene
   `_recon_loss_fn.log_theta` vector. Only `nb_softplus` and `zinb_softplus`
   register that parameter; a `poisson` run cannot produce it, and stage 2
   (which *is* `poisson`) does not have it.
3. **`best_val_loss`.** The checkpoint records `0.8786115260685191`. The old doc
   quotes `0.0162444713881568`, which belongs to the other run. Losses are not
   comparable across `nb_softplus` and `poisson` anyway.

### Chain of custody

Stage 2's checkpoint carries
`finetuning_config.checkpoint = .../exp_045_training_2026-04-28_10-22-41/checkpoints/final_model.pth`.
The stage-1 file was final at Apr 28 16:28; stage 2 ran Apr 30. So stage 2 read
the `nb_softplus` weights, and stage 3 read stage 2's output. **The entire
released chain sits on an `nb_softplus` encoder with an MLW ramp of 0.7 → 0.9.**

### What this costs

The stage-1 benchmark numbers the old doc reports (perturbation 38.3469,
scIB 0.608339, combined 49.59) belong to weights that **were never used
downstream**. They are not quoted in `MODEL_CARD.md`; re-measure with
`occam-benchmark` instead. Stage 2's and stage 3's own numbers are unaffected —
those runs genuinely consumed the on-disk files.

If you specifically want the *documented* `poisson` recipe, a later re-run of
the same config exists in the research tree at
`outputs/experiments/exp_082_training_2026-05-03_10-30-59`
(`neurips_repr_poisson_lr0.0004`, poisson, MLW 0.0 → 0.7, lr 4e-4,
`target_sum` 2000, `small`, full `final_model.pth`). It is a different run, not
the one the old doc names.

---

## Collision 2 — stage 3 is lr 3e-6

`decoder_results/ihabnet_decoder_20260501_172850/` was written by **five
colliding array tasks** of one job. At the time, decoder run directories were
named `ihabnet_decoder_{timestamp}` with no random suffix — the `_{hex6}`
suffix was added later, in response to this — and all five launched inside the
same second:

| array task | lr | epochs | outcome | best val loss it reported |
|---|---|---|---|---|
| **0** | **3e-6** | 200 | completed 05:51 | **0.001179** |
| 1 | 1e-5 | 200 | completed 05:41 | 0.001116 |
| 2 | 3e-5 | 200 | completed 05:41 | 0.001098 |
| 3 | 3e-6 | 300 | crashed | — |
| 4 | 1e-5 | 300 | crashed | — |

All three survivors printed
`Results saved to: ./decoder_results/ihabnet_decoder_20260501_172850`.

**The surviving weights are task 0.** `best_decoder.pth` records
`epoch = 185`, `best_val_loss = 0.0011785769730683088`,
`training_config['lr'] = 3e-06`, and
`training_config['wandb_run_name'] = 'decoder_lr_sweep_3e-6_ep200'`. That val
loss matches task 0's stdout (`0.001179`) exactly.

The `run_config.json` sitting in the same directory says `lr: 1e-05` — it
belongs to task 1, which wrote it at launch. The old doc copied it. The wandb
run for this arm was later deleted, so the checkpoint is the only primary
record. That file is preserved here as
`checkpoints/release/occam_small_encoder_decoder.run_config.json.UNTRUSTED`;
the suffix is deliberate.

### A caveat that belongs on the model card

Task 0's val loss (0.001179) was the **worst** of the three survivors
(3e-5: 0.001098 < 1e-5: 0.001116 < 3e-6: 0.001179). The released decoder is not
the best arm of its own sweep — it is the arm whose checkpoint happened to
survive the collision.

### Not an isolated incident

The earlier decoder sweep collided too: `ihabnet_decoder_20260501_105512` is
claimed by three different learning rates (3e-5, 3e-4, 1e-3) and
`…_105516` by two (5e-4, 3e-3). This is a systematic defect of timestamp-only
run directories, not a one-off.

Stage 2 (`exp_062_finetuning_…_2026-04-30_17-54-02`) was checked and is
**clean** — written by exactly one array task.

### What Occam does about it

* Run directories are deterministic (`runs/<config>/<stage>/`) with no global
  registry and no sequential ids, and the launcher **refuses to write into a
  non-empty run directory**.
* Every checkpoint carries its own `config_hash`, `git_sha` and parent artifact
  path, so provenance is intrinsic to the artifact rather than to a mutable
  sidecar file.
* `scripts/write_manifest.py` records the *verified* config from each
  checkpoint's own embedded config, never from a neighbouring JSON.

---

## The reference environment

The released weights were produced under the `scmae` pyenv on this cluster:
**Python 3.8.19, torch 2.4.1+cu121**. Verified package by package on
2026-09-10:

| package | version | package | version |
|---|---|---|---|
| torch | 2.4.1+cu121 | anndata | 0.9.2 |
| numpy | 1.24.4 | scanpy | 1.9.6 |
| scipy | 1.9.3 | h5py | 3.9.0 |
| scikit-learn | 1.1.3 | wandb | 0.21.1 |
| pandas | 2.0.3 | torch-geometric | 2.6.1 |

`pyproject.toml` pins this stack. That is not a stale pin — **the forward pass
is not bit-stable across torch versions.** Replaying `golden/release_forward.npz`
under torch 2.1.2 instead of 2.4.1 shifts ~52% of latent elements by up to
1.4e-6. So `scripts/verify_checkpoint.py` requires bit-exactness only when the
torch version matches the fixture, and falls back to a tolerance otherwise,
reporting the substitution.

Note the launchers invoked a bare `python`, so the interpreter is inferred from
the package versions matching this environment exactly, not from the scripts.

---

## Symbol map

| research tree | Occam |
|---|---|
| `IhabNetTrainer` | `occam.mae.trainer.OccamTrainer` |
| `IhabNetAdataDataset` | `occam.data.datasets.OccamAdataDataset` |
| `OptimizedIhabNetDataset` | `occam.data.datasets.OccamChunkedDataset` |
| `IhabNetTransform` | `occam.transforms.OccamTransform` |
| `IhabNetInference` | `occam.inference.embed.OccamEmbedder` |
| `IhabNetReconstruction` | `occam.inference.reconstruct.OccamReconstructor` |
| `IhabNetPreprocessor` / `IhabNetFeatureExtractor` | `occam.decoder.discriminator.Occam*` |
| `create_ihabnet_model` | `occam.model.create_occam_model` |
| `train_ihabnet` | `occam.mae.stages.run_pretrain` |
| `finetune_ihabnet` | `occam.mae.stages.run_finetune` |
| `train_ihabnet_decoder` | `occam.decoder_train.stages.run_decoder` |
| `TrainingConfig` | `occam.mae.config.PretrainConfig` |
| `FinetuningConfig` | `occam.mae.finetune_config.FinetuneConfig` |
| `DecoderTrainingConfig` | `occam.decoder_train.config.DecoderTrainConfig` |
| `ihabnet_checkpoint` (config key) | `encoder_checkpoint` |
| `ihabnet_checkpoint_path` (checkpoint key) | `parent_checkpoint_path` |
| `IHABNET_BMDB_DIR` / `SCMAE_BMDB_DIR` | `OCCAM_BMDB_DIR` |
| `IHABNET_EVAL_H5AD` / `SCMAE_EVAL_H5AD` | `OCCAM_EVAL_H5AD` |
| `IHABNET_EVAL_FREQ` / `SCMAE_EVAL_FREQ` | `OCCAM_EVAL_FREQ` |
| wandb project `scMAE` | `occam` |
| `X_ihabnet` (obsm key) | `X_occam` |

`occam._compat` reads every old *persisted* name (config keys, checkpoint keys,
environment variables) so existing artifacts still load. Class and function
aliases are deliberately **not** provided — importing an old name raises
`ImportError`.

New in Occam, with no counterpart in the research tree:

| symbol | why it exists |
|---|---|
| `occam.model.OccamEncoder` | The encoder half alone, so `encoder_state_dict` can load with `strict=True`. The research tree had to load into the full `AutoEncoder` non-strictly and so could not tell "these keys are legitimately absent" from "the architecture is wrong". |
| `occam.model.build_encoder_modules` | Shared by `AutoEncoder` and `OccamEncoder` so their layer ordering — which fixes the integer indices in every checkpoint key — cannot drift. |
| `occam.ddp.stage_fence` | Closes out a stage: barrier, tear down the process group, release CUDA memory. Needed because each stage tears down its own group *before* running its end-of-training benchmark. |
| `occam._wandb_guard.ensure_no_shadow` | The research tree renamed a `wandb/` directory at **import time**. Now an explicit call. |
| `occam.paths` | One resolver for what was four separate `_get_bmdb_dir()` implementations plus a dozen hardcoded absolute dataset paths. |

---

## Dropped deliberately

* **Tandem finetuning** (`--finetune` alongside pretraining). The released chain
  ran stage 2 standalone, and the two paths differ: standalone loads with
  `strict=False`, dropping `_recon_loss_fn.log_theta` and re-initialising the
  loss, while tandem keeps the live module in memory. Tandem also had a latent
  double-transform bug when `ft_batch_size != batch_size`.
* **The XCell dataset modes and splits.** Occam's virtual-cell evaluation is
  corpus-agnostic; see `docs/evaluation.md`.
* **cell-eval2.** Occam scores with cell-eval v1 only.
* **The `0.5818` headline.** The research tree's own findings log records that
  measuring the exact quadrant that number claims returns 0.5041 — chance — and
  asks for resolution before it is quoted again. With the XCell splits dropped,
  this repo cannot recompute it either. It is retired, and
  `tests/test_rename_hygiene.py` fails if it reappears.
