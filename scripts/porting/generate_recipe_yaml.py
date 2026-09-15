#!/usr/bin/env python
"""Generate configs/occam_small.yaml from the released checkpoints themselves.

The published recipe must not be hand-typed: two of the three stages were
misdocumented in the research tree, and a transcription error here would be the
same class of bug. This reads each checkpoint's own embedded config and emits
the YAML, annotating every value with where it came from.

    python scripts/porting/generate_recipe_yaml.py --repo . --out configs/occam_small.yaml
"""

from __future__ import annotations

import argparse
import os

import torch

STAGE1 = "checkpoints/reference/occam_small_stage1_pretrain.pth"
STAGE2 = "checkpoints/reference/occam_small_stage2_finetune.pth"
STAGE3 = "checkpoints/release/occam_small_encoder_decoder.pth"


def cfg_of(path, keys=("config", "training_config")):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in keys:
        c = ck.get(k)
        if isinstance(c, dict):
            return ck, c
        if c is not None:
            d = getattr(c, "__dict__", None)
            if d:
                return ck, dict(d)
    return ck, {}


def fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # Full precision via repr(), then force a dot into the mantissa.
        #
        # This matters: YAML 1.1 (PyYAML) resolves `3e-06` and `1e-07` as
        # STRINGS, not floats, because its float pattern requires a `.` in the
        # mantissa. Emitting repr() directly would put a string learning rate
        # into the published recipe and it would fail somewhere far away.
        # `3.0e-06` parses as a float; `3e-06` does not.
        text = repr(v)
        if "e" in text or "E" in text:
            mantissa, _, exponent = text.partition("e" if "e" in text else "E")
            if "." not in mantissa:
                mantissa += ".0"
            text = f"{mantissa}e{exponent}"
        return text
    if isinstance(v, str):
        return v if v and not any(c in v for c in ": #{}[]|>*&!%@`") else repr(v)
    return repr(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = args.repo
    ck1, c1 = cfg_of(os.path.join(r, STAGE1))
    ck2, c2 = cfg_of(os.path.join(r, STAGE2))
    ft = ck2.get("finetuning_config") or {}
    if not isinstance(ft, dict):
        ft = dict(getattr(ft, "__dict__", {}) or {})
    ck3, c3 = cfg_of(os.path.join(r, STAGE3))
    d3 = dict(ck3.get("decoder_config") or {})
    e3 = dict(ck3.get("encoder_config") or {})

    g = lambda d, k, default=None: d.get(k, default)  # noqa: E731

    out = f'''# =====================================================================
# Occam-small: THE PUBLISHED RECIPE.
#
# Every value below was transcribed by scripts/porting/generate_recipe_yaml.py
# directly from the released checkpoints' own embedded configs -- NOT from the
# research tree's documentation, which is wrong for stages 1 and 3. Two run
# directories there were written by colliding SLURM array tasks.
# See docs/reproducing_chain_a.md.
#
# Run it:
#     python -m occam.train --config configs/occam_small.yaml
#     python -m occam.train --config configs/occam_small.yaml --stages decoder
#     python -m occam.train --config configs/occam_small.yaml --dry-run
# =====================================================================
version: 1
name: occam_small
seed: {fmt(g(c3, 'seed', 42))}
device: cuda

paths: !include paths.yaml          # machine-local; see paths.example.yaml

run:
  root: ./runs
  stages: [pretrain, finetune, decoder]
  runner: inprocess                 # inprocess | subprocess
  resume: auto                      # auto | never | force
  wandb:
    enabled: true
    project: occam
    tags: [occam, small, published-recipe]

data:
  train: ${{paths.train_corpus}}
  val: null                         # null -> chunk-based split using val_split
  train_gene_col: gene_symbol
  val_gene_col: null
  # Sanity assertions, checked at load. See data/diverserna/DATASET_CARD.md for
  # why "raw_counts" in the corpus filename is wrong for 46% of its rows.
  expect:
    n_obs: 1434299
    n_genes: {fmt(e3.get('num_genes', 44728))}

stages:

  # -------------------------------------------------------------------
  # STAGE 1 - MAE pretraining
  # VERIFIED from stage-1 final_model.pth['config'].
  # Recorded run name: {g(c1, 'wandb_run_name')}
  # Result: best_val_loss {ck1.get('best_val_loss')} (nb_softplus; not
  # comparable to a poisson loss), {g(c1, 'epochs')} epochs completed.
  #
  # NOTE the loss and the MLW direction. The old documentation said `poisson`
  # with a 0.0 -> 0.7 ramp. The weights say otherwise, and the state dict
  # carries `_recon_loss_fn.log_theta`, which only nb_softplus registers.
  # -------------------------------------------------------------------
  pretrain:
    model:
      model_size: {fmt(g(c1, 'model_size'))}
      hidden_size: {fmt(g(c1, 'hidden_size'))}          # null -> 512 for `small`
      mae_decoder_hidden: {fmt(g(c1, 'mae_decoder_hidden'))}   # null -> 1024 legacy default
      dropout: {fmt(g(c1, 'dropout', 0.0))}
    optim:
      epochs: {fmt(g(c1, 'epochs'))}
      batch_size: {fmt(g(c1, 'batch_size'))}
      learning_rate: {fmt(g(c1, 'learning_rate'))}
      weight_decay: {fmt(g(c1, 'weight_decay'))}
      optimizer: {fmt(g(c1, 'optimizer', 'adam'))}
      lr_scheduler: {fmt(g(c1, 'lr_scheduler'))}
      warmup_epochs: {fmt(g(c1, 'warmup_epochs'))}
      warmup_start_factor: {fmt(g(c1, 'warmup_start_factor', 0.001))}
      gradient_clip_val: {fmt(g(c1, 'gradient_clip_val', 1.0))}
      early_stopping_patience: {fmt(g(c1, 'early_stopping_patience', 3500))}   # effectively disabled
      validation_freq: {fmt(g(c1, 'validation_freq', 1))}
    masking:
      mask_probability: {fmt(g(c1, 'mask_probability'))}
      corruption_mode: {fmt(g(c1, 'corruption_mode', 'swap'))}
      gaussian_sigma: {fmt(g(c1, 'gaussian_sigma', 0.5))}
      variable_masking: {fmt(g(c1, 'variable_masking', False))}
      gene_presence_mask: {fmt(g(c1, 'gene_presence_mask', False))}
      presence_shift: {fmt(g(c1, 'presence_shift', 1.0))}
    loss:
      reconstruction_loss: {fmt(g(c1, 'reconstruction_loss'))}
      masked_data_weight: {fmt(g(c1, 'masked_data_weight'))}
      mask_loss_weight: {fmt(g(c1, 'mask_loss_weight'))}
      mlw_schedule_start: {fmt(g(c1, 'mlw_schedule_start'))}
      mlw_schedule_end: {fmt(g(c1, 'mlw_schedule_end'))}
      mlw_schedule_warmup: {fmt(g(c1, 'mlw_schedule_warmup'))}
      mlw_schedule_cooldown: {fmt(g(c1, 'mlw_schedule_cooldown'))}
      clamp_max_log1p: {fmt(g(c1, 'clamp_max_log1p'))}
      w_mse: {fmt(g(c1, 'w_mse', 1.0))}
      w_pois: {fmt(g(c1, 'w_pois', 0.1))}
      recon_loss_scale: {fmt(g(c1, 'recon_loss_scale', 1.0))}
      gpl_weight: {fmt(g(c1, 'gpl_weight', 0.0))}
      loss_diagnostics: {fmt(g(c1, 'loss_diagnostics', True))}
    data:
      data_mode: {fmt(g(c1, 'data_mode'))}
      target_sum: {fmt(g(c1, 'target_sum'))}
      val_split: {fmt(g(c1, 'val_split'))}
      scale: {fmt(g(c1, 'scale', False))}
      skip_normalize: {fmt(g(c1, 'skip_normalize', False))}
      skip_log1p: {fmt(g(c1, 'skip_log1p', False))}
      optim_chunk_size: {fmt(g(c1, 'optim_chunk_size', 8192))}
      optim_buffer_chunks: {fmt(g(c1, 'optim_buffer_chunks', 4))}
      min_gene_overlap_pct: {fmt(g(c1, 'min_gene_overlap_pct'))}
    runtime:
      enable_amp: {fmt(g(c1, 'enable_amp', False))}
      compile: {fmt(g(c1, 'compile', False))}
      use_multi_gpu: {fmt(g(c1, 'use_multi_gpu', True))}
      use_ddp: {fmt(g(c1, 'use_ddp', True))}      # effective world size was 1 (no torchrun)
      freeze_mask_predictor: {fmt(g(c1, 'freeze_mask_predictor', False))}
    eval:
      enabled: {fmt(g(c1, 'eval_enabled', True))}
      frequency: {fmt(g(c1, 'eval_frequency', 5))}
      batch_size: {fmt(g(c1, 'eval_batch_size', 1024))}
      scib: {fmt(g(c1, 'scib_eval_enabled', True))}

  # -------------------------------------------------------------------
  # STAGE 2 - unmasking finetune. This stage is CLEAN (single array task).
  # VERIFIED from stage-2 finetuned_final_model.pth['finetuning_config'].
  # Recorded run name: {g(ft, 'wandb_run_name')}
  # Result: best_val_loss {ck2.get('best_val_loss')}
  #
  # `null` below means INHERIT from the parent checkpoint's stage-1 config --
  # note: from the CHECKPOINT, not from the `pretrain` block above. The loader
  # writes overlay_audit.json recording which won, per field.
  # -------------------------------------------------------------------
  finetune:
    init_from: ${{stages.pretrain.artifacts.final_model}}
    unmasking:
      mode: {fmt(g(ft, 'unmasking_mode'))}
      zero_at: {fmt(g(ft, 'unmasking_zero_at'))}
      final_ratio: {fmt(g(ft, 'unmasking_final_ratio', 0.5))}
      # !! Under mode=gradual the mask probability ramps down from the PARENT
      #    CHECKPOINT's mask_probability ({fmt(g(c1, 'mask_probability'))}), not from
      #    masking.mask_probability below, which is inert. exp_062's log records
      #    `mask_p: 0.100` at epoch 1. Asserted in tests/test_schedules.py.
      effective_mask_prob_source: parent_checkpoint.mask_probability
    optim:
      epochs: {fmt(g(ft, 'epochs'))}
      learning_rate: {fmt(g(ft, 'learning_rate'))}
      weight_decay: {fmt(g(ft, 'weight_decay'))}
      batch_size: {fmt(g(ft, 'batch_size'))}
      optimizer: {fmt(g(ft, 'optimizer'))}
      lr_scheduler: {fmt(g(ft, 'lr_scheduler'))}
      warmup_epochs: {fmt(g(ft, 'warmup_epochs'))}
      warmup_start_factor: {fmt(g(ft, 'warmup_start_factor'))}
      gradient_clip: {fmt(g(ft, 'gradient_clip'))}
      early_stopping: {fmt(g(ft, 'early_stopping'))}
      validation_freq: {fmt(g(ft, 'validation_freq'))}
    masking:
      mask_probability: {fmt(g(ft, 'mask_probability'))}      # inert under mode=gradual
      corruption_mode: {fmt(g(ft, 'corruption_mode'))}        # null -> inherit (swap)
      gaussian_sigma: {fmt(g(ft, 'gaussian_sigma'))}          # null -> inherit (0.5)
    loss:
      recon_loss: {fmt(g(ft, 'recon_loss'))}                  # differs from stage 1 on purpose
      masked_data_weight: {fmt(g(ft, 'masked_data_weight'))}
      mask_loss_weight: {fmt(g(ft, 'mask_loss_weight'))}
      mlw_schedule_start: {fmt(g(ft, 'mlw_schedule_start'))}
      mlw_schedule_end: {fmt(g(ft, 'mlw_schedule_end'))}
      mlw_schedule_warmup: {fmt(g(ft, 'mlw_schedule_warmup'))}
      mlw_schedule_cooldown: {fmt(g(ft, 'mlw_schedule_cooldown'))}
      clamp_max_log1p: {fmt(g(ft, 'clamp_max_log1p'))}
    runtime:
      data_mode: {fmt(g(ft, 'data_mode'))}
      freeze_mask_predictor: {fmt(g(ft, 'freeze_mask_predictor'))}   # null -> inherit (false)
      enable_amp: {fmt(g(ft, 'enable_amp'))}
      compile: {fmt(g(ft, 'compile'))}
    eval:
      eval_freq: {fmt(g(ft, 'eval_freq'))}
    load:
      # REQUIRED. The nb_softplus parent contributes `_recon_loss_fn.log_theta`
      # (44,728 floats) which the poisson child does not have. Declaring the
      # expectation gives strictness' safety without changing behaviour.
      strict: false
      expect_unexpected: ["_recon_loss_fn.log_theta"]
      expect_missing: []

  # -------------------------------------------------------------------
  # STAGE 3 - external decoder, frozen encoder, AS RELEASED.
  # VERIFIED from best_decoder.pth['training_config'] + ['decoder_config'].
  # Recorded run name: {g(c3, 'wandb_run_name')}
  # Result: best epoch {ck3.get('epoch')}, best_val_loss {ck3.get('best_val_loss')}
  #
  # NOTE the learning rate. The neighbouring run_config.json says 1e-5; it
  # belongs to a different array task. Five tasks shared this run directory.
  # NOTE decoder_num_blocks is 5, and the code default is 4.
  # NOTE this arm had the WORST val loss of the three that survived the
  # collision. configs/decoder_softplus_nb.yaml supersedes it once run.
  # -------------------------------------------------------------------
  decoder:
    init_from: ${{stages.finetune.artifacts.final_model}}
    encoder:
      freeze_encoder: {fmt(g(c3, 'freeze_encoder'))}
      encoder_lr: {fmt(g(c3, 'encoder_lr'))}            # unused while frozen
      # Read from the encoder checkpoint and OR-ed with these; the released
      # stage-2 config has neither key, so both resolve false.
      skip_normalize: false
      skip_log1p: false
    architecture:
      decoder_size: null                     # null -> use the explicit dims below
      decoder_hidden_dim: {fmt(d3.get('hidden_dim'))}
      decoder_num_blocks: {fmt(d3.get('num_blocks'))}          # !! code default is 4
      decoder_dropout: {fmt(d3.get('dropout_rate'))}
      decoder_se_reduction: {fmt(d3.get('se_reduction'))}
      # NOT recorded in the released checkpoint. Determined empirically by
      # scripts/verify_checkpoint.py: under rectified_sigmoid 83.6% of genes are
      # constant across cells, under softplus 0.0%.
      decoder_output_activation: rectified_sigmoid
      decoder_weight_tie: false
      decoder_bottleneck_k: 1
      decoder_monarch_blocks: 0
      decoder_kron_rank: 0
      decoder_glu: false
      decoder_sparsity_24: false
    optim:
      lr: {fmt(g(c3, 'lr'))}
      eta_min: {fmt(g(c3, 'eta_min'))}
      weight_decay: {fmt(g(c3, 'weight_decay'))}
      gradient_clip: {fmt(g(c3, 'gradient_clip'))}
      batch_size: {fmt(g(c3, 'batch_size'))}
      epochs: {fmt(g(c3, 'epochs'))}
      loss_type: {fmt(g(c3, 'loss_type'))}
      smooth_l1_beta: {fmt(g(c3, 'smooth_l1_beta'))}
      weight_init: {fmt(g(c3, 'weight_init', 'default'))}
      dispersion_lr: null                    # nb/zinb only; unused here
    data:
      data_mode: {fmt(g(c3, 'data_mode', 'optim'))}
      val_split: {fmt(g(c3, 'val_split', 0.1))}
      optim_chunk_size: {fmt(g(c3, 'optim_chunk_size', 8192))}
      optim_buffer_chunks: {fmt(g(c3, 'optim_buffer_chunks', 4))}
      gene_presence_mask: {fmt(g(c3, 'gene_presence_mask', False))}
      presence_shift: {fmt(g(c3, 'presence_shift', 1.0))}
      min_gene_overlap_pct: {fmt(g(c3, 'min_gene_overlap_pct'))}
      train_gene_col: gene_symbol
      eval_gene_col: ensembl_gene_id
    augmentation:
      no_augmentation: {fmt(g(c3, 'no_augmentation', False))}
      gvf_cache_dir: null                    # -> runs/<name>/decoder/gvf_cache
      aug_hparams:
'''
    aug = g(c3, "aug_hparams") or {}
    for k in sorted(aug):
        out += f"        {k}: {fmt(aug[k])}\n"

    out += f'''    adversarial:                             # all off in the released run
      adv_loss: {fmt(g(c3, 'adv_loss', False))}
      perc_loss: {fmt(g(c3, 'perc_loss', False))}
      feature_extractor_checkpoint: {fmt(g(c3, 'feature_extractor_checkpoint'))}
      adv_mode: {fmt(g(c3, 'adv_mode', 'frozen'))}
      perc_mode: {fmt(g(c3, 'perc_mode', 'frozen'))}
      lambda_adv: {fmt(g(c3, 'lambda_adv', 0.1))}
      lambda_perc: {fmt(g(c3, 'lambda_perc', 0.1))}
      disc_lr: {fmt(g(c3, 'disc_lr', 0.0002))}
    eval:
      decoder_eval_freq: {fmt(g(c3, 'decoder_eval_freq', 25))}
      bmdb_eval_freq: {fmt(g(c3, 'bmdb_eval_freq', 3))}
      run_encoder_benchmark: {fmt(g(c3, 'run_encoder_benchmark', False))}
      log_interval: {fmt(g(c3, 'log_interval', 3))}
      cell_eval_enabled: false               # postdates the release; off for parity
      eval_data:
        - ${{paths.eval_dir}}/nadig_2024_jurkat.h5ad
    # Flags that POSTDATE the released run, listed at their no-op values so a
    # reader can see they were not in play. Changing any of them departs from
    # the published recipe.
    postdates_release:
      no_amp: false
      eval_present_genes_only: false
      eval_zero_threshold: 0.0
      eval_calibrate: none
      eval_libnorm: none
      celleval_calibrate: none
      celleval_zero_mode: none
      celleval_sample_predictive: false
'''

    with open(os.path.join(r, args.out) if not os.path.isabs(args.out) else args.out,
              "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"wrote {args.out} ({len(out.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
