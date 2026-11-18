# Finetuning the released encoder on your own data (label-free)

Adapt Occam's encoder to a new corpus with a few epochs of its own self-supervised
objective. No labels are needed. The result is a plain MAE checkpoint that
`occam.api.load_encoder` embeds with, like any other.

## Which checkpoint to start from

**The stage-2 MAE:** `checkpoints/reference/occam_small_stage2_finetune.pth` (HF:
`checkpoints/stage2/occam_small_stage2_finetune.pth`).

| file | encoder | MAE decoder + mask predictor | use it to finetune? |
|---|---|---|---|
| `occam_small_encoder_decoder.pth` (released) | yes | **no**: it carries the external softplus decoder instead | no |
| `occam_small_stage2_finetune.pth` (stage 2) | yes, byte-identical to the released one | yes (Poisson), 21 tensors | **yes** |

Stage 2 is the last training of the MAE itself. The released decoder was trained later,
on the frozen encoder, and is a different module. Finetuning needs the MAE's own decoder
and mask predictor, which only the stage-2 file keeps.

## The recipe

`configs/finetune_ssl.yaml` is stage 2 of `configs/occam_small.yaml`, with only
**epochs 100 → 5** changed:
- Poisson reconstruction; `masked_data_weight` 0.
- Mask-loss weight 0.7 → 0 (warmup 10%, cooldown 10%).
- lr 6e-4, cosine, no warmup; batch 1024.
- Swap-noise masking; gradual unmasking 0.1 → 0 by 90% of training; `target_sum` 2000.

`tests/test_finetuning.py` fails if the two configs drift apart anywhere else.

## Steps

```bash
# 1. init copy: same weights, mask ramp starts at 0.1 (see gotcha 1)
python scripts/finetune/make_finetune_init.py \
    --parent checkpoints/reference/occam_small_stage2_finetune.pth \
    --out runs/finetune_init/occam_small_stage2_mask0.1.pth

# 2. data: raw counts, genes in the checkpoint's vocabulary, dense X (gotcha 2)
python scripts/finetune/densify_h5ad.py --src mine.h5ad --dst mine_dense.h5ad

# 3. train (~10 min/epoch per 1.3 M rows on one H100, bound by reading the file)
python -m occam.train --config configs/finetune_ssl.yaml --set data.train=mine_dense.h5ad

# 4. embed with the result
python -c "from occam.api import load_encoder; \
enc = load_encoder('runs/finetune_ssl/finetune/finetuned_final_model.pth'); print(enc)"
```

**Data requirements.**
- `X` holds **raw counts** (or another linear, non-negative scale).
- Genes are named by the checkpoint vocabulary: 44,560 of its 44,728 entries are Ensembl
  ids. Use `var_names`, or name a `.var` column with `data.train_gene_col`.
- Vocabulary genes missing from your file are zero-filled. Genes outside the vocabulary
  are ignored.
- Several of your genes that map to one vocabulary gene must be **summed** before
  writing.
- Rows should be shuffled, because the loader only shuffles within 8,192-row chunks and a
  4-chunk buffer.

## Gotchas

1. **Masking would silently be off.** `gradual` unmasking ramps down from the **parent
   checkpoint's** `config.mask_probability`, not from `stages.finetune.masking`. Stage 2
   ended fully unmasked and records **0.0**, so rerunning the recipe on it masks nothing.
   `make_finetune_init.py` writes a copy that records 0.1 and checks all 21 tensors are
   unchanged. Confirm in the log: epoch 1 must show `mask_p: 0.100`.
2. **The `optim` loader needs a dense `X`.** A sparse file fails at the first read with
   `'Group' object has no attribute 'shape'`. Densify it (size = rows × genes × 4 bytes).
3. **wandb follows `run.wandb`.** Before 2026-09-24, stage 2 ignored `run.wandb.enabled`
   and inherited the parent's flag, so a finetune logged to the parent's project under
   the parent's run name. Fixed in `OccamConfig.build_finetune_config`.
4. **End-of-training eval.** `eval.eval_freq: 5` runs the perturbation and scIB benchmark
   after the last epoch (slow). `finetuned_final_model.pth` is written **before** it, so
   the checkpoint is safe to use even if you stop the eval.
5. **Losses are not comparable to stage 2's.** They include the mask-BCE term at the
   scheduled weight. Only the last epoch (weight 0) matches stage 2's logging.

## Worked example (2026-09-24)

1,186,883 single cells + 6,111 bulk samples (×20) from 48 patient-cohort datasets. 5
epochs, one H100, about 52 min.

| epoch | train loss | val loss | mask_p | MLW |
|---|---|---|---|---|
| 1 | 0.8796 | 0.4922 | 0.100 | 0.700 |
| 5 | 0.0208 | 0.0213 | 0.000 | 0.000 |

The encoder moved substantially (the first linear layer's weights changed by 57% in
relative norm). It did not collapse: pbmc_12k kNN15 cell-type accuracy was 0.8826 raw /
0.9565 after PCA-50, against 0.8817 / 0.9494 for the released encoder.
