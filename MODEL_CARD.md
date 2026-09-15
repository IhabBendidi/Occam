---
license: other
license_name: see-repository
pipeline_tag: feature-extraction
tags:
  - single-cell
  - transcriptomics
  - masked-autoencoder
library_name: occam
---

# Occam-small

A masked autoencoder for single-cell transcriptomes, with a high-capacity
external decoder. Encodes a 44,728-gene expression profile to a
512-dimensional embedding and reconstructs it.

* **File**: `occam_small_encoder_decoder.pth` (928,212,950 bytes)
* **sha256**: `22c05a18e1fd29c4c4619fa5748baa47efd1a6d7cbcf921d1e87dfc623bfe9d0`
* **Parameters**: encoder 94,279,864 + decoder 137,393,859 = **231,673,723**
* Verify with `python scripts/verify_checkpoint.py` (25 checks)

## Architecture

**Encoder** — MLP, `44,728 → 2048 → 1024 → 512`, each block
Linear → LayerNorm → Mish, dropout 0.0. Latent **512**.

**Decoder** — `HybridNormGlobalSkipDecoder`: `512 → 2048`, then
**5** squeeze-and-excitation residual blocks (dropout
0.3, SE reduction 16), a global skip from the
projected input, and an output head to 44,728 genes.

⚠️ `num_blocks` is **5**, not the code default of 4. Rebuilding with
the default fails to load.

⚠️ The released file does **not** record its output activation. It is
`rectified_sigmoid`, determined empirically rather than assumed: under `rectified_sigmoid`
83.6% of genes come out constant across cells
and 84.9% of outputs are exactly zero, while under
softplus both are 0.0%. Checkpoints Occam
writes today record the field.

## Input and output contract

Raw counts → library-normalise to `target_sum = 2000.0` → `log1p`.
Both transforms apply (`skip_normalize` and `skip_log1p` are false).

Output is in log1p(normalised counts) space, non-negative, and bounded above by
`log1p(2000) ≈ 7.601` because of the rectified-sigmoid head.

**Gene order is part of the contract.** The checkpoint carries `gene_names`
(44,728 entries) and `gene_to_index`; callers must reindex their input to
that order. The vocabulary is byte-identical to the DiverseRNA release's
`genes.parquet.gene_id`.

## Training recipe

Every value below was read from the checkpoints themselves, not from run logs.
See [`docs/reproducing_chain_a.md`](../docs/reproducing_chain_a.md) for why that
distinction matters here. The full recipe is
[`configs/occam_small.yaml`](../configs/occam_small.yaml).

| | stage 1 — pretrain | stage 2 — finetune | stage 3 — decoder |
|---|---|---|---|
| loss | `nb_softplus` | `poisson` | `smooth_l1` (β 2.0) |
| lr | 0.0004 | 0.0006 | **3e-06** |
| epochs | 100 | 100 | 200 (best at 185) |
| batch size | 1024 | 1024 | 1024 |
| scheduler | cosine, warmup 15 | cosine, warmup 0 | cosine to 1e-07 |
| mask probability | 0.1 | 0.0 (inert, see below) | — |
| mask-loss weight | 0.7, ramp 0.7 → 0.9 | ramp 0.7 → 0.0 | — |
| masked-data weight | 0.75 | 0.0 | — |
| unmasking | — | `gradual`, zero at 0.9 | — |
| weight decay | 1.81890136e-09 | 1.81890136e-09 | 0.01 |
| best val loss | 0.8786115260685191 | 0.05102195278467501 | 0.0011785769730683088 |

Validation losses are **not comparable across stages** — three different losses.

⚠️ Stage 2 sets `mask_probability = 0.0`, and it is inert.
Under `unmasking_mode: gradual` the ramp starts from the **parent checkpoint's**
mask probability (0.1), so masking is live for most of the
finetune. Pinned in `tests/test_schedules.py`.

**Data**: DiverseRNA, 44,728 genes, 90/10 chunk-wise split (chunk 8192,
seed 42). Read [the dataset card](../data/diverserna/DATASET_CARD.md) before
assuming anything about it — it is not uniformly raw counts and not uniformly
single-cell.

## Intended use

Representation learning and expression reconstruction on this gene axis:
embeddings for clustering, retrieval, or as features for a downstream model.

**Not** a perturbation-response model. Predicting the effect of a perturbation
is `occam.vcell`, which builds on these weights but is a separate system with
its own evaluation. Not for clinical use.

## Limitations

1. **The released decoder was not the best arm of its own sweep.** Five SLURM
   array tasks wrote one run directory. Of the three that survived, this one had
   the *worst* validation loss (3e-5: 0.001098 < 1e-5: 0.001116 < 3e-6:
   0.001179); it is simply the one whose checkpoint survived the
   collision. See `docs/reproducing_chain_a.md`.
2. **The stage-1 recipe was misdocumented for four months.** Two wandb runs claim
   the same experiment directory. The weights are `nb_softplus` with an
   MLW ramp of 0.7 → 0.9, not the
   `poisson` / 0.0 → 0.7 that was published. The parameter counts corroborate:
   stage 1 carries one extra 44,728-length tensor, the per-gene NB
   dispersion, that stage 2's poisson run does not.
3. **The decoder run's tracking record was deleted**, and the `run_config.json`
   beside the weights belongs to a different array task. The checkpoint's own
   embedded config is the only primary record; it ships as
   `occam_small_encoder_decoder.run_config.json.UNTRUSTED`.
4. **`model_state_dict` is the decoder, not the model** — byte-identical to
   `decoder_state_dict`. Use `occam.inference.loader.load_encoder_wrapper`.
5. **The output activation is not recorded** (see above).
6. **The rectified-sigmoid head is exactly zero below threshold, with zero
   gradient there.** On this checkpoint 83.6% of genes
   are constant across cells. `configs/decoder_softplus_nb.yaml` is a prepared
   replacement; it is not yet run.
7. **No calibration is applied.** On the decoder-only panels, control-fit
   quantile calibration is worth about 0.37 pearson_delta and the loss/head
   choice about 0.02 — so an uncalibrated comparison of heads measures almost
   nothing.
8. **Pretraining-corpus caveats**: 43.6% of rows are non-integer, 2.4% are bulk
   RNA-seq, and 7 of 9 constituent datasets are unidentified.
9. **No encoder benchmark numbers are quoted here.** The figures previously
   published for stage 1 belong to the *other* run that claimed that directory —
   weights never used downstream. Re-measure with `occam-benchmark`.

## Load it

```python
import torch
from occam.inference.loader import load_encoder_wrapper

w = load_encoder_wrapper("occam_small_encoder_decoder.pth", device="cuda")
# w.model.feature(x) -> (n_cells, 512) embeddings
# x must be library-normalised to 2000.0 then log1p, in w._gene_names order
```

Both halves load `strict=True`; the golden forward reproduces bit-exactly under
torch 2.4.1+cu121. Across torch versions expect ~1e-6 drift.

## Provenance chain

```
mlw_nb_softplus_s0.7_e0.9_cd0.0
  └─ occam_small_stage1_pretrain.pth   sha256 21c519b1ad686a01…
       └─ ft_mlw_B_ramp_lr0.0006
            └─ occam_small_stage2_finetune.pth   sha256 9ea9d48dec462c63…
                 └─ decoder_lr_sweep_3e-6_ep200
                      └─ occam_small_encoder_decoder.pth   sha256 22c05a18e1fd29c4…
```
