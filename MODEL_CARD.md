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

* **File**: `occam_small_encoder_decoder.pth` (892,816,810 bytes)
* **sha256**: `ef82433f8a6e…` — full value in `MANIFEST.json`
* **Parameters**: encoder 94,279,864 + decoder 128,456,385 = **222,736,249**
* Verify with `python scripts/verify_checkpoint.py` (27 checks)

> **Updated 2026-09-16.** The decoder half is now the **softplus + negative
> binomial** decoder. It replaces the `rectified_sigmoid` + smooth-L1 decoder
> that shipped until now, which scored **0.1985** calibrated `pearson_delta`
> against this one's **0.5732** on the same panel — a measured +0.3747. The
> superseded file is kept at
> `checkpoints/legacy/occam_small_encoder_decoder_rectified_sigmoid.pth`, is
> **not** published, and exists only because the golden fixtures pin it and it
> is therefore the repository's port-equivalence evidence. Full comparison:
> [`docs/decoder_retrain_jurkat.md`](../docs/decoder_retrain_jurkat.md).
>
> **The encoder did not change.** Its 13 tensors are byte-identical to the
> previous release's, and the golden latent still reproduces bit-exactly
> (`max|diff| = 0`). Only the decoder was retrained.

## Architecture

**Encoder** — MLP, `44,728 → 2048 → 1024 → 512`, each block
Linear → LayerNorm → Mish, dropout 0.0. Latent **512**.

**Decoder** — `HybridNormGlobalSkipDecoder`: `512 → 2048`, then **4**
squeeze-and-excitation residual blocks (dropout 0.3, SE reduction 16), a global
skip from the projected input, and an output head to 44,728 genes.
Output activation **`softplus`**, recorded in the checkpoint.

`num_blocks: 4` matches the code default and every H1 arm ever measured.
(The superseded decoder was 5 blocks, which is *not* the default — a trap only
the legacy file carries.)

## Input and output contract

Raw counts → library-normalise to `target_sum = 2000.0` → `log1p`.
Both transforms apply (`skip_normalize` and `skip_log1p` are false).

Output is in log1p(normalised counts) space and **strictly positive** —
`softplus` has no exact-zero floor, unlike the superseded head. It is also
**unbounded above**, where the old head was capped at `log1p(2000) ≈ 7.601`.
Code that relied on that ceiling, or on exact zeros marking undetected genes,
must be revisited.

**Gene order is part of the contract.** The checkpoint carries `gene_names`
(44,728 entries) and `gene_to_index`; callers must reindex their input to
that order. The vocabulary is byte-identical to the DiverseRNA release's
`genes.parquet.gene_id`.

## Training recipe

Every value below was read from the checkpoints themselves, not from run logs.
See [`docs/reproducing_chain_a.md`](../docs/reproducing_chain_a.md) for why that
distinction matters here.

Stages 1–2 are [`configs/occam_small.yaml`](../configs/occam_small.yaml).
**Stage 3 is [`configs/decoder_softplus_nb.yaml`](../configs/decoder_softplus_nb.yaml)** —
that config, not the stage-3 block of `occam_small.yaml`, describes the shipped
decoder. `tests/test_config_frozen.py` asserts both correspondences.

| | stage 1 — pretrain | stage 2 — finetune | stage 3 — decoder |
|---|---|---|---|
| loss | `nb_softplus` | `poisson` | **`nb`**, learned per-gene `log_theta` |
| output head | — | — | **`softplus`** |
| lr | 0.0004 | 0.0006 | **5e-05** |
| epochs | 100 | 100 | 200 requested, **best at 124** |
| batch size | 1024 | 1024 | 1024 |
| scheduler | cosine, warmup 15 | cosine, warmup 0 | cosine to 1e-07 |
| mask probability | 0.1 | 0.0 (inert, see below) | — |
| mask-loss weight | 0.7, ramp 0.7 → 0.9 | ramp 0.7 → 0.0 | — |
| masked-data weight | 0.75 | 0.0 | — |
| unmasking | — | `gradual`, zero at 0.9 | — |
| weight decay | 1.81890136e-09 | 1.81890136e-09 | 0.01 |
| best val loss | 0.8786115260685191 | 0.05102195278467501 | 0.053064 |

Validation losses are **not comparable across stages** — three different
losses, and stage 3's is now an NB likelihood rather than a smooth-L1, so it is
not comparable to the superseded decoder's 0.0011785769730683088 either.

Learned NB dispersion ships in the checkpoint (`loss_state_dict.log_theta`):
`theta` mean 20.52, range 0.78–237.1.

⚠️ Stage 2 sets `mask_probability = 0.0`, and it is inert.
Under `unmasking_mode: gradual` the ramp starts from the **parent checkpoint's**
mask probability (0.1), so masking is live for most of the
finetune. Pinned in `tests/test_schedules.py`.

**Data**: DiverseRNA, 44,728 genes, 90/10 chunk-wise split (chunk 8192,
seed 42). Read [the dataset card](../data/diverserna/DATASET_CARD.md) before
assuming anything about it — it is not uniformly raw counts and not uniformly
single-cell.

## Measured performance

Decoder reconstruction scored with ARC cell-eval v1 on
`nadig_2024_jurkat` (262,956 cells, 2,394 perturbations, 12,013 controls,
8,873 of the 44,728 genes present), `injctrl` convention, control-fit per-gene
quantile calibration, present genes only. Full table and caveats in
[`docs/decoder_retrain_jurkat.md`](../docs/decoder_retrain_jurkat.md).

| metric | superseded (rect_sigmoid) | **released (softplus + nb)** |
|---|---|---|
| `pearson_delta` | 0.1985 | **0.5732** |
| `discrimination_score_l2` | 0.9652 | **0.9867** |
| `de_direction_match` | 0.8212 | **0.8705** |
| `de_nsig_counts_pred` *(truth 262.2)* | 3,871.0 | **1,822.8** |
| `mse` / `mae` *(lower better)* | 0.0169 / 0.0678 | **0.0014 / 0.0239** |
| `de_sig_genes_recall` | **0.6341** | 0.5130 |

Two numbers to read carefully. The DE over-call factor falls 14.8× → 6.9×,
which is the mechanism the head change was expected to fix and the reason to
believe the gain is real rather than incidental. But
**`de_sig_genes_recall` is lower**: calling ~2,000 fewer genes significant
recovers fewer of the true ones. If your use case needs recall over precision,
the superseded decoder scores higher on it, for the uninteresting reason that
it called nearly everything significant.

**Calibration is load-bearing, and is not part of the checkpoint.** On these
panels control-fit quantile calibration is worth roughly 0.37 `pearson_delta`
and the head/loss choice about 0.02. An uncalibrated comparison of decoder
heads measures almost nothing. See `occam.vcell.panel.calibrate`.

**Integer emission was tested and rejected.** Re-emitting predictions as
counts (`expm1` → round → `log1p`) costs 0.034 `pearson_delta`, and rescaling
the library size back afterwards recovers none of it (−0.010). At
`target_sum` 2,000 rounding drives the zero fraction to 0.907 against the
target's 0.606. Use the raw log1p output.

## Intended use

Representation learning and expression reconstruction on this gene axis:
embeddings for clustering, retrieval, or as features for a downstream model.

**Not** a perturbation-response model. The numbers above measure whether a
reconstruction *preserves* perturbation deltas, not whether a model *predicts*
them. Predicting the effect of a perturbation is `occam.vcell`, a separate
system with its own evaluation. Not for clinical use.

## Limitations

1. **The decoder result is n = 1, on one panel, in one cell line.** No seed
   replicates. The +0.3747 `pearson_delta` gap is large enough to survive that,
   but the epoch-124-over-epoch-49 margin (+0.0096) is not a basis for anything.
2. **Its training run did not finish.** A cluster filesystem outage killed the
   job at epoch ~195 of 200 (`FAILED`, exit 120). The shipped weights are the
   selected epoch-124 checkpoint, written hours earlier and verified intact
   afterwards; the two later evals were *worse* (64.69 at 149, 65.21 at 174), so
   selection had already peaked. Nothing was recovered from the lost epochs.
3. **Selection is on reconstruction, not on perturbation quality.**
   `best_decoder.pth` is chosen by combined Pearson/Spearman/mse_inv with
   `eval_present_genes_only: true`. cell-eval takes hours per run, so per-epoch
   selection on it is not implementable. This project has a documented case of
   those two criteria ranking decoders *oppositely* (3072×10 wins
   reconstruction, loses virtual-cell transfer 6/6).
4. **Softplus output is unbounded and never exactly zero.** Any consumer
   relying on the old `log1p(2000)` ceiling or on exact zeros as "gene not
   detected" needs changing.
5. **`model_state_dict` is the decoder, not the model** — byte-identical to
   `decoder_state_dict`. Use `occam.inference.loader.load_encoder_wrapper`.
6. **The stage-1 recipe was misdocumented for four months.** Two wandb runs claim
   the same experiment directory. The weights are `nb_softplus` with an
   MLW ramp of 0.7 → 0.9, not the
   `poisson` / 0.0 → 0.7 that was published. The parameter counts corroborate:
   stage 1 carries one extra 44,728-length tensor, the per-gene NB
   dispersion, that stage 2's poisson run does not.
7. **Pretraining-corpus caveats**: 43.6% of rows are non-integer, 2.4% are bulk
   RNA-seq, and 7 of 9 constituent datasets are unidentified. Because of the
   last point, no claim in this card can assert that any evaluation cell line is
   an *unseen* context.
8. **No encoder benchmark numbers are quoted here.** The figures previously
   published for stage 1 belong to the *other* run that claimed that directory —
   weights never used downstream. Re-measure with `occam-benchmark`.
9. **Superseded-decoder defects, retained for the record**, since it is still in
   the repository under `checkpoints/legacy/`: it was not the best arm of its own
   sweep (five array tasks wrote one directory; of the three survivors it had the
   *worst* validation loss); its tracking record was deleted and the
   `run_config.json` beside it belongs to a different task; it does not record
   its output activation; and its rectified-sigmoid head leaves 83.6% of genes
   constant across cells. See `docs/reproducing_chain_a.md`.

## Load it

```python
import torch
from occam.inference.loader import load_encoder_wrapper

w = load_encoder_wrapper("occam_small_encoder_decoder.pth", device="cuda")
# w.model.feature(x) -> (n_cells, 512) embeddings
# x must be library-normalised to 2000.0 then log1p, in w._gene_names order
```

Both halves load `strict=True`. The golden **latent** reproduces bit-exactly
under torch 2.4.1+cu121, which is what pins this encoder to the weights the
port was verified against. Across torch versions expect ~1e-6 drift.

## Provenance chain

```
mlw_nb_softplus_s0.7_e0.9_cd0.0
  └─ occam_small_stage1_pretrain.pth   sha256 21c519b1ad686a01…
       └─ ft_mlw_B_ramp_lr0.0006
            └─ occam_small_stage2_finetune.pth   sha256 9ea9d48dec462c63…
                 ├─ occam_small_decoder_softplus_nb  (epoch 124, RELEASED)
                 │    └─ occam_small_encoder_decoder.pth   sha256 ef82433f8a6e…
                 └─ decoder_lr_sweep_3e-6_ep200  (SUPERSEDED, not published)
                      └─ legacy/occam_small_encoder_decoder_rectified_sigmoid.pth
                           sha256 22c05a18e1fd29c4…
```

Both decoders hang off the same stage-2 encoder, which is why the encoder half
is byte-identical between them.
