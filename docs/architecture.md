# Architecture

## The masked autoencoder

```
input (num_genes)
   │
   ├─ Dropout → Linear → LayerNorm → Mish        ×3          Encoder
   │                                                          │
   ▼                                                          ▼
latent (hidden_size)                                     512 for `small`
   │
   ├────────────────────────────┐
   ▼                            ▼
Mask predictor              [latent ‖ mask_pred]
Linear → Sigmoid                  │
(num_genes)                       ▼
                          Linear → LayerNorm → Mish → Linear
                                  │
                                  ▼
                          reconstruction (num_genes)
```

Two design choices distinguish it:

**All genes, not a highly-variable subset.** The encoder's input width is the
full vocabulary of the training corpus (44,728 for the released model), so the
gene axis is fixed at training time and is part of the checkpoint contract.

**Swap noise, not zero masking.** Masked entries are replaced with values drawn
from the same gene in other cells, not with zeros. A zero is a plausible
observation in this data, so zero-masking cannot be distinguished from a real
dropout — which is also why the mask predictor is a supervised head rather than
an inference.

Loss: `(1 − mask_loss_weight) · recon + mask_loss_weight · mask_bce`, everything
in log1p space. `masked_data_weight` sets how much of the reconstruction term
comes from masked versus unmasked genes.

### Sizes

| size | encoder | hidden | MAE decoder hidden | encoder | mask pred. | decoder | total |
|---|---|---|---|---|---|---|---|
| small | 44728 → 2048 → 1024 → 512 | 512 | 1024 | 94,235,136 | 22,945,464 | 92,175,032 | **209,355,632** |
| base | 44728 → 4096 → 2048 → 768 | 768 | 1536 | 193,188,096 | 34,395,832 | 138,633,400 | **366,217,328** |
| large | 44728 → 8192 → 4096 → 1024 | 1024 | 2048 | 404,200,448 | 45,846,200 | 185,353,912 | **635,400,560** |

Counts are at 44,728 genes and exclude loss-module parameters such as the
per-gene NB dispersion.

The MAE's own decoder scales with model size (`2 × hidden_size`). It was once
fixed at 1024 for every size, which forced a `large` encoder through the same
reconstruction bottleneck as a `small` one. Because `2 × 512 == 1024`, **`small`
is bit-identical to every pre-scaling checkpoint**, which is why the released
weights load unchanged.

Reloading resolves the width in a fixed order — the config key, then inferred
from the decoder's first weight, then a legacy default of 1024 — and **never**
from the size preset, since a pre-scaling `base` checkpoint used 1024 regardless.

## The external decoder

A separate, larger decoder trained on the frozen encoder's embeddings.
Deliberately decoupled from `model_size`, so a small encoder can be paired with
a large decoder and vice versa — which is what a decoder-capacity ablation needs.

```
latent (512) → Linear (hidden_dim)
                  │
                  ├──────────────────────── global skip ─────┐
                  ▼                                          │
        SE residual block  × num_blocks                      │
                  ▼                                          │
                  └──────────────── + ──────────────────────┘
                                    ▼
              BatchNorm → GELU → Linear → activation → (num_genes)
```

| preset | hidden | blocks | params @ 512-in, 44,728-out |
|---|---|---|---|
| small | 1024 | 3 | 53.1 M |
| base | 2048 | 4 | 128.4 M |
| large | 3072 | 6 | 259.5 M |
| xlarge | 4096 | 8 | 470.8 M |

The **released** decoder is 2048 × **5** (137.4 M), which is not a preset.
Resolution order is explicit flag > preset > legacy default (2048 × 4).

### Output activation

Two options, and the choice matters more than it looks.

`rectified_sigmoid` — `upper_bound · relu(2·sigmoid(x/2e) − 1)`. Bounded above
by `log1p(2000) ≈ 7.6`, and **exactly zero for every non-positive
pre-activation, with exactly zero gradient there.** Genes that are rarely
detected get pinned and never recover: on the released checkpoint, 83.6% of the
44,728 genes come out constant across cells.

`softplus` — strictly positive, nowhere-zero gradient, unbounded above.

The released model uses `rectified_sigmoid`.
`configs/decoder_softplus_nb.yaml` prepares the replacement, paired with a
negative-binomial loss: softplus with smooth-L1 showed ~38% validation-loss
spikes, while softplus with NB trained smoothly to epoch 200, so the likelihood
appears to tame the unbounded output.

⚠️ The activation holds no parameters, so **a decoder trained with one loads
`strict=True` under the other** and silently runs the wrong nonlinearity.
Checkpoints Occam writes record the field; for older ones,
`scripts/verify_checkpoint.py` determines it empirically.

### Geometry: do not scale it on reconstruction alone

Wider and deeper decoders win decoder-only reconstruction metrics — 3072 × 10
beats 2048 × 4 on `pearson_delta` on both panels — and **lose** virtual-cell
transfer by 1.6–2.9 standard deviations even after their own learning-rate
sweep, negative in 6 of 6 seed-paired arms, with one arm diverging 3/3 with
`loss=nan` from epoch 1.

Decoder-only quality does not predict transfer. Every panel checkpoint uses
2048 × 4.

## The perturbation model

`occam.vcell` composes the frozen encoder, a perturbation embedding, and the
external decoder:

```
control cell ──▶ encoder ──▶ z_ctrl ──┐
                                      ├──▶ interaction ──▶ z_pert ──▶ decoder ──▶ predicted cell
perturbation ──▶ graph/adjacency ──▶ e_pert ──┘
```

The interaction is FiLM by default. The perturbation embedding comes either from
a GNN over prior knowledge graphs or from a learned adjacency map.

## Data pipeline

1. `.h5ad` with raw counts in `X`
2. `NormalizePerCell(target_sum)` — 2000 for the released model, not the 1e4 default
3. `Log1p`
4. optional z-scoring
5. swap-noise masking

Three loading modes: `in_memory` (fast, ~245 GB for a large corpus), `backed`
(low memory, slow), `optim` (chunked HDF5 with prefetching — what the released
model used).

In `optim` mode the normalise and log1p steps are **deferred to the GPU**:
datasets yield raw rows and the transforms run after `.to(device)` but before
masking. Any new code path consuming optim-mode data must call
`_apply_gpu_transforms`, or it will train on raw counts.
