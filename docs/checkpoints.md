# Checkpoints

Two shapes, and confusing them is a real failure mode.

## Plain MAE (stages 1 and 2)

```
config              the training config
model_state_dict    the WHOLE autoencoder: encoder + MAE decoder + mask predictor
gene_names          ordered vocabulary
dataset_info        includes target_sum
current_epoch, best_val_loss, training_history, scheduler_state_dict
```

Stage 2 adds `finetuning_config`, which records the checkpoint it started from —
the chain of custody, inside the artifact.

## Unified encoder + decoder (stage 3, and the release)

```
encoder_state_dict  the encoder half only (13 tensors)
encoder_config      {model_size, hidden_size, num_genes}
decoder_state_dict  the external decoder (81 tensors for the released 2048x4;
                    99 for the legacy 2048x5)
decoder_config      {input_dim, output_dim, hidden_dim, num_blocks, dropout_rate,
                     se_reduction, output_activation, ...}
model_state_dict    ⚠️ byte-identical to decoder_state_dict, NOT the model
gene_names, gene_to_index, target_sum, training_config
parent_checkpoint_path
is_unified_checkpoint: True
```

⚠️ **`model_state_dict` is the decoder.** Anything treating it as the whole model
gets a silent encoder-shaped hole. Use the loader:

```python
from occam.inference.loader import load_encoder_wrapper
w = load_encoder_wrapper(path, device="cuda")   # handles both shapes
```

## Loading strictly

`occam.model.OccamEncoder` holds exactly the encoder and the `pos_weight`
buffer, so its `state_dict()` keys are precisely the 13 in a unified
checkpoint's `encoder_state_dict`:

```python
from occam.model import OccamEncoder
encoder, info = OccamEncoder.from_checkpoint(ck, strict=True)
assert info["missing"] == [] and info["unexpected"] == []
```

This exists because loading an encoder-only state dict into a full
`AutoEncoder` cannot be strict — the autoencoder also owns `decoder.*` and
`mask_predictor.*` — so the research tree used `strict=False` and could not
distinguish "legitimately absent" from "wrong architecture".

## Fields that are easy to get wrong

| field | value in the release | why it matters |
|---|---|---|
| `decoder_config.num_blocks` | **4** | the code default; the legacy decoder is 5 and will not load at 4 |
| `target_sum` | **2000.0** | read from `dataset_info` before `config` |
| `skip_normalize` / `skip_log1p` | both False | inherit-then-OR: a CLI flag can only force a transform *off* |
| `decoder_config.output_activation` | **`softplus`** | recorded in the release; **absent** in the legacy file and older research checkpoints, where the default `rectified_sigmoid` makes a softplus decoder load cleanly and emit ~99% exact zeros |
| `mae_decoder_hidden` | absent → 1024 | resolved from the config key, then the state dict, then a legacy default — never from the size preset |

`scripts/verify_checkpoint.py` checks all of these. For the legacy file,
`--profile legacy` **determines the output activation empirically** rather than
trusting the default: under `rectified_sigmoid` 83.6% of genes come out constant
across cells, under softplus 0.0%.

## Provenance

Every checkpoint Occam writes records `parent_checkpoint_path`, its config hash
and the git SHA, so provenance travels with the artifact rather than living in a
neighbouring JSON file that another job can overwrite.

That is a direct response to how two of the three released checkpoints became
unattributable. See [reproducing_chain_a.md](reproducing_chain_a.md).

`checkpoints/MANIFEST.json` records sha256, size, parameter counts and the
**verified** config for each released file — read from the checkpoint itself,
never from a sidecar.
