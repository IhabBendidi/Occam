# Training

```bash
python -m occam.train --config configs/occam_small.yaml
```

Three stages, chained through checkpoint files on disk.

## Why files, not memory

The research tree also had a "tandem" mode that handed the live model from
pretraining straight into finetuning. The released weights did **not** come from
it, and the two paths are not equivalent:

1. Standalone loads with `strict=False`, which drops the NB dispersion parameter
   and re-initialises the loss module. Tandem keeps the live module in memory
   until finetuning swaps it. Different parameter set at step 0.
2. Standalone resets `best_val_loss` and starts a fresh run directory.
3. Tandem had a latent double-transform bug: with a different finetuning batch
   size it rebuilt the dataset with CPU transforms while GPU deferral was still
   on, applying normalise+log1p twice.

So the orchestrator reproduces the standalone path only.

## The ledger

`runs/<name>/ledger.json` records, per stage, its status, artifacts, config
hash, git SHA and parent checkpoint.

A stage is **reused** only if all four hold:

* it completed
* its per-stage config hash is unchanged
* its parent checkpoint is unchanged
* its artifacts still exist on disk

Otherwise it **blocks**, with a field-level diff of what changed. It does not
silently reuse weights trained under different hyperparameters — that is the
failure this repo exists to prevent.

`init_from` is deliberately excluded from the stage hash. It begins as a
deferred `${stages.X.artifacts.Y}` reference and becomes a path once upstream
runs, so including it would make every re-run look like a change. The parent is
tracked separately, which also lets the tool say *which* of the two things
changed.

```bash
--resume auto    # default: reuse what is valid
--resume never   # fail if anything already exists
--resume force   # re-run, archiving the old stage directory
```

## Run directories

`runs/<config_name>/<stage>/` — deterministic, no global counter, and the
orchestrator **refuses to write into a populated directory**.

The research tree allocated `exp_<NNN>` from a shared registry and named decoder
runs by timestamp alone. Both collided between concurrent array tasks, and two
of the three released checkpoints became unattributable as a result. See
[reproducing_chain_a.md](reproducing_chain_a.md).

## Stage 2 inherits from the checkpoint, not the config

`FinetuneConfig.resolve()` overlays the finetuning config's non-`None` fields
onto a config rebuilt from the **parent checkpoint**. So stage 2 inherits
`target_sum`, `corruption_mode`, `model_size`, `mae_decoder_hidden` and `scale`
from the weights it is finetuning, not from the `pretrain:` block of the same
YAML file.

That is what produced the released chain, so it is preserved — and made visible.
Every stage-2 run writes `runs/<name>/finetune/config/overlay_audit.json`
recording, per field, what the YAML said, what was inherited, and which won:

```json
"recon_loss": {
  "maps_to": "reconstruction_loss",
  "yaml_value": "poisson",
  "inherited_from_parent": "nb_softplus",
  "effective": "poisson",
  "source": "config"
}
```

Architecture-affecting mismatches are fatal; everything else warns.

## The gradual-unmasking trap

Stage 2 sets `mask_probability: 0.0`. **It is inert.** Under
`unmasking_mode: gradual` the ramp starts from the *parent checkpoint's*
`mask_probability` (0.1), so masking is live for most of the finetune.

Ground truth: the original stage-2 log records `mask_p: 0.100` at epoch 1. The
orchestrator prints the effective value at stage start, the config carries a
read-only `effective_mask_prob_source` field, and `tests/test_schedules.py`
fails if this ever changes.

## Resume within a stage

Stages checkpoint optimizer, scheduler, epoch and best-so-far at every eval
epoch. Both schedules are pure functions of `(epoch, total_epochs)` and the
chunk shuffle is epoch-seeded, so they resume exactly.

RNG state is now saved too — but **checkpoints written before that are not
bit-exact on resume**, only approximately so.

## Smoke test

```bash
python scripts/make_smoke_corpus.py --out /tmp/occam_smoke.h5ad
python -m occam.train --config configs/occam_small_smoke.yaml \
    --set data.train=/tmp/occam_smoke.h5ad
```

~1 minute on CPU, 4,096 cells × 512 genes, all three stages. Exercises the
chunked reader, both loss modules, the schedules, the stage fence, the ledger,
artifact hand-off and the unified-checkpoint writer. Not a scientific
configuration.
