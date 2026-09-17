#!/usr/bin/env bash
# Shared settings for the PerturbDiff zarr-benchmark runs.
# Sourced by the train / eval / canary sbatch scripts.
#
# Hyperparameters are upstream PerturbDiff's own Replogle recipe
# (README.md, COMMON_REPLOGLE_TRAIN): use_cell_set=32, micro_batch_size=128,
# lr=0.002, replogle_gene_encoding=genept -- NOT the PBMC/Tahoe 256/2048.
# Step budget and checkpoint cadence match ../STATE/state/train_state_zarr.sh
# (max_steps=80000, ckpt every 2000) so the two competitors are comparable.

PD_ROOT=/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff
PD_OUT=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out
PD_PY="${PD_ROOT}/.venv/bin/python"

# The grouping A/B. On this data the two differ ~13x in samples per epoch, so
# they are compared at a MATCHED STEP COUNT, never matched epochs.
# 4 runs by default: the grouping A/B is the question being asked, lr is the one
# secondary axis worth probing (changing the grouping changes the effective
# batch statistics a lot, so upstream's tuned lr may not transfer).
# Perturbation encoding is fixed to upstream's Replogle choice, genept, which
# resolves 1980/2023 (97.9%) after the Ensembl->symbol translation.
#
# Each axis is overridable from the environment so a follow-up sweep can reuse
# these scripts without editing them. Export in the submitting shell and use a
# bare --export=ALL:
#
#   export PD_GROUPS_CSV=pert_celltype PD_LRS_CSV=0.001,0.003,0.005
#   export PD_RUN_SUFFIX=_ema
#   sbatch --export=ALL --array=0-2%2 launch/perturbdiff_zarr_train.sh
#
# Do NOT inline the assignments as --export=ALL,VAR=a,b,c : sbatch splits that
# string on commas, so a value containing commas is silently truncated to its
# first element and the rest are read as (nonexistent) variable names. That
# turns a 5-point sweep into a 1-point one and makes the other array tasks die
# on `set -u`.
IFS=',' read -r -a PD_GROUPS <<< "${PD_GROUPS_CSV:-pert_celltype_batch,pert_celltype}"
IFS=',' read -r -a PD_LRS    <<< "${PD_LRS_CSV:-0.002,0.0005}"
IFS=',' read -r -a PD_PENCS  <<< "${PD_PENCS_CSV:-genept}"
# Cell-set size. Sweepable; when more than one value is given the run name gains
# a _cs<N> tag so a sweep cannot collide with the single-value runs already on
# disk. micro_batch_size must stay divisible by it (collate_fn reshapes to
# [B/S, S, G]).
IFS=',' read -r -a PD_CELL_SETS <<< "${PD_CELL_SETS_CSV:-${PD_CELL_SET:-32}}"
# Set PD_FORCE_CS_TAG=1 to emit the _cs<N> tag even for a single cell-set value.
# Needed when re-running one non-default cell set (e.g. a seed sweep at cs16):
# without it the name collides with the cs32 runs already on disk.
PD_FORCE_CS_TAG=${PD_FORCE_CS_TAG:-0}

# Random seed. pl.seed_everything(optimization.seed) in BOTH entrypoints, so a
# seed varies training init/ordering AND the diffusion noise + control pairing
# at sampling time -- i.e. the spread across seeds is the honest error bar for
# "rerun the whole pipeline", which is what every single-seed number here needs.
# The _s<N> tag appears only when more than one seed is given, so the existing
# seed-42 runs keep their names and count as a free extra replicate.
IFS=',' read -r -a PD_SEEDS <<< "${PD_SEEDS_CSV:-42}"

# EMA. Upstream's COMMON_TRAIN_RUNTIME sets decay=0.99 / update_steps=10 for
# every published recipe; configs/lightning/base.yaml ships decay=0, which
# disables the callback entirely (training_runtime_helpers.build_callbacks only
# adds EMAWeightAveraging when decay > 0). The first benchmark sweep ran without
# it -- an unintended deviation -- so it is explicit here.
PD_EMA_DECAY=${PD_EMA_DECAY:-0.99}
PD_EMA_UPDATE=${PD_EMA_UPDATE:-10}

# Appended to every run name. Set it when re-running a grid you already have
# results for, otherwise the outputs are overwritten.
PD_RUN_SUFFIX=${PD_RUN_SUFFIX:-}

PD_MAX_STEPS=${PD_MAX_STEPS:-80000}
PD_CKPT_EVERY=${PD_CKPT_EVERY:-2000}
PD_MICRO_BATCH=${PD_MICRO_BATCH:-128}
PD_WORKERS=${PD_WORKERS:-4}

NO_WANDB=(
  lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger
  '~lightning.logger.project' '~lightning.logger.save_dir' '~lightning.logger.name'
)

# combo index -> (pert-encoding, lr, group, cell-set, seed), in that nesting order
pd_combo() {
  local idx=$1
  local n_penc=${#PD_PENCS[@]}
  local n_lr=${#PD_LRS[@]}
  local n_grp=${#PD_GROUPS[@]}
  local n_cs=${#PD_CELL_SETS[@]}
  local n_seed=${#PD_SEEDS[@]}
  PD_PENC=${PD_PENCS[$(( idx % n_penc ))]}
  PD_LR=${PD_LRS[$(( (idx / n_penc) % n_lr ))]}
  PD_GROUP=${PD_GROUPS[$(( (idx / (n_penc * n_lr)) % n_grp ))]}
  PD_CELL_SET=${PD_CELL_SETS[$(( (idx / (n_penc * n_lr * n_grp)) % n_cs ))]}
  PD_SEED=${PD_SEEDS[$(( idx / (n_penc * n_lr * n_grp * n_cs) ))]}
  # Short, filesystem-safe run name. The _cs and _s tags appear only when that
  # axis is being swept, so single-value runs keep the names already on disk.
  local g_short=full ; [ "$PD_GROUP" = "pert_celltype" ] && g_short=noBatch
  local cs_tag="" ; { [ "$n_cs" -gt 1 ] || [ "$PD_FORCE_CS_TAG" = "1" ] ; } && cs_tag="_cs${PD_CELL_SET}"
  local s_tag="" ; [ "$n_seed" -gt 1 ] && s_tag="_s${PD_SEED}"
  PD_RUN_NAME="pd_${g_short}_lr${PD_LR}_${PD_PENC}${cs_tag}${s_tag}${PD_RUN_SUFFIX}"
}

pd_n_combos() {
  echo $(( ${#PD_GROUPS[@]} * ${#PD_LRS[@]} * ${#PD_PENCS[@]} * ${#PD_CELL_SETS[@]} * ${#PD_SEEDS[@]} ))
}
