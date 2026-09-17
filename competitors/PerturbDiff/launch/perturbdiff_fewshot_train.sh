#!/usr/bin/env bash
#SBATCH --job-name=pd_fewshot
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/fewshot_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/fewshot_%A_%a.out
#SBATCH --time=0-16:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=100G
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# PerturbDiff on Replogle Nadig FEW SHOT, 5 seeds, under the shared selection protocol.
#
# PROTOCOL (identical across model families, which is the whole point):
#   1. cell-eval on the VALIDATION split during training
#   2. keep the single best checkpoint by validation selection_score
#      = mean(pearson_delta, discrimination_score_l1), mode=max
#   3. evaluate exactly once, from that checkpoint
#   4. 5 seeds
#
# WHAT WAS WRONG BEFORE. The benchmark's original split file has an empty `validation` list, so
# PerturbDiff's val_dataloader for that split yielded nothing and its only live validation metric
# was `validation_test_loss_epoch` -- a loss, on TEST. Any checkpoint choice was a choice made on
# test. This points at split_replogle_nadig_fewshot__v1.json, which has 4,825 real validation cells.
#
# CADENCE. val_check_interval = PD_MAX_STEPS / PD_N_VALIDATIONS (default 5 validations), so the
# last validation lands exactly on the final step. The script refuses counts that do not divide,
# because the final checkpoint would otherwise never be scored and so could never be selected.
#
# ⚠ THE CHECKPOINT IS DELIBERATELY *NOT* ON `every_n_train_steps`. Lightning saves that variant in
# `on_train_batch_end`, which runs BEFORE the validation loop for the same step -- so it ranks using
# the PREVIOUS validation's score. A smoke run made this concrete: validation scores were 0.35278
# then 0.40559, and it kept the 0.35278 checkpoint, because at the final step the only score in
# callback_metrics was still the earlier one. With `every_n_train_steps=null` and
# `every_n_epochs=1`, ModelCheckpoint saves in `on_validation_end` instead -- after the metric has
# been published -- so each validation is ranked against its own score.
#
# (An epoch here spans the whole multi-cell-line training set, so current_epoch stays 0 and
# `every_n_epochs=1` does not gate anything; `on_validation_end` fires for every mid-epoch
# validation that `val_check_interval` triggers.)
#
# Validation covers the WHOLE validation split. Diffusion sampling is expensive, so the cost is
# controlled by validating RARELY rather than by sampling a subset -- truncating would add noise to
# the one signal that decides which checkpoint is kept.
#
# NO MODELLING CHANGE. PerturbDiff stays in gene space; pca_for_decode is None and no PCA is
# introduced. Because it has no latent codec, the reconstruct-controls ("decoded") convention does
# not apply to it -- controls are always the real measured cells. That cell stays empty in the
# cross-model table, by nature of the method rather than by omission.
#
#   sbatch launch/perturbdiff_fewshot_train.sh
#   PD_MAX_STEPS=20000 sbatch --export=ALL,PD_MAX_STEPS launch/perturbdiff_fewshot_train.sh
# ============================================================================================

# Absolute path, matching the other launchers: SLURM runs the batch script from a per-job spool
# copy, so BASH_SOURCE points at /cm/local/.../job<id>/ and cannot locate the repo.
# shellcheck disable=SC1091
source /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff/launch/_pd_common.sh
cd "${PD_ROOT}"

# One seed per array task, based at 42 so task 0 matches every other arm's first seed.
SEED_BASE="${SEED_BASE:-42}"
PD_SEED=$(( SEED_BASE + ${SLURM_ARRAY_TASK_ID:-0} ))

# Validation covers the WHOLE validation split (all 4,825 cells), but runs only a handful of times
# across the run. Diffusion sampling is the expensive part, so the cost is controlled by how OFTEN
# we validate, not by truncating what we validate -- a subset estimate would make the selection
# signal noisier exactly where it matters, and generate_samples logs a PARTIAL EVAL warning for it.
# Empty/unset => null => whole split.
PD_VAL_BATCHES="${PD_VAL_BATCHES-}"

# Number of validation passes over the whole run. The step cadence is derived from it so that
# val_check_interval and the checkpoint cadence stay equal by construction; picking them separately
# is how a monitored metric goes stale.
PD_N_VALIDATIONS="${PD_N_VALIDATIONS:-5}"
if [ "${PD_N_VALIDATIONS}" -lt 1 ]; then
  echo "PD_N_VALIDATIONS must be >= 1" >&2; exit 1
fi
if [ $(( PD_MAX_STEPS % PD_N_VALIDATIONS )) -ne 0 ]; then
  echo "PD_MAX_STEPS=${PD_MAX_STEPS} is not divisible by PD_N_VALIDATIONS=${PD_N_VALIDATIONS}." >&2
  echo "The last validation would not land on the final step, so the final checkpoint could" >&2
  echo "never be selected. Pick values that divide." >&2
  exit 1
fi
PD_VAL_EVERY=$(( PD_MAX_STEPS / PD_N_VALIDATIONS ))

MANIFEST="${PD_ROOT}/configs/manifests/ihabxcell_hepg2_fewshot_zarr.toml"
[ -f "${MANIFEST}" ] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }

# Single-arm config: the grouping/lr sweep axes are pinned to the values the zarr arm settled on,
# because this run is a protocol comparison across models, not a hyperparameter search.
PD_GROUP="${PD_GROUP:-pert_celltype}"
PD_LR="${PD_LR:-0.002}"
PD_PENC="${PD_PENC:-genept}"
# 16, not 32: ZARR_BENCHMARK.md records the best config as use_cell_set=16 (0.4104 +/- 0.0233,
# n=4), while cs32 scores 0.3857. The median (pert, cell_type) group holds 53 cells, so cs16 fills
# from real cells whereas cs32 needs replacement-padding. An earlier fewshot round mistakenly ran
# cs32 and reproduced 0.38303 +/- 0.0229 -- correct for cs32, but not PerturbDiff's best setup.
PD_CELL_SET="${PD_CELL_SET:-16}"

PD_RUN_NAME="pd_fewshot_${PD_GROUP}_lr${PD_LR}_${PD_PENC}_cs${PD_CELL_SET}_s${PD_SEED}"
CKPT_DIR="${PD_OUT}/ckpt/${PD_RUN_NAME}"
mkdir -p "${PD_OUT}/logs" "${CKPT_DIR}"

VAL_BATCH_OVERRIDE="lightning.callbacks.validation_celleval.num_sampled_batches=${PD_VAL_BATCHES:-null}"

echo "=== PerturbDiff fewshot ==="
echo "    seed=${PD_SEED}  steps=${PD_MAX_STEPS}  validations=${PD_N_VALIDATIONS}  every=${PD_VAL_EVERY} steps"
echo "    val_batches=${PD_VAL_BATCHES:-<whole split>}  manifest=$(basename "${MANIFEST}")"
echo "    run=${PD_RUN_NAME}"
echo "    node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PD_PY}" ./src/apps/run/rawdata_diffusion_training.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  path.manifest_file="${MANIFEST}" \
  cov_encoding.replogle_gene_encoding="${PD_PENC}" \
  run_name="${PD_RUN_NAME}" \
  trainer.devices=[0] trainer.max_steps="${PD_MAX_STEPS}" \
  trainer.num_sanity_val_steps=0 \
  trainer.check_val_every_n_epoch=null \
  trainer.val_check_interval="${PD_VAL_EVERY}" \
  trainer.limit_val_batches=1.0 \
  optimization.micro_batch_size="${PD_MICRO_BATCH}" \
  optimization.optimizer.lr="${PD_LR}" \
  optimization.seed="${PD_SEED}" \
  data.use_cell_set="${PD_CELL_SET}" data.num_workers="${PD_WORKERS}" \
  data.cell_set_group_by="${PD_GROUP}" \
  lightning.ema.decay="${PD_EMA_DECAY}" \
  lightning.ema.update_steps="${PD_EMA_UPDATE}" \
  lightning.callbacks.validation_celleval.enabled=true \
  lightning.callbacks.validation_celleval.interval=1 \
  "${VAL_BATCH_OVERRIDE}" \
  lightning.callbacks.checkpoint.monitor="validation/celleval/selection_score" \
  lightning.callbacks.checkpoint.mode=max \
  lightning.callbacks.checkpoint.save_top_k=1 \
  lightning.callbacks.checkpoint.every_n_train_steps=null \
  lightning.callbacks.checkpoint.every_n_epochs=1 \
  lightning.callbacks.checkpoint.dirpath="${CKPT_DIR}" \
  sampling.output_dir="${PD_OUT}/valscore/${PD_RUN_NAME}" \
  "${NO_WANDB[@]}"

echo "=== TRAIN DONE: ${PD_RUN_NAME} ==="
echo "--- checkpoint kept (save_top_k=1) ---"
ls -la "${CKPT_DIR}" || true
MARKER="${PD_OUT}/runs/${PD_RUN_NAME}.best_ckpt_path"
[ -f "${MARKER}" ] && echo "--- evaluated checkpoint recorded: $(cat "${MARKER}") ---"
