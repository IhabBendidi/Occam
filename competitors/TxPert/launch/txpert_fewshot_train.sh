#!/bin/bash
#SBATCH --job-name=txpert_fewshot
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/fewshot_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/fewshot_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --time=0-10:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=200G
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# TxPert on Replogle Nadig Few Shot, 5 seeds, under the uniform selection protocol.
#
# PROTOCOL (identical for every model family on this benchmark):
#   1. cell-eval on the whole 4,825-cell VALIDATION split every 10 epochs, final epoch included
#   2. keep the single best checkpoint by validation selection_score
#      = mean(pearson_delta, discrimination_score_l1), mode=max
#   3. run the TEST evaluation exactly once, from that checkpoint (txpert_fewshot_eval.sh)
#   4. 5 seeds, 42-46
#
# WHY THIS EXISTS. TxPert had no validation split on this benchmark. Upstream monitored
# `test_pearson_delta` -- selection on the test set -- and the vendored arm kept the last epoch
# instead, which is honest but leaves the 50-epoch budget unchosen. Worse, the validation hook
# scored the TEST loader on every epoch, so test metrics were visible throughout training. The
# fewshot config turns both off and selects on a real validation split.
#
# THE TEST NUMBER DOES NOT COME FROM THIS JOB. `test_after_fit` and `celleval.enable` are both
# false in the config, so this job reports no test metric at all -- deliberately, because the
# reported number must come from the selected checkpoint, not the final weights.
#
# SEEDS 42-46, not TxPert's historical 1-3, so the arm shares its seed set with hooke-tx, STATE
# and PerturbDiff. Seed noise on this benchmark (~0.009-0.016 sd) is larger than the sampling
# noise floor, so seeds must be matched to compare arms.
#
# Sizing: ~3 min/epoch on an H100 plus setup, so 50 epochs is ~4-5 h. The five validation passes
# each run cell-eval over 4,825 cells (~1 min), so budget ~10 h.
#
#   sbatch launch/txpert_fewshot_train.sh
#   EPOCHS=20 CHECK_VAL=10 sbatch --array=0 --export=ALL,EPOCHS,CHECK_VAL \
#     launch/txpert_fewshot_train.sh        # smoke: 2 validations, second one final
# ============================================================================================

# Absolute, like ../PerturbDiff/launch/_pd_common.sh: SLURM copies the batch script into its
# own spool directory, so `dirname "$0"` points at /cm/local/apps/slurm/var/spool, not here.
TXPERT_ROOT="${TXPERT_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/TxPert}"
cd "${TXPERT_ROOT}"

EPOCHS="${EPOCHS:-50}"
CHECK_VAL="${CHECK_VAL:-10}"

# The final epoch is validated only when epochs % check_val_every_n_epoch == 0, and a final epoch
# that is never scored can never be selected -- the run would silently be unable to choose its own
# last checkpoint. Assert rather than trust.
if (( EPOCHS % CHECK_VAL != 0 )); then
  echo "FATAL: epochs=${EPOCHS} is not a multiple of check_val_every_n_epoch=${CHECK_VAL}," >&2
  echo "       so the final epoch would never be validated or selectable." >&2
  exit 1
fi

SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
# TAG_SUFFIX keeps a smoke run out of the real run's directories. Without it a seed-42 smoke
# writes into ckpt/fewshot_s42 alongside the real result, and save_top_k=1 only prunes files it
# tracked itself -- which is how PerturbDiff ended up with a smoke checkpoint sitting next to a
# real one in the same dir.
TAG="fewshot_s${SEED}${TAG_SUFFIX:-}"
OUT_ROOT="/rxrx/scratch/sandbox/${USER}/txpert_out"

# uv's default cache is on NFS while the venv is on GPFS; unpacking torch across that
# boundary takes hours instead of a minute. See README_HOOKE_TX.md.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"
# Per-task WANDB_DIR plus a long service wait: sharing one WANDB_DIR across simultaneous
# wandb.init() calls hung a task indefinitely inside init on the lr sweep.
export WANDB_DIR="${OUT_ROOT}/wandb/${TAG}"
export WANDB__SERVICE_WAIT=300
mkdir -p "${WANDB_DIR}" "${OUT_ROOT}/logs"

CKPT_DIR="${OUT_ROOT}/ckpt/${TAG}"
VAL_DIR="${OUT_ROOT}/valscore/${TAG}"
mkdir -p "${CKPT_DIR}" "${VAL_DIR}"

echo "=== TxPert fewshot train ==="
echo "    seed=${SEED}  tag=${TAG}  epochs=${EPOCHS}  validate every ${CHECK_VAL}"
echo "    ckpt=${CKPT_DIR}"
echo "    valscore=${VAL_DIR}"
echo "    node=$(hostname)"

# `.venv/bin/python`, NOT `uv run python`: uv re-verifies the environment and locks the venv on
# every invocation, and two tasks hung indefinitely inside uv before python ever started.
#
# Keep every comment ABOVE this invocation. A `#` comment inside a backslash-continued command
# terminates it: that silently dropped 29 of 32 overrides on STATE, which then fell back to its
# defaults (name=debug) and auto-resumed an unrelated checkpoint, exiting 0.
"${TXPERT_ROOT}/.venv/bin/python" -u main.py \
  --config-name=config-txpert-zarr-fewshot \
  training.seed="${SEED}" \
  training.epochs="${EPOCHS}" \
  training.check_val_every_n_epoch="${CHECK_VAL}" \
  wandb.name="txpert-fewshot-s${SEED}" \
  training.checkpoint.dirpath="${CKPT_DIR}" \
  training.validation_celleval.output_dir="${VAL_DIR}"

echo "=== TRAIN DONE ==="
MARKER="${CKPT_DIR}/best_ckpt_path.txt"
if [[ -f "${MARKER}" ]]; then
  echo "selected checkpoint: $(cat "${MARKER}")"
else
  echo "WARNING: no marker at ${MARKER}; the eval leg will refuse to run" >&2
fi
ls -la "${CKPT_DIR}"
