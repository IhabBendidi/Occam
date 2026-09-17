#!/usr/bin/env bash
#SBATCH --job-name=txpert_ctrl
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/ctrl_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/ctrl_%A_%a.out
#SBATCH --time=0-10:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=120G
#SBATCH --array=0-5
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# Control-conditioning ablation for TxPert (IHABXCELL_CELLEVAL_SCORES.md section 31.4, caveat 1).
#
# QUESTION: TxPert beats every other arm on the headline (pearson_delta 0.5092 vs hooke-tx 0.4795,
# STATE 0.4313). It is also handed the *mean of the controls from each cell's own (cell_type,
# batch)* as its basal input. How much of the win is that batch baseline rather than the model?
#
# `_get_control_data` (gspp/data/datamodule.py) picks the basal input four ways:
#
#   match_cntr  avg_cntr   basal input given to the model
#   ----------  --------   ------------------------------------------------------------
#   true        true       mean of controls in the SAME (cell_type, batch)   <- section 31 baseline
#   false       true       mean of ALL controls of that cell type            <- arm B here
#   false       false      one random control cell of that cell type         <- arm C here
#   true        false      one random control cell from the same batch       (not run)
#
# Arm B removes batch information but keeps a clean averaged baseline. Arm C additionally removes
# the averaging, so the model sees a single noisy control -- closest to "generic control cell".
# Baseline (match=true, avg=true) is section 31 and is NOT re-run; its 3 seeds already exist.
#
# Everything else is identical to section 31: lr 3e-4, 50 epochs, same split, same seeds 1-3, and
# the same cell-eval protocol. `match_cntr_for_eval` stays true in every arm so the internal
# metric is computed the same way and only the MODEL's input changes. cell-eval is unaffected
# either way -- it injects real controls into pred and real itself.
#
# Read the result as: baseline_minus_armB = the value of batch-matched controls. If arm B lands
# near hooke-tx's 0.4795, the section 31.3 ranking is largely a control-information artefact and
# must be re-framed. If arm B stays near 0.50, the ranking stands on the model.
#
# Sizing: identical to the LR sweep -- ~2 h 35 m per run on an H100.

TXPERT_ROOT="${TXPERT_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/TxPert}"
cd "${TXPERT_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"

# 0,1,2 -> arm B (nomatch-avg) ; 3,4,5 -> arm C (nomatch-single)
MATCH=(false false false false false false)
AVG=(true true true false false false)
ARM=(nomatch-avg nomatch-avg nomatch-avg nomatch-single nomatch-single nomatch-single)
SEEDS=(1 2 3 1 2 3)

IDX=${SLURM_ARRAY_TASK_ID:-0}
M=${MATCH[$IDX]}; A=${AVG[$IDX]}; SEED=${SEEDS[$IDX]}
TAG="ctrl-${ARM[$IDX]}-seed${SEED}"

# Per-task wandb tree: nine simultaneous wandb.init() calls against one WANDB_DIR hung a task.
export WANDB_DIR="${WANDB_DIR:-/rxrx/scratch/sandbox/$USER/txpert_out/wandb}/ctrl${IDX}"
export WANDB__SERVICE_WAIT=300
mkdir -p "$WANDB_DIR" /rxrx/scratch/sandbox/"$USER"/txpert_out/logs

echo "[$(date -Is)] task ${IDX}: match_cntr=${M} avg_cntr=${A} seed=${SEED} tag=${TAG}"

# `-u` so progress is visible in the log (stdout is fully buffered to a file otherwise), and
# `.venv/bin/python` rather than `uv run` (uv re-locks the venv per invocation and has hung).
"${TXPERT_ROOT}/.venv/bin/python" -u main.py \
  --config-name=config-txpert-zarr-adjmap \
  datamodule.match_cntr="${M}" \
  datamodule.avg_cntr="${A}" \
  training.lr=0.0003 \
  training.seed="${SEED}" \
  wandb.name="txpert-zarr-adjmap-${TAG}" \
  training.checkpoint.dirpath="$PWD/cache/checkpoints/txpert-zarr-adjmap/${TAG}" \
  celleval.output_dir="$PWD/outputs/celleval/${TAG}"
