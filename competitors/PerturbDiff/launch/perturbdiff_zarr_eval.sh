#!/usr/bin/env bash
#SBATCH --job-name=pd_eval
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/eval_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/eval_%A_%a.out
#SBATCH --time=0-04:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=100G
#SBATCH --array=0-3
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# cell-eval scoring for each run of perturbdiff_zarr_train.sh, on the FULL test
# split (sampling.num_sampled_batches=null -> 26,878 perturbed + 4,976 control
# = 31,854 rows, exactly STATE's).
#
# Submit with a dependency so each eval waits for its training task:
#   sbatch --dependency=aftercorr:<TRAIN_JOBID> perturbdiff_zarr_eval.sh
# aftercorr pairs array element N with training element N.
#
# Writes <out>/eval/<run>/cell_eval_<ts>/<CellType>_results.csv, the same metric
# set STATE's tx predict produces. cell-eval is pinned to 0.7.2 to match STATE's
# installed version.
#
# NOTE: real controls are injected into both adatas, i.e. STATE's
# --inject-real-controls convention -- compare against STATE runs using it.

source /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff/launch/_pd_common.sh
cd "${PD_ROOT}"
mkdir -p "${PD_OUT}/logs"

pd_combo "${SLURM_ARRAY_TASK_ID}"
# Score the checkpoint the TRAINING job actually evaluated, not simply the final weights.
# run_pipeline writes that path to <run_name>.best_ckpt_path: it is ModelCheckpoint's
# best_model_path when a monitored validation metric fired, and the final weights otherwise. Without
# this the eval job would silently score a different model than the one selected on validation.
MARKER="${PD_OUT}/runs/${PD_RUN_NAME}.best_ckpt_path"
if [ -n "${PD_CKPT_OVERRIDE:-}" ]; then
  CKPT="${PD_CKPT_OVERRIDE}"
  echo "using PD_CKPT_OVERRIDE=${CKPT}"
elif [ -f "${MARKER}" ]; then
  CKPT="$(tr -d '\n' < "${MARKER}")"
  echo "using the checkpoint recorded by training: ${CKPT}"
else
  CKPT="${PD_OUT}/runs/${PD_RUN_NAME}"
  echo "no ${MARKER}; falling back to the final weights: ${CKPT}" >&2
fi
[ -e "${CKPT}" ] || { echo "checkpoint does not exist: ${CKPT}" >&2; exit 1; }
echo "=== task ${SLURM_ARRAY_TASK_ID}: eval ${PD_RUN_NAME} ==="
echo "    checkpoint=${CKPT}"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: no final checkpoint at ${CKPT}; did training finish?" >&2
  exit 1
fi

"${PD_PY}" ./src/apps/run/rawdata_diffusion_sampling.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  cov_encoding.replogle_gene_encoding="${PD_PENC}" \
  run_name="${PD_RUN_NAME}_eval" \
  model_checkpoint_path="${CKPT}" \
  device=cuda trainer.devices=[0] \
  optimization.micro_batch_size="${PD_MICRO_BATCH}" \
  optimization.seed="${PD_SEED}" \
  data.use_cell_set="${PD_CELL_SET}" data.num_workers="${PD_WORKERS}" \
  data.cell_set_group_by="${PD_GROUP}" \
  sampling.num_sampled_batches=null sampling.progress=false \
  sampling.run_cell_eval=true \
  sampling.output_dir="${PD_OUT}/eval/${PD_RUN_NAME}" \
  "${NO_WANDB[@]}"

echo "=== EVAL DONE: ${PD_OUT}/eval/${PD_RUN_NAME} ==="
ls -la "${PD_OUT}/eval/${PD_RUN_NAME}"/cell_eval_*/ 2>/dev/null || true
