#!/usr/bin/env bash
#SBATCH --job-name=pd_fewshot_eval
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/fewshot_eval_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/fewshot_eval_%A_%a.out
#SBATCH --time=0-06:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=100G
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# TEST cell-eval for the Replogle Nadig FEW SHOT arm, from the validation-selected checkpoint.
#
# WHY THIS EXISTS. perturbdiff_fewshot_train.sh produces validation cell-eval and a test *loss*
# (trainer.validate iterates both loaders and logs validation_test_loss_epoch), but the only
# cell-eval in the training process is the callback, which hard-codes split="validation". So
# training alone yields ZERO test cell-eval metrics. This job is the test leg.
#
# WHY NOT REUSE perturbdiff_zarr_eval.sh. Two silent traps:
#   1. It never sets path.manifest_file, so it inherits the ORIGINAL manifest and would build
#      train/validation from the wrong fold. (The test ids are byte-identical between the two
#      split files, so the scored cells happen to be right -- but the fold is not.)
#   2. Its run name comes from pd_combo, which can only emit `pd_noBatch_...`, never
#      `pd_fewshot_...`. The marker lookup would miss and it would fall back to a ZARR-ARM
#      checkpoint, reporting a different model entirely.
#
# A MISSING MARKER IS FATAL HERE, unlike the zarr script. For this protocol the marker is how the
# validation-selected checkpoint is identified; falling back to final weights would silently report
# a model that was never selected.
#
# CONTROLS: real measured controls, always. PerturbDiff works in gene space -- pca_for_decode is
# None at its only call site -- so there is no codec to reconstruct controls through and the
# "decoded" convention is not definable for it. That cell stays empty in cross-model tables.
#
#   sbatch --dependency=aftercorr:<TRAIN_JOBID> launch/perturbdiff_fewshot_eval.sh
# aftercorr pairs array element N with training element N, lining up tasks 0-4 with seeds 42-46.
# ============================================================================================

source /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff/launch/_pd_common.sh
cd "${PD_ROOT}"
mkdir -p "${PD_OUT}/logs"

SEED_BASE="${SEED_BASE:-42}"
PD_SEED=$(( SEED_BASE + ${SLURM_ARRAY_TASK_ID:-0} ))

MANIFEST="${PD_ROOT}/configs/manifests/ihabxcell_hepg2_fewshot_zarr.toml"
[ -f "${MANIFEST}" ] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }

# Pinned to match perturbdiff_fewshot_train.sh. Deliberately NOT pd_combo: that helper renames
# pert_celltype -> noBatch and cannot produce the fewshot prefix, so it would build a name that
# resolves to the wrong run's artifacts.
PD_GROUP="${PD_GROUP:-pert_celltype}"
PD_LR="${PD_LR:-0.002}"
PD_PENC="${PD_PENC:-genept}"
# 16, not 32: ZARR_BENCHMARK.md records the best config as use_cell_set=16 (0.4104 +/- 0.0233,
# n=4), while cs32 scores 0.3857. The median (pert, cell_type) group holds 53 cells, so cs16 fills
# from real cells whereas cs32 needs replacement-padding. An earlier fewshot round mistakenly ran
# cs32 and reproduced 0.38303 +/- 0.0229 -- correct for cs32, but not PerturbDiff's best setup.
PD_CELL_SET="${PD_CELL_SET:-16}"
PD_RUN_NAME="pd_fewshot_${PD_GROUP}_lr${PD_LR}_${PD_PENC}_cs${PD_CELL_SET}_s${PD_SEED}"

MARKER="${PD_OUT}/runs/${PD_RUN_NAME}.best_ckpt_path"
if [ -n "${PD_CKPT_OVERRIDE:-}" ]; then
  CKPT="${PD_CKPT_OVERRIDE}"
  echo "using PD_CKPT_OVERRIDE=${CKPT}"
else
  if [ ! -f "${MARKER}" ]; then
    echo "ERROR: no marker at ${MARKER}." >&2
    echo "Training either has not finished or its selection step failed. Refusing to score the" >&2
    echo "final weights instead -- that would report a model that was never selected." >&2
    echo "Override deliberately with PD_CKPT_OVERRIDE=<path> if you know what you want." >&2
    exit 1
  fi
  CKPT="$(tr -d '\n' < "${MARKER}")"
  echo "checkpoint selected on validation: ${CKPT}"
fi
[ -e "${CKPT}" ] || { echo "ERROR: checkpoint does not exist: ${CKPT}" >&2; exit 1; }

# Hydra cannot parse '=' inside an override value, and Lightning's filename template here is
# "{epoch}-{step}-{validation_random_Perturbseq_loss:.2f}", which produces
# epoch=15-step=80000-validation_random_Perturbseq_loss=0.00.ckpt -- three of them. Pass a
# symlink with an '='-free name instead.
SAFE="/tmp/pd_fewshot_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.ckpt"
ln -sfn "${CKPT}" "${SAFE}"
trap 'rm -f "${SAFE}"' EXIT

OUT_DIR="${PD_OUT}/eval_fewshot/${PD_RUN_NAME}"
echo "=== PerturbDiff fewshot TEST eval ==="
echo "    seed=${PD_SEED}  run=${PD_RUN_NAME}"
echo "    ckpt=${CKPT}"
echo "    out=${OUT_DIR}"
echo "    node=$(hostname)"

# generate_samples defaults to split="test", so this scores the test split; num_sampled_batches=null
# covers all of it (26,878 perturbed + 4,976 control = 31,854 rows).
"${PD_PY}" ./src/apps/run/rawdata_diffusion_sampling.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  path.manifest_file="${MANIFEST}" \
  cov_encoding.replogle_gene_encoding="${PD_PENC}" \
  run_name="${PD_RUN_NAME}_eval" \
  model_checkpoint_path="${SAFE}" \
  device=cuda trainer.devices=[0] \
  optimization.micro_batch_size="${PD_MICRO_BATCH}" \
  optimization.seed="${PD_SEED}" \
  data.use_cell_set="${PD_CELL_SET}" data.num_workers="${PD_WORKERS}" \
  data.cell_set_group_by="${PD_GROUP}" \
  sampling.num_sampled_batches=null sampling.progress=false \
  sampling.run_cell_eval=true \
  "sampling.output_dir=${OUT_DIR}" \
  "${NO_WANDB[@]}"

echo "=== EVAL DONE: ${OUT_DIR} ==="
ls -la "${OUT_DIR}"/cell_eval_*/ 2>/dev/null | grep -E "agg_results|results.csv" || true
