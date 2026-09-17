#!/bin/bash
#SBATCH --job-name=txpert_fewshot_eval
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/fewshot_eval_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/fewshot_eval_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --time=0-04:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=200G
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# TEST evaluation for the TxPert fewshot arm, from the VALIDATION-selected checkpoint, scored
# under BOTH predicted-control conventions.
#
# ONE prediction pass, scored twice. `export_for_celleval` can only produce the inject-real
# convention -- it injects the real controls into pred AND real itself -- so this instead writes
# adata_pred.h5ad + adata_real.h5ad once (gspp/export_artifacts.py) and hands them to the shared
# scorer twice. Predicting twice into one directory is what left ../STATE's test_replogle_zarr/
# with two agg CSVs that survive only because someone renamed them by hand.
#
# THE TWO CONVENTIONS FOR TXPERT:
#   decoded ("reconstruct controls") -- control rows on the PRED side are MODEL OUTPUTS, produced
#                                      with perturbation id -1. TxPert already trains for this:
#                                      embed_cntr puts ctrl -> -1 in pert2id and control cells are
#                                      training targets, so this is in-distribution and needed no
#                                      training change.
#   real    ("inject controls")      -- the measured controls, swapped in by the scorer, so pred
#                                      and real share an identical baseline.
#
# Like STATE, this is NOT the PCA-decode sense of "reconstruct" used for hooke-tx -- TxPert emits
# a genuinely predicted control. There is no codec here.
#
# Validation selection used inject-real only, so both test numbers come from the same checkpoint
# and differ only in scoring convention.
#
# A MISSING MARKER IS FATAL. Under this protocol, no marker means selection failed; scoring the
# final weights instead would silently report a different model than the one that was chosen.
#
#   sbatch --dependency=aftercorr:<train_job_id> launch/txpert_fewshot_eval.sh
# ============================================================================================

TXPERT_ROOT="${TXPERT_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/TxPert}"
HOOKE_TX_DIR="${HOOKE_TX_DIR:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx}"
cd "${TXPERT_ROOT}"

SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
# TAG_SUFFIX keeps a smoke run out of the real run's directories. Without it a seed-42 smoke
# writes into ckpt/fewshot_s42 alongside the real result, and save_top_k=1 only prunes files it
# tracked itself -- which is how PerturbDiff ended up with a smoke checkpoint sitting next to a
# real one in the same dir.
TAG="fewshot_s${SEED}${TAG_SUFFIX:-}"
OUT_ROOT="/rxrx/scratch/sandbox/${USER}/txpert_out"
CKPT_DIR="${OUT_ROOT}/ckpt/${TAG}"
EVAL_DIR="${OUT_ROOT}/eval/${TAG}"
MARKER="${CKPT_DIR}/best_ckpt_path.txt"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"
export WANDB_DIR="${OUT_ROOT}/wandb/${TAG}_eval"
export WANDB__SERVICE_WAIT=300
mkdir -p "${WANDB_DIR}" "${EVAL_DIR}" "${OUT_ROOT}/logs"

[[ -f "${MARKER}" ]] || {
  echo "FATAL: no best-checkpoint marker at ${MARKER}." >&2
  echo "       Training either did not finish or published no validation score, so no" >&2
  echo "       checkpoint was selected. Refusing to score the final weights instead." >&2
  exit 1
}
CHECKPOINT="$(cat "${MARKER}")"
[[ -f "${CHECKPOINT}" ]] || { echo "FATAL: marker points at a missing file: ${CHECKPOINT}" >&2; exit 1; }

# Hydra cannot parse '=' inside an override value, and Lightning checkpoint names routinely
# contain them. The fewshot config sets auto_insert_metric_name=false and filename=best-ep{epoch}
# precisely so this one does not -- assert it rather than discover it as a parse error four hours
# into a dependency chain, which is how the PerturbDiff eval array lost all five tasks.
case "${CHECKPOINT}" in
  *=*) echo "FATAL: checkpoint path contains '=', which Hydra cannot accept as an override" >&2
       echo "       value: ${CHECKPOINT}" >&2
       exit 1 ;;
esac

echo "=== TxPert fewshot eval ==="
echo "    seed=${SEED}  tag=${TAG}"
echo "    checkpoint=${CHECKPOINT}"
echo "    eval_dir=${EVAL_DIR}"
echo "    node=$(hostname)"

# 1. One prediction pass -> row-aligned pred/real AnnData. Controls on the pred side are model
#    outputs, which IS the `decoded` convention; `real` is applied at scoring time.
#    mode=test loads the checkpoint; the config's test_after_fit/celleval.enable are both false,
#    so nothing scores the final weights or writes a single-convention result.
"${TXPERT_ROOT}/.venv/bin/python" -u main.py \
  --config-name=config-txpert-zarr-fewshot \
  mode=test \
  checkpoint_path="${CHECKPOINT}" \
  training.seed="${SEED}" \
  wandb.name="txpert-fewshot-eval-s${SEED}" \
  training.checkpoint.dirpath="${CKPT_DIR}" \
  export_artifacts.enable=true \
  export_artifacts.dir="${EVAL_DIR}" \
  export_artifacts.split=test

SCORER="${HOOKE_TX_DIR}/hooke_tx/eval/celleval_artifacts.py"

# TWO INTERPRETERS, on purpose. TxPert's venv runs TxPert. The shared scorer is a hooke-tx tool
# and needs hooke-tx's dependency set: pointing a competitor's python at it failed first on
# `No module named hooke_tx`, then -- once PYTHONPATH was fixed -- on `No module named loguru`.
# Rather than chase transitive deps into a venv that does not own them, score with hooke-tx's own
# interpreter, which already has loguru, cell_eval, anndata, numpy and pandas.
SCORER_PY="${SCORER_PY:-${HOOKE_TX_DIR}/.venv/bin/python}"
[[ -x "${SCORER_PY}" ]] || { echo "missing scorer interpreter: ${SCORER_PY}" >&2; exit 1; }

# The scorer defaults to --num-threads 16, which is MORE than this job's allocation. numba caps
# NUMBA_NUM_THREADS at the cgroup CPU count and then raises "The number of threads must be between
# 1 and 12" from inside pdex, after the artifacts are already written. `nproc` respects the cgroup
# affinity, which is the same reason gspp/celleval_eval.py carries effective_cpu_count().
THREADS="$(nproc)"
echo "    scoring threads=${THREADS}"

# Clear the manifests from BOTH levels before re-initializing. `score` writes one into each
# results dir as well as the artifact dir, and it refuses to run when a stored manifest does not
# match the current artifacts -- correctly, since that is how a number from one artifact set would
# get reported against another. A re-run after any change to the export therefore hits the guard
# unless the stale manifests go first. Only the manifests are removed, never the results.
rm -f -- "${EVAL_DIR}/celleval_manifest.json"
rm -f -- "${EVAL_DIR}"/results_*/celleval_manifest.json

# `--stored-pred-control-mode generated` declares what is IN the artifacts: model-predicted
# controls. The scorer refuses `--pred-control-mode generated` otherwise, which is the guard that
# stops a real-control artifact from being reported as a predicted-control number.
#
# Cell type is `hepg2` (lowercase) here, versus `HepG2` in STATE's outputs. Cosmetic, but it means
# the result filenames are not name-identical across arms.
PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" initialize \
  --artifact-dir "${EVAL_DIR}" \
  --pert-col target_gene \
  --control-pert empty \
  --cell-type-col cell_type \
  --stored-pred-control-mode generated \
  --profile full \
  --skip-metrics pearson_edistance clustering_agreement

# 2. Score twice into separate results dirs, so neither convention overwrites the other.
#    `generated` is reported as the `decoded` arm to match the column name the other arms use.
for MODE in generated real; do
  case "${MODE}" in
    generated) ARM=decoded ;;
    real)      ARM=real ;;
  esac
  RESULTS_DIR="${EVAL_DIR}/results_${ARM}"
  mkdir -p "${RESULTS_DIR}"
  echo "=== scoring controls=${MODE} -> ${RESULTS_DIR} (arm=${ARM}) ==="
  PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" score \
    --artifact-dir "${EVAL_DIR}" \
    --cell-type hepg2 \
    --pred-control-mode "${MODE}" \
    --num-threads "${THREADS}" \
    --results-dir "${RESULTS_DIR}"

  PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" summarize \
    --artifact-dir "${EVAL_DIR}" \
    --results-dir "${RESULTS_DIR}"
done

echo "=== EVAL DONE ==="
for ARM in decoded real; do
  echo "--- ${ARM} ---"; ls "${EVAL_DIR}/results_${ARM}" 2>/dev/null | head -5
done
