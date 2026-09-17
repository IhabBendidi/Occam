#!/bin/bash
#SBATCH --job-name=state_fewshot_pred
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/state_out/logs/fewshot_pred_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/state_out/logs/fewshot_pred_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=20
#SBATCH --mem=320G
#SBATCH --time=0-08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# TEST evaluation for the STATE fewshot arm, from the VALIDATION-selected checkpoint, scored
# under BOTH predicted-control conventions.
#
# ONE prediction pass, scored twice. `state tx predict` derives its results_dir solely from
# --output-dir plus the checkpoint basename and does not expose an override, so running predict
# twice would write both conventions into the same directory and the second would overwrite the
# first -- exactly what happened in test_replogle_zarr/, where the two agg CSVs survive only
# because someone renamed them by hand. Instead: predict once with --predict-only, then score the
# artifacts twice with the shared hooke-tx scorer into separate results dirs.
#
# THE TWO CONVENTIONS FOR STATE:
#   generated ("reconstruct controls") -- STATE's DEFAULT. Control rows are model outputs: they are
#                                        members of the split and go through predict_step like any
#                                        other row. There is no flag for this; it is what you get.
#   real      ("inject controls")      -- real measured controls on the predicted side, so both
#                                        adatas share an identical real baseline.
#
# Note this is NOT the PCA-decode sense of "reconstruct" used for hooke-tx; STATE genuinely emits a
# predicted control. The shared scorer rejects --reconstruct-pred-controls for STATE-shaped
# artifacts for exactly that reason.
#
# Validation selection used inject-real only, so both test numbers come from the same checkpoint
# and differ only in scoring convention.
#
#   sbatch --dependency=aftercorr:<TRAIN_JOBID> predict_state_fewshot.sh
# ============================================================================================

# SLURM runs the batch script from a per-job spool copy, so BASH_SOURCE points at
# /cm/local/.../job<id>/ and cannot locate the checkout. Try the plausible roots and take the first
# that actually contains this package -- and hardcode nothing, because train_state_zarr.sh:24 still
# points at notebooks/STATE/state, which no longer exists.
if [ -z "${STATE_DIR:-}" ]; then
  for _cand in \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
    "/rxrx/scratch/sandbox/${USER}/bc-hooke/src/hooke-tx/competitors/STATE/state"
  do
    if [ -n "${_cand}" ] && [ -f "${_cand}/pyproject.toml" ] && [ -d "${_cand}/src/state" ]; then
      STATE_DIR="${_cand}"; break
    fi
  done
fi
if [ -z "${STATE_DIR:-}" ] || [ ! -d "${STATE_DIR}/src/state" ]; then
  echo "Could not locate the STATE checkout. Set it explicitly:" >&2
  echo "  STATE_DIR=/path/to/competitors/STATE/state sbatch --export=ALL $0" >&2
  exit 1
fi
cd "${STATE_DIR}"

HOOKE_TX_DIR="$(cd "${STATE_DIR}/../../.." && pwd)"
SCORER="${HOOKE_TX_DIR}/hooke_tx/eval/celleval_artifacts.py"
TOML="${STATE_DIR}/ihabxcell_hepg2_fewshot_zarr.toml"

SEED_BASE="${SEED_BASE:-42}"
SEED=$(( SEED_BASE + ${SLURM_ARRAY_TASK_ID:-0} ))
OUTPUT_DIR="${OUTPUT_DIR:-/rxrx/scratch/sandbox/${USER}/state_out/replogle_nadig_fewshot}"
RUN_NAME="fewshot_s${SEED}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
# The checkpoint selected on validation cell-eval, NOT `best.ckpt` (which is val_loss-selected) and
# not `snapshot_step.ckpt` (which predict_state_zarr.sh names and which does not exist anywhere).
CHECKPOINT="${CHECKPOINT:-best_celleval.ckpt}"
PROFILE="${PROFILE:-full}"
EVAL_DIR="${RUN_DIR}/eval_${CHECKPOINT}"

[[ -f "${SCORER}" ]] || { echo "missing scorer: ${SCORER}" >&2; exit 1; }
[[ -f "${TOML}"   ]] || { echo "missing toml: ${TOML}" >&2; exit 1; }
if [[ ! -f "${RUN_DIR}/checkpoints/${CHECKPOINT}" ]]; then
  echo "ERROR: no ${CHECKPOINT} in ${RUN_DIR}/checkpoints/" >&2
  echo "Training either has not finished or the validation metric never fired, so no checkpoint" >&2
  echo "was selected. Refusing to score best.ckpt or last.ckpt instead -- those are selected on" >&2
  echo "val_loss and on nothing respectively, and would not be the model this protocol chose." >&2
  ls -la "${RUN_DIR}/checkpoints/" >&2 2>/dev/null || true
  exit 1
fi

echo "=== STATE fewshot TEST eval ==="
echo "    seed=${SEED}  run=${RUN_NAME}  ckpt=${CHECKPOINT}  profile=${PROFILE}"
echo "    eval_dir=${EVAL_DIR}"
echo "    node=$(hostname)"

# HOW STATE IS INVOKED, and why it looks awkward.
#
# `.venv/bin/state` has a stale shebang pointing at the vanished
# notebooks/STATE/state/.venv/bin/python3, so running it -- including via `uv run state` -- fails
# with a pyenv exec error. The editable install's .pth is broken the same way, so plain
# `python -m state` reports "No module named state". Putting src on PYTHONPATH and running the
# module with the venv's own interpreter works and touches nothing.
#
# A `uv sync` on a login node would regenerate both correctly; until then this is the reliable route.
STATE_PY=("${STATE_DIR}/.venv/bin/python" -m state)
export PYTHONPATH="${STATE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset VIRTUAL_ENV
rm -f -- "${EVAL_DIR}/celleval_manifest.json"

# 1. One prediction pass. Controls stay as the model produced them, which is the `generated`
#    convention; the `real` convention is applied at scoring time without re-predicting.
"${STATE_PY[@]}" tx predict \
  --output-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --toml "${TOML}" \
  --profile "${PROFILE}" \
  --predict-only

PY="${STATE_DIR}/.venv/bin/python"   # .venv/bin/state has a stale shebang; python itself is fine

# TWO INTERPRETERS, on purpose.
#
# STATE's venv runs `state tx predict`. The shared scorer is a hooke-tx tool and needs hooke-tx's
# dependency set: pointing STATE's python at it failed first on `No module named hooke_tx`, then --
# once PYTHONPATH was fixed -- on `No module named loguru`. Rather than chase transitive deps into a
# venv that does not own them (and cannot be uv-synced on a compute node without ADC), score with
# hooke-tx's own interpreter, which already has loguru, cell_eval, anndata, numpy and pandas.
SCORER_PY="${SCORER_PY:-${HOOKE_TX_DIR}/.venv/bin/python}"
[[ -x "${SCORER_PY}" ]] || { echo "missing scorer interpreter: ${SCORER_PY}" >&2; exit 1; }

PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" initialize \
  --artifact-dir "${EVAL_DIR}" \
  --pert-col gene \
  --control-pert empty \
  --cell-type-col cell_type \
  --stored-pred-control-mode generated \
  --profile "${PROFILE}" \
  --skip-metrics pearson_edistance clustering_agreement

# 2. Score twice into separate results dirs, so neither convention overwrites the other.
for MODE in generated real; do
  RESULTS_DIR="${EVAL_DIR}/results_${MODE}"
  mkdir -p "${RESULTS_DIR}"
  echo "=== scoring controls=${MODE} -> ${RESULTS_DIR} ==="
  PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" score \
    --artifact-dir "${EVAL_DIR}" \
    --cell-type HepG2 \
    --pred-control-mode "${MODE}" \
    --results-dir "${RESULTS_DIR}"

  # summarize needs the SAME --results-dir as score: it averages the per-cell-type
  # *_agg_results.csv files, which live in the per-convention dir, not in EVAL_DIR. Without it this
  # exits with "No scored cell types in <EVAL_DIR>; missing ['HepG2']". Not `|| true` -- a missing
  # summary means the CSVs the bundle reads are not where they are expected to be.
  PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" summarize \
    --artifact-dir "${EVAL_DIR}" \
    --results-dir "${RESULTS_DIR}"
done

echo "=== EVAL DONE ==="
for MODE in generated real; do
  echo "--- ${MODE} ---"; ls "${EVAL_DIR}/results_${MODE}" 2>/dev/null | head -4
done
