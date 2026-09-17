#!/bin/bash
#SBATCH --job-name=cellflow_fewshot_pred
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/cellflow/logs/fewshot_pred_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/cellflow/logs/fewshot_pred_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --time=0-04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# TEST evaluation for the CellFlow few-shot arm, from the VALIDATION-SELECTED state, scored
# under BOTH predicted-control conventions.
#
# ONE prediction pass, scored twice by the SHARED scorer. CellFlow's own `_run_celleval` cannot
# do this: its output paths are fixed (`celleval_{split}/`, `celleval_{split}_metrics.json`) and
# it `shutil.rmtree`s the directory on every call, so a second convention would overwrite the
# first. It also passes no `pdex_kwargs`, where the shared scorer always passes
# {exp_post_agg, is_log1p}, and this venv resolves pdex 0.3.0 against 0.2.2 elsewhere -- two
# independent reasons its numbers would not be comparable to the other four arms.
#
# THE TWO CONVENTIONS FOR CELLFLOW:
#   real    ("inject controls")      -- raw measured control cells on the pred side. This is what
#                                      CellFlow does today.
#   decoded ("reconstruct controls") -- the pred-side controls are PCA-encoded and decoded through
#                                      the run's own frozen basis.
#
# CellFlow is the only competitor whose `decoded` means the SAME thing as hooke-tx's, because it
# is the only one with a real PCA codec (STATE's and TxPert's second arm is `generated`, i.e.
# model-predicted controls, which CellFlow never produces -- it never asks the model for the
# control condition).
#
# `decoded` is not a cosmetic second column here: today the perturbed predictions are PCA-decoded
# and clipped to [0, 14] while their control baseline is raw gene space, so every delta metric is
# computed across two different representations. Decoding the controls removes that asymmetry.
#
# The selected state is already what CellFlow.pkl holds -- the selection callback restores the
# best params before save() -- so there is no checkpoint to choose here.
#
#   sbatch --dependency=aftercorr:<train_job_id> predict_replogle_nadig_fewshot.sh
# ============================================================================================

CELLFLOW_ROOT="${CELLFLOW_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/CellFlow}"
HOOKE_TX_DIR="${HOOKE_TX_DIR:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx}"
CF_PKG="${CELLFLOW_ROOT}/cellflow"

SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
TAG="fewshot_s${SEED}${TAG_SUFFIX:-}"
OUT_ROOT="${OUT_ROOT:-/rxrx/scratch/sandbox/${USER}/cellflow/replogle_nadig_fewshot}"
RUN_DIR="${OUT_ROOT}/${TAG}"
EVAL_DIR="${RUN_DIR}/artifacts"

mkdir -p "/rxrx/scratch/sandbox/${USER}/cellflow/logs"

[[ -f "${RUN_DIR}/CellFlow.pkl" ]] || {
  echo "FATAL: no trained model at ${RUN_DIR}/CellFlow.pkl" >&2
  exit 1
}
# run_meta.json records which state was selected. No selection block means the run predates the
# selection callback or it never published a score -- in which case the pickle holds the FINAL
# state, not a chosen one, and reporting it under this protocol would be wrong.
"${CF_PKG}/.venv/bin/python" - "${RUN_DIR}" <<'PY' || exit 1
import json, sys
from pathlib import Path
meta = json.loads((Path(sys.argv[1]) / "run_meta.json").read_text())
sel = meta.get("selection")
if not sel or not sel.get("best"):
    sys.exit(
        "FATAL: run_meta.json has no selection record, so CellFlow.pkl holds the final state "
        "rather than a validation-selected one."
    )
best = sel["best"]
print(f"    selected state: {best['tag']} selection_score={best['selection_score']:.5f} "
      f"(of {sel['n_scored_states']} scored)")
print(f"    seed={meta.get('seed')} n_comps={meta.get('n_comps')}")
PY

export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/${USER}/.uv-cache}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
export HATCH_VCS_PRETEND_VERSION=0.1.0
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

echo "=== CellFlow fewshot eval ==="
echo "    seed=${SEED}  tag=${TAG}"
echo "    run_dir=${RUN_DIR}"
echo "    node=$(hostname)"

# The scorer defaults to 16 threads; numba caps at the cgroup CPU count and then raises from
# inside pdex, after the artifacts are already written.
THREADS="$(nproc)"
echo "    scoring threads=${THREADS}"

# 1. One prediction pass. --skip-eval because CellFlow's own cell-eval is not what gets reported;
#    --artifact-dir writes the pred/real h5ads plus the converted PCA basis for the scorer.
#    Keep every comment ABOVE the invocation: a '#' inside a backslash-continued command
#    terminates it, which silently dropped 29 of 32 overrides on STATE.
cd "${CF_PKG}"
"${CF_PKG}/.venv/bin/python" -u ../predict_cellflow.py \
  --run-dir "${RUN_DIR}" \
  --split-key test \
  --artifact-dir "${EVAL_DIR}" \
  --skip-eval \
  --max-control-cells "${MAX_CONTROL_CELLS:-512}" \
  --seed "${SEED}"

SCORER="${HOOKE_TX_DIR}/hooke_tx/eval/celleval_artifacts.py"
SCORER_PY="${SCORER_PY:-${HOOKE_TX_DIR}/.venv/bin/python}"
[[ -x "${SCORER_PY}" ]] || { echo "missing scorer interpreter: ${SCORER_PY}" >&2; exit 1; }

# Clear manifests at BOTH levels. `score` writes one into each results dir as well as the
# artifact dir and refuses to mix a stored manifest with different artifacts -- correctly, but it
# makes any re-run fail unless the stale ones go first.
rm -f -- "${EVAL_DIR}/celleval_manifest.json"
rm -f -- "${EVAL_DIR}"/results_*/celleval_manifest.json

PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" initialize \
  --artifact-dir "${EVAL_DIR}" \
  --pert-col pert_str \
  --control-pert empty \
  --cell-type-col cell_type \
  --profile full \
  --skip-metrics pearson_edistance clustering_agreement

CELL_TYPE="$("${SCORER_PY}" -c '
import json,sys
m=json.load(open(sys.argv[1]))
print(m["cell_types"][0])' "${EVAL_DIR}/celleval_manifest.json")"
echo "=== cell type: ${CELL_TYPE} ==="

# 2. Score twice into separate results dirs.
#    real     -> --pred-control-mode real  (validates control alignment; the rows are already real)
#    decoded  -> --reconstruct-pred-controls, which PCA-decodes pred-side controls only, leaving
#                adata_real raw. The two flags are mutually exclusive, hence two invocations.
for ARM in real decoded; do
  RESULTS_DIR="${EVAL_DIR}/results_${ARM}"
  mkdir -p "${RESULTS_DIR}"
  echo "=== scoring arm=${ARM} -> ${RESULTS_DIR} ==="
  if [[ "${ARM}" == "real" ]]; then
    MODE_ARGS=(--pred-control-mode real)
  else
    MODE_ARGS=(--reconstruct-pred-controls --pca-state "${EVAL_DIR}/pca_state.npz")
  fi
  PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" score \
    --artifact-dir "${EVAL_DIR}" \
    --cell-type "${CELL_TYPE}" \
    --num-threads "${THREADS}" \
    --results-dir "${RESULTS_DIR}" \
    "${MODE_ARGS[@]}"

  PYTHONPATH="${HOOKE_TX_DIR}" "${SCORER_PY}" "${SCORER}" summarize \
    --artifact-dir "${EVAL_DIR}" \
    --results-dir "${RESULTS_DIR}"
done

echo "=== EVAL DONE ==="
for ARM in real decoded; do
  echo "--- ${ARM} ---"; ls "${EVAL_DIR}/results_${ARM}" 2>/dev/null | head -5
done
