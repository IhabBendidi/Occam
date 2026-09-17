#!/bin/bash
#SBATCH --job-name=cellflow_fewshot
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/cellflow/logs/fewshot_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/cellflow/logs/fewshot_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --time=0-08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --array=0-4
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# CellFlow on Replogle Nadig Few Shot, 5 seeds, under the uniform selection protocol.
#
# PROTOCOL (identical for every model family on this benchmark):
#   1. cell-eval on the whole 4,825-cell VALIDATION split every 20,000 iterations
#   2. keep the best state by validation selection_score
#      = mean(pearson_delta, discrimination_score_l1), mode=max
#   3. run the TEST evaluation exactly once, from that state
#      (predict_replogle_nadig_fewshot.sh)
#   4. 5 seeds, 42-46
#
# WHY THIS EXISTS. CellFlow had no way to choose a state without test data: train() returns the
# last iteration's solver and the wrapper passed no callbacks. It also could not read this split
# at all -- load_fold rejected the file, the multi-source path refused a validation split, and
# 8 of the 380 test perturbations had no one-hot column so predict aborted outright.
#
# SIX VALIDATIONS, NOT FIVE. valid_freq=20000 over 100,000 iterations gives 5 in-loop passes, and
# the trainer's on_train_end pass scores the final state as well. So an early-peaking run and a
# still-improving one are both selectable.
#
# THE SAVED PICKLE IS THE SELECTED STATE. The callback restores the best params before save(),
# so there is one ~5 GB pickle per run rather than five, and the eval leg needs no checkpoint
# argument. It refuses to run if run_meta.json carries no selection record.
#
# SEEDS 42-46, matching every other arm. Note this arm required a fix to mean anything: the
# trainer hardcoded PRNGKey(0)/default_rng(0) and never received the configured seed, so five
# runs would have shared their training-step and batch-sampling streams.
#
# TIMING. Measured on a 1,200-iteration smoke: ~90 it/s training, ~230 s per validation scoring.
# 100k iterations therefore projects to ~20 min of training plus ~25 min of scoring, so under an
# hour per seed. 8 h is headroom for the 256-component PCA fit, which the smoke did not exercise
# (it ran at 64).
#
#   sbatch train_replogle_nadig_fewshot.sh
#   ITERATIONS=2000 VALID_FREQ=500 TAG_SUFFIX=_smoke sbatch --array=0 \
#     --export=ALL,ITERATIONS,VALID_FREQ,TAG_SUFFIX train_replogle_nadig_fewshot.sh
# ============================================================================================

# Absolute: SLURM copies the batch script into its own spool directory, so `dirname "$0"` points
# at /cm/local/apps/slurm/var/spool rather than here. The configs also use ${hydra:runtime.cwd}
# equivalents via $PWD, so the working directory matters.
CELLFLOW_ROOT="${CELLFLOW_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/CellFlow}"
CF_PKG="${CELLFLOW_ROOT}/cellflow"

ITERATIONS="${ITERATIONS:-100000}"
VALID_FREQ="${VALID_FREQ:-20000}"

# A validation cadence that does not divide the budget leaves the last stretch unscored. The
# final state is still scored by on_train_end, so this is a warning rather than fatal -- but an
# uneven cadence makes the validation history harder to compare across seeds.
if (( ITERATIONS % VALID_FREQ != 0 )); then
  echo "WARNING: iterations=${ITERATIONS} is not a multiple of valid_freq=${VALID_FREQ}" >&2
fi

SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
# TAG_SUFFIX keeps a smoke run out of the real run's directories. The run dir has no seed or
# timestamp of its own and save(overwrite=True) would clobber silently.
TAG="fewshot_s${SEED}${TAG_SUFFIX:-}"
OUT_ROOT="${OUT_ROOT:-/rxrx/scratch/sandbox/${USER}/cellflow/replogle_nadig_fewshot}"

mkdir -p "/rxrx/scratch/sandbox/${USER}/cellflow/logs" "${OUT_ROOT}"

[[ -x "${CF_PKG}/.venv/bin/python" ]] || {
  echo "FATAL: no CellFlow venv at ${CF_PKG}/.venv" >&2
  echo "       cd ${CF_PKG} && uv sync --python 3.12" >&2
  exit 1
}

# uv's cache defaults to NFS while the venv is on GPFS; see README_HOOKE_TX-style note in
# ../TxPert. Kept even though we invoke .venv/bin/python directly, since jax reads it on import.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/${USER}/.uv-cache}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
export HATCH_VCS_PRETEND_VERSION=0.1.0
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

echo "=== CellFlow fewshot train ==="
echo "    seed=${SEED}  tag=${TAG}"
echo "    iterations=${ITERATIONS}  valid_freq=${VALID_FREQ}"
echo "    out=${OUT_ROOT}/${TAG}"
echo "    node=$(hostname)"

# `.venv/bin/python`, not `uv run`: uv re-verifies and locks the venv on every invocation, which
# hangs when several array tasks start together on shared GPFS.
#
# Keep every comment ABOVE this invocation. A '#' inside a backslash-continued command
# terminates it, which silently dropped 29 of 32 overrides on STATE.
cd "${CF_PKG}"
"${CF_PKG}/.venv/bin/python" -u ../train_cellflow.py \
  --config "${CELLFLOW_ROOT}/configs/replogle_nadig_fewshot.yaml" \
  --output-dir "${OUT_ROOT}" \
  --run-name "${TAG}" \
  --num-iterations "${ITERATIONS}" \
  --set "train.valid_freq=${VALID_FREQ}" \
  --set "model.seed=${SEED}"

echo "=== TRAIN DONE ==="
RUN_DIR="${OUT_ROOT}/${TAG}"
"${CF_PKG}/.venv/bin/python" - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
meta = json.loads((Path(sys.argv[1]) / "run_meta.json").read_text())
sel = meta.get("selection") or {}
best = sel.get("best")
if not best:
    print("WARNING: no selection record; the pickle holds the FINAL state", file=sys.stderr)
else:
    print(f"selected: {best['tag']} selection_score={best['selection_score']:.5f} "
          f"(of {sel['n_scored_states']} scored)")
    for h in sel.get("history", []):
        print(f"    {h['tag']:<8} {h['selection_score']:.5f}")
PY
ls -la "${RUN_DIR}"
