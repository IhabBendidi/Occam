#!/usr/bin/env bash
# ⛔ THE BRIDGE GATE. No competitor training launches until this passes.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict scripts/slurm/rhj_competitor_gate.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was
# never measured, from a single control cell.
#
# This job's question: does the two-phase competitor bridge add any signal or bias of its
# own? A competitor's number is only interpretable if the pipe it travels through is inert.
#
# Varied: nothing. Held fixed: everything. The "competitor" here is the IDENTITY function on
# its own control input -- Phase A feeds it the drawn half-B control cells and it returns
# them unchanged, so the exported pred.npy IS the `ctrlcell` baseline, routed the long way
# round (export to pred.npy, then scripts/eval_competitor.py -> protocol.run_split).
#
# WHAT EACH OUTCOME MEANS
#   * It reproduces the directly-measured `ctrlcell` numbers -> the bridge is inert, and any
#     later competitor score is the model's, not the plumbing's.
#   * It does NOT reproduce them -> the bridge is injecting or destroying signal, and every
#     competitor number measured through it would be uninterpretable. Fix before training.
#
# THE TARGET, split 0, from docs/rhj_baseline_results.md (arm `ctrlcell`, uncalibrated):
#     pearson_delta            -0.0018      (all-5-split mean; split 0 is the row to match)
#     overlap_at_N              0.0000
#     discrimination_score_l1   0.5038      (0.5 is chance)
# Anchors for scale: meanpert 0.1324 · transfer 0.2323 · experimental 0.3722.
#
# ⚠️ This is a CPU job. No model is loaded: `run_split` is called with model=None,
# device="cpu", calibrate="none". The cost is cell-eval's Wilcoxon pass, ~25-40 min.
#
# ⚠️ Do NOT put this script in the scratchpad. /tmp is NODE-LOCAL, so a compute node cannot
# see a login-node /tmp path -- job 18042891 failed in 2 s on exactly that
# ("gate.sh: No such file or directory"). Launchers live in the repo, on GPFS.

#
# ⚠️ NO `srun`. On the `cpu` partition srun refuses with "Invalid generic resource
# (gres) specification" and the job FAILS in 2 s -- measured, job 18042983. No
# launcher in this repo uses srun; the batch step IS the task.
#SBATCH --job-name=rhj_cgate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/rhj_cgate_%j.out
#SBATCH --error=runs/slurm/rhj_cgate_%j.err

set -euo pipefail

_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then
            echo "$c"; return 0
        fi
    done
    echo "cannot locate the Occam repo; sbatch from the repo root or set OCCAM_REPO" >&2
    return 1
}
HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

ARM=_gate_ctrlcell
SEED=42
SPLIT=0
PRED_DIR="runs/competitors/${ARM}/s${SEED}/split_${SPLIT}"
OUT_ROOT="runs/rhj_splithalf/compet_${ARM}_s${SEED}"
PY=/mnt/ps/home/CORP/ihab.bendidi/.pyenv/versions/scmae/bin/python

echo "[gate] repo     $HERE"
echo "[gate] host     $(hostname)"
echo "[gate] pred     $PRED_DIR"
echo "[gate] out      $OUT_ROOT"
echo "[gate] target   pearson_delta -0.0018 · overlap_at_N 0.0000 · disc_l1 0.5038"

if [ ! -f "$PRED_DIR/pred.npy" ]; then
    echo "[gate] no pred.npy -- Phase A has not run. Export it first:" >&2
    echo "  PYTHONPATH=src:competitors/shared $PY scripts/export_gate_ctrlcell.py" >&2
    exit 1
fi

export PYTHONPATH="$HERE/src:$HERE/competitors/shared"

"$PY" scripts/eval_competitor.py \
    --arm "$ARM" \
    --seed "$SEED" \
    --split "$SPLIT" \
    --pred-dir "$PRED_DIR" \
    --out-root "$OUT_ROOT" \
    --num-threads "${SLURM_CPUS_PER_TASK:-16}"

echo "[gate] done. Compare the three numbers above against the printed result."
