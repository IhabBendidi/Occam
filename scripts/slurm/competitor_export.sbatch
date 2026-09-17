#!/usr/bin/env bash
# Phase A: one arm's selected checkpoint predicts all 5 splits, in ITS OWN venv.
#
#   ARM=txpert sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4 \
#       --export=ALL,ARM scripts/slurm/competitor_export.sbatch
#
# One array task per SEED (42-46); each task loops over all 5 splits, because loading the
# checkpoint and the graph is the expensive part and the 5 splits share it.
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This leg's question: nothing on its own. It is the transport. It produces, per (seed, split),
# an `(18575, 6983)` float32 matrix in OUR evaluation space, which competitor_score.sbatch then
# scores through OUR cell-eval 0.6.0 path.
#
# WHY TWO PHASES. An arm's checkpoint loads only in the interpreter that pinned it, and our
# scorer runs only in ours. Two interpreters cannot share a Python callable, so the bridge is a
# file: `pred.npy` + `labels.json` + `manifest.json`.
#
# WHAT IS FIXED HERE, and cannot be opted out of:
#   * the drawn half-B control cells are the model's ONLY input -- same draw as our own arms,
#     rng = default_rng(42 + 1000 * split), reproduced in load_panel_split
#   * the row order is the half-A perturbed cells by argsort(labels, kind="stable"), written to
#     labels.json so Phase B verifies rather than assumes
#   * the emission is mapped into our evaluation space ONCE, at this boundary, and
#     assert_same_space REFUSES rather than warns
#
# ⚠️ THE ADAPTER IS PER ARM and lives at competitors/<Arm>/export_occam.py. It must expose
# `predict(ctrl, pert_names) -> (n, 6983)`. If it is absent this job exits 1 with the path --
# it does not invent a fallback, because a silently-wrong prediction matrix produces a
# plausible number from a meaningless comparison.
#
# ⚠️ EMISSION SPACE IS DECLARED, NEVER GUESSED. Each adapter states its `from_target_sum` and
# `from_log1p`; the exact 10k -> 2k map is `log1p(expm1(x) * 0.2)`, exact because both use the
# same full-panel row denominator. Never let cell-eval decide: its lognorm heuristic reads 500
# UNSEEDED cells and cannot tell log1p from library-normed linear.
#
# SIZING: inference over 18,575 cells x 5 splits. Minutes on a GPU for every arm; 2 h is
# margin, not estimate. CellFlow's PCA decode is the slowest of the four.

#SBATCH --job-name=cmp_export
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/cmp_export_%A_%a.out
#SBATCH --error=runs/slurm/cmp_export_%A_%a.err

source "$(dirname "${BASH_SOURCE[0]}")/_competitor_common.sh" 2>/dev/null || \
    source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/_competitor_common.sh"

HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

ARM="${ARM:?set ARM=txpert|state|cellflow|perturbdiff and pass --export=ALL,ARM}"
SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
N_SPLITS="${N_SPLITS:-5}"

case "$ARM" in
    txpert)      ARM_DIR="$HERE/competitors/TxPert" ;;
    state)       ARM_DIR="$HERE/competitors/STATE" ;;
    cellflow)    ARM_DIR="$HERE/competitors/CellFlow" ;;
    perturbdiff) ARM_DIR="$HERE/competitors/PerturbDiff" ;;
    *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac

ADAPTER="$ARM_DIR/export_occam.py"
if [ ! -f "$ADAPTER" ]; then
    echo "FATAL: no Phase-A adapter at $ADAPTER" >&2
    echo "       It must expose predict(ctrl, pert_names) -> (n, 6983) and call" >&2
    echo "       competitors/shared/export_predictions.py::export_split." >&2
    exit 1
fi

PY="$(competitor_venv_python "$ARM" "$HERE")" || exit 1
competitor_banner "$ARM" export "$SEED" "$HERE"
echo "  adapter $ADAPTER"
echo "  splits  0..$(( N_SPLITS - 1 ))"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE/competitors/shared${PYTHONPATH:+:$PYTHONPATH}"
unset VIRTUAL_ENV

FAILED=0
for K in $(seq 0 $(( N_SPLITS - 1 ))); do
    OUT="$HERE/runs/competitors/$ARM/s${SEED}/split_${K}"
    echo "--- split $K -> $OUT"
    if "$PY" -u "$ADAPTER" --seed "$SEED" --split "$K" --out-dir "$OUT" \
            --panel-dir "$HERE/data/benchmarks/rhj_fewshot/splithalf" \
            --bench-dir "$HERE/data/benchmarks/rhj_fewshot"; then
        echo "--- split $K OK"
    else
        echo "--- split $K FAILED (exit $?)" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done

echo "=== EXPORT DONE: $(( N_SPLITS - FAILED ))/$N_SPLITS splits ==="
if [ "$FAILED" -ne 0 ]; then
    echo "⚠️ $FAILED split(s) did not export. A partial arm is reported as partial, never" >&2
    echo "   averaged over whatever survived." >&2
    exit 1
fi
echo "next: ARM=$ARM SEED=$SEED sbatch --array=0-4 scripts/slurm/competitor_score.sbatch"
