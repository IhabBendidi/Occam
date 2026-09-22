#!/usr/bin/env bash
# Phase B: score one arm's exported predictions through OUR path. One task per SPLIT.
#
#   ARM=txpert SEED=42 sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4 \
#       --export=ALL,ARM,SEED scripts/slurm/competitor_score.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This leg's question: what does the arm score, measured on exactly the instrument our own arms
# are measured on -- same 611-perturbation cohort, same 5 splits, same 6,983-gene axis, same
# drawn half-B controls, same cell-eval 0.6.0, same five baselines?
#
# The competitor inherits, without being able to opt out: `_assert_in_pert2id`,
# `assert_disjoint_halves`, `assert_cohort` (which ENFORCES the frozen 611), the label-order
# check, the non-finite guard, the mandatory zero-signal anchors, `pred_ctrl = ctrl_B`, and the
# sha256-keyed real-DE cache shared with every other arm.
#
# ⚠️ THEIR OWN SCORER IS NEVER RUN. It clips to [0,14], skips `pearson_edistance` and
# `clustering_agreement`, labels controls "empty", and runs cell-eval 0.7.2 / pdex 0.2.2-0.3.0
# against our 0.6.0 / pdex 0.1.24. It is kept as reference only.
#
# ⚠️ UNCALIBRATED ONLY. Their models emit gene space directly and were never calibrated
# upstream; on our own model, control-fit quantile calibration cost 0.047 pearson_delta and
# quadrupled mae (docs/rhj_splithalf.md F12). `model_nocal` is the like-for-like column.
#
# READ THE RESULT AGAINST: ctrlcell -0.0018 · ctrlmean 0.0016 · meanpert 0.1324 ·
# transfer 0.2323 · experimental 0.3722 (pearson_delta, all 5 splits).
#
# ⚠️ ONE SPLIT IS NOT A RESULT. Split-to-split std is 0.001-0.003, so a single split looks
# precise and is not a mean. Run all 5, then scripts/collect_competitors.py.
#
# SIZING: CPU-only. No model is loaded (model=None, device="cpu", calibrate="none"); the cost
# is cell-eval's Wilcoxon pass, ~25-40 min per split. 2 h is margin.

#
# ⚠️ NO `srun`. On the `cpu` partition srun refuses with "Invalid generic resource
# (gres) specification" and the job FAILS in 2 s -- measured, job 18042983. No
# launcher in this repo uses srun; the batch step IS the task.
#SBATCH --job-name=cmp_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
# ⚠️ 128G, NOT 64G. Split 4 needs more than 64: both TxPert chains hit OUT_OF_MEMORY there
# while splits 0-3 fit, and the retry measured MaxRSS 65.6 GB / 69.6 GB. 64 GB is the worst
# possible request -- it fails late, on one split out of five, and only sometimes.
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/cmp_score_%A_%a.out
#SBATCH --error=runs/slurm/cmp_score_%A_%a.err
#SBATCH --exclude=hop39
# ⚠️ hop39's GPU is DEAD -- nvidia-smi there returns "Unable to determine the device
# handle for GPU0 ... Unknown Error" / "No devices were found" -- while sinfo still
# reports it `mix`, so Slurm keeps scheduling onto it and every task fails with
# "CUDA unknown error". REMOVE this line once the node is drained/fixed.

source "$(dirname "${BASH_SOURCE[0]}")/_competitor_common.sh" 2>/dev/null || \
    source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/_competitor_common.sh"

HERE="$(_find_repo)" || exit 1
cd "$HERE"
# Resolves BENCH / BENCH_DIR / PANEL_DIR / N_AXIS_GENES and the three digests.
# BENCH defaults to rhj_fewshot, so every pre-existing invocation is unchanged.
competitor_resolve_bench "$HERE"
mkdir -p runs/slurm

ARM="${ARM:?set ARM=txpert|state|cellflow|perturbdiff and pass --export=ALL,ARM,SEED}"
SEED="${SEED:?set SEED=42..46 and pass --export=ALL,ARM,SEED}"
SPLIT="${SLURM_ARRAY_TASK_ID:-0}"

# ⚠️ ARM_SUFFIX separates VARIANTS OF ONE ARM -- see the note in competitor_export.sbatch.
# Two configurations of the same arm and seed would otherwise share one output directory and
# the second would silently overwrite the first.
ARM_SUFFIX="${ARM_SUFFIX:-}"
ARM_OUT="${ARM}${ARM_SUFFIX}$(competitor_bench_tag)"
PRED_DIR="$HERE/runs/competitors/$ARM_OUT/s${SEED}/split_${SPLIT}"
OUT_ROOT="$HERE/$(competitor_score_root)/compet_${ARM_OUT}_s${SEED}"

competitor_banner "$ARM" score "$SEED" "$HERE"
echo "  split   $SPLIT"
echo "  out arm $ARM_OUT"
echo "  pred    $PRED_DIR"
echo "  out     $OUT_ROOT"

if [ ! -f "$PRED_DIR/pred.npy" ]; then
    echo "FATAL: no pred.npy in $PRED_DIR. Phase A has not run for this (seed, split):" >&2
    echo "  ARM=$ARM sbatch --array=$(( SEED - 42 )) --export=ALL,ARM \\" >&2
    echo "      scripts/slurm/competitor_export.sbatch" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE/src:$HERE/competitors/shared"

"$OCCAM_PY" scripts/eval_competitor.py \
    --arm "$ARM" \
    --seed "$SEED" \
    --split "$SPLIT" \
    --pred-dir "$PRED_DIR" \
    --out-root "$OUT_ROOT" \
    --panel-dir "$PANEL_DIR" \
    --num-threads "${SLURM_CPUS_PER_TASK:-16}"

echo "=== SCORE DONE: $ARM s$SEED split $SPLIT ==="
echo "next, once all 5 splits are in:"
echo "  PYENV_VERSION=scmae python scripts/collect_competitors.py \\"
echo "      --out docs/rhj_competitors_results.md"
