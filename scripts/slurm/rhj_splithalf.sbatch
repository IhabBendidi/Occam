#!/usr/bin/env bash
# The repeated split-half paired evaluation -- one array task per split.
#
#   CKPT=runs/rhj_fewshot_a03__.../vcell/checkpoints/best_model.pth \
#     sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4 \
#            scripts/slurm/rhj_splithalf.sbatch
#
# qos and wckey must MATCH. One task per split, all 5 in parallel: the splits share
# nothing, and the real-side DE cache is keyed on each split's own target-file sha256, so
# cross-split reuse is impossible by construction and concurrency is free.
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was
# never measured, from a single control cell.
#
# This job's question, per split: scored against the same half-A reference, how does the
# model fed half-B control cells compare with the half-B perturbed cells themselves?
#
# Varied: which cells land in half A and which in half B (5 independent, batch-stratified
# splits). Held FIXED: the 611-perturbation cohort, the 6,983-gene axis, the checkpoint,
# the arm set, and the per-perturbation row counts -- m(p) = floor(n_p/2) depends only on
# n_p, so every arm in every split emits the same rows for the same perturbation.
#
# THE ARMS (5 per split, one shared target file)
#   ctrlmean      B-control mean, tiled          zero-signal anchor
#   ctrlcell      the drawn B-control cell       zero-signal anchor, the load-bearing floor
#   experimental  the B-perturbed cells          replicate agreement at m cells
#   model_cal     model fed the drawn B controls THE MEASUREMENT
#   meanpert      ctrl + (train grand mean - B ctrl mean)   "it learned the average knockdown"
#
# HOW TO READ IT
# --------------
# * **Positive gap = the experimental arm is better.** That is the expected result and its
#   size is the headline. A NEGATIVE gap is a red flag, not a triumph: at median m = 22 the
#   reference and the experimental prediction are BOTH noisy pseudobulks while a
#   low-variance predictor is attenuated on one side only, so a model can beat the
#   "ceiling" from noise alone. The count-stratified gap must SHRINK as m grows.
# * One split is not a result. The claim is the mean of the 5 per-split PAIRED differences
#   (std with ddof=1), which is what scripts/paired_report.py computes.
# * `ctrlmean`'s pearson_delta is undefined here by construction -- with B controls on the
#   pred side its delta is exactly zero. `ctrlcell` is the floor instead.
#
# ANCHORS, next to every number
#   ctrlcell  = control-sampling noise at m cells (this panel's zero-signal floor)
#   meanpert  = the mean training delta, the "no per-perturbation information" baseline
#   experimental = replicate agreement at m cells (NOT an upper bound -- see above)
#
# COST: measured by scripts/slurm/rhj_splithalf_probe.sbatch before this was launched.
# Read runs/rhj_splithalf/probe/split_00/metrics_probe.json for the real-DE/pred-DE split.

#SBATCH --job-name=rhj_splithalf
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/rhj_splithalf_%A_%a.out
#SBATCH --error=runs/slurm/rhj_splithalf_%A_%a.err

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
mkdir -p runs/slurm runs/rhj_splithalf

export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${CKPT:-}" ]; then
    echo "CKPT is required: CKPT=<path to a vcell checkpoint> sbatch ..." >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "CKPT does not exist: $CKPT" >&2
    exit 1
fi

SPLIT="${SLURM_ARRAY_TASK_ID:-0}"
# The arm tag goes in the output root so two checkpoints never write into one directory --
# every metrics.json would otherwise be overwritten by whichever task finished last.
TAG="${TAG:-$(basename "$(dirname "$(dirname "$CKPT")")")__$(basename "$CKPT" .pth)}"
OUT="runs/rhj_splithalf/${TAG}"
mkdir -p "$OUT"

echo "=== rhj split-half · split ${SPLIT} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm   : job ${SLURM_ARRAY_JOB_ID:-?} task ${SPLIT}  qos ${SLURM_JOB_QOS:-?}"
echo "ckpt    : $CKPT"
echo "out     : $OUT/split_$(printf %02d "$SPLIT")"
echo

cat > "$OUT/slurm_job_${SPLIT}.json" <<JSON
{"split": ${SPLIT}, "checkpoint": "${CKPT}", "tag": "${TAG}",
 "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}", "qos": "${SLURM_JOB_QOS:-}",
 "host": "$(hostname)", "started": "$(date -Is)"}
JSON

python scripts/eval_splithalf.py \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --out-root "$OUT" \
    --calibrate "${CALIBRATE:-quantile}" \
    --num-threads 16

echo
echo "finished: $(date -Is)"
echo
echo "When all 5 splits are done, the ONLY valid summary is:"
echo "    python scripts/paired_report.py --run-dir $OUT"
