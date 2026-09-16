#!/usr/bin/env bash
# The split-half TIMING PROBE -- one split, anchors only, no checkpoint.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict scripts/slurm/rhj_splithalf_probe.sbatch
#
# qos and wckey must MATCH.
#
# WHY THIS RUNS BEFORE THE REAL EVALUATION
# ----------------------------------------
# Overall goal: report how far the model is from experimental reproducibility on the 611
# held-out HepG2 perturbations, using the reviewer's repeated paired split-half protocol.
#
# This job's question: what does ONE cell-eval invocation on this panel actually cost, and
# how much of that cost is the real-side DE that all 5 arms of a split share?
#
# Varied: nothing. Held fixed: everything (the frozen panel, cohort and split 0).
#
# Why it needs no checkpoint: cell-eval's runtime depends on the panel -- 42,402 cells,
# 6,983 genes, 612 groups, 21,063 rows per scored file -- and not at all on the model. The
# probe's "prediction" is the identity on the control input, so `model_nocal` collapses onto
# the `ctrlcell` anchor and nothing about the numbers is interpretable. They are stamped
# `probe: true` and written to metrics_probe.json so they cannot be read as a result.
#
# HOW TO READ IT
# --------------
#   arm 1 (cold) = real-side DE + pred-side DE + anndata metrics
#   arm 2 (warm) = pred-side DE + anndata metrics        (real DE read from the sha256 cache)
#   real-side DE = cold - warm
# Then: one split of 5 arms costs (real DE + 5 x warm), and the 5-split array costs 5x that
# in total but only ONE of those per task in wall clock.
#
# The prior estimate being tested is ~2.0-2.8 h per split, extrapolated across a 5x change
# in panel size from the decoder-stage runs. That extrapolation is exactly why this probe
# exists.

#SBATCH --job-name=rhj_sh_probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/rhj_sh_probe_%j.out
#SBATCH --error=runs/slurm/rhj_sh_probe_%j.err

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
SPLIT="${SPLIT:-0}"

echo "=== rhj split-half timing probe ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm   : job ${SLURM_JOB_ID:-?}  qos ${SLURM_JOB_QOS:-?}"
echo "split   : ${SPLIT}"
echo

python scripts/eval_splithalf.py --probe --split "$SPLIT" --num-threads 16

echo
echo "finished: $(date -Is)"
