#!/usr/bin/env bash
# The norman_combo REFERENCE pass: every non-learned baseline, on all 5 splits.
#
#   sbatch --qos=vco --wckey=vco --array=0-4 scripts/slurm/norman_reference.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that exact perturbation was
# never measured. This benchmark's form of it is COMPOSITION -- train on single-gene
# perturbations, predict PAIRS.
#
# This leg scores the five arms that need NO checkpoint, which have to exist before any model
# arm is trained, or the model's number has nothing to be read against:
#   ctrlcell        one drawn B control cell            the zero-signal floor
#   ctrlmean        the B control mean                  pearson_delta is NaN by construction
#   meanpert        mean effect over the 105 singles    "ignore which pair this is"
#   sum_of_singles  ctrl + delta_A + delta_B            ⭐ THE ADDITIVITY BASELINE
#   experimental    the half-B perturbed cells          the empirical reference, NOT a ceiling
#
# ⚠️ `sum_of_singles` IS our architecture's own composition operator in closed form -- the
# model sums the perturbations' graph embeddings. A model at or below it has learned nothing
# about interaction. On the quick pseudobulk check it already reaches pearson(delta) 0.93 on
# the additive stratum and 0.84 on the non-additive one.
#
# ⚠️ `transfer` does NOT exist here: one cell line, nothing to transfer from. The reference
# arm list is resolved against the panel artifact, so it is absent rather than zero.
#
# HOW TO READ THE RESULT
# ----------------------
#   python scripts/norman/norman_report.py
# The SCORED panels are `additive` (36 pairs) and `nonadditive` (36), effect-controlled
# residual terciles -- user decision 2026-09-21. `all` (111) is computed and reported as
# context, not as the headline.
#
# Space: library-normalised to 2,000 then log1p, the row sum taken over ALL 33,694 source
# genes before restriction to the 8,000-gene axis -- identical to the model's own input space,
# so `mae` is comparable between every baseline, our model and every competitor.

#SBATCH --job-name=norman_ref
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=hop39
#SBATCH --output=runs/slurm/norman_ref_%A_%a.out
#SBATCH --error=runs/slurm/norman_ref_%A_%a.err

set -euo pipefail

# ⚠️ NOT `dirname "${BASH_SOURCE[0]}"`. SLURM runs a COPY of this script from its spool
# directory, so BASH_SOURCE points at /var/spool/... and the relative `../..` resolved to an
# unwritable path -- the first launch died with `mkdir: cannot create directory 'runs':
# Permission denied` before running anything. Same helper as the other launchers, and it
# VALIDATES the candidate instead of trusting it.
_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" "$(pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then
            echo "$c"; return 0
        fi
    done
    echo "cannot locate the Occam repo; sbatch from the repo root or set OCCAM_REPO" >&2
    return 1
}
HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm runs/norman_combo
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export OCCAM_CELLEVAL_ATTEMPTS="${OCCAM_CELLEVAL_ATTEMPTS:-3}"

SPLIT="${SLURM_ARRAY_TASK_ID:-0}"
CACHE="runs/norman_combo/_de_real_shared"
mkdir -p "$CACHE"

echo "=== norman_combo reference pass · split ${SPLIT} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

python scripts/eval_splithalf.py \
    --mode reference \
    --panel-dir data/benchmarks/norman_combo/splithalf \
    --split "$SPLIT" \
    --out-root runs/norman_combo/reference \
    --de-real-cache-dir "$CACHE" \
    --calibrate none \
    --num-threads 16

echo
echo "finished: $(date -Is)"
