#!/usr/bin/env bash
# The five MODEL-INDEPENDENT arms, once per split. Run this BEFORE the model sweep.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4 \
#          scripts/slurm/rhj_splithalf_reference.sbatch
#
# WHY THIS EXISTS SEPARATELY
# --------------------------
# Five of the six arms do not depend on the checkpoint:
#   ctrlmean      the half-B control mean
#   ctrlcell      the drawn half-B control cell
#   experimental  the real half-B perturbed cells
#   transfer      ctrl + this perturbation's Jurkat/RPE1 delta
#   meanpert      ctrl + (training grand mean - half-B control mean)
# `ctrl_used` comes from a seeded draw (seed + 1000*split) that is bit-identical across
# every run of a split, so these five matrices are the same for all 27 checkpoints.
# Scoring them once per split instead of once per checkpoint turns 27 x 5 x 6 = 810
# cell-eval runs into 25 + 135, and it also guarantees the model arm is compared against
# byte-identical references rather than re-drawn ones.
#
# No checkpoint is needed or accepted: the prediction is the identity on the control input,
# which is what produces `ctrl_used`, and the model arm is not scored here.
#
# HOW TO READ THE RESULT
# ----------------------
# `transfer` is the one to watch. On the validation side it beat every one of the 24 FiLM
# arms by 0.20 (docs/rhj_fewshot.md F5), and it is the NON-LEARNED version of exactly what
# the model is asked to do: take the same perturbation's measured effect in the two other
# cell lines and add it to a HepG2 control cell.
#
# `experimental` is replicate agreement at m cells (median m = 22), NOT an upper bound --
# both it and the reference are noisy pseudobulks while a low-variance predictor is
# attenuated on one side only. `ctrlcell` is the zero-signal floor for both metric
# families; `ctrlmean` over-calls DE by 27x and brackets it from the other side.
#
# COST: 8.5 min real-side DE (shared, cached by target sha256) + 11.0 min per arm
# = ~64 min per split, all 5 splits in parallel.

#SBATCH --job-name=rhj_sh_ref
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/rhj_sh_ref_%A_%a.out
#SBATCH --error=runs/slurm/rhj_sh_ref_%A_%a.err

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
SPLIT="${SLURM_ARRAY_TASK_ID:-0}"
CACHE="${DE_CACHE:-runs/rhj_splithalf/_de_real_shared}"
mkdir -p "$CACHE"

echo "=== rhj split-half REFERENCE arms · split ${SPLIT} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm   : job ${SLURM_ARRAY_JOB_ID:-?} task ${SPLIT}  qos ${SLURM_JOB_QOS:-?}"
echo "de cache: $CACHE  (shared across splits AND checkpoints)"
echo

# ARMS=ctrlmean retries a single arm after the transient pdex shared-memory race
# (FileNotFoundError: /psm_...) without repaying the other four.
ARMS_ARG=()
if [ -n "${ARMS:-}" ]; then
    ARMS_ARG=(--arms "$ARMS")
    echo "arms    : ${ARMS}  (restricted pass -- merge, do not overwrite)"
fi

python scripts/eval_splithalf.py \
    --mode reference \
    --split "$SPLIT" \
    "${ARMS_ARG[@]+"${ARMS_ARG[@]}"}" \
    --out-root runs/rhj_splithalf/reference \
    --de-real-cache-dir "$CACHE" \
    --num-threads 16

echo
echo "finished: $(date -Is)"
