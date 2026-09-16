#!/usr/bin/env bash
# rhj_fewshot ablation grid -- one array task per arm.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-12%7 scripts/slurm/rhj_fewshot_grid.sbatch
#   sbatch --qos=vco          --wckey=vco          --array=13-25%6 scripts/slurm/rhj_fewshot_grid.sbatch
#
# qos and wckey must MATCH. Split across both so neither queue's headroom is the
# bottleneck; check it first with:
#   squeue -h -o "%q %T %b" | awk '$1=="vco"||$1=="hooke-predict"'
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was
# never measured, from a single control cell.
#
# This grid's question: which parts of the FiLM conditioning actually matter on THIS
# benchmark, and does the negative-binomial loss beat smooth-L1 here?
#
# Varied (configs/rhj_fewshot_grid.tsv, one row per arm):
#     film_mode {full, multiplicative_only, additive_only}
#   x film_hidden_dim {0, 256}
#   x film_mean_free {0, 1}
#   x loss_type {smooth_l1, nb}                            = 24 arms
#   + 2 no-FiLM anchors (interaction_type=additive)        = 26 total
#
# Held fixed: the frozen benchmark artifact, film_scale 1.0, 30 epochs, seed 42,
# target_pb_k 1, ctrl_avg_k 1, frozen encoder, stream_v2 + proportional interleave,
# num_workers 8 (part of the determinism contract), source_weight none.
#
# HOW TO READ IT
# --------------
# There are NO imported anchors. Research-tree numbers came from a different panel and a
# different scorer; they justify this starting configuration and nothing more. The anchors
# are the two no-FiLM arms here, and then the zero-signal/replicate arms measured by
# scripts/eval_splithalf.py.
#
# Pre-registered expectations, so a null is readable:
#   * additive_only cannot be context-dependent (no (gamma-1)*z term). Its score is the
#     floor that makes "multiplicative_only is the useful part" mean something.
#   * additive_only x film_mean_free x film_hidden_dim=0 is analytically near-degenerate
#     (W(e_p - ebar) is embedding centring, which the bias can absorb). If that arm moves a
#     lot, suspect the film_mean_free port, not biology.
#   * nb arms: expect completion with grad-norm max ~100x the smooth_l1 arms, and a score
#     BELOW smooth_l1. Recorded per arm as a number, not an impression.
#   * loss_type changes TWO things at cosine_loss_weight 0.5: smooth_l1's recon term is
#     O(0.1-1), nb's NLL is O(1-10), so the smooth_l1 arms are genuinely half-cosine. Do
#     not rescale to equalise -- that is tuning inside an ablation. Report the ratio.
#
# COST -- MEASURED ON THIS BENCHMARK, 2026-09-16, job 18021830.
#   608,686 samples/epoch at **1,580 samples/s** = **386 s/epoch**, so 30 epochs is
#   **~3.2 h** of training plus ~10 min of GO-graph load and the control-embedding scan.
#   Budget ~3.5 h per arm. The limit below is 8 h: margin, not estimate.
#
# ⚠️ The research tree's 14,135 samples/s does NOT transfer and must not be used to size
# this job. It was measured on a different corpus with a different estimand, and the first
# launch of this grid was given a 3 h limit on the strength of it -- every arm would have
# been killed at about epoch 23, which silently confounds the ablation with training
# length. The 8.9x discrepancy is most likely I/O contention: 13 concurrent streaming jobs
# reading the same three h5ad files over GPFS.

#SBATCH --job-name=rhj_grid
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/rhj_grid_%A_%a.out
#SBATCH --error=runs/slurm/rhj_grid_%A_%a.err

set -euo pipefail

# Slurm COPIES the batch script into a node spool directory, so ${BASH_SOURCE[0]} does not
# point into the repo. Prefer the submit dir, and verify whichever candidate wins.
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
mkdir -p runs/slurm runs/rhj_fewshot

export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

GRID="${GRID:-configs/rhj_fewshot_grid.tsv}"
CONFIG="${CONFIG:-configs/rhj_fewshot.yaml}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Row IDX of the data rows (comments and the header are skipped). Read from a checked-in
# file rather than computed, so an arm cannot silently become a duplicate of its neighbour.
ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((IDX + 1))p")"
if [ -z "$ROW" ]; then
    echo "no grid row at index $IDX in $GRID" >&2
    exit 1
fi
read -r ARM KIND TAG INTERACT FMODE FHID FMF LOSS SEED <<<"$ROW"

# Deterministic run dir: arm id first so `ls` sorts in grid order, every varied axis in the
# name, no timestamp and no randomness -- re-running an arm returns to the same directory,
# which is what makes --resume auto and the ledger's block-on-hash-change work.
SUFFIX="${ARM}__${TAG}"

echo "=== rhj_fewshot grid arm ${ARM} (${KIND}) ==="
echo "host      : $(hostname)"
echo "started   : $(date -Is)"
echo "git sha   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm     : job ${SLURM_ARRAY_JOB_ID:-?} task ${IDX}  qos ${SLURM_JOB_QOS:-?}"
echo "arm       : interaction=${INTERACT} film_mode=${FMODE} film_hidden_dim=${FHID}"
echo "            film_mean_free=${FMF} loss=${LOSS} seed=${SEED}"
echo "run dir   : runs/rhj_fewshot_${SUFFIX}/vcell"
echo

# The job id goes INSIDE the run dir, never in its name, so a requeued array task does not
# fork a second directory.
mkdir -p "runs/rhj_fewshot_${SUFFIX}/vcell"
cat > "runs/rhj_fewshot_${SUFFIX}/slurm_job.json" <<JSON
{"arm_id": "${ARM}", "kind": "${KIND}", "tag": "${TAG}",
 "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}", "slurm_task_id": "${IDX}",
 "qos": "${SLURM_JOB_QOS:-}", "host": "$(hostname)", "started": "$(date -Is)"}
JSON

FMF_BOOL=$([ "$FMF" = "1" ] && echo true || echo false)

python -m occam.train --config "$CONFIG" --stages vcell \
    --run-suffix "$SUFFIX" \
    --set "stages.vcell.interaction.interaction_type=${INTERACT}" \
    --set "stages.vcell.interaction.film_mode=${FMODE}" \
    --set "stages.vcell.interaction.film_hidden_dim=${FHID}" \
    --set "stages.vcell.interaction.film_mean_free=${FMF_BOOL}" \
    --set "stages.vcell.loss.loss_type=${LOSS}" \
    --set "seed=${SEED}"

echo
echo "finished  : $(date -Is)"
echo
echo "Collect every arm -- including the ones that did not run -- with:"
echo "    python scripts/collect_rhj_results.py"
