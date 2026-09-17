#!/usr/bin/env bash
# The adjacency-map sweep: optimisation knobs x FiLM axes, at batch 2048. 486 arms.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-485%27 \
#          scripts/slurm/rhj_adjmap_sweep.sbatch
#   sbatch --qos=vco --wckey=vco --array=0-485%12 scripts/slurm/rhj_adjmap_sweep.sbatch
#
# qos and wckey must MATCH. Arms come from the CHECKED-IN TSV, one row per array index.
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was
# never measured, from a single control cell.
#
# This sweep's question: with the SIMPLE ADJACENCY MAP as the perturbation model and batch
# size fixed at 2048, which optimisation setting and which FiLM configuration work here?
#
# Varied (486 arms): lr x decoder_lr x cosine_loss_weight x data_perts_only x film_mode
# x film_hidden_dim x film_mean_free, fully crossed, minus the 162 cells where
# data_perts_only=1 meets film_mean_free=1 -- confounded by construction, because mean-free
# subtracts a reference averaged over the whole node table and data_perts_only restricts
# that table.
#
# HELD FIXED: pert_model_type=adjacency_map (standing decision), batch_size=2048,
# loss_type=smooth_l1 (the only F6 result that survived F7: 12/12 over nb),
# film_scale=1.0, epochs=30, seed=42, and the frozen benchmark artifact.
#
# WHY BATCH SIZE IS NOT AN AXIS
# ----------------------------
# Measured (docs/rhj_fewshot.md F8): batch 256 -> 2048 is 6.46x faster, and the adjacency
# map alone is 5.89x at batch 256, because the 4-layer GATv2 was a fixed PER-STEP cost.
# Workers do nothing (+0.7%), so this was never I/O bound. But batch size also changes the
# effective learning rate, so crossing it with lr would leave neither axis readable. It is
# fixed at the research job's value and lr is swept around it instead.
#
# ⚠️ HOW TO READ THIS -- F7 IS THE OPEN QUESTION
# ---------------------------------------------
# The 27-arm grid's validation metric swung +/-0.20 between CONSECUTIVE epochs, so its
# `best_metric` was the peak of an oscillation and no arm ordering survived. Two candidates:
# learning rate, and eval-time BatchNorm statistics from an unbalanced epoch tail.
#
# **lr is an axis here precisely to separate them.** Before ranking any arm, check the
# per-epoch curve in this log:
#     grep "pearson_delta: mean=" runs/slurm/rhj_adjmap_<job>_<task>.out
#   * low-lr curves smooth, high-lr curves alternating -> F7 is a learning-rate artefact,
#     and the low-lr arms' rankings are trustworthy.
#   * every curve still alternating -> F7 is not lr. Suspect the 11 BatchNorm1d layers in
#     the decoder and fix those before believing any arm ordering, here or in the old grid.
#
# COST: at 10,435 samples/s (gnn @ 2048) 30 epochs is ~29 min; the adjacency map should be
# faster still. The 2 h limit is margin. Disk: ~1.7 GB of checkpoints per arm, ~830 GB
# total.

#SBATCH --job-name=rhj_adjmap
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/rhj_adjmap_%A_%a.out
#SBATCH --error=runs/slurm/rhj_adjmap_%A_%a.err

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

export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

GRID="${GRID:-configs/rhj_adjmap_sweep.tsv}"
CONFIG="${CONFIG:-configs/rhj_fewshot.yaml}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((IDX + 1))p")"
if [ -z "$ROW" ]; then
    echo "no row at index $IDX in $GRID" >&2
    exit 1
fi
read -r ARM KIND TAG LR DLR COS DPO FMODE FHID FMF SEED <<<"$ROW"

DPO_BOOL=$([ "$DPO" = "1" ] && echo true || echo false)
FMF_BOOL=$([ "$FMF" = "1" ] && echo true || echo false)
SUFFIX="${ARM}__${TAG}"

echo "=== rhj adjacency-map sweep arm ${ARM} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm   : job ${SLURM_ARRAY_JOB_ID:-?} task ${IDX}  qos ${SLURM_JOB_QOS:-?}"
echo "varied  : lr=${LR} decoder_lr=${DLR} cosine=${COS} data_perts_only=${DPO_BOOL}"
echo "          film_mode=${FMODE} film_hidden_dim=${FHID} film_mean_free=${FMF_BOOL}"
echo "fixed   : pert_model=adjacency_map batch=2048 loss=smooth_l1 epochs=30 seed=${SEED}"
echo "run dir : runs/rhj_adjmap_${SUFFIX}/vcell"
echo

mkdir -p "runs/rhj_adjmap_${SUFFIX}/vcell"
cat > "runs/rhj_adjmap_${SUFFIX}/slurm_job.json" <<JSON
{"arm_id": "${ARM}", "kind": "${KIND}", "tag": "${TAG}",
 "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}", "slurm_task_id": "${IDX}",
 "qos": "${SLURM_JOB_QOS:-}", "host": "$(hostname)", "started": "$(date -Is)"}
JSON

python -m occam.train --config "$CONFIG" --stages vcell \
    --run-suffix "$SUFFIX" \
    --set "stages.vcell.graph.pert_model_type=adjacency_map" \
    --set "stages.vcell.optim.batch_size=2048" \
    --set "stages.vcell.optim.lr=${LR}" \
    --set "stages.vcell.optim.decoder_lr=${DLR}" \
    --set "stages.vcell.loss.loss_type=smooth_l1" \
    --set "stages.vcell.loss.cosine_loss_weight=${COS}" \
    --set "stages.vcell.graph.data_perts_only=${DPO_BOOL}" \
    --set "stages.vcell.interaction.interaction_type=film" \
    --set "stages.vcell.interaction.film_mode=${FMODE}" \
    --set "stages.vcell.interaction.film_hidden_dim=${FHID}" \
    --set "stages.vcell.interaction.film_mean_free=${FMF_BOOL}" \
    --set "seed=${SEED}"

echo
echo "finished: $(date -Is)"
echo
echo "Collect with:  python scripts/collect_rhj_results.py --grid ${GRID} --runs runs"
echo "⚠️ Check the per-epoch curve BEFORE ranking (F7):"
echo "    grep 'pearson_delta: mean=' runs/slurm/rhj_adjmap_${SLURM_ARRAY_JOB_ID:-JOB}_${IDX}.out"
