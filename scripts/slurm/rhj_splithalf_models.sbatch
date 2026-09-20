#!/usr/bin/env bash
# The model arm for every (checkpoint, split). 27 arms x 5 splits = 135 array tasks.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-134%20 \
#          scripts/slurm/rhj_splithalf_models.sbatch
#
# Index arithmetic: task = arm_index * 5 + split. The arm list comes from the CHECKED-IN
# grid TSV in row order, so an arm cannot silently become its neighbour.
#
# ⛔ RUN scripts/slurm/rhj_splithalf_reference.sbatch FIRST. It scores the five
# model-independent arms and populates the shared real-side DE cache. Without it every task
# here recomputes an 8.5 min DE it could have read, and there is nothing to compare against.
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was
# never measured, from a single control cell.
#
# This sweep's question: on the 611 held-out HepG2 perturbations, scored by cell-eval
# against a half-A reference, where does each of the 27 trained arms sit relative to the
# zero-signal floor, the mean-training-delta baseline, the cross-cell-line TRANSFER
# baseline, and 1-cell replicate agreement?
#
# Varied: the checkpoint (27) x the split (5). Held fixed: the 611-perturbation cohort, the
# 6,983-gene axis, the arm construction, the drawn control cells (seeded per split, so the
# model and every baseline see the SAME controls), and control-fit quantile calibration.
#
# ⚠️ HOW TO READ IT -- F7 MATTERS HERE
# -----------------------------------
# Every checkpoint below was selected on `best_metric`, a validation metric that oscillates
# +/-0.20 between consecutive epochs (docs/rhj_fewshot.md F7). So each best_model.pth is
# whichever high-parity epoch happened to land highest, and BETWEEN-ARM differences here may
# be dominated by which epoch each arm got lucky on rather than by its configuration. The
# cause is not yet established -- learning rate and eval-time BatchNorm statistics are both
# live suspects, and an LR sweep is planned.
#
# What this sweep CAN still answer, F7 notwithstanding:
#   * where the whole population of 27 sits against `transfer` -- if all 27 fall short, that
#     is a statement about the approach, not about one lucky epoch;
#   * whether the val metric and cell-eval even agree on an ordering (this project has a
#     documented case of two such metrics ranking models OPPOSITELY);
#   * the smooth_l1 vs nb contrast, which was 12/12 consistent on validation.
# final_model.pth is deliberately NOT scored: at epoch 30 every arm sits in the LOW phase of
# the oscillation, so it is systematically the unlucky weights rather than a fair
# alternative.

#SBATCH --job-name=rhj_sh_model
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/rhj_sh_model_%A_%a.out
#SBATCH --error=runs/slurm/rhj_sh_model_%A_%a.err

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

# GRID also accepts configs/rhj_adjmap_sweep.tsv: the adjacency-map sweep writes to the
# same runs/rhj_fewshot_<arm>__<tag>/ prefix, because the orchestrator derives the run dir
# from the config's `name` and not from the launcher.
GRID="${GRID:-configs/rhj_fewshot_grid.tsv}"
N_SPLITS="${N_SPLITS:-5}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARM_I=$(( IDX / N_SPLITS ))
SPLIT=$(( IDX % N_SPLITS ))

ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((ARM_I + 1))p")"
if [ -z "$ROW" ]; then
    echo "no grid row at arm index $ARM_I (task $IDX, $N_SPLITS splits per arm)" >&2
    exit 1
fi
read -r ARM KIND TAG REST <<<"$ROW"

CKPT="runs/rhj_fewshot_${ARM}__${TAG}/vcell/checkpoints/best_model.pth"
if [ ! -f "$CKPT" ]; then
    echo "checkpoint missing for arm ${ARM}: $CKPT" >&2
    exit 1
fi

CACHE="${DE_CACHE:-runs/rhj_splithalf/_de_real_shared}"
mkdir -p "$CACHE"
OUT="runs/rhj_splithalf/${OUT_PREFIX:-}${ARM}__${TAG}"
mkdir -p "$OUT"

echo "=== rhj split-half MODEL arm · ${ARM} (${KIND}) · split ${SPLIT} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm   : job ${SLURM_ARRAY_JOB_ID:-?} task ${IDX} -> arm ${ARM_I}, split ${SPLIT}"
echo "ckpt    : $CKPT"
echo "calib   : ${CALIBRATE:-none}   center_delta=${CENTER_DELTA:-0}"
echo "de cache: $CACHE"
echo "out     : $OUT/split_$(printf %02d "$SPLIT")"
echo

cat > "$OUT/slurm_job_${SPLIT}.json" <<JSON
{"arm_id": "${ARM}", "kind": "${KIND}", "tag": "${TAG}", "split": ${SPLIT},
 "checkpoint": "${CKPT}", "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
 "slurm_task_id": "${IDX}", "qos": "${SLURM_JOB_QOS:-}", "host": "$(hostname)",
 "started": "$(date -Is)"}
JSON

python scripts/eval_splithalf.py \
    --mode model \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --out-root "$OUT" \
    --de-real-cache-dir "$CACHE" \
    --calibrate "${CALIBRATE:-none}" \
    ${CENTER_DELTA:+--center-delta} \
    --num-threads 16

echo
echo "finished: $(date -Is)"
