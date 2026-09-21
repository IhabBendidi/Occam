#!/usr/bin/env bash
# Score ONE per-epoch checkpoint on ONE split. Selection stage of the checkpoint sweep.
#
#   GRID=configs/rhj_ckptsweep_eval.tsv sbatch --array=0-89%6 scripts/slurm/rhj_ckptsweep_eval.sbatch
#
# WHAT THIS ANSWERS
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This leg's question: does picking a checkpoint on CELL-EVAL beat picking it on
# `val_pearson_delta`, the per-cell metric the trainer uses -- and is it even a different
# checkpoint? pipeline.py already records that the current criterion "was MEASURED to rank
# checkpoints wrongly on the real panel ... it picked one arm and put the eventual bake-off
# winner last", and that cell-eval was simply too slow to run in-loop.
#
# ⚠️ SELECTION SPLIT vs MEASUREMENT SPLITS. This scores SPLIT 0 ONLY. The chosen checkpoint is
# then scored on splits 1-4, which selection never saw, so the reported number is NOT
# selection-on-test -- the same concern recorded against TxPert's hyperparameters (T9).
#
# ⚠️ THIS IS NOT THE VALIDATION SET. The faithful design would select on the 103-perturbation
# validation cohort, but cell-eval needs a Panel artifact and no validation panel exists; one
# would have to be built with its own disjoint control pools. Split 0 of the test panel is a
# legitimate held-out selection set, but it is a DIFFERENT one: 611 perturbations, not 103.

#SBATCH --job-name=rhj_cksweep
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/rhj_cksweep_%A_%a.out
#SBATCH --error=runs/slurm/rhj_cksweep_%A_%a.err
#SBATCH --exclude=hop39

set -euo pipefail
_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then echo "$c"; return 0; fi
    done
    echo "cannot locate the Occam repo" >&2; return 1
}
HERE="$(_find_repo)" || exit 1
cd "$HERE"; mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

GRID="${GRID:-configs/rhj_ckptsweep_eval.tsv}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SPLIT="${SELECT_SPLIT:-0}"
ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((IDX + 1))p")"
[ -n "$ROW" ] || { echo "no row at $IDX" >&2; exit 1; }
read -r ARM RUNDIR CKPT <<<"$ROW"

CK="${RUNDIR}/vcell/checkpoints/${CKPT}"
[ -f "$CK" ] || { echo "missing checkpoint $CK" >&2; exit 1; }
OUT="runs/rhj_splithalf/CKSWEEP_${ARM}/${CKPT%.pth}"
mkdir -p "$OUT"
echo "=== ckpt-sweep · ${ARM} · ${CKPT} · selection split ${SPLIT} ==="
echo "host: $(hostname)   ckpt: $CK   out: $OUT"

python scripts/eval_splithalf.py --mode model --checkpoint "$CK" --split "$SPLIT" \
    --out-root "$OUT" --de-real-cache-dir runs/rhj_splithalf/_de_real_shared \
    --calibrate none --num-threads 16
