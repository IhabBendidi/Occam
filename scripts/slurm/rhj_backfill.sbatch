#!/usr/bin/env bash
# F-infra1 backfill: re-score ONLY the (arm, split, mode) triples whose cell-eval score is missing.
#
#   python scripts/build_backfill_grid.py --out configs/rhj_backfill.tsv --exclude-running <jobid>
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-20%5 scripts/slurm/rhj_backfill.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a cellular context where that perturbation
# was never measured, from a single control cell.
#
# This leg's question: nothing new. It repairs the record. F-infra1 -- a pdex 0.1.24 shared-
# memory race (`FileNotFoundError: '/psm_<hex>'` at pdex/_single_cell.py:140) -- made cell-eval
# return nothing for 31 triples while the SLURM task still exited 0, so several arm means are
# over fewer than 5 splits and `D03`, the weight the validation sweep pre-registered as optimal,
# cannot be reported at all.
#
# ⚠️ THROTTLE AT %5 AND DO NOT RAISE IT. The race is concurrency-driven, measured per node in
# job 18179618: hop44 ran 10 tasks and 9 failed; hop54 ran 5 and 4 failed; nodes running 1-4
# tasks almost never failed. Dropping the throttle from %20 to %5 cut the rate from 18/60 to
# 1/5. A bigger throttle does not finish sooner, it just re-creates the thing being repaired.
#
# ⚠️ EACH TASK SCORES ONLY THE MISSING MODES, from the grid's own `calibrate` column:
# `none` -> model_nocal, `quantile` -> model_cal, `both` -> both. Re-running a mode that already
# succeeded would burn ~8 min AND risk replacing a good score with a freshly raced one.
#
# ⚠️ An empty cell-eval result is now a HARD FAILURE (src/occam/vcell/panel/run.py), so a task
# that loses the race shows as FAILED here instead of COMPLETED-with-a-missing-file. Re-run the
# grid builder afterwards; anything still missing will reappear in it.
#
# HOW TO READ THE RESULT: you do not read this job. Re-run
# `python scripts/scoreboard.py --out docs/rhj_scoreboard.md` and check that every row says n=5.

#SBATCH --job-name=rhj_backfill
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/rhj_backfill_%A_%a.out
#SBATCH --error=runs/slurm/rhj_backfill_%A_%a.err
#SBATCH --exclude=hop39

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

GRID="${GRID:-configs/rhj_backfill.tsv}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((IDX + 1))p")"
if [ -z "$ROW" ]; then
    echo "no backfill row at index $IDX in $GRID" >&2
    exit 1
fi
read -r ARM TAG SPLIT CALIB MODES <<<"$ROW"

CKPT="runs/rhj_fewshot_${ARM}__${TAG}/vcell/checkpoints/best_model.pth"
[ -f "$CKPT" ] || { echo "checkpoint missing for ${ARM}: $CKPT" >&2; exit 1; }

CACHE="${DE_CACHE:-runs/rhj_splithalf/_de_real_shared}"
mkdir -p "$CACHE"
OUT="runs/rhj_splithalf/${ARM}__${TAG}"
mkdir -p "$OUT"

echo "=== F-infra1 BACKFILL · ${ARM} · split ${SPLIT} · restoring: ${MODES} (--calibrate ${CALIB}) ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "ckpt    : $CKPT"
echo "out     : $OUT/split_$(printf %02d "$SPLIT")"
echo

python scripts/eval_splithalf.py \
    --mode model \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --out-root "$OUT" \
    --de-real-cache-dir "$CACHE" \
    --calibrate "$CALIB" \
    --num-threads 16

echo
echo "finished: $(date -Is)"
