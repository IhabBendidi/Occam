#!/usr/bin/env bash
# norman_combo MODEL eval: one (arm, composition mode, split) per array task.
#
#   python scripts/norman/build_mode_grid.py            # writes configs/norman_mode_eval.tsv
#   GRID=configs/norman_mode_eval.tsv \
#   sbatch --qos=vco --wckey=vco --array=0-359%6 scripts/slurm/norman_models.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that exact perturbation was
# never measured. Here the form is COMPOSITION -- train on single-gene perturbations, predict
# PAIRS.
#
# This leg asks TWO questions at once, on one set of weights each:
#   1. which loss base composes best (M03 / M06 / Q02), and does the 150-epoch schedule that
#      won on rhj (+0.0260 there) transfer to a different dataset AND a different task;
#   2. which of the 12 COMPOSITION MODES turns a singles-trained model into a double
#      prediction best.
#
# ⚠️ THE MODES NEED NO RETRAINING. They are inference strategies and are algebraically
# identical on a single perturbation, which is all training ever sees
# (tests/test_composition_modes.py pins that identity). So the same checkpoint answers all 12
# ways and the comparison is perfectly controlled.
#
# ⚠️ TWO EMISSION FAMILIES, and a number from one is NOT directly comparable to a number from
# the other. `delta_sum_realctrl` and `resid_seq_*` add their deltas to the REAL control cell
# (the `injctrl` convention); the other eight add them to the model's own reconstruction. On a
# single perturbation the families differ by the decoder's reconstruction error, which
# F-recon1 measured as LARGER than the effect itself. Read `resid_seq_*` against
# `delta_sum_realctrl` -- that pair isolates SEQUENCING with emission held fixed -- and never
# against `sum` alone.
#
# HOW TO READ THE RESULT
# ----------------------
#   python scripts/norman/norman_report.py --runs runs/norman_combo/eval --also-all
# Scored panels are `additive` and `nonadditive` (36 pairs each, effect-size controlled).
# THE BAR IS `sum_of_singles`: 0.8824 / 0.7757 pearson_delta, 0.0126 / 0.0168 mae,
# 0.2865 / 0.2750 ovl@N. It is this architecture's own operator in closed form, so an arm at
# or below it has learned nothing about interaction. `experimental` is 0.9165 / 0.8377.

#SBATCH --job-name=norman_eval
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=hop39
#SBATCH --output=runs/slurm/norman_eval_%A_%a.out
#SBATCH --error=runs/slurm/norman_eval_%A_%a.err

set -euo pipefail
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

GRID="${GRID:-configs/norman_mode_eval.tsv}"
N_SPLITS="${N_SPLITS:-5}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ROW_I=$(( IDX / N_SPLITS ))
SPLIT=$(( IDX % N_SPLITS ))

ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((ROW_I + 1))p")"
if [ -z "$ROW" ]; then
    echo "no grid row at index $ROW_I (task $IDX, $N_SPLITS splits per row)" >&2
    exit 1
fi
# ⚠️ COMPOSE is the LAST field read, so it absorbs any extra column. The `read -r` tail-absorb
# rule is what put "1e-7<TAB>M03" inside an eta_min once; keep the column count at 4.
read -r ARM TAG COMPOSE REST <<<"$ROW"
if [ -n "${REST:-}" ]; then
    echo "grid row $ROW_I has extra fields (${REST}); this launcher reads exactly 4" >&2
    exit 1
fi

CKPT="runs/norman_combo_${ARM}__${TAG}/vcell/checkpoints/best_model.pth"
[ -f "$CKPT" ] || { echo "checkpoint missing for ${ARM}: $CKPT" >&2; exit 1; }

CACHE="runs/norman_combo/_de_real_shared"
mkdir -p "$CACHE"
# the mode MUST be in the output path: 12 modes share one checkpoint, and without it they
# would all write into the same split dir and overwrite each other's scores
OUT="runs/norman_combo/eval/${ARM}__${COMPOSE}"
mkdir -p "$OUT"

echo "=== norman_combo eval · ${ARM} · compose=${COMPOSE} · split ${SPLIT} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "ckpt    : $CKPT"
echo "out     : $OUT/split_$(printf %02d "$SPLIT")"
echo

python scripts/eval_splithalf.py \
    --mode model \
    --panel-dir data/benchmarks/norman_combo/splithalf \
    --checkpoint "$CKPT" \
    --compose "$COMPOSE" \
    --split "$SPLIT" \
    --out-root "$OUT" \
    --de-real-cache-dir "$CACHE" \
    --calibrate none \
    --num-threads 16

echo
echo "finished: $(date -Is)"
