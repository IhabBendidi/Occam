#!/usr/bin/env bash
#SBATCH --job-name=norman_repro_eval
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=hop39
#SBATCH --array=0-4%6
#SBATCH --output=runs/slurm/norman_repro_eval_%A_%a.out
#SBATCH --error=runs/slurm/norman_repro_eval_%A_%a.err
# =====================================================================
# REPRODUCE the norman_combo main result (published arm NQ02) -- EVALUATION
#
# WHAT THIS MEASURES
#   two effect-size-controlled panels of 36 DOUBLE perturbations each (additive / nonadditive), predicted from singles only, under the paired split-half protocol: half-A of each
#   perturbation's cells is the reference the metrics are computed against,
#   half-B is the target, and the model is fed half-B CONTROLS -- so it never
#   sees a cell that appears on the scoring side. 5 seeded splits, cell-eval
#   0.6.0, --calibrate none (nocal). One array task per split.
#
# WHAT EACH OUTCOME MEANS
#   Reproduced: additive pearson_delta 0.8169 +- 0.0009, nonadditive 0.7380 +- 0.0018, under compose=decode_seq_mean
#   At or below meanpert: the arm learned nothing usable on this panel.
#
# ANCHORS: meanpert 0.4634/0.4016 . sum_of_singles 0.8824/0.7757 <- THE BAR . experimental 0.9165/0.8377
#   All four headline metrics matter, and `mae` is an ERROR -- LOWER is better.
#   `disc_l1` chance is 0.5. Do NOT rank on `overlap_at_N` alone: a zero-signal
#   predictor beats a real model on it (the emission-floor artifact).
#
# USAGE
#   sbatch scripts/repro/norman_eval.sh                      # the published checkpoint
#   sbatch scripts/repro/norman_eval.sh path/to/best_model.pth   # your own
#   MODE=reference sbatch scripts/repro/norman_eval.sh       # score the baselines instead
#   MODE=reference ARMS=meanpert,experimental sbatch ...  # a subset of the baselines
#   COMPOSE=sum sbatch scripts/repro/norman_eval.sh       # a different composition mode
#
# COMPOSE defaults to `decode_seq_mean`, the adopted default
#   (docs/default_recipe.md). The 12 modes share ONE checkpoint: they differ
#   only in how a PAIR is composed at inference, so switching mode needs no
#   retraining. The mode is part of the result's identity -- always report it
#   next to the number.
# WARNING: THE %6 ARRAY THROTTLE IS LOAD-BEARING. cell-eval's Wilcoxon backend
#   (pdex) races on /dev/shm; at %12 roughly half of the marginal tasks failed.
#   Do not raise it for evaluation.
#
# COST  ~35 min per split on one H100; 5 splits at 5 concurrent = ~40 min wall.
# =====================================================================
set -euo pipefail

# SLURM_SUBMIT_DIR first: under sbatch BASH_SOURCE is the spool copy, outside the repo.
HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE"
mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export OCCAM_CELLEVAL_ATTEMPTS="${OCCAM_CELLEVAL_ATTEMPTS:-3}"

SPLIT="${SLURM_ARRAY_TASK_ID:-0}"
PANEL="${PANEL:-data/benchmarks/norman_combo/splithalf}"
CKPT="${1:-runs/norman_combo_NQ02__smooth_l1_nbw0_nbs1.0_dcos0.32977_mag0.0400866_lmse0.00198826_e150_em3e-4_s42/vcell/checkpoints/best_model.pth}"
MODE="${MODE:-model}"
OUT="${OUT:-runs/norman_combo/repro_eval}"
CACHE="${CACHE:-runs/norman_combo/_de_real_shared}"
mkdir -p "$OUT" "$CACHE"

if [ "$MODE" != model ] && [ "$MODE" != reference ]; then
    echo "MODE must be 'model' or 'reference', got '$MODE'" >&2
    exit 2
fi

# The argument list is built explicitly. An inline `$( ... && echo --flag )`
# drops the flag silently when the test is false, and under `set -e` a bare
# `[ ... ] && echo` aborts the script on the false branch.
ARGS=(--mode "$MODE" --panel-dir "$PANEL" --split "$SPLIT"
      --out-root "$OUT" --de-real-cache-dir "$CACHE"
      --calibrate none --num-threads 16)

if [ "$MODE" = model ]; then
    if [ ! -f "$CKPT" ]; then
        echo "checkpoint not found: $CKPT" >&2
        echo "train one with scripts/repro/norman_train.sh, or download the published" >&2
        echo "checkpoint from the HuggingFace repo (see docs/hf_release.md)" >&2
        exit 1
    fi
    ARGS+=(--checkpoint "$CKPT")
    ARGS+=(--compose "${COMPOSE:-decode_seq_mean}")
fi

# `--arms` restricts which REFERENCE arms are scored. eval_splithalf.py reads it only
# inside its reference branch, so passing it in model mode is accepted by argparse and
# then silently ignored -- refuse instead of pretending it did something.
if [ -n "${ARMS:-}" ]; then
    if [ "$MODE" != reference ]; then
        echo "ARMS only applies with MODE=reference (it selects which BASELINES to" >&2
        echo "score). In model mode it would be silently ignored." >&2
        exit 2
    fi
    ARGS+=(--arms "$ARMS")
fi

echo "=== norman_combo repro eval . split $SPLIT . mode $MODE ==="
echo "host   : $(hostname)"
echo "started: $(date -Is)"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "panel  : $PANEL"
if [ "$MODE" = model ]; then echo "ckpt   : $CKPT"; fi
echo "out    : $OUT/split_$(printf %02d "$SPLIT")"
echo

python scripts/eval_splithalf.py "${ARGS[@]}"

echo
echo "finished: $(date -Is)"
echo "next: python scripts/export_split_metrics.py --out data/release/metrics"
