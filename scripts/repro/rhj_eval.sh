#!/usr/bin/env bash
#SBATCH --job-name=rhj_repro_eval
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=hop39
#SBATCH --array=0-4%6
#SBATCH --output=runs/slurm/rhj_repro_eval_%A_%a.out
#SBATCH --error=runs/slurm/rhj_repro_eval_%A_%a.err
# Evaluate the published rhj all-source model (ABL_N_ALL), five split-half evaluations.
# Set SEED=42..46 to select a released checkpoint; pass a path as the first argument for your own.
# MODE=reference scores baselines and experimental reproducibility. Keep the %6 array throttle.
set -euo pipefail

# SLURM_SUBMIT_DIR first: under sbatch BASH_SOURCE is the spool copy, outside the repo.
HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE"
mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export OCCAM_CELLEVAL_ATTEMPTS="${OCCAM_CELLEVAL_ATTEMPTS:-3}"

SPLIT="${SLURM_ARRAY_TASK_ID:-0}"
PANEL="${PANEL:-data/benchmarks/rhj_fewshot/splithalf}"
CKPT="${1:-checkpoints/vcell/rhj/seed_${SEED:-42}/best_model.pth}"
MODE="${MODE:-model}"
OUT="${OUT:-runs/rhj_fewshot/S${SEED:-42}_ABL_N_ALL__repro}"
CACHE="${CACHE:-runs/rhj_fewshot/_de_real_shared}"
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
        echo "train one with scripts/repro/rhj_train.sh, or download the published" >&2
        echo "checkpoint from the HuggingFace repo (see docs/hf_release.md)" >&2
        exit 1
    fi
    ARGS+=(--checkpoint "$CKPT")
fi

# `--arms` restricts which REFERENCE arms are scored. eval_splithalf.py reads it only
# inside its reference branch, so passing it in model mode is accepted by argparse and
# then silently ignored -- refuse instead of pretending it did something.
if [ "${OOD:-0}" = 1 ]; then ARGS+=(--ood-context); fi

if [ -n "${ARMS:-}" ]; then
    if [ "$MODE" != reference ]; then
        echo "ARMS only applies with MODE=reference (it selects which BASELINES to" >&2
        echo "score). In model mode it would be silently ignored." >&2
        exit 2
    fi
    ARGS+=(--arms "$ARMS")
fi

echo "=== rhj_fewshot repro eval . split $SPLIT . mode $MODE ==="
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
