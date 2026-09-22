#!/usr/bin/env bash
#SBATCH --job-name=norman_repro_train
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --exclude=hop39
#SBATCH --output=runs/slurm/norman_repro_train_%j.out
#SBATCH --error=runs/slurm/norman_repro_train_%j.err
# =====================================================================
# REPRODUCE the norman_combo main checkpoint (published arm NQ02) -- TRAINING
#
# WHAT WE ARE TRYING TO ACHIEVE OVERALL
#   Predict a perturbation's transcriptional effect in a context where it was
#   never measured, from a SINGLE control cell.
#
# WHAT THIS SCRIPT DOES
#   Trains one model on norman_combo with the DEFAULT RECIPE and nothing else:
#   Q02 loss base, 150 epochs, eta_min 3e-4, batch 2048, lr/decoder_lr 1e-3.
#   There are no --set overrides on purpose -- configs/norman_combo.yaml ships those
#   values (docs/default_recipe.md), so this command IS the recipe. If a
#   future edit changes them, tests/test_default_recipe.py fails.
#
# WHAT EACH OUTCOME MEANS
#   Matching the published number (0.8169 additive / 0.7380 non-additive pearson_delta, nocal, 5 splits)
#   reproduces the result. A LOWER number with an identical config points at
#   the environment or the data, not the recipe -- check the benchmark sha256
#   the run prints against the one in the checkpoint card. A HIGHER number is
#   seed noise unless it exceeds the +-0.002 SEM by a clear margin.
#
# ANCHORS, norman_combo: sum_of_singles 0.8824 / 0.7757
#
# COST  one H100, ~3 h wall clock. Writes
#   runs/norman_combo_repro__*/vcell/checkpoints/{best_model,final_model}.pth (2.2 GB each).
#   final_model.pth is written ONLY at the end, so it is the completion marker.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

SUFFIX="${SUFFIX:-repro_NQ02_s${SEED:-42}}"

echo "=== norman_combo repro training (arm NQ02) ==="
echo "host   : $(hostname)"
echo "started: $(date -Is)"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "config : configs/norman_combo.yaml  (no overrides -- the config IS the recipe)"
echo "suffix : $SUFFIX"
echo

python -m occam.train \
    --config configs/norman_combo.yaml \
    --stages vcell \
    --run-suffix "$SUFFIX" \
    ${SEED:+--set "seed=$SEED"}

echo
echo "finished: $(date -Is)"
echo "checkpoint: runs/norman_combo_$SUFFIX/vcell/checkpoints/best_model.pth"
echo "next: sbatch scripts/repro/norman_eval.sh runs/norman_combo_$SUFFIX/vcell/checkpoints/best_model.pth"
