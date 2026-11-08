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
# REPRODUCE the norman_combo main checkpoint (official arm NF32ALLA) -- TRAINING
#
# WHAT WE ARE TRYING TO ACHIEVE OVERALL
#   Predict a perturbation's transcriptional effect in a context where it was
#   never measured, from a SINGLE control cell.
#
# WHAT THIS SCRIPT DOES
#   Trains NF32ALLA from configs/norman_combo_official.yaml: GO+STRING, GenePT, ESM2;
#   smooth_l1 + 0.0255 * delta-cos; batch 32, 150 epochs, eta_min 3e-4.
#
# WHAT EACH OUTCOME MEANS
#   Matching the five-seed result (0.8955 additive / 0.8013 non-additive Pearson delta,
#   nocal, decode_seq_mean, 5 splits per seed)
#   reproduces the result. A LOWER number with an identical config points at
#   the environment or the data, not the recipe -- check the benchmark sha256
#   the run prints against the one in the checkpoint card. A HIGHER number is
#   seed noise unless it exceeds the measured seed spread by a clear margin.
#
# ANCHORS, norman_combo: sum_of_singles 0.8822 / 0.7749
#
# COST  one H100, ~3 h wall clock. Writes
#   runs/norman_combo_repro__*/vcell/checkpoints/{best_model,final_model}.pth (2.2 GB each).
#   final_model.pth is written ONLY at the end, so it is the completion marker.
# =====================================================================
set -euo pipefail

# SLURM_SUBMIT_DIR first: under sbatch BASH_SOURCE is the spool copy, outside the repo.
HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE"
mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

SUFFIX="${SUFFIX:-repro_NF32ALLA_s${SEED:-42}}"

echo "=== norman_combo repro training (arm NF32ALLA) ==="
echo "host   : $(hostname)"
echo "started: $(date -Is)"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "config : configs/norman_combo_official.yaml"
echo "suffix : $SUFFIX"
echo

python -m occam.train \
    --config configs/norman_combo_official.yaml \
    --stages vcell \
    --run-suffix "$SUFFIX" \
    ${SEED:+--set "seed=$SEED"}

echo
echo "finished: $(date -Is)"
echo "checkpoint: runs/norman_combo_$SUFFIX/vcell/checkpoints/best_model.pth"
echo "next: sbatch scripts/repro/norman_eval.sh runs/norman_combo_$SUFFIX/vcell/checkpoints/best_model.pth"
