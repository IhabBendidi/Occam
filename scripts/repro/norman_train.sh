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
# Train the published Norman all-source model (NF32ALLA).
# GO + STRING + GenePT + ESM2; 150 epochs, batch 32, delta cosine weight 0.0255.
# Set SEED=42..46; settings are in configs/norman_combo_official.yaml.
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
