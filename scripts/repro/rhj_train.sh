#!/usr/bin/env bash
#SBATCH --job-name=rhj_repro_train
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --exclude=hop39
#SBATCH --output=runs/slurm/rhj_repro_train_%j.out
#SBATCH --error=runs/slurm/rhj_repro_train_%j.err
# Train the published RHJ all-source model (ABL_N_ALL).
# GO + STRING + GenePT + ESM2; 180 epochs, batch 2048, delta cosine weight 0.5.
# Set SEED=42..46; all scientific settings are in configs/rhj_fewshot_official.yaml.
set -euo pipefail

# SLURM_SUBMIT_DIR first: under sbatch BASH_SOURCE is the spool copy, outside the repo.
HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE"
mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

SUFFIX="${SUFFIX:-repro_ABL_N_ALL_s${SEED:-42}}"

echo "=== rhj_fewshot repro training (arm ABL_N_ALL) ==="
echo "host   : $(hostname)"
echo "started: $(date -Is)"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "config : configs/rhj_fewshot_official.yaml"
echo "suffix : $SUFFIX"
echo

python -m occam.train \
    --config configs/rhj_fewshot_official.yaml \
    --stages vcell \
    --run-suffix "$SUFFIX" \
    ${SEED:+--set "seed=$SEED"}

echo
echo "finished: $(date -Is)"
echo "checkpoint: runs/rhj_fewshot_$SUFFIX/vcell/checkpoints/best_model.pth"
echo "next: sbatch scripts/repro/rhj_eval.sh runs/rhj_fewshot_$SUFFIX/vcell/checkpoints/best_model.pth"
