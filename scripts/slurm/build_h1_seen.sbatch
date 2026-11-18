#!/usr/bin/env bash
#SBATCH --job-name=build_h1_seen
#SBATCH --partition=cpu
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/build_h1_seen_%j.out
#SBATCH --error=runs/slurm/build_h1_seen_%j.err
# Build data/benchmarks/h1_seen/ (seen perturbation, unseen cell line). See the docstring
# of scripts/build_h1_seen_panel.py for what it contains and why.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
# H1_TRAIN / H1_VAL / H1_TEST: the three VCC 2025 files
PYENV_VERSION=scmae python -u scripts/build_h1_seen_panel.py \
    --h1-files "${H1_TRAIN:?}" "${H1_VAL:?}" "${H1_TEST:?}" ${FORCE:+--force}
