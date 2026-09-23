#!/usr/bin/env bash
#SBATCH --job-name=build_ood
#SBATCH --partition=cpu
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/build_ood_%j.out
#SBATCH --error=runs/slurm/build_ood_%j.err
# One seen-perturbation / unseen-cell-line panel build. BUILD_ARGS is passed verbatim to
# scripts/build_h1_seen_panel.py, e.g.
#   BUILD_ARGS="--dataset k562"                         -> data/benchmarks/k562_seen
#   BUILD_ARGS="--dataset k562 --from-raw data/benchmarks/k562_seen/k562_seen_raw.h5ad \
#               --eligible h1set --bench-id k562_seen_h1set"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
python -u scripts/build_h1_seen_panel.py ${BUILD_ARGS:?set BUILD_ARGS}
