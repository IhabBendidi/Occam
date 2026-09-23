#!/usr/bin/env bash
#SBATCH --job-name=compress_raw
#SBATCH --partition=cpu
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/compress_raw_%j.out
#SBATCH --error=runs/slurm/compress_raw_%j.err
# IN -> IN (compressed), keeping the original as IN.uncompressed until the rebuild verifies.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
IN="${IN:?}"
python -u scripts/compress_raw_copy.py "$IN" "${IN%.h5ad}.gz_tmp.h5ad"
mv "$IN" "$IN.uncompressed"
mv "${IN%.h5ad}.gz_tmp.h5ad" "$IN"
ls -la "$IN" "$IN.uncompressed"
sha256sum "$IN"
