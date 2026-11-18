#!/bin/bash
# Compute ESM2 embeddings from protein sequences. Set FASTA and optionally OUT.
# To reproduce the released models, download the released knowledge assets instead.
#SBATCH --job-name=esm2_table
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/esm2_table_%j.out
#SBATCH --error=runs/slurm/esm2_table_%j.err

set -euo pipefail

_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then
            echo "$c"; return 0
        fi
    done
    echo "cannot locate the Occam repo; sbatch from the repo root or set OCCAM_REPO" >&2
    return 1
}
HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

: "${FASTA:?set FASTA=<UniProt human reference proteome .fasta>}"
MODEL="${MODEL:-esm2_t33_650M_UR50D}"
export PYENV_VERSION="${PYENV_VERSION:-aido}"   # needs torch + (fair-esm or transformers) + pandas

echo "host=$(hostname) model=${MODEL} fasta=${FASTA} env=${PYENV_VERSION} started=$(date -Is)"
python scripts/build_esm2_embeddings.py \
    --source compute --fasta "${FASTA}" --model "${MODEL}" --device cuda \
    ${OUT:+--out "${OUT}"}
echo "finished: $(date -Is)"
