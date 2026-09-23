#!/bin/bash
# =====================================================================================
# Build the per-gene ESM2 table for ablation arm A5 BY RUNNING ESM2 (GPU path).
#
# WHY. Arm A5 of docs/ABLATION_PLAN.md replaces Y06's GO+STRING adjacency row with a frozen
# per-gene ESM2 protein embedding (FrozenEmbeddingModel -> the same MLP). Overall goal: the
# best model on SEEN perturbations in an UNSEEN cell line; this arm asks whether protein
# sequence alone carries what the graph prior carries.
#
# ⚠️ NOT NEEDED FOR THE CURRENT A5 TABLE. data/assets/esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl
#    was built on CPU (2026-09-23) from the precomputed genbio-ai/foundation-models-perturbation
#    ESM2 (D=1280) table: 18,149/18,216 vocab genes, 2,381/2,381 benchmark perturbations present.
#    Run this only to regenerate from sequences (e.g. to pin the model/pooling, which the
#    precomputed file does not record).
#
# INPUTS NOT ON THE CLUSTER (as of 2026-09-23) -- provide both before submitting:
#   FASTA : UniProt human reference proteome, one protein per gene
#           https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/
#           reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz  (gunzip)
#   ESM2 weights: fair-esm (`pip install fair-esm`, downloads to ~/.cache/torch/hub) or
#           transformers `facebook/esm2_t33_650M_UR50D` (HF cache or network). No env has
#           fair-esm; `aido` / `jupyter` / `state` pyenvs have transformers.
#
# OUTPUT: data/assets/${MODEL}_occam_rhj_fewshot.pkl (+ .json coverage report). Set OUT to
#   avoid overwriting the precomputed-derived table.
#
#   FASTA=/path/UP000005640_9606.fasta OUT=data/assets/esm2_t33_650M_UR50D_computed_occam_rhj_fewshot.pkl \
#     sbatch scripts/ablation/build_esm2_embeddings.sbatch
# =====================================================================================
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
python scripts/ablation/build_esm2_embeddings.py \
    --source compute --fasta "${FASTA}" --model "${MODEL}" --device cuda \
    ${OUT:+--out "${OUT}"}
echo "finished: $(date -Is)"
