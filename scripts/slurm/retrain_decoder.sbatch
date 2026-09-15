#!/usr/bin/env bash
# Stage-3 retrain with the softplus + negative-binomial recipe.
#
# ⚠️ NOT SUBMITTED BY ANYTHING. Run it when you decide to:
#
#       sbatch scripts/slurm/retrain_decoder.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: a decoder that reconstructs expression from Occam embeddings
# faithfully enough to support perturbation prediction in an unseen cellular
# context.
#
# This run's question: does replacing the released decoder's rectified_sigmoid
# head and smooth-L1 loss with softplus + NB improve calibrated cell-eval on
# the small/512 encoder, as it did on the base/768 one?
#
# Varied:     output head (rectified_sigmoid -> softplus) and loss
#             (smooth_l1 -> nb), with lr moved 3e-6 -> 5e-5 to match.
# Held fixed: the frozen small/512 encoder, 2048x4 geometry, dropout, SE
#             reduction, augmentation, batch size, epochs, seed.
#
# How to read the result, with anchors:
#   released decoder (rectified_sigmoid + smooth_l1, calibrated)  nadig pΔ 0.2285
#   softplus + smooth_l1, calibrated                              nadig pΔ 0.5807
#   softplus + nb, calibrated   <- what this run should reproduce  nadig pΔ 0.6009
#   softplus + nb, NO calibration                                 nadig pΔ 0.2336
#
#   A calibrated nadig pearson_delta near 0.60 reproduces the gain.
#   At or below 0.2285 the gain does not survive the encoder change, and the
#   released decoder should stay.
#
#   Report pearson_delta and discrimination_score_l2. NOT overlap_at_N -- a
#   zero-signal predictor beats the real model on it.
#
# Cost: ~12 h on one H100 for 200 epochs over 1.43 M cells.

#SBATCH --job-name=occam_decoder_softplus_nb
#SBATCH --partition=hopper
#SBATCH --qos=vco
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=runs/slurm/occam_decoder_softplus_nb_%j.out
#SBATCH --error=runs/slurm/occam_decoder_softplus_nb_%j.err

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p runs/slurm

export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "=== occam decoder retrain: softplus + nb ==="
echo "host      : $(hostname)"
echo "started   : $(date -Is)"
echo "git sha   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

python -m occam.train --config configs/decoder_softplus_nb.yaml --stages decoder

echo
echo "finished  : $(date -Is)"
echo
echo "Record the calibrated cell-eval numbers against the anchors in this"
echo "script's header before drawing any conclusion, and update MODEL_CARD.md"
echo "if this decoder supersedes the released one."
