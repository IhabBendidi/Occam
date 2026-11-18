#!/usr/bin/env bash
# Stage-3 retrain with the softplus + negative-binomial recipe.
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
# EVALUATION
# ----------
# One file: data/eval/pert/nadig_2024_jurkat.h5ad -- 262,956 cells, 2,394
# perturbations, 12,013 'non-targeting' controls, 8,882 of the encoder's
# 44,728 genes present. Scored with ARC cell-eval v1, injctrl, per-gene
# quantile calibration fit on the control cells, present genes only.
#
# The baseline arm is the RELEASED decoder scored the same way, through the
# same method, with the same flags:
#
#   sbatch scripts/slurm/celleval_decoder.sbatch      # defaults to the release
#
# Run it: the anchors below come from a DIFFERENT encoder (base/768) and a
# different eval panel, so they set the expected direction, not the expected
# number. The comparison that decides anything is this run against that one.
#
# How to read the result, with anchors (nadig pearson_delta, base/768 encoder):
#   released decoder (rectified_sigmoid + smooth_l1, calibrated)   0.2285
#   softplus + smooth_l1, calibrated                               0.5807
#   softplus + nb, calibrated   <- what this run should reproduce  0.6009
#   softplus + nb, NO calibration                                  0.2336
#
#   Above the measured released-decoder-on-jurkat baseline: the head+loss
#   change survives the move to the small/512 encoder, and this decoder should
#   supersede the released one.
#   At or below it: the gain does not survive, and the released decoder stays.
#   Either outcome is publishable; the run is worth it for the second as much
#   as the first.
#
#   Report pearson_delta and discrimination_score_l2. NOT overlap_at_N -- a
#   zero-signal predictor beats the real model on it.
#
# Cost: ~12 h on one H100 for 200 epochs over 1.43 M cells, plus ~5 h for the
# end-of-training cell-eval (2,394 perts x 8,882 genes is ~4.6x the DE work of
# the 255-perturbation H1 panel, which takes ~60 min at 16 threads).

#SBATCH --job-name=occam_decoder_softplus_nb
#SBATCH --partition=hopper
# qos and wckey must MATCH -- the cluster rejects a submission whose wckey is
# not the qos it runs under. hooke-predict has a 128-H100 group limit, vco 32,
# so this defaults to the roomier one. Override both together:
#     sbatch --qos=vco --wckey=vco scripts/slurm/...
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
# cell-eval on a large eval file holds TWO dense (n_cells x 44,728) float32
# arrays -- predictions and targets -- plus the in-memory dataset, and
# np.concatenate doubles one of them transiently. On nadig_2024_jurkat
# (262,956 cells) that peaked at 188 GB and was killed at a 200 GB request.
# Restriction to present genes happens after the forward pass, so it does not
# bound the peak.
#SBATCH --mem=700G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/occam_decoder_softplus_nb_%j.out
#SBATCH --error=runs/slurm/occam_decoder_softplus_nb_%j.err

set -euo pipefail

# Locate the repo. Slurm COPIES the batch script into a spool directory on the
# compute node, so ${BASH_SOURCE[0]} does not point into the repo under sbatch
# -- it points at the spool copy, and cd'ing relative to it lands somewhere
# unwritable. Prefer the submit directory, and verify whichever candidate wins
# actually is the repo rather than trusting it.
_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then
            echo "$c"; return 0
        fi
    done
    echo "cannot locate the Occam repo. sbatch from the repo root, or set" >&2
    echo "OCCAM_REPO=/path/to/Occam." >&2
    return 1
}
HERE="$(_find_repo)" || exit 1
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
echo "Record the calibrated cell-eval numbers in"
echo "docs/decoder_retrain_jurkat.md -- next to the released decoder's numbers"
echo "on the same file -- BEFORE drawing any conclusion, and update"
echo "MODEL_CARD.md if this decoder supersedes the released one."
