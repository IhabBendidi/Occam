#!/usr/bin/env bash
# Score one decoder checkpoint on one eval file with ARC cell-eval v1.
#
#       sbatch scripts/slurm/celleval_decoder.sbatch
#       CKPT=runs/.../best_decoder.pth ACT=softplus \
#           sbatch scripts/slurm/celleval_decoder.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: a decoder that reconstructs expression from Occam embeddings
# faithfully enough to support perturbation prediction in an unseen cellular
# context.
#
# This run's question: how well does an EXISTING decoder checkpoint recover
# perturbation effects, measured the way the training pipeline measures its own
# decoders at the end of training?
#
# Defaults score the RELEASED decoder, which is the baseline arm for the
# softplus+NB retrain (scripts/slurm/retrain_decoder.sbatch). Both arms call
# the same method with the same flags, so the difference between them is the
# decoder and nothing else.
#
# Varied:     nothing within one run -- this measures a single checkpoint.
# Held fixed: eval file, perturbation column, calibration, mode, gene subset,
#             cell-eval version and profile.
#
# HOW TO READ IT, with anchors (nadig pearson_delta, base/768 encoder, a
# different panel -- these set the expected DIRECTION, not the number):
#   rectified_sigmoid + smooth_l1, calibrated   0.2285
#   softplus + nb, calibrated                   0.6009
#   softplus + nb, NO calibration               0.2336
#
# Read pearson_delta and discrimination_score_l2. NOT overlap_at_N: a
# zero-signal predictor beat the real model on it (TREK 0.3360 vs 0.2030) at
# chance-level discrimination.
#
# ⚠️ Calibration is fit on the eval file's CONTROL cells. It cannot manufacture
#    perturbation signal -- it only removes the decoder's per-gene shift -- but
#    it is fit on real data and is the dominant term in this measurement
#    (0.23 -> 0.60 on nadig). State CAL next to any number this produces.
#
# Cost: ~15 min on GPU for the forward pass, ~25 min to write the two ~9 GB
# h5ad, then a few hours of CPU for cell-eval itself.
#
# ⚠️ --de-batch-size is not a tuning knob here, it is load-bearing.
#    cell-eval's default of 100 (perturbation, gene) pairs per DE task turns
#    2,394 perturbations x 8,873 genes into ~212,000 pool tasks, each
#    re-attaching to the single POSIX shared-memory segment holding the
#    expression matrix. That run died with FileNotFoundError: /psm_... --
#    CPython unlinks shared-memory segments early (fixed in 3.13; cell-eval
#    runs under 3.10). Threads are back to 16, which is what every successful
#    cell-eval run on this cluster has used.

#SBATCH --job-name=occam_celleval
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
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/occam_celleval_%j.out
#SBATCH --error=runs/slurm/occam_celleval_%j.err

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

CKPT="${CKPT:-checkpoints/release/occam_small_encoder_decoder.pth}"

# Output activation. Only the RELEASED checkpoint needs this passed: it predates
# the field, and occam-celleval refuses to guess because a softplus decoder
# loads cleanly under rectified_sigmoid and emits ~99% exact zeros.
# (occam-verify determined it from a forward pass; see docs/checkpoints.md.)
#
# Anything this repo trained records the field, and occam-celleval REFUSES a
# flag that contradicts the record -- so the flag must be absent for those.
# Hence `${ACT+set}`, which distinguishes "caller did not set ACT" from
# "caller set ACT to empty on purpose"; `${ACT:-...}` would collapse the two
# and silently reimpose rectified_sigmoid on a softplus checkpoint.
if [ -z "${ACT+set}" ]; then
    case "$CKPT" in
        */occam_small_encoder_decoder.pth) ACT="rectified_sigmoid" ;;
        *)                                 ACT="" ;;
    esac
fi
EVAL_DATA="${EVAL_DATA:-data/eval/pert/nadig_2024_jurkat.h5ad}"
PERT_COL="${PERT_COL:-gene_name}"
CALIBRATE="${CALIBRATE:-quantile}"
OUT="${OUT:-runs/celleval/$(basename "${CKPT%.pth}")__$(basename "${EVAL_DATA%.h5ad}")__${CALIBRATE}}"

echo "=== occam cell-eval ==="
echo "host      : $(hostname)"
echo "started   : $(date -Is)"
echo "git sha   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "checkpoint: $CKPT"
echo "eval data : $EVAL_DATA"
echo "calibrate : $CALIBRATE"
echo "output    : $OUT"
echo

python -m occam.cli.celleval \
    --checkpoint "$CKPT" \
    --eval-data "$EVAL_DATA" \
    --output-dir "$OUT" \
    --pert-col "$PERT_COL" \
    --control-label non-targeting \
    --eval-gene-col ensembl_gene_id \
    --calibrate "$CALIBRATE" \
    --modes injctrl \
    --num-threads 16 \
    --de-batch-size 2000 \
    --de-real-cache runs/celleval/_de_real_cache \
    --timeout 36000 \
    ${ACT:+--output-activation "$ACT"}

echo
echo "finished  : $(date -Is)"
echo
echo "Record the number in docs/decoder_retrain_jurkat.md, with the checkpoint,"
echo "the calibration setting and the eval file next to it."
