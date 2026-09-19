#!/usr/bin/env bash
# TxPert on OUR rhj_fewshot data, 5 seeds. Official hyperparameters, our dataloader.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4%2 \
#       scripts/slurm/competitor_train_txpert.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This job's question: what does TxPert score on OUR instrument -- our three h5ads, our frozen
# 6,983-gene axis, our 611-perturbation split-half cohort, our cell-eval 0.6.0?
#
# VARIED: the seed (42-46). HELD FIXED: every hyperparameter from the upstream recipe
# (reproducibility/txpert__replogle_nadig_fewshot/launch.train.sh), transcribed unchanged --
# 50 epochs, lr 3e-4, validation cell-eval every 10 epochs, checkpoint on
# `val/celleval/selection_score` = mean(pearson_delta, discrimination_score_l1), mode=max.
#
# FOUR DELIBERATE DEVIATIONS, all recorded in docs/rhj_competitors.md:
#   1. `occam: true` / `zarr: false` -- it reads OUR h5ads through
#      competitors/shared/occam_data.py, not a the upstream benchmark zarr dump. No intermediate format.
#   2. `kg_list: [go, string]` -- pxmap-top1 and txmap-top1 REMOVED, so TxPert and Occam see
#      the same prior knowledge (Occam runs graph_types: [go, string]). ⚠️ Their published
#      0.50787 was measured with FOUR KGs; this arm is therefore not their number.
#   3. `hvg_number: 6983` -- our axis, not their 2,000.
#   4. Normalisation is ours-for-them: log1p(counts / rowsum_FULL_PANEL * 10_000). Their zarr
#      arrived pre-normalised at 1e4, which is why every config passes target_sum=null,
#      log1p=false. Our h5ads are raw counts, so occam_data does that step.
#
# WHAT EACH OUTCOME MEANS
#   * above `transfer` 0.2323 -> it is earning its parameters on this benchmark
#   * between `meanpert` 0.1324 and `transfer` -> it learned something perturbation-specific
#     but less than copying a measured neighbouring context
#   * at or below `meanpert`, with `discrimination_score_l1` near 0.5 -> it is predicting a
#     perturbation-independent mean, whatever its loss curve says
#   * far outside 0.0-0.4 -> suspect the PORT, not the model. Their own half-split
#     reproducibility ceiling was 0.386 on their panel; ours is 0.3722.
#
# ⚠️ THE TEST NUMBER DOES NOT COME FROM THIS JOB. `test_after_fit` and `celleval.enable` are
# false. This leg only trains and selects on validation; the reported number comes from
# competitor_export.sbatch -> competitor_score.sbatch, through OUR scorer.
#
# SIZING: upstream measured ~3 min/epoch on an H100, 50 epochs plus 5 validation cell-eval
# passes = ~10 h. Our axis is 3.5x wider (6,983 vs 2,000) and our training set larger, so 10 h
# is the floor, not the budget. Submit at 16 h and record what it actually takes.

#SBATCH --job-name=cmp_txpert
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=runs/slurm/cmp_txpert_%A_%a.out
#SBATCH --error=runs/slurm/cmp_txpert_%A_%a.err
#SBATCH --exclude=hop39
# ⚠️ hop39's GPU is DEAD -- nvidia-smi there returns "Unable to determine the device
# handle for GPU0 ... Unknown Error" / "No devices were found" -- while sinfo still
# reports it `mix`, so Slurm keeps scheduling onto it and every task fails with
# "CUDA unknown error". REMOVE this line once the node is drained/fixed.

source "$(dirname "${BASH_SOURCE[0]}")/_competitor_common.sh" 2>/dev/null || \
    source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/_competitor_common.sh"

HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

ARM=txpert
SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
EPOCHS="${EPOCHS:-50}"
CHECK_VAL="${CHECK_VAL:-10}"
TAG="occam_s${SEED}${TAG_SUFFIX:-}"

# ─── the control regime the model is TRAINED on ────────────────────────────────────────────
# TxPert's published config trains on the (cell_type, batch) MEAN control:
# `datamodule.match_cntr=true, avg_cntr=true` -> `_get_control_data` returns
# `control_data[control_ct_batch_indices[ct][batch]].mean(0)`. But our benchmark protocol
# scores every arm from ONE drawn half-B control cell, so a mean-trained TxPert would be fed
# an input regime it never saw, AND would be the only arm receiving a denoised basal if we
# matched inference to its training instead.
#
# So this is an explicit A/B, both arms inferenced on single cells:
#   TXPERT_CNTR=single  (match_cntr=false, avg_cntr=false)  train == inference regime
#   TXPERT_CNTR=mean    (match_cntr=true,  avg_cntr=true)   TxPert's published regime
# Everything else is identical, so the difference IS the train/inference regime mismatch.
TXPERT_CNTR="${TXPERT_CNTR:-single}"
case "$TXPERT_CNTR" in
    single) MATCH_CNTR=false ; AVG_CNTR=false ;;
    mean)   MATCH_CNTR=true  ; AVG_CNTR=true  ;;
    *) echo "FATAL: TXPERT_CNTR must be 'single' or 'mean', got '$TXPERT_CNTR'" >&2; exit 1 ;;
esac

# ⚠️ OMIT THE CONTROL TOKEN, on BOTH sides. `-1` is the second slot of TxPert's GEARS-format
# condition string -- `data_utils.py` rewrites every single-gene perturbation as "SYMBOL+ctrl"
# to fill an unused combinatorial slot -- and our cohort is single-gene throughout. With
# `omit_cntr=true`, `gears_derivative.py:261` skips it, so training and our export agree
# without the export having to synthesise a token that carries no signal on this data.
# This IS a deviation from TxPert's published `omit_cntr: false`; recorded as one.
OMIT_CNTR="${OMIT_CNTR:-true}"
OUT_ROOT="$HERE/runs/competitors/txpert"
CKPT_DIR="$OUT_ROOT/ckpt/$TAG"
VAL_DIR="$OUT_ROOT/valscore/$TAG"

# A final epoch that is never validated can never be selected, so the run would silently be
# unable to choose its own last checkpoint. Assert rather than trust.
if (( EPOCHS % CHECK_VAL != 0 )); then
    echo "FATAL: epochs=$EPOCHS is not a multiple of check_val_every_n_epoch=$CHECK_VAL," >&2
    echo "       so the final epoch would never be validated or selectable." >&2
    exit 1
fi

PY="$(competitor_venv_python "$ARM" "$HERE")" || exit 1
competitor_banner "$ARM" train "$SEED" "$HERE"
echo "  epochs $EPOCHS, validate every $CHECK_VAL · ckpt $CKPT_DIR"

mkdir -p "$CKPT_DIR" "$VAL_DIR" "$OUT_ROOT/wandb/$TAG"
export WANDB_DIR="$OUT_ROOT/wandb/$TAG"
export WANDB__SERVICE_WAIT=300
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

# ⚠️ Stale manifests break re-runs. Every upstream script rm -f's this first.
rm -f "$VAL_DIR/celleval_manifest.json"

# `.venv/bin/python`, NOT `uv run python`: uv re-verifies and LOCKS the venv on every
# invocation, and two upstream tasks hung indefinitely inside uv before python started.
#
# Keep every comment ABOVE this invocation. See the warning in _competitor_common.sh.
cd "$HERE/competitors/TxPert"
"$PY" -u main.py \
    --config-name=config-txpert-occam \
    training.seed="$SEED" \
    training.epochs="$EPOCHS" \
    training.check_val_every_n_epoch="$CHECK_VAL" \
    datamodule.occam_bench_dir="$HERE/data/benchmarks/rhj_fewshot" \
    datamodule.match_cntr="$MATCH_CNTR" \
    datamodule.avg_cntr="$AVG_CNTR" \
    model.omit_cntr="$OMIT_CNTR" \
    wandb.name="txpert-occam-s${SEED}" \
    training.checkpoint.dirpath="$CKPT_DIR" \
    training.validation_celleval.output_dir="$VAL_DIR"

echo "=== TRAIN DONE ==="
MARKER="$CKPT_DIR/best_ckpt_path.txt"
if [ -f "$MARKER" ]; then
    echo "selected checkpoint: $(cat "$MARKER")"
    echo "next: ARM=txpert SEED=$SEED sbatch --array=0-4 scripts/slurm/competitor_export.sbatch"
else
    echo "WARNING: no marker at $MARKER; the export leg will refuse to run" >&2
fi
ls -la "$CKPT_DIR"
