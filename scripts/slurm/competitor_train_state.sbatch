#!/usr/bin/env bash
# STATE on OUR rhj_fewshot data, 5 seeds. Official hyperparameters, our dataloader.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4%2 \
#       scripts/slurm/competitor_train_state.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This job's question: what does STATE score on OUR instrument -- our three h5ads, our frozen
# 6,983-gene axis, our 611-perturbation split-half cohort, our cell-eval 0.6.0?
#
# VARIED: the seed (42-46), passed to BOTH `training.train_seed` and `data.kwargs.random_seed`.
# HELD FIXED: every hyperparameter from reproducibility/state__replogle_nadig_fewshot, unchanged
# -- 80,000 steps, lr 1e-3, batch_size 64, cell_set_len 64, hidden_dim 336, batch_encoder on,
# 5 validation cell-eval passes, checkpoint kept as `best_celleval.ckpt` on
# selection_score = mean(pearson_delta, discrimination_score_l1), mode=max.
#
# TWO DELIBERATE DEVIATIONS, recorded in docs/rhj_competitors.md:
#   1. `data=occam_h5ad` -- it reads OUR h5ads through competitors/shared/occam_data.py, not a
#      hooke-tx zarr dump. Its own ZarrPerturbationDataModule already has an `anndata`
#      source_format, which is the branch this uses. No intermediate format.
#   2. Our 6,983-gene axis instead of their 2,000 selected Ensembl ids.
#
# `target_sum=null, log1p=false` is KEPT, exactly as upstream: it means "the input arrives
# already normalised", and occam_data does that step for us at
# log1p(counts / rowsum_FULL_PANEL * 10_000). Setting these to anything else would
# double-transform.
#
# WHAT EACH OUTCOME MEANS
#   * above `transfer` 0.2323 -> earning its parameters here
#   * between `meanpert` 0.1324 and `transfer` -> perturbation-specific but weaker than copying
#     a measured neighbouring context
#   * at or below `meanpert` with `discrimination_score_l1` near 0.5 -> a
#     perturbation-independent mean
#   * far outside 0.0-0.4 -> suspect the PORT, not the model. Their own half-split
#     reproducibility ceiling was 0.386 on their panel; our `experimental` arm is
#     0.3722 on ours.
#
# ⚠️ ITS VENV HAS A BROKEN EDITABLE INSTALL. The shebang in .venv/bin/state is stale and the
# editable .pth does not resolve, so upstream invokes `.venv/bin/python -m state` with
# PYTHONPATH pointing at src/ and VIRTUAL_ENV unset. That workaround is kept VERBATIM below --
# it is not cargo cult, the arm does not start without it.
#
# SIZING: upstream ran 24 h at 320 G for 80,000 steps on a 2,000-gene panel. Our axis is 3.5x
# wider. 24 h is the transcribed budget; if it truncates, record the step count reached rather
# than silently shortening the run.

#SBATCH --job-name=cmp_state
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/cmp_state_%A_%a.out
#SBATCH --error=runs/slurm/cmp_state_%A_%a.err

source "$(dirname "${BASH_SOURCE[0]}")/_competitor_common.sh" 2>/dev/null || \
    source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/_competitor_common.sh"

HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

ARM=state
SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
MAX_STEPS="${MAX_STEPS:-80000}"
N_VALIDATIONS="${N_VALIDATIONS:-5}"
VAL_FREQ=$(( MAX_STEPS / N_VALIDATIONS ))
RUN_NAME="occam_s${SEED}${TAG_SUFFIX:-}"
OUTPUT_DIR="$HERE/runs/competitors/state"
STATE_DIR="$HERE/competitors/STATE/state"

PY="$(competitor_venv_python "$ARM" "$HERE")" || exit 1
competitor_banner "$ARM" train "$SEED" "$HERE"
echo "  steps $MAX_STEPS · $N_VALIDATIONS validations · val_freq $VAL_FREQ"
echo "  out $OUTPUT_DIR/$RUN_NAME"

mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${STATE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset VIRTUAL_ENV
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rm -f "$OUTPUT_DIR/$RUN_NAME/celleval_manifest.json"

# Keep every comment ABOVE this invocation: a `#` inside a backslash continuation is what
# dropped 29 of 32 overrides upstream and sent this very arm to `name=debug`.
"$PY" -m state tx train \
    data=occam_h5ad \
    data.kwargs.occam_bench_dir="$HERE/data/benchmarks/rhj_fewshot" \
    data.kwargs.num_workers="${NUM_WORKERS:-0}" \
    data.kwargs.embed_key='X_hvg' \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="batch_center" \
    data.kwargs.cell_type_key="cell_type" \
    data.kwargs.control_pert="empty" \
    data.kwargs.target_sum=null \
    data.kwargs.log1p=false \
    data.kwargs.feature_loading=ram \
    +data.kwargs.random_seed="$SEED" \
    training.max_steps="$MAX_STEPS" \
    training.val_freq="$VAL_FREQ" \
    training.train_seed="$SEED" \
    training.batch_size=64 \
    training.lr=1e-3 \
    training.celleval_validation.enabled=true \
    training.celleval_validation.profile=minimal \
    training.celleval_validation.every_n_validations=1 \
    model.kwargs.cell_set_len=64 \
    model.kwargs.hidden_dim=336 \
    model.kwargs.batch_encoder=True \
    model=state \
    wandb.entity="valencelabs" \
    wandb.project="Hooke-External" \
    wandb.tags="[rhj_fewshot]" \
    output_dir="$OUTPUT_DIR" \
    name="$RUN_NAME"

echo "=== TRAIN DONE: $OUTPUT_DIR/$RUN_NAME ==="
ls -la "$OUTPUT_DIR/$RUN_NAME/checkpoints/" || true
echo "next: ARM=state SEED=$SEED sbatch --array=0-4 scripts/slurm/competitor_export.sbatch"
