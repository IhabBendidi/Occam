#!/usr/bin/env bash
# PerturbDiff on OUR rhj_fewshot data, 5 seeds. Official hyperparameters, our dataloader.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4%2 \
#       scripts/slurm/competitor_train_perturbdiff.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This job's question: what does PerturbDiff score on OUR instrument -- our three h5ads, our
# frozen 6,983-gene axis, our 611-perturbation split-half cohort, our cell-eval 0.6.0?
#
# VARIED: the seed (42-46). HELD FIXED: every hyperparameter from
# reproducibility/perturbdiff__replogle_nadig_fewshot/launch.real.train.sh, transcribed
# unchanged -- 80,000 steps, lr 2e-3, micro_batch 128, cell_set 16 grouped by pert_celltype,
# EMA decay 0.99 every 10 steps, genept perturbation encoding, 5 validation cell-eval passes,
# checkpoint on `validation/celleval/selection_score`, mode=max, save_top_k=1.
#
# TWO DELIBERATE DEVIATIONS, recorded in docs/rhj_competitors.md:
#   1. `data=occam_h5ad` with `source_format: h5ad` -- it reads OUR h5ads through
#      competitors/shared/occam_data.py, not a hooke-tx zarr dump. Its h5ad dataset IO already
#      exists, and it RAISES on a missing gene rather than zero-filling, which is the behaviour
#      we want and is kept.
#   2. Our 6,983-gene axis instead of their 2,000. ⚠️ THREE values must move together:
#      `data.pad_length` (in the config), `model.input_dim` and `model.hidden_num[0]`
#      (overridden below). `data.embed_shape` interpolates `model.input_dim` and the
#      collate ASSERTS embed_shape == pad_length, so a half-applied change fails at the
#      first batch rather than training on a silently wrong width.
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
# ⚠️ ITS OWN SEED NOISE IS ±0.023 on their panel -- the largest of the four arms. A single-seed
# gap under ~0.05 against another arm means nothing for PerturbDiff. That is why this leg is an
# array over 5 seeds and not a single run.
#
# ⚠️ EMISSION SPACE. It emits gene space through `output_fn: relu`, so its raw output is
# non-negative but NOT guaranteed to be log1p-shaped. Its Phase-A adapter declares
# `from_log1p` explicitly and competitors/shared/emission_space.py::assert_same_space REFUSES
# rather than warns. Never let cell-eval guess: the 100-cell heuristic is unseeded.
#
# SIZING: upstream 16 h at 100 G for 80,000 steps on a 2,000-gene panel. Our axis is 3.5x
# wider; 20 h with 128 G is the transcribed budget plus margin for the wider axis.

#SBATCH --job-name=cmp_pdiff
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=runs/slurm/cmp_pdiff_%A_%a.out
#SBATCH --error=runs/slurm/cmp_pdiff_%A_%a.err

source "$(dirname "${BASH_SOURCE[0]}")/_competitor_common.sh" 2>/dev/null || \
    source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/_competitor_common.sh"

HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

ARM=perturbdiff
PD_SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
PD_MAX_STEPS="${PD_MAX_STEPS:-80000}"
PD_N_VALIDATIONS="${PD_N_VALIDATIONS:-5}"
PD_MICRO_BATCH="${PD_MICRO_BATCH:-128}"
PD_LR="${PD_LR:-0.002}"
PD_PENC="${PD_PENC:-genept}"
PD_CELL_SET="${PD_CELL_SET:-16}"
PD_GROUP="${PD_GROUP:-pert_celltype}"
PD_EMA_DECAY="${PD_EMA_DECAY:-0.99}"
PD_EMA_UPDATE="${PD_EMA_UPDATE:-10}"
PD_WORKERS="${PD_WORKERS:-4}"

if (( PD_MAX_STEPS % PD_N_VALIDATIONS != 0 )); then
    echo "FATAL: PD_MAX_STEPS=$PD_MAX_STEPS is not divisible by " >&2
    echo "       PD_N_VALIDATIONS=$PD_N_VALIDATIONS, so the cadence would not land on the" >&2
    echo "       final step and the last checkpoint could never be selected." >&2
    exit 1
fi
PD_VAL_EVERY=$(( PD_MAX_STEPS / PD_N_VALIDATIONS ))

PD_RUN_NAME="pd_occam_${PD_GROUP}_lr${PD_LR}_${PD_PENC}_cs${PD_CELL_SET}_s${PD_SEED}${TAG_SUFFIX:-}"
PD_OUT="$HERE/runs/competitors/perturbdiff"
CKPT_DIR="$PD_OUT/ckpt/$PD_RUN_NAME"

PY="$(competitor_venv_python "$ARM" "$HERE")" || exit 1
competitor_banner "$ARM" train "$PD_SEED" "$HERE"
echo "  steps $PD_MAX_STEPS · $PD_N_VALIDATIONS validations · every $PD_VAL_EVERY steps"
echo "  run $PD_RUN_NAME"

mkdir -p "$CKPT_DIR" "$PD_OUT/valscore/$PD_RUN_NAME" "$PD_OUT/runs"
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rm -f "$PD_OUT/valscore/$PD_RUN_NAME/celleval_manifest.json"

# Keep every comment ABOVE this invocation. See the warning in _competitor_common.sh.
cd "$HERE/competitors/PerturbDiff"
"$PY" ./src/apps/run/rawdata_diffusion_training.py \
    data=occam_h5ad path=occam_path cov_encoding=occam_h5ad \
    path.occam_bench_dir="$HERE/data/benchmarks/rhj_fewshot" \
    model.input_dim=6983 \
    "model.hidden_num=[6983,512]" \
    cov_encoding.replogle_gene_encoding="$PD_PENC" \
    run_name="$PD_RUN_NAME" \
    trainer.devices=[0] trainer.max_steps="$PD_MAX_STEPS" \
    trainer.num_sanity_val_steps=0 \
    trainer.check_val_every_n_epoch=null \
    trainer.val_check_interval="$PD_VAL_EVERY" \
    trainer.limit_val_batches=1.0 \
    optimization.micro_batch_size="$PD_MICRO_BATCH" \
    optimization.optimizer.lr="$PD_LR" \
    optimization.seed="$PD_SEED" \
    data.use_cell_set="$PD_CELL_SET" data.num_workers="$PD_WORKERS" \
    data.cell_set_group_by="$PD_GROUP" \
    lightning.ema.decay="$PD_EMA_DECAY" \
    lightning.ema.update_steps="$PD_EMA_UPDATE" \
    lightning.callbacks.validation_celleval.enabled=true \
    lightning.callbacks.validation_celleval.interval=1 \
    lightning.callbacks.validation_celleval.num_sampled_batches=null \
    lightning.callbacks.checkpoint.monitor="validation/celleval/selection_score" \
    lightning.callbacks.checkpoint.mode=max \
    lightning.callbacks.checkpoint.save_top_k=1 \
    lightning.callbacks.checkpoint.every_n_train_steps=null \
    lightning.callbacks.checkpoint.every_n_epochs=1 \
    lightning.callbacks.checkpoint.dirpath="$CKPT_DIR" \
    sampling.output_dir="$PD_OUT/valscore/$PD_RUN_NAME"

echo "=== TRAIN DONE: $PD_RUN_NAME ==="
echo "--- checkpoint kept (save_top_k=1) ---"
ls -la "$CKPT_DIR" || true
MARKER="$PD_OUT/runs/${PD_RUN_NAME}.best_ckpt_path"
[ -f "$MARKER" ] && echo "recorded checkpoint: $(cat "$MARKER")"
echo "next: ARM=perturbdiff SEED=$PD_SEED sbatch --array=0-4 scripts/slurm/competitor_export.sbatch"
