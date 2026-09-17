#!/usr/bin/env bash
#SBATCH --job-name=pd_train
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/train_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/train_%A_%a.out
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=100G
#SBATCH --array=0-3
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# PerturbDiff on the hooke-tx zarr benchmark: the cell-set grouping A/B plus one
# secondary axis, 2 x 2 = 4 runs.
#
#   cell_set_group_by      pert_celltype_batch (upstream) | pert_celltype (STATE-style)
#   optimizer.lr           0.002 (upstream Replogle)      | 0.0005
#
# Everything else is upstream's Replogle recipe (use_cell_set=32,
# micro_batch_size=128, replogle_gene_encoding=genept) at STATE's step budget
# (80k steps, ckpt every 2000).
#
# Sizing from the GPU canary (job 16987937, H100): 300 steps + a 20-batch eval
# + cell-eval took 2m27s wall, peak RSS 7.5 GB. 80k steps extrapolates to ~3.5 h,
# so 12 h is generous headroom.
#
# The two grouping modes differ ~13x in samples per epoch on this data, so they
# are compared at a MATCHED STEP COUNT. Equal epochs would not be equal compute.
#
# All tasks run concurrently. Add a throttle (e.g. --array=0-3%2) if the
# cluster is busy; an already-submitted array can be throttled or released with
#   scontrol update JobId=<id> ArrayTaskThrottle=<n>   (0 = unlimited)
#
# ModelCheckpoint gets a per-run dirpath. With the config default (dirpath:
# null) Lightning resolves it to <default_root_dir>/checkpoints, which is SHARED
# by every concurrent run: they collide on the {epoch}-{step} filename and get
# -v1/-v2 suffixes assigned by an existence check that is not atomic across
# processes. Results were never at risk -- eval reads the final checkpoint
# trainer.save_checkpoint() writes to the unique <runs>/<run_name> -- but the
# shared directory accumulated 57 GB of unread intermediates and two runs could
# in principle clobber each other's periodic file.

source /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff/launch/_pd_common.sh
cd "${PD_ROOT}"
mkdir -p "${PD_OUT}/logs"

pd_combo "${SLURM_ARRAY_TASK_ID}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: ${PD_RUN_NAME} ==="
echo "    grouping=${PD_GROUP}  lr=${PD_LR}  pert_encoding=${PD_PENC}"
echo "    use_cell_set=${PD_CELL_SET}  micro_batch=${PD_MICRO_BATCH}  max_steps=${PD_MAX_STEPS}"
echo "    seed=${PD_SEED}"
echo "    ema_decay=${PD_EMA_DECAY}  ema_update_steps=${PD_EMA_UPDATE}"
echo "    node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PD_PY}" ./src/apps/run/rawdata_diffusion_training.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  cov_encoding.replogle_gene_encoding="${PD_PENC}" \
  run_name="${PD_RUN_NAME}" \
  trainer.devices=[0] trainer.max_steps="${PD_MAX_STEPS}" \
  trainer.limit_val_batches=20 trainer.num_sanity_val_steps=0 \
  optimization.micro_batch_size="${PD_MICRO_BATCH}" \
  optimization.optimizer.lr="${PD_LR}" \
  optimization.seed="${PD_SEED}" \
  data.use_cell_set="${PD_CELL_SET}" data.num_workers="${PD_WORKERS}" \
  data.cell_set_group_by="${PD_GROUP}" \
  lightning.ema.decay="${PD_EMA_DECAY}" \
  lightning.ema.update_steps="${PD_EMA_UPDATE}" \
  lightning.callbacks.checkpoint.monitor=null \
  lightning.callbacks.checkpoint.save_top_k=1 \
  lightning.callbacks.checkpoint.every_n_train_steps="${PD_CKPT_EVERY}" \
  lightning.callbacks.checkpoint.dirpath="${PD_OUT}/ckpt/${PD_RUN_NAME}" \
  "${NO_WANDB[@]}"

echo "=== TRAIN DONE: ${PD_RUN_NAME} -> ${PD_OUT}/runs/${PD_RUN_NAME} ==="
