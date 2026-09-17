#!/usr/bin/env bash
#SBATCH --job-name=pd_canary
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/canary_%j.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/logs/canary_%j.out
#SBATCH --time=0-01:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=100G
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# GPU canary: everything so far was verified on CPU only. Runs a short training
# job and a truncated eval to prove the CUDA path works and to measure step time,
# so the real sweep's --time can be sized from data instead of guessed.

source /rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/PerturbDiff/launch/_pd_common.sh
cd "${PD_ROOT}"
mkdir -p "${PD_OUT}/logs"

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PD_PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

STEPS=300
echo "=== train ${STEPS} steps on GPU (grouping=pert_celltype, genept) ==="
/usr/bin/time -v "${PD_PY}" ./src/apps/run/rawdata_diffusion_training.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  cov_encoding.replogle_gene_encoding=genept \
  run_name=pd_canary \
  trainer.devices=[0] trainer.max_steps=${STEPS} \
  trainer.limit_val_batches=2 trainer.num_sanity_val_steps=0 \
  optimization.micro_batch_size=${PD_MICRO_BATCH} \
  optimization.optimizer.lr=0.002 \
  data.use_cell_set=${PD_CELL_SET} data.num_workers=${PD_WORKERS} \
  data.cell_set_group_by=pert_celltype \
  lightning.callbacks.checkpoint.monitor=null \
  lightning.callbacks.checkpoint.save_top_k=1 \
  lightning.callbacks.checkpoint.every_n_train_steps=${PD_CKPT_EVERY} \
  "${NO_WANDB[@]}" 2>&1 | grep -viE "grad_2\.0_norm" | tail -40

echo
echo "=== eval: 20 batches + cell-eval (timing probe, NOT a benchmark number) ==="
"${PD_PY}" ./src/apps/run/rawdata_diffusion_sampling.py \
  data=ihabxcell_zarr path=hooke_tx_path cov_encoding=hooke_tx_zarr \
  cov_encoding.replogle_gene_encoding=genept \
  run_name=pd_canary_eval \
  model_checkpoint_path="${PD_OUT}/runs/pd_canary" \
  device=cuda trainer.devices=[0] \
  optimization.micro_batch_size=${PD_MICRO_BATCH} \
  data.use_cell_set=${PD_CELL_SET} data.num_workers=${PD_WORKERS} \
  data.cell_set_group_by=pert_celltype \
  sampling.num_sampled_batches=20 sampling.progress=false \
  sampling.run_cell_eval=true \
  sampling.output_dir="${PD_OUT}/eval/pd_canary" \
  "${NO_WANDB[@]}" 2>&1 | tail -25

echo "=== CANARY DONE ==="
