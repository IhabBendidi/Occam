#!/usr/bin/env bash
#SBATCH --job-name=txpert_zarr
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/train_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/train_%A_%a.out
#SBATCH --time=0-08:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=200G
#SBATCH --array=0-2
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# TxPert (adjacency map) on the hooke-tx replogle_nadig zarr benchmark, 3 seeds.
#
# Sizing from the 1-epoch canary on an H100: data load ~35 s, setup (DE genes) ~125 s,
# 9,633 train steps at ~57 it/s = ~3 min/epoch, then per-epoch evaluation over the 380
# held-out perturbations. 50 epochs therefore lands around 4-5 h; 8 h is headroom.
#
# Memory: the four zarr datasets are read fully into RAM as one dense float32 matrix
# (643,413 x 2,000 = ~5.1 GB) and the 9,887 x 9,887 adjacency adds ~390 MB, so 200 G is
# generous. Peak observed in the canary was well under half of that.

# Absolute, like ../PerturbDiff/launch/_pd_common.sh: SLURM copies the batch script into its
# own spool directory, so `dirname "$0"` points at /cm/local/apps/slurm/var/spool, not here.
TXPERT_ROOT="${TXPERT_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/TxPert}"
cd "${TXPERT_ROOT}"

# uv's default cache is on NFS while the venv is on GPFS; unpacking torch across that
# boundary takes hours instead of a minute. See README_HOOKE_TX.md.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"
export WANDB_DIR="${WANDB_DIR:-/rxrx/scratch/sandbox/$USER/txpert_out/wandb}"
mkdir -p "$WANDB_DIR" "$(dirname "${SLURM_SUBMIT_DIR:-.}")" || true

SEED=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))

# Each seed gets its own checkpoint dir and cell-eval output dir: with the config default
# they would share one path and race on filenames.
# `.venv/bin/python`, NOT `uv run python`: uv re-verifies the environment and locks the venv on
# every invocation, which can hang when several array tasks start at once on shared GPFS.
"${TXPERT_ROOT}/.venv/bin/python" -u main.py \
  --config-name=config-txpert-zarr-adjmap \
  training.seed="${SEED}" \
  wandb.name="txpert-zarr-adjmap-seed${SEED}" \
  training.checkpoint.dirpath="$PWD/cache/checkpoints/txpert-zarr-adjmap/seed${SEED}" \
  celleval.output_dir="$PWD/outputs/celleval/seed${SEED}"
