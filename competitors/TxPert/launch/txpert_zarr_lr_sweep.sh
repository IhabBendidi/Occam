#!/usr/bin/env bash
#SBATCH --job-name=txpert_lr
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/lr_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/txpert_out/logs/lr_%A_%a.out
#SBATCH --time=0-10:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=120G
#SBATCH --array=0-8
#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# TxPert (adjacency map) on the hooke-tx replogle_nadig zarr benchmark.
#
# 3 learning rates x 3 seeds = 9 runs.
#   lr 3e-4  the sweep-xcell-v2-training-params optimum (test_pearson_delta 0.4779). These
#            three seeds are the numbers to put in the cross-method table.
#   lr 1e-3  upstream's shipped default; best observed with this dropout/hidden/latent combo
#            was 0.4727, so it is a genuine contender rather than a throwaway.
#   lr 1e-4  the sweep never probed below 3e-4. If 1e-4 wins, the optimum was on the boundary
#            and the LR axis needs re-opening.
# Everything else is config-txpert-zarr-adjmap.yaml.
#
# Sizing, measured on the 1-epoch H100 canary: ~35 s zarr load, ~125 s DE setup, ~3 min/epoch
# train, and a full 380-perturbation evaluation every epoch (check_val_every_n_epoch=1) at
# ~2 min. 50 epochs therefore lands near 4.5 h; 10 h leaves room for a busy node.
#
# Memory: the four datasets load fully into RAM (643,413 x 2,000 float32 = 5.1 GB) plus the
# 9,887^2 adjacency (~390 MB). The preprocessing caches are disabled on this path (guarded by
# xcell_v2 in _save_cache / _load_or_calculate_de_adata), so concurrent array tasks recompute
# independently and cannot race on a shared cache file -- at the cost of ~3 min setup each.

# Absolute, like ../PerturbDiff/launch/_pd_common.sh: SLURM copies the batch script into its
# own spool directory, so `dirname "$0"` points at /cm/local/apps/slurm/var/spool, not here.
TXPERT_ROOT="${TXPERT_ROOT:-/rxrx/scratch/sandbox/ihab.bendidi/bc-hooke/src/hooke-tx/competitors/TxPert}"
cd "${TXPERT_ROOT}"

# uv's default cache is on NFS while the venv is on GPFS; unpacking across that boundary takes
# hours instead of a minute. See README_HOOKE_TX.md.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"
LRS=(0.0003 0.001 0.0001)
SEEDS=(1 2 3)

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Per-task wandb directory. Sharing one WANDB_DIR across nine simultaneous `wandb.init()`
# calls hung one task indefinitely inside init (no output at all, on two different nodes,
# while the other eight ran). Give each task its own tree and bound the service handshake.
export WANDB_DIR="${WANDB_DIR:-/rxrx/scratch/sandbox/$USER/txpert_out/wandb}/task${IDX}"
export WANDB__SERVICE_WAIT=300
mkdir -p "$WANDB_DIR" /rxrx/scratch/sandbox/"$USER"/txpert_out/logs
LR=${LRS[$(( IDX / 3 ))]}
SEED=${SEEDS[$(( IDX % 3 ))]}
TAG="lr${LR}-seed${SEED}"

echo "[$(date -Is)] task ${IDX}: lr=${LR} seed=${SEED}"

# Per-run checkpoint and cell-eval directories: with the config defaults all nine tasks would
# share one path and collide on filenames.
# `.venv/bin/python`, NOT `uv run python`: uv re-verifies the environment and takes a lock on
# the venv on every invocation. With nine array tasks starting at once against one venv on
# shared GPFS, two of them hung indefinitely inside uv before python ever started. The venv is
# already built, so skip uv at run time entirely.
"${TXPERT_ROOT}/.venv/bin/python" -u main.py \
  --config-name=config-txpert-zarr-adjmap \
  training.lr="${LR}" \
  training.seed="${SEED}" \
  wandb.name="txpert-zarr-adjmap-${TAG}" \
  training.checkpoint.dirpath="$PWD/cache/checkpoints/txpert-zarr-adjmap/${TAG}" \
  celleval.output_dir="$PWD/outputs/celleval/${TAG}"
