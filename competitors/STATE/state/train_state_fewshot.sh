#!/bin/bash
#SBATCH --job-name=state_fewshot
#SBATCH --output=/rxrx/scratch/sandbox/ihab.bendidi/state_out/logs/fewshot_%A_%a.out
#SBATCH --error=/rxrx/scratch/sandbox/ihab.bendidi/state_out/logs/fewshot_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=20
#SBATCH --mem=320G
#SBATCH --time=0-24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=0-4
#SBATCH --partition=hopper
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
set -euo pipefail

# ============================================================================================
# STATE on Replogle Nadig FEW SHOT, 5 seeds, under the shared selection protocol.
#
#   1. cell-eval on the VALIDATION split, 5 times over the run, whole split
#   2. keep the best checkpoint by val/celleval/selection_score
#      = mean(pearson_delta, discrimination_score_l1), mode=max  -> best_celleval.ckpt
#   3. evaluate TEST once from that checkpoint, under BOTH control conventions
#   4. 5 seeds (42-46)
#
# WHAT WAS WRONG BEFORE. The old split has `validation: 0`, so val_datasets was empty and
# zarr_perturbation.val_dataloader() silently returned the TEST loader. val_loss -- and therefore
# ModelCheckpoint(monitor="val_loss") -- was computed and selected on TEST for every replogle run
# to date. That fallback is now a hard error, and this points at the fewshot split.
#
# ⚠ BUDGET: 80,000 steps, compute-matched to PerturbDiff. That is 20% of STATE's published 400,000
# default, so STATE is under-trained relative to its own recipe here. State that next to any
# number from this arm. (train_state_zarr.sh also used 80,000, so this is not a regression
# against STATE's previous results on this benchmark.)
#
# ⚠ BOTH seeds must be set. training.train_seed drives torch init; data.kwargs.random_seed drives
# batch composition and basal-control pairing and is NOT wired to it. Setting only the first would
# give five runs with identical batches and pairings.
#
#   sbatch train_state_fewshot.sh
#   MAX_STEPS=8000 N_VALIDATIONS=2 sbatch --array=0 --export=ALL,MAX_STEPS,N_VALIDATIONS train_state_fewshot.sh
# ============================================================================================

# SLURM runs the batch script from a per-job spool copy, so BASH_SOURCE points at
# /cm/local/.../job<id>/ and cannot locate the checkout. Try the plausible roots and take the first
# that actually contains this package -- and hardcode nothing, because train_state_zarr.sh:24 still
# points at notebooks/STATE/state, which no longer exists.
if [ -z "${STATE_DIR:-}" ]; then
  for _cand in \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
    "/rxrx/scratch/sandbox/${USER}/bc-hooke/src/hooke-tx/competitors/STATE/state"
  do
    if [ -n "${_cand}" ] && [ -f "${_cand}/pyproject.toml" ] && [ -d "${_cand}/src/state" ]; then
      STATE_DIR="${_cand}"; break
    fi
  done
fi
if [ -z "${STATE_DIR:-}" ] || [ ! -d "${STATE_DIR}/src/state" ]; then
  echo "Could not locate the STATE checkout. Set it explicitly:" >&2
  echo "  STATE_DIR=/path/to/competitors/STATE/state sbatch --export=ALL $0" >&2
  exit 1
fi
cd "${STATE_DIR}"

HOOKE_TX_DIR="$(cd "${STATE_DIR}/../../.." && pwd)"  # competitors/STATE/state -> hooke-tx
GENES="${HOOKE_TX_DIR}/configs/metadata/replogle_nadig_selected_ens_ids.txt"
TOML="${STATE_DIR}/ihabxcell_hepg2_fewshot_zarr.toml"

MAX_STEPS="${MAX_STEPS:-80000}"
N_VALIDATIONS="${N_VALIDATIONS:-5}"
SEED_BASE="${SEED_BASE:-42}"
SEED=$(( SEED_BASE + ${SLURM_ARRAY_TASK_ID:-0} ))
OUTPUT_DIR="${OUTPUT_DIR:-/rxrx/scratch/sandbox/${USER}/state_out/replogle_nadig_fewshot}"
RUN_NAME="fewshot_s${SEED}"

# Preflight. Fail in the first seconds rather than after hours of queue time.
[[ -f "${TOML}"  ]] || { echo "missing toml: ${TOML}" >&2; exit 1; }
[[ -f "${GENES}" ]] || { echo "missing gene list: ${GENES}" >&2; exit 1; }
if [ "${N_VALIDATIONS}" -lt 1 ]; then echo "N_VALIDATIONS must be >= 1" >&2; exit 1; fi
if [ $(( MAX_STEPS % N_VALIDATIONS )) -ne 0 ]; then
  echo "MAX_STEPS=${MAX_STEPS} is not divisible by N_VALIDATIONS=${N_VALIDATIONS}." >&2
  echo "The last validation would not land on the final step, so the final checkpoint could" >&2
  echo "never be scored and therefore never selected." >&2
  exit 1
fi
VAL_FREQ=$(( MAX_STEPS / N_VALIDATIONS ))

mkdir -p "/rxrx/scratch/sandbox/${USER}/state_out/logs" "${OUTPUT_DIR}"

echo "=== STATE fewshot ==="
echo "    seed=${SEED} (train_seed and data random_seed)"
echo "    steps=${MAX_STEPS}  validations=${N_VALIDATIONS}  val_freq=${VAL_FREQ}"
echo "    toml=$(basename "${TOML}")"
echo "    out=${OUTPUT_DIR}/${RUN_NAME}"
echo "    node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# HOW STATE IS INVOKED, and why it looks awkward.
#
# `.venv/bin/state` has a stale shebang pointing at the vanished
# notebooks/STATE/state/.venv/bin/python3, so running it -- including via `uv run state` -- fails
# with a pyenv exec error. The editable install's .pth is broken the same way, so plain
# `python -m state` reports "No module named state". Putting src on PYTHONPATH and running the
# module with the venv's own interpreter works and touches nothing.
#
# A `uv sync` on a login node would regenerate both correctly; until then this is the reliable route.
STATE_PY=("${STATE_DIR}/.venv/bin/python" -m state)
export PYTHONPATH="${STATE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset VIRTUAL_ENV
# NOTE on the override list below -- do NOT put '#' comments inside it. A comment on a
# backslash-continued line terminates the command: the escaped newline joins the next line, the '#'
# swallows it, and every remaining override silently becomes a separate shell command. That is how
# an earlier run reached Hydra with only the first three overrides, fell back to the config
# defaults (name=debug, output_dir=./debugging), and auto-resumed an unrelated leftover checkpoint.
#
#   num_workers: 0 by default. Workers pass tensors through /dev/shm and one died with "could not
#     unlink the shared memory file" while /dev/shm was 1%% used -- a teardown race, not capacity.
#   +data.kwargs.random_seed: '+' because it is a DataModule constructor default, not a declared
#     config key; plain assignment raises "Key 'random_seed' is not in struct".
"${STATE_PY[@]}" tx train \
    data=perturbation_zarr \
    data.kwargs.toml_config_path="${TOML}" \
    data.kwargs.selected_ensembl_gene_ids_path="${GENES}" \
    data.kwargs.num_workers=${NUM_WORKERS:-0} \
    data.kwargs.embed_key='X_hvg' \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="batch_center" \
    data.kwargs.cell_type_key="cell_type" \
    data.kwargs.control_pert="empty" \
    data.kwargs.target_sum=null \
    data.kwargs.log1p=false \
    data.kwargs.feature_loading=ram \
    +data.kwargs.random_seed="${SEED}" \
    training.max_steps="${MAX_STEPS}" \
    training.val_freq="${VAL_FREQ}" \
    training.train_seed="${SEED}" \
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
    wandb.tags="[replogle_nadig_fewshot]" \
    output_dir="${OUTPUT_DIR}" \
    name="${RUN_NAME}"

echo "=== TRAIN DONE: ${OUTPUT_DIR}/${RUN_NAME} ==="
ls -la "${OUTPUT_DIR}/${RUN_NAME}/checkpoints/" || true
