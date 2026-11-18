#!/usr/bin/env bash
# Throughput micro-benchmark: WHY is this 8.9x slower than the tuned research job?
#
#   sbatch --qos=vco --wckey=vco --array=0-5 scripts/slurm/rhj_throughput_probe.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# The grid trains at 1,580 samples/s. The research job reported 14,135 -- 8.9x faster. Two
# facts are already measured, and they do not add up to 8.9x on their own:
#
#   * `target_pb_k` is 1 in BOTH configs, so the estimand is not the difference.
#   * the 4-layer GATv2 over 18,216 nodes / 515,063 edges costs 48.8 ms forward+backward,
#     which is 30% of the measured 161 ms step -- and it is a PER-STEP cost, independent of
#     batch size, because the GNN is being trained and cannot be cached.
#   * the decoder arithmetic accounts for <1% of the step (~0.5 ms of 161 ms on an H100),
#     so the remaining ~70% is pipeline overhead, not compute.
#
# The two config differences that could explain the rest:
#     batch_size   research 2048   ours 256   (8x -- and the GNN cost amortises over it)
#     num_workers  research 14     ours 8
#
# This grid separates them. Each task runs 200 real training batches and prints samples/s.
#
# WHAT EACH OUTCOME MEANS
#   * throughput scales ~linearly with batch size  -> fixed per-step overhead dominates
#     (GNN + Python + host sync). Raise batch_size; the sweep should use 1024-2048.
#   * throughput flat in batch size, rises with workers -> per-job I/O bound. Raise workers.
#   * neither helps -> the bottleneck is inside the step and needs a profiler, not a knob.
#   * adjacency_map much faster than gnn -> the GNN is the fixed cost, and the pert-model
#     choice is a throughput decision as well as a modelling one.
#
# ⚠️ These are THROUGHPUT numbers only. 200 batches says nothing about final quality, and
# batch size is not a free parameter -- it changes the effective learning rate. Any batch
# size adopted here needs its own lr, which is why the lr sweep comes after this.

#SBATCH --job-name=rhj_tput
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/rhj_tput_%A_%a.out
#SBATCH --error=runs/slurm/rhj_tput_%A_%a.err

set -euo pipefail

_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then
            echo "$c"; return 0
        fi
    done
    echo "cannot locate the Occam repo; sbatch from the repo root or set OCCAM_REPO" >&2
    return 1
}
HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# arm: BS WORKERS PERTMODEL   -- one row per array index, checked in, no index arithmetic
CONFIGS=(
    "256  8  gnn"             # 0: the grid's configuration, the reference point
    "1024 8  gnn"             # 1: batch size alone, 4x
    "2048 8  gnn"             # 2: batch size alone, 8x -- the research value
    "256  16 gnn"             # 3: workers alone, 2x
    "2048 16 gnn"             # 4: both, i.e. the research configuration
    "256  8  adjacency_map"   # 5: the pert model alone, at the grid's batch size
)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
read -r BS WORKERS PMODEL <<<"${CONFIGS[$IDX]}"
BATCHES="${BATCHES:-200}"

echo "=== throughput probe ${IDX}: batch=${BS} workers=${WORKERS} pert_model=${PMODEL} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "batches : ${BATCHES}"
echo
echo "reference points: grid 1,580 samples/s · research job 14,135 samples/s"
echo "measured already: GNN fwd+bwd 48.8 ms/step (30% of a 161 ms step at batch 256)"
echo

python -m occam.train --config configs/rhj_fewshot.yaml --stages vcell \
    --run-suffix "tput_bs${BS}_w${WORKERS}_${PMODEL}" \
    --set "stages.vcell.optim.batch_size=${BS}" \
    --set "stages.vcell.stream.num_workers=${WORKERS}" \
    --set "stages.vcell.graph.pert_model_type=${PMODEL}" \
    --set "stages.vcell.guards.bench_batches=${BATCHES}" \
    --set "run.wandb.enabled=false"

echo
echo "finished: $(date -Is)"
