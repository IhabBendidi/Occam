#!/usr/bin/env bash
#SBATCH --job-name=repro_baselines
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=hop39
#SBATCH --array=0-4%6
#SBATCH --output=runs/slurm/repro_baselines_%A_%a.out
#SBATCH --error=runs/slurm/repro_baselines_%A_%a.err
# =====================================================================
# REPRODUCE every BASELINE on either benchmark
#
# WHY THE BASELINES ARE THE POINT
#   A pearson_delta is a rank within its own panel. 0.40 is excellent on
#   rhj_fewshot and would be catastrophic on norman_combo. The baselines ARE the
#   scale, and two of them are the only honest tests of whether a model earned
#   its parameters:
#     meanpert         predict the same average delta for every perturbation.
#                      Known to score competitively in this field -- a model at
#                      meanpert has learned nothing perturbation-specific.
#     sum_of_singles   (norman_combo only) d_A + d_B measured from the singles.
#                      This IS our architecture's hypothesis in closed form: an
#                      arm at or below it has learned nothing about interaction.
#   The rest bound the scale:
#     ctrlcell / ctrlmean  zero-signal floors (the drawn control cell / the
#                          control mean, unchanged). ~0 by construction.
#     transfer         (rhj_fewshot only) the SAME perturbation's delta measured
#                      in another cell line -- the "just copy a neighbouring
#                      context" baseline.
#     experimental     replicate agreement at the panel's own cell counts.
#                      ⚠️ NOT an upper bound: a model predicting the group mean
#                      can beat a noisy empirical half, and on rhj_fewshot ours
#                      does (0.4043 vs 0.3722).
#
# WHICH ARMS EXIST DEPENDS ON THE PANEL, and is resolved from the artifact rather
#   than hardcoded: `transfer` needs transfer_deltas.npz (rhj_fewshot has it; a
#   single-cell-line combinatorial panel cannot have it), `sum_of_singles` needs
#   sum_of_singles.npz (norman_combo has it; single-gene perturbations have
#   nothing to compose). Asking for an arm the panel cannot build used to leave it
#   silently absent, which reads as "scored 0" rather than "not applicable".
#
# EXPECTED VALUES, pearson_delta, nocal, mean of 5 splits
#   rhj_fewshot   ctrlcell -0.0018 . ctrlmean 0.0016 . meanpert 0.1324 .
#                 transfer 0.2323 . experimental 0.3722
#   norman_combo  (additive / nonadditive)  meanpert 0.4634 / 0.4016 .
#                 sum_of_singles 0.8824 / 0.7757 . experimental 0.9165 / 0.8377
#   These are deterministic given the panel and the seed: the control draw is
#   rng = default_rng(42 + 1000 * split), identical to the one the model arms use,
#   so meanpert and transfer cancel the model's control noise exactly. A baseline
#   that does NOT reproduce means the panel differs -- rebuild it with
#   scripts/repro/build_benchmark_{rhj,norman}.sh and check the sha256.
#
# USAGE
#   sbatch scripts/repro/baselines.sh                              # rhj_fewshot
#   BENCHMARK=norman_combo sbatch scripts/repro/baselines.sh
#   ARMS=meanpert,experimental sbatch scripts/repro/baselines.sh   # a subset
#
# ⚠️ %6 array throttle: cell-eval's Wilcoxon backend (pdex) races on /dev/shm.
# COST  ~25 min per split; 5 splits concurrent = ~30 min wall.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p runs/slurm
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export OCCAM_CELLEVAL_ATTEMPTS="${OCCAM_CELLEVAL_ATTEMPTS:-3}"

BENCHMARK="${BENCHMARK:-rhj_fewshot}"
case "$BENCHMARK" in
    rhj_fewshot|norman_combo) ;;
    *) echo "BENCHMARK must be rhj_fewshot or norman_combo, got '$BENCHMARK'" >&2
       exit 2 ;;
esac

SPLIT="${SLURM_ARRAY_TASK_ID:-0}"
PANEL="data/benchmarks/$BENCHMARK/splithalf"
OUT="${OUT:-runs/$BENCHMARK/repro_baselines}"
CACHE="${CACHE:-runs/$BENCHMARK/_de_real_shared}"

if [ ! -d "$PANEL" ]; then
    echo "panel not found: $PANEL" >&2
    if [ "$BENCHMARK" = rhj_fewshot ]; then
        echo "build it: bash scripts/repro/build_benchmark_rhj.sh" >&2
    else
        echo "build it: bash scripts/repro/build_benchmark_norman.sh" >&2
    fi
    exit 1
fi
mkdir -p "$OUT" "$CACHE"

ARGS=(--mode reference --panel-dir "$PANEL" --split "$SPLIT"
      --out-root "$OUT" --de-real-cache-dir "$CACHE"
      --calibrate none --num-threads 16)
if [ -n "${ARMS:-}" ]; then
    ARGS+=(--arms "$ARMS")
fi

echo "=== $BENCHMARK baselines . split $SPLIT ==="
echo "host   : $(hostname)"
echo "started: $(date -Is)"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "panel  : $PANEL"
echo "arms   : ${ARMS:-all the panel can build}"
echo

python scripts/eval_splithalf.py "${ARGS[@]}"

echo
echo "finished: $(date -Is)"
echo "next: python scripts/export_split_metrics.py --out data/release/metrics"
