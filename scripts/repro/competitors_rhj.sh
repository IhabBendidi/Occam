#!/usr/bin/env bash
# =====================================================================
# REPRODUCE any competitor on rhj_fewshot -- train, export, score
#
# WHAT THIS ANSWERS
#   Is Occam's 0.4043 good absolutely, or only relative to our own baselines?
#   Four published methods were ported to train on OUR data (the frozen
#   6,983-gene axis, our splits) and be scored by OUR cell-eval 0.6.0 under the
#   same split-half protocol, from the same single drawn control cell.
#
# MEASURED, pearson_delta, nocal, mean +- SEM of 5 splits (details and every
# deliberate deviation: docs/rhj_competitors_results.md)
#   Occam Y06          0.4043 +- 0.0019     <- ours
#   TxPert             0.3463 +- 0.0035   mae 0.0227  ovl 0.0582  disc_l1 0.6704
#   PerturbDiff genept 0.2449 +- 0.0038   mae 0.0287  ovl 0.0734  disc_l1 0.6650
#   STATE              0.2362 +- 0.0019   mae 0.0291  ovl 0.0849  disc_l1 0.6639
#   CellFlow           0.2154 +- 0.0039   mae 0.0251  ovl 0.0315  disc_l1 0.5436
#   anchors            meanpert 0.1324 . transfer 0.2323 . experimental 0.3722
#
# ⚠️ NONE of these reproduces its method's own published figure, and they are not
#   meant to: each is retrained on our data at our axis, and each carries
#   deliberate, recorded deviations. Read them as "how this architecture does on
#   this task under this protocol", never as a reimplementation score.
#
# ⚠️ DELIBERATE DEVIATIONS, stated here because a number without them is a
#   misattribution:
#     txpert      trained AND inferenced on SINGLE-CELL controls
#                 (match_cntr/avg_cntr false, against its published config) so it
#                 gets the same input every other arm gets -- a (cell_type, batch)
#                 MEAN control would be a denoised input nobody else receives.
#                 Also omit_cntr: true on both sides. Its BatchNorm is untouched.
#                 ⛔ An earlier 0.3077 for this arm is WITHDRAWN (C28): it was
#                 trained on mean controls, inferenced on single cells, and missing
#                 the control token -- three faults at once.
#     state       one sentence per predict_step, no padding, per-cell batch
#                 covariate. Its first number (0.1306, exactly meanpert) came from
#                 an export that packed 4 pure sentences into one 256-cell sentence,
#                 so ~8 perturbations attended to each other.
#     cellflow    retrained: its selection metric had scored PCA-RECONSTRUCTED
#                 ground truth, and each validation pass scored a different random
#                 60 of 103 conditions. Its PCA basis uses random_state=0, so all
#                 seeds share one representation.
#     perturbdiff the genept arm is the headline; onehot is a separate arm.
#
# USAGE  one arm, one seed, all 5 splits, chained with Slurm dependencies
#   ARM=txpert bash scripts/repro/competitors_rhj.sh
#   ARM=state SEED=43 bash scripts/repro/competitors_rhj.sh
#   ARM=txpert SKIP_TRAIN=1 bash scripts/repro/competitors_rhj.sh   # export+score only
#
# This submits three legs and returns immediately:
#   1. train   one GPU task            (arm-specific cost, 4-10 h)
#   2. export  afterok:1, one task     predictions for all 5 splits, ~20 min
#   3. score   afterok:2, array 0-4%6  our cell-eval path, one task per split
#
# ⚠️ ONE SPLIT IS NOT A RESULT. Split-to-split std is 0.001-0.003, so a single
#   split looks decisive when it is not. Wait for all five.
#
# ⚠️ WANDB_MODE=offline is set for every competitor leg, deliberately: an
#   unidentified wandb race hung a PerturbDiff run for 10 h at 0% GPU.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p runs/slurm

ARM="${ARM:?set ARM=txpert|state|cellflow|perturbdiff}"
case "$ARM" in
    txpert|state|cellflow|perturbdiff) ;;
    *) echo "unknown ARM=$ARM; expected txpert|state|cellflow|perturbdiff" >&2
       exit 2 ;;
esac
SEED="${SEED:-42}"
TASK=$(( SEED - 42 ))
if [ "$TASK" -lt 0 ] || [ "$TASK" -gt 4 ]; then
    echo "SEED must be 42..46 (the array index is SEED-42), got $SEED" >&2
    exit 2
fi
QOS_ARGS=(${QOS:+--qos=$QOS} ${WCKEY:+--wckey=$WCKEY})

if [ ! -d data/benchmarks/rhj_fewshot/splithalf ]; then
    echo "panel not found: data/benchmarks/rhj_fewshot/splithalf" >&2
    echo "build it: bash scripts/repro/build_benchmark_rhj.sh" >&2
    exit 1
fi

echo "=== competitor repro: $ARM, seed $SEED (array task $TASK) ==="
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

DEP=""
if [ -z "${SKIP_TRAIN:-}" ]; then
    TRAIN_SCRIPT="scripts/slurm/competitor_train_${ARM}.sbatch"
    [ -f "$TRAIN_SCRIPT" ] || { echo "missing $TRAIN_SCRIPT" >&2; exit 1; }
    J1=$(sbatch --parsable "${QOS_ARGS[@]}" --array="$TASK" \
         --export=ALL,SEED_BASE=42 "$TRAIN_SCRIPT")
    echo "1/3 train  : job $J1  ($TRAIN_SCRIPT)"
    DEP="--dependency=afterok:$J1"
else
    echo "1/3 train  : SKIPPED (SKIP_TRAIN set) -- expecting a trained checkpoint on disk"
fi

J2=$(sbatch --parsable "${QOS_ARGS[@]}" $DEP --array="$TASK" \
     --export=ALL,ARM="$ARM",SEED_BASE=42 scripts/slurm/competitor_export.sbatch)
echo "2/3 export : job $J2  (predictions for all 5 splits, in OUR evaluation space)"

J3=$(sbatch --parsable "${QOS_ARGS[@]}" --dependency=afterok:"$J2" --array=0-4%6 \
     --export=ALL,ARM="$ARM",SEED="$SEED" scripts/slurm/competitor_score.sbatch)
echo "3/3 score  : job $J3  (one task per split, %6 for the pdex /dev/shm race)"

echo
echo "watch: squeue -j ${J1:-}${J1:+,}$J2,$J3"
echo "then : python scripts/export_split_metrics.py --out data/release/metrics"
echo "       and read metrics_wide__rhj_fewshot__all.csv"
