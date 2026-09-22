#!/usr/bin/env bash
# =====================================================================
# LAUNCH the competitors on norman_combo -- DOUBLE perturbations
#
# WHAT WE ARE TRYING TO ACHIEVE
#   Train on SINGLE-gene perturbations; predict a PAIR where both genes were seen
#   individually and the pair never was. Occam's own composition operator SUMS the
#   perturbation embeddings, so this is the direct test of it -- and the question
#   this script answers is whether four published methods do better.
#
# THE ONE THING THAT MAKES THE NUMBERS COMPARABLE
#   Every arm reads the SAME frozen artifact Occam is scored on:
#     data/benchmarks/norman_combo   axis 8,000 genes  sha256 6efa91af1af5c8b8...
#                                    benchmark         sha256 8ac7da0d5adb3ae4...
#                                    cohort            sha256 dc1dd7806ba60d34...
#   The axis width is READ from that artifact and substituted into every model
#   dimension (`competitor_resolve_bench` in scripts/slurm/_competitor_common.sh);
#   no launcher carries a gene count. Three configs used to hardcode 6983, which on
#   this benchmark would have trained a 6,983-wide model against 8,000-wide targets
#   -- that does not crash, it just scores badly and looks like the method being bad.
#   Each leg prints all three digests, so "same genes, same splits" is checkable
#   rather than asserted.
#
# HOW EACH ARM COMPOSES A PAIR -- ITS OWN WAY, or the number measures our rule
#   txpert       NATIVE additive. Its GEARS condition is already `"A+B"` and
#                gears_derivative.py sums the two perturbation embeddings before
#                decoding. Its paper evaluates exactly this split on Norman.
#   cellflow     NATIVE set aggregation (attention / DeepSets), switched on by
#                `max_combination_length: 2` in configs/occam_norman_combo.yaml. Its
#                paper covers "novel combinations of genes individually observed".
#   state        decode_seq_mean. No genetic protocol published, but its own drug
#                combination experiment runs "consecutive forward passes" -- so this
#                is OUR EXTENSION OF ITS OWN DESIGN, applied at the export boundary.
#   perturbdiff  decode_seq_mean. It has NO combinatorial protocol at all, so this
#                rule is OURS and is a deviation as large as any recorded in
#                docs/rhj_competitors_results.md.
#   Full evidence, with quotes and code references: docs/competitors_on_doubles.md
#
# HOW TO READ THE RESULT, stated before the run
#   `pearson_delta`, nocal, mean of 5 splits, on the two effect-size-controlled
#   panels (additive / nonadditive):
#     meanpert                0.4634 / 0.4016   predict one constant delta
#     Occam NQ02 decode_seq_mean 0.8169 / 0.7380
#     sum_of_singles          0.8824 / 0.7757   <- THE BAR: d_A + d_B in closed form
#     experimental            0.9165 / 0.8377   replicate agreement
#   AT OR BELOW the bar  -> that arm has learned nothing about interaction; it is
#                           doing no better than adding the two single effects.
#   ABOVE the bar on `nonadditive` -> the first evidence in this panel that any
#                           method captures interaction. That is the result worth
#                           having, and none of ours has produced it.
#   ⚠️ Read all four metrics. `mae` is an ERROR (lower better) and `disc_l1` has
#      chance at 0.5. Occam beats the bar on `mae` and `overlap_at_N` while losing
#      on `pearson_delta`, so a one-metric summary of this panel is misleading.
#
# USAGE
#   bash scripts/repro/competitors_norman.sh                 # all four arms, seed 42
#   ARMS=txpert,cellflow bash scripts/repro/competitors_norman.sh
#   SEED=43 bash scripts/repro/competitors_norman.sh
#   SKIP_TRAIN=1 bash scripts/repro/competitors_norman.sh    # export + score only
#   DRY_RUN=1 bash scripts/repro/competitors_norman.sh       # print, submit nothing
#
# Each arm becomes three chained Slurm jobs: train -> (afterok) export -> (afterok)
# score, the last an array 0-4%6 over the splits. The script returns immediately.
#
# ⚠️ ONE SPLIT IS NOT A RESULT. Split-to-split std is 0.001-0.003. Wait for all five.
# ⚠️ %6 on the score array is the pdex /dev/shm race limit. Do not raise it.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p runs/slurm

export BENCH=norman_combo
SEED="${SEED:-42}"
TASK=$(( SEED - 42 ))
if [ "$TASK" -lt 0 ] || [ "$TASK" -gt 4 ]; then
    echo "SEED must be 42..46 (the array index is SEED-42), got $SEED" >&2
    exit 2
fi
ARMS="${ARMS:-txpert,cellflow,state,perturbdiff}"
# This cluster REFUSES a submission without a wckey ("A wckey is required for all job
# submissions"), so these are defaulted rather than left empty -- an empty QOS_ARGS made
# the first launch fail on the first sbatch with nothing submitted. `hooke-predict` is
# what every competitor launcher's own header documents and what 243 of this account's
# recent jobs used. Override with QOS= / WCKEY= for a different allocation.
QOS="${QOS:-hooke-predict}"
WCKEY="${WCKEY:-hooke-predict}"
QOS_ARGS=(--qos="$QOS" --wckey="$WCKEY")

if [ ! -f "data/benchmarks/$BENCH/splithalf/panel.npz" ]; then
    echo "no panel at data/benchmarks/$BENCH/splithalf/panel.npz" >&2
    echo "download benchmarks/$BENCH from the Hub, or build it:" >&2
    echo "  bash scripts/repro/build_benchmark_norman.sh" >&2
    exit 1
fi
if [ ! -f "data/benchmarks/$BENCH/files/norman_2019_k562.json" ]; then
    echo "no per-source manifest at data/benchmarks/$BENCH/files/" >&2
    echo "the competitors' shared reader needs it:" >&2
    echo "  python scripts/norman/norman_write_file_manifest.py" >&2
    exit 1
fi

# Print the identity every arm will train against, once, before anything is submitted.
source scripts/slurm/_competitor_common.sh
competitor_resolve_bench "$HERE"
echo "=================================================================="
echo "  competitors on $BENCH   seed $SEED   arms: $ARMS"
echo "  axis   : $N_AXIS_GENES genes   sha256 ${AXIS_SHA256:0:16}"
echo "  bench  : ${BENCH_SHA256:0:16}   cohort ${COHORT_SHA256:0:16}"
echo "  splits : $BENCH_N_SPLITS        run tag: '${TAG_SUFFIX}'"
echo "  git    : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=================================================================="
echo

submit() {   # submit <parsable sbatch args...>; honours DRY_RUN
    if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN would submit: sbatch $*" >&2; echo "0"
    else sbatch --parsable "$@"; fi
}

IFS=',' read -r -a ARM_LIST <<<"$ARMS"
for ARM in "${ARM_LIST[@]}"; do
    case "$ARM" in
        txpert|state|cellflow|perturbdiff) ;;
        *) echo "unknown arm '$ARM'; expected txpert|state|cellflow|perturbdiff" >&2
           exit 2 ;;
    esac
    case "$ARM" in
        txpert)      HOW="native additive sum (its own mechanism)" ;;
        cellflow)    HOW="native set aggregation, max_combination_length=2" ;;
        state)       HOW="decode_seq_mean (our extension of its consecutive forward passes)" ;;
        perturbdiff) HOW="decode_seq_mean (OURS -- it publishes no combinatorial protocol)" ;;
    esac
    echo "--- $ARM : $HOW"

    DEP=""
    if [ -z "${SKIP_TRAIN:-}" ]; then
        TRAIN="scripts/slurm/competitor_train_${ARM}.sbatch"
        [ -f "$TRAIN" ] || { echo "missing $TRAIN" >&2; exit 1; }
        J1=$(submit "${QOS_ARGS[@]}" --array="$TASK" \
             --export=ALL,BENCH="$BENCH",SEED_BASE=42 "$TRAIN")
        echo "    1/3 train  job $J1"
        DEP="--dependency=afterok:$J1"
    else
        echo "    1/3 train  SKIPPED (SKIP_TRAIN)"
    fi

    J2=$(submit "${QOS_ARGS[@]}" $DEP --array="$TASK" \
         --export=ALL,BENCH="$BENCH",ARM="$ARM",SEED_BASE=42 \
         scripts/slurm/competitor_export.sbatch)
    echo "    2/3 export job $J2  (all $BENCH_N_SPLITS splits, into OUR eval space)"

    J3=$(submit "${QOS_ARGS[@]}" --dependency=afterok:"$J2" --array=0-4%6 \
         --export=ALL,BENCH="$BENCH",ARM="$ARM",SEED="$SEED" \
         scripts/slurm/competitor_score.sbatch)
    echo "    3/3 score  job $J3  (one task per split)"
    echo
done

echo "watch : squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "then  : python scripts/export_split_metrics.py --scope all \\"
echo "            --out data/release/metrics_all"
echo "        and read metrics_wide__norman_combo__{additive,nonadditive}.csv"
