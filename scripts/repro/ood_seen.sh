#!/usr/bin/env bash
# =====================================================================
# Score EXISTING rhj_fewshot checkpoints on a SEEN-perturbation, UNSEEN-cell-line panel
#
#   BENCH=h1_seen           H1 hESC (VCC 2025): unseen assay + unseen cell line, 80 perts
#   BENCH=k562_seen         K562 (Replogle 2022 essential): seen assay, unseen line, the 528
#                           RHJ test perts            -> compare with rhj_fewshot
#   BENCH=k562_seen_h1set   K562, the SAME 80 perts as h1_seen -> compare with h1_seen
#
# OBJECTIVE  How far do models trained on HepG2 + Jurkat + RPE1 carry a perturbation they HAVE
#   seen into a cell line none of them saw? No training. Same split-half protocol as
#   rhj_fewshot: 5 splits, halves A/B drawn from the panel's own cells, model input = one
#   half-B control cell, cell-eval 0.6.0, nocal. Gene axis: the rhj_fewshot axis genes the
#   panel measures, identical for every arm (docs/h1_seen.md, docs/k562_seen.md).
#
# WHAT EACH OUTCOME MEANS
#   model pΔ near/above `transfer`, well above `ctrlcell` -> the learned effect carries over;
#   near `ctrlcell` / `meanpert` -> it does not leave the training cell lines.
#   k562_seen_h1set vs h1_seen (same 80 perts): the gap is the assay shift, not the set.
#
# SUBMITS (all afterok on BUILD_JOB when set)
#   1. baselines   scripts/repro/baselines.sh at control-draw seeds SEEDS (default 42..46)
#   2. Occam       Y06, Q02, M06, M03 x 5 splits, --ood-context (configs/ood_seen_occam.tsv)
#   3. competitors s42 txpert_single, state, cellflow, perturbdiff_genept: export with proxy
#                  cell type HEPG2 (from the rhj bench) and, for STATE/PerturbDiff, a seeded
#                  per-cell draw of HepG2 batches -> score
#
# USAGE   BENCH=k562_seen BUILD_JOB=<jobid> bash scripts/repro/ood_seen.sh
#         ARMS=baselines,occam,competitors (default all)  SEEDS="42 43"  DRY_RUN=1  QOS=vco
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
BENCH="${BENCH:?set BENCH=h1_seen|k562_seen|k562_seen_h1set}"
PANEL="data/benchmarks/$BENCH/splithalf"
OUTB="runs/$BENCH"
mkdir -p runs/slurm "$OUTB"
[ -n "${BUILD_JOB:-}" ] || [ -f "$PANEL/panel.npz" ] || { echo "no panel at $PANEL" >&2; exit 1; }

# The Occam and baseline legs run `python` from the SUBMITTING environment (--export=ALL).
if ! python -c "import numpy, torch" 2>/dev/null; then
    echo "the current \`python\` cannot import numpy/torch; activate the Occam env" >&2
    exit 1
fi

ARMS="${ARMS:-baselines,occam,competitors}"
SEEDS="${SEEDS:-42 43 44 45 46}"
DEP="${BUILD_JOB:+--dependency=afterok:$BUILD_JOB}"
QOS="${QOS:-hooke-predict}"
Q="--qos=$QOS --wckey=$QOS"
sub() {
    if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN sbatch $*" >&2; echo 0; return; fi
    sbatch --parsable $Q "$@" 2>&1 | tail -1
}
echo "=== $BENCH: $ARMS ==="

if [[ ",$ARMS," == *",baselines,"* ]]; then
    for s in $SEEDS; do
        out="$OUTB/reference"; [ "$s" = 42 ] || out="$OUTB/reference_s$s"
        J=$(sub $DEP --array=0-4%6 \
            --export=ALL,BENCHMARK=$BENCH,SEED=$s,OUT=$out,CACHE=$OUTB/_de_real_shared \
            scripts/repro/baselines.sh)
        echo "baselines seed $s : $J -> $out"
    done
fi

if [[ ",$ARMS," == *",occam,"* ]]; then
    J=$(sub $DEP --array=0-19%6 \
        --export=ALL,GRID=configs/ood_seen_occam.tsv,N_SPLITS=5,CALIBRATE=none,OUT_BASE=$OUTB,PANEL_DIR=$PANEL,EXTRA_ARGS=--ood-context,DE_CACHE=$OUTB/_de_real_shared \
        scripts/slurm/rhj_splithalf_models.sbatch)
    echo "occam 4 x 5 splits : $J"
fi

if [[ ",$ARMS," == *",competitors,"* ]]; then
    C=runs/competitors
    declare -A CKPT=(
        [txpert]="$C/txpert/ckpt/occam_s42_single/best-ep49.ckpt"
        [state]="$C/state/occam_s42/checkpoints/best_celleval.ckpt"
        [cellflow]="$C/cellflow/occam_s42"
        [perturbdiff]="$(ls $C/perturbdiff/ckpt/pd_occam_pert_celltype_lr0.002_genept_cs16_s42_genept/epoch=*.ckpt 2>/dev/null | head -1)"
    )
    declare -A SUFFIX=([txpert]=_single [state]= [cellflow]= [perturbdiff]=_genept)
    declare -A EXTRA=([txpert]= [state]=--proxy-batch-draw [cellflow]= [perturbdiff]="--proxy-batch-draw --penc genept")
    for ARM in txpert state cellflow perturbdiff; do
        ck="${CKPT[$ARM]}"
        [ -e "$ck" ] || { echo "missing checkpoint for $ARM: $ck" >&2; exit 1; }
        common="ARM=$ARM,ARM_SUFFIX=${SUFFIX[$ARM]},PANEL_OVERRIDE=$HERE/$PANEL,PANEL_TAG=__$BENCH"
        JE=$(sub $DEP --array=0 --time=04:00:00 \
             --export="ALL,$common,SEED_BASE=42,ADAPTER_ARGS=--checkpoint $HERE/$ck ${EXTRA[$ARM]}" \
             scripts/slurm/competitor_export.sbatch)
        JS=$(sub --dependency=afterok:$JE --array=0-4%6 \
             --export="ALL,$common,SEED=42,SCORE_ROOT=$OUTB" \
             scripts/slurm/competitor_score.sbatch)
        echo "$ARM${SUFFIX[$ARM]}  export $JE -> score $JS"
    done
fi
echo "results: $OUTB/{reference*,<arm>__<tag>,compet_*}/split_NN/metrics.json"
