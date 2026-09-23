#!/usr/bin/env bash
# =====================================================================
# h1_seen -- score EXISTING rhj_fewshot checkpoints on H1 (seen perturbation, UNSEEN line)
#
# OBJECTIVE  How far do models trained on HepG2 + Jurkat + RPE1 carry a perturbation they
#   HAVE seen into a cell line none of them saw -- not its perturbations, not its controls
#   (H1 hESC, VCC 2025)? No training. Same split-half protocol as rhj_fewshot: 5 splits,
#   halves A/B drawn from the H1 cells, model input = half-B controls, cell-eval 0.6.0, nocal.
#
# PANEL  data/benchmarks/h1_seen/ (scripts/build_h1_seen_panel.py): 99 perturbations that
#   are eligible in rhj_fewshot (70 trained in all 3 RHJ lines, 29 RHJ val/test perts trained
#   only in Jurkat/RPE1), all 38,176 H1 controls, 6,535 of the 6,983 axis genes.
#
# WHAT EACH OUTCOME MEANS
#   model pΔ near `transfer` and well above `ctrlcell` -> the learned effect carries over
#   to a new context; near `ctrlcell` / `meanpert` -> it does not leave the training lines.
#
# SUBMITS (all afterok on BUILD_JOB when set)
#   1. baselines          scripts/repro/baselines.sh, BENCHMARK=h1_seen, 5 splits
#   2. Occam              Y06, Q02, M06, M03 x 5 splits, --ood-context
#   3. competitors s42    txpert_single, state, cellflow, perturbdiff_genept:
#                         export (proxy cell type = HEPG2 from the rhj bench, per-cell proxy
#                         batch draw for STATE/PerturbDiff) -> score
#
# USAGE   BUILD_JOB=<jobid> bash scripts/repro/h1_seen.sh       DRY_RUN=1 to print only
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p runs/slurm runs/h1_seen

PANEL="data/benchmarks/h1_seen/splithalf"
# ⚠️ the Occam and baseline legs run `python` from the SUBMITTING environment (--export=ALL).
# Refuse here rather than let 25 tasks die on `import numpy` (2026-09-23, exactly that).
if ! python -c "import numpy, torch, occam" 2>/dev/null; then
    echo "the current \`python\` cannot import numpy/torch/occam; activate the Occam env" >&2
    echo "(README step 1) before running this launcher" >&2
    exit 1
fi
DEP="${BUILD_JOB:+--dependency=afterok:$BUILD_JOB}"
QOS="${QOS:-hooke-predict}"
Q="--qos=$QOS --wckey=$QOS"
sub() {
    if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN sbatch $*" >&2; echo 0; return; fi
    sbatch --parsable $Q "$@" 2>&1 | tail -1
}

# 1. baselines
J=$(sub $DEP --array=0-4%6 --export=ALL,BENCHMARK=h1_seen,OUT=runs/h1_seen/reference,CACHE=runs/h1_seen/_de_real_shared \
    scripts/repro/baselines.sh)
echo "baselines          : $J"

# 2. Occam
J=$(sub $DEP --array=0-19%6 \
    --export=ALL,GRID=configs/h1_seen_occam.tsv,N_SPLITS=5,CALIBRATE=none,OUT_BASE=runs/h1_seen,PANEL_DIR=$PANEL,EXTRA_ARGS=--ood-context,DE_CACHE=runs/h1_seen/_de_real_shared \
    scripts/slurm/rhj_splithalf_models.sbatch)
echo "occam 4 x 5 splits : $J"

# 3. competitors, seed 42, headline variants
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
    common="ARM=$ARM,ARM_SUFFIX=${SUFFIX[$ARM]},PANEL_OVERRIDE=$HERE/$PANEL,PANEL_TAG=__h1_seen"
    JE=$(sub $DEP --array=0 --time=04:00:00 \
         --export="ALL,$common,SEED_BASE=42,ADAPTER_ARGS=--checkpoint $HERE/$ck ${EXTRA[$ARM]}" \
         scripts/slurm/competitor_export.sbatch)
    JS=$(sub --dependency=afterok:$JE --array=0-4%6 \
         --export="ALL,$common,SEED=42,SCORE_ROOT=runs/h1_seen" \
         scripts/slurm/competitor_score.sbatch)
    echo "$ARM${SUFFIX[$ARM]}  export $JE -> score $JS"
done
echo "results: runs/h1_seen/{reference,<arm>__<tag>,compet_*}/split_NN/metrics.json"
