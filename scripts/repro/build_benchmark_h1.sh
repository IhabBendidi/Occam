#!/usr/bin/env bash
#SBATCH --job-name=build_h1_bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/build_h1_bench_%j.out
#SBATCH --error=runs/slurm/build_h1_bench_%j.err
# =====================================================================
# REBUILD the h1_seen benchmark from the published local copy -- EXACTLY as we use it
#
# WHAT THIS BENCHMARK IS
#   rhj_fewshot checkpoints (trained on HepG2 + Jurkat + RPE1) scored on H1 hESC from the
#   VCC 2025 dataset -- a cell line none of them saw, not even its controls, in an assay they
#   did not see either -- on 80 perturbations that ARE in rhj_fewshot training ("seen
#   perturbation, unseen assay, unseen cell line"). The 80 are the H1 perts also measured in
#   the K562 essential screen (data/benchmarks/shared/h1_k562_perts.json, in git), so
#   k562_seen_h1set scores the same set in a seen assay.
#   Same split-half protocol as rhj_fewshot: 5 seeded splits into halves A/B of the H1
#   cells; the model gets half-B controls, half A is the reference. docs/h1_seen.md.
#
# WHAT IT PRODUCES   data/benchmarks/h1_seen/ (or OUT)
#   benchmark.json, gene_axis.csv (6,535 genes), perturbations.csv (99 + RHJ stratum)
#   splithalf/  panel.npz, splits.npz (5 splits), cohort.json, splits_manifest.json,
#               train_grand_mean.npy (meanpert), transfer_deltas.npz (transfer)
#
# INPUTS, from the HuggingFace repo (docs/hf_release.md)
#   benchmarks/h1_seen/h1_seen_raw.h5ad   -> data/benchmarks/h1_seen/
#       raw counts on all 18,080 H1 genes: the 80 perturbations + 38,176 controls, already
#       selected from the 3 VCC 2025 files, in build order.
#       sha256 73ea83694d7c34fc2eaad94697b35fa7758c2606ca5a056b8913b096482d76d0
#   benchmarks/rhj_fewshot/               -> data/benchmarks/rhj_fewshot/
#       the axis the checkpoints were trained on, and which perturbations are "seen"
#   eval/pert/{nadig_2024_hepg2,nadig_2024_jurkat,replogle_2022_rpe1_e}.h5ad
#                                         -> data/eval/pert/   (meanpert + transfer)
#   (the published panel itself, benchmarks/h1_seen/splithalf/, to verify against)
#
# WHAT EACH OUTCOME MEANS
#   MATCH    -> axis, cohort, every cell, all 5 A/B splits and both baselines are ours;
#               scores are comparable to docs/h1_seen.md.
#   MISMATCH -> do not compare; the printed field names what moved.
#
# USAGE
#   OUT=/tmp/h1_check bash scripts/repro/build_benchmark_h1.sh      # rebuild + verify
#   FORCE=1 bash scripts/repro/build_benchmark_h1.sh                 # replace in place
#   (either can be sbatch'ed; the header asks for 240 GB on the cpu partition)
#
# COST  ~10 min, ~120 GB RAM peak (the 8.8 GB copy is read whole). No GPU.
#
# ⚠️ Rebuilding from the ORIGINAL three VCC 2025 files is also possible
#   (python scripts/build_h1_seen_panel.py --h1-files TRAIN VAL TEST) and is how the
#   published copy was made; the local copy exists so nobody else needs the 35 GB.
# =====================================================================
set -euo pipefail

HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

RAW_SHA256="73ea83694d7c34fc2eaad94697b35fa7758c2606ca5a056b8913b096482d76d0"
PUBLISHED="${PUBLISHED:-data/benchmarks/h1_seen}"
OUT="${OUT:-data/benchmarks/h1_seen}"
RAW="${RAW:-data/benchmarks/h1_seen/h1_seen_raw.h5ad}"

if [ -f "$OUT/benchmark.json" ] && [ -z "${FORCE:-}" ]; then
    echo "$OUT already holds a frozen benchmark.json." >&2
    echo "To VERIFY it (what you almost certainly want):" >&2
    echo "    OUT=/tmp/h1_check bash $0" >&2
    echo "To REPLACE it in place:  FORCE=1 bash $0" >&2
    exit 2
fi

for f in "$RAW" data/benchmarks/rhj_fewshot/benchmark.json data/benchmarks/shared/h1_k562_perts.json \
         data/eval/pert/nadig_2024_hepg2.h5ad data/eval/pert/nadig_2024_jurkat.h5ad \
         data/eval/pert/replogle_2022_rpe1_e.h5ad; do
    if [ ! -f "$f" ]; then
        echo "missing input: $f -- download it from the HuggingFace repo (docs/hf_release.md)" >&2
        exit 1
    fi
done

echo "=== rebuilding h1_seen ==="
echo "started: $(date -Is)   git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "input  : $RAW"
have="$(sha256sum "$RAW" | cut -d' ' -f1)"
if [ "$have" != "$RAW_SHA256" ]; then
    echo "  sha256 $have != published $RAW_SHA256 -- not the published local copy" >&2
    exit 3
fi
echo "  sha256 OK ($RAW_SHA256)"

VERIFY=()
if [ -f "$PUBLISHED/splithalf/panel.npz" ] && \
   [ "$(cd "$OUT" 2>/dev/null && pwd)" != "$(cd "$PUBLISHED" && pwd)" ]; then
    VERIFY=(--verify-against "$PUBLISHED")
else
    echo "  (no separate published panel to compare against; building without verification)"
fi

python -u scripts/build_h1_seen_panel.py --from-raw "$RAW" --out-dir "$OUT" \
    ${FORCE:+--force} "${VERIFY[@]}"

echo
echo "finished: $(date -Is)"
echo "next: BUILD_JOB= bash scripts/repro/h1_seen.sh   (score the rhj_fewshot checkpoints)"
