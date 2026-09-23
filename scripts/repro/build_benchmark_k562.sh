#!/usr/bin/env bash
#SBATCH --job-name=build_k562_bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/build_k562_bench_%j.out
#SBATCH --error=runs/slurm/build_k562_bench_%j.err
# =====================================================================
# REBUILD the two K562 benchmarks from the published local copy -- EXACTLY as we use them
#
# WHAT THEY ARE
#   rhj_fewshot checkpoints scored on K562 (Replogle 2022 ESSENTIAL CRISPRi screen): seen
#   perturbation, SEEN assay (same screen family as the rhj_fewshot RPE1 file), UNSEEN cell
#   line. Same split-half protocol as rhj_fewshot. docs/k562_seen.md.
#     k562_seen          the 528 rhj_fewshot TEST perturbations      -> read against rhj_fewshot
#     k562_seen_h1set    the 80 perturbations shared with h1_seen     -> read against h1_seen
#                        (data/benchmarks/shared/h1_k562_perts.json)
#
# INPUTS, from the HuggingFace repo (docs/hf_release.md)
#   benchmarks/k562_seen/k562_seen_raw.h5ad -> data/benchmarks/k562_seen/
#       raw counts, all 8,563 K562 genes: the RHJ test perts + the 80 + 10,691 controls
#   benchmarks/rhj_fewshot/                 -> data/benchmarks/rhj_fewshot/
#   eval/pert/{nadig_2024_hepg2,nadig_2024_jurkat,replogle_2022_rpe1_e}.h5ad -> data/eval/pert/
#   data/benchmarks/shared/h1_k562_perts.json (in git)
#
# WHAT EACH OUTCOME MEANS
#   MATCH    -> axis, cohort, every cell, all 5 A/B splits, meanpert and transfer are ours.
#   MISMATCH -> do not compare; the printed field names what moved.
#
# USAGE
#   OUT=/tmp/k562_check bash scripts/repro/build_benchmark_k562.sh   # -> /tmp/k562_check/{k562_seen,k562_seen_h1set}
#   FORCE=1 bash scripts/repro/build_benchmark_k562.sh                # replace in place
# COST  ~10 min per panel, ~120 GB RAM peak. No GPU.
# =====================================================================
set -euo pipefail
HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

RAW_SHA256="59f2ea90df681853739dffee1232947d47f16f7da92bf3c7213f26d760dfd211"
RAW="${RAW:-data/benchmarks/k562_seen/k562_seen_raw.h5ad}"
OUT="${OUT:-data/benchmarks}"

for f in "$RAW" data/benchmarks/rhj_fewshot/benchmark.json data/benchmarks/shared/h1_k562_perts.json \
         data/eval/pert/nadig_2024_hepg2.h5ad data/eval/pert/nadig_2024_jurkat.h5ad \
         data/eval/pert/replogle_2022_rpe1_e.h5ad; do
    [ -f "$f" ] || { echo "missing input: $f -- download it from the HF repo (docs/hf_release.md)" >&2; exit 1; }
done
echo "=== rebuilding k562_seen + k562_seen_h1set ==="
echo "started: $(date -Is)   git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
have="$(sha256sum "$RAW" | cut -d' ' -f1)"
[ "$have" = "$RAW_SHA256" ] || { echo "  sha256 $have != published $RAW_SHA256" >&2; exit 3; }
echo "  input sha256 OK ($RAW_SHA256)"

rc=0
for spec in "k562_seen:rhj_test" "k562_seen_h1set:h1set"; do
    id="${spec%%:*}"; rule="${spec##*:}"; dst="$OUT/$id"; pub="data/benchmarks/$id"
    if [ -f "$dst/benchmark.json" ] && [ -z "${FORCE:-}" ]; then
        echo "$dst already holds a frozen benchmark.json; OUT=/tmp/k562_check to verify, FORCE=1 to replace" >&2
        exit 2
    fi
    VERIFY=()
    if [ -f "$pub/splithalf/panel.npz" ] && \
       [ "$(cd "$dst" 2>/dev/null && pwd)" != "$(cd "$pub" && pwd)" ]; then
        VERIFY=(--verify-against "$pub")
    fi
    echo; echo "--- $id (cohort: $rule)"
    python -u scripts/build_h1_seen_panel.py --dataset k562 --from-raw "$RAW" \
        --eligible "$rule" --bench-id "$id" --out-dir "$dst" ${FORCE:+--force} "${VERIFY[@]}" || rc=$?
done
echo; echo "finished: $(date -Is)  exit $rc"
echo "next: BENCH=k562_seen bash scripts/repro/ood_seen.sh   (and BENCH=k562_seen_h1set)"
exit $rc
