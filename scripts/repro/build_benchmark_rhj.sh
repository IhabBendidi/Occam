#!/usr/bin/env bash
# =====================================================================
# REBUILD the rhj_fewshot benchmark from the published data -- EXACTLY as we use it
#
# WHAT THIS ANSWERS
#   "Are the genes, perturbations and splits you scored on the same ones I get?"
#   The answer has to be a checksum, not a description: this script rebuilds the
#   benchmark from the three published .h5ad files and then ASSERTS the sha256
#   against the value the published checkpoint records. If a single gene, cell or
#   perturbation differed, the digest would differ and this would fail.
#
# WHAT IT PRODUCES
#   data/benchmarks/rhj_fewshot/
#     benchmark.json     the frozen definition -- sha256 d934211e5eb1e37e...
#     gene_axis.csv      the 6,983-gene scoring axis (sha256 ef64f2ba7ff7816e...)
#     perturbations.csv  the cohort, per cell line, with cell counts
#     splits.json        train / val / test perturbations
#     exclusions.json    perturbations dropped by cell, and why
#     files/*.json       per-source provenance: path, sha256, n_obs, batches
#     splithalf/         the 5 seeded split-halves + panel.npz + the baselines
#
# INPUTS, all from the HuggingFace repo (docs/hf_release.md)
#   eval/pert/{nadig_2024_hepg2,nadig_2024_jurkat,replogle_2022_rpe1_e}.h5ad
#                                          -> data/eval/pert/
#   graphs/master_gene_set_sorted.csv      -> data/graphs/
#   checkpoints/occam_small_encoder_decoder.pth -> checkpoints/release/
#   benchmarks/rhj_fewshot/REQUESTED.json  -> data/benchmarks/rhj_fewshot/
#     ^ THE SPLIT DEFINITION. The val/test perturbation lists as originally
#       specified. Everything else here is derived from it and the data.
#
# WHAT EACH OUTCOME MEANS
#   sha256 matches  -> you have our exact genes, perturbations and splits, and any
#                      number you produce is comparable to ours.
#   sha256 differs  -> do NOT compare scores. The script prints which component
#                      moved (axis / cohort / splits); the usual cause is a
#                      different .h5ad revision, which its own sha256 will show.
#
# COST  ~25 min, ~120 GB RAM at the peak (three h5ads, 6,983 columns each).
#       No GPU. The split-half panel step is the long half.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

# The reference to compare a rebuild against. Its own digests, for the record:
#   benchmark.json  d934211e5eb1e37e28c893fb8c5ff4211d607a1d77b0103d5c2b79221d21677f
#   gene_axis.csv   ef64f2ba7ff7816ed947796066cd8a68014884c6176619d750ef79462c78d4c0
PUBLISHED="${PUBLISHED:-data/benchmarks/rhj_fewshot}"
OUT="${OUT:-data/benchmarks/rhj_fewshot}"
# REQUESTED.json is the SPLIT DEFINITION, i.e. an input. Read from its own path so that
# `OUT=/tmp/check` verifies the published artifact without copying anything.
REQUESTED="${REQUESTED:-data/benchmarks/rhj_fewshot/REQUESTED.json}"

# ⛔ REFUSE to rebuild in place over a frozen benchmark. The default OUT is the
# published artifact directory, and a rebuild from a different source revision would
# overwrite it with something that no longer matches the checkpoints, the configs or
# the scoreboard -- and the digest check below would then fail on a file the reader
# can no longer restore. Rebuild into a scratch directory and compare, or set FORCE=1
# if replacing the frozen artifact is genuinely what you mean.
if [ -f "$OUT/benchmark.json" ] && [ -z "${FORCE:-}" ]; then
    echo "$OUT already holds a frozen benchmark.json." >&2
    echo >&2
    echo "To VERIFY the published artifact (what you almost certainly want):" >&2
    echo "    OUT=/tmp/bench_check bash $0" >&2
    echo >&2
    echo "To REPLACE it in place, knowing it invalidates every score measured" >&2
    echo "against it unless the digests still match:" >&2
    echo "    FORCE=1 bash $0" >&2
    exit 2
fi

for f in data/eval/pert/nadig_2024_hepg2.h5ad \
         data/eval/pert/nadig_2024_jurkat.h5ad \
         data/eval/pert/replogle_2022_rpe1_e.h5ad \
         data/graphs/master_gene_set_sorted.csv \
         checkpoints/release/occam_small_encoder_decoder.pth \
         "$REQUESTED"; do
    if [ ! -f "$f" ]; then
        echo "missing input: $f" >&2
        echo "download it from the HuggingFace repo -- see docs/hf_release.md for the" >&2
        echo "folder -> local destination table" >&2
        exit 1
    fi
done

echo "=== rebuilding rhj_fewshot ==="
echo "started: $(date -Is)"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

# Step 1/2 -- the benchmark: axis, cohort, splits, exclusions, provenance.
# Every argument is left at its default on purpose; the defaults ARE the frozen
# definition (sources, holdout file nadig_2024_hepg2.h5ad, gene_col
# ensembl_gene_id, pert_col gene_name, control label non-targeting).
echo "--- step 1/2: benchmark definition"
python scripts/build_rhj_fewshot.py --out-dir "$OUT" \
    --splits-source "$REQUESTED"

# Step 2/2 -- the 5 seeded split-halves, the model-visible control pool and the
# reference-arm deltas. seed 42 and min_cells are part of the identity.
echo
echo "--- step 2/2: split-half panel (5 splits, seed 42)"
python scripts/build_splithalf_panel.py --bench-dir "$OUT" --out-dir "$OUT/splithalf"

echo
echo "--- verifying against the published benchmark"
# ⚠️ NOT a byte hash of benchmark.json. That file embeds `git_sha`, `built` and
# `command`, so two builds of the SAME benchmark on different days differ in bytes while
# being identical in content -- measured on 2026-09-22, where a bit-identical norman
# rebuild was reported as a MISMATCH by the first version of this check. The verifier
# compares the scientific quantities instead: the axis digest, the builder's own
# per-file content digests, the counts, the input .h5ad checksums and the cohort.
if [ "$(cd "$OUT" && pwd)" = "$(cd "$PUBLISHED" && pwd)" ]; then
    echo "  OUT is the published directory itself; nothing to compare against."
    echo "  Rebuild into a scratch dir to verify:  OUT=/tmp/bench_check bash $0"
else
    python scripts/repro/verify_benchmark.py --rebuilt "$OUT" --published "$PUBLISHED"
fi

echo
echo "finished: $(date -Is)"
echo "next: sbatch scripts/repro/rhj_train.sh   (or evaluate the published checkpoint:"
echo "      sbatch scripts/repro/rhj_eval.sh)"
