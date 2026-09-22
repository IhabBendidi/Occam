#!/usr/bin/env bash
# =====================================================================
# REBUILD the norman_combo benchmark from the published data -- EXACTLY as we use it
#
# WHAT THIS ANSWERS
#   Same question as build_benchmark_rhj.sh, answered the same way: rebuild, then
#   assert the sha256 the published checkpoint records. The composition benchmark
#   has more moving parts than the few-shot one -- an 8,000-gene axis chosen under
#   two constraints, a 111-pair test set, and two effect-size-controlled strata --
#   so "same data" has to mean "same digest", not "same description".
#
# WHAT IT PRODUCES
#   data/benchmarks/norman_combo/
#     benchmark.json    the frozen definition -- sha256 8ac7da0d5adb3ae4...
#     axis.npz/.json    the 8,000-gene scoring axis (sha256 6efa91af1af5c8b8...)
#     gene_axis.csv     the same axis, readable
#     splits.json       105 single-gene train conditions, 20 val pairs, 111 test pairs
#     exclusions.json   the 131 doubles, dropped by cell from training
#     loss_scales.json  the measured term magnitudes the loss weights are matched to
#     splithalf/        5 seeded split-halves, panel.npz, the baselines,
#                       additivity.npz and splits_manifest.json (the 7 panels)
#
# THE AXIS, because it is the part people get wrong
#   8,000 genes that are BOTH in Occam's Ensembl vocabulary AND detected in >8% of
#   the QC'd cells, PLUS all 102 resolvable perturbation targets force-included.
#   41 of those 102 FAIL the detection filter, because Norman is CRISPRa and the
#   activated transcription factors are off at baseline. Without the exception the
#   benchmark cannot see the gene that was perturbed.
#
# THE TWO TEST STRATA, because only two of the seven panels are scored
#   additive / nonadditive, 36 pairs each. The additivity residual
#   ||d_A + d_B - d_real|| / ||d_real|| is terciled WITHIN effect-size terciles, so
#   the strata are matched on how large the true effect is. The *_raw panels tercile
#   globally and are therefore confounded with effect size -- they exist for
#   completeness and must not be quoted. `middle` is the middle tercile, held out by
#   design. `all` mixes the strata.
#
# INPUTS, from the HuggingFace repo (docs/hf_release.md)
#   eval/pert/norman_2019_k562.h5ad        -> data/eval/pert/
#     91,168 x 33,694 raw CSR int32, rebuilt from GEO GSE133344.
#     sha256 c6cad008202666ef...  ⚠️ NOT the upstream norman_2016_a549 file, which
#     is mislabelled (it is 2019 and K562 -- docs/norman_benchmark.md).
#   graphs/master_gene_set_sorted.csv      -> data/graphs/
#   checkpoints/occam_small_encoder_decoder.pth -> checkpoints/release/
#
# WHAT EACH OUTCOME MEANS
#   sha256 matches -> your genes, pairs and strata are ours; scores are comparable.
#   sha256 differs -> do not compare. The printed component names what moved.
#
# COST  ~15 min, ~60 GB RAM. No GPU.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

EXPECT_BENCH_SHA="8ac7da0d5adb3ae49e3a3dfe084494f4dce051286d219e346b4ea11c467cb3cf"
EXPECT_AXIS_SHA="6efa91af1af5c8b88fa5b024c683a53962ab9b7aa0c02b82173babb45966c77d"
OUT="${OUT:-data/benchmarks/norman_combo}"
SRC="${SRC:-data/eval/pert/norman_2019_k562.h5ad}"

for f in "$SRC" data/graphs/master_gene_set_sorted.csv \
         checkpoints/release/occam_small_encoder_decoder.pth; do
    if [ ! -f "$f" ]; then
        echo "missing input: $f" >&2
        echo "download it from the HuggingFace repo -- see docs/hf_release.md" >&2
        exit 1
    fi
done

echo "=== rebuilding norman_combo ==="
echo "started: $(date -Is)"
echo "source : $SRC"
echo "git sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

# The published axis.npz is shipped with the benchmark, so step 1 is a no-op unless
# you are rebuilding it from the raw GEO matrix. Re-deriving it needs the streaming
# gene-stats pass over the GEO mtx (scripts/norman/norman_gene_stats.py) and then
# norman_pick_axis.py; both are in the repo. Keeping the published axis is the
# reproducible choice -- picking it again from a different source file would move it.
if [ ! -f "$OUT/axis.npz" ]; then
    echo "$OUT/axis.npz is missing." >&2
    echo "It ships inside benchmarks/norman_combo/ on the Hub. Re-deriving it needs" >&2
    echo "the raw GEO matrix: scripts/norman/norman_gene_stats.py then" >&2
    echo "scripts/norman/norman_pick_axis.py --n-genes 8000 --min-detection 0.08" >&2
    exit 1
fi

echo "--- step 1/2: split-half panel (5 splits, seed 42, min_cells 40, 20 val pairs)"
python scripts/norman/norman_build_panel.py \
    --source "$SRC" --axis "$OUT/axis.npz" --out-dir "$OUT/splithalf"

echo
echo "--- step 2/2: benchmark definition (stamps the panel manifest)"
python scripts/norman/norman_build_benchmark.py \
    --source "$SRC" --axis "$OUT/axis.npz" --panel "$OUT/splithalf" --out-dir "$OUT"

echo
echo "--- verifying against the published digests"
GOT_BENCH="$(python - "$OUT/benchmark.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
GOT_AXIS="$(python - "$OUT/axis.npz" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
FAIL=0
if [ "$GOT_BENCH" != "$EXPECT_BENCH_SHA" ]; then
    echo "  benchmark.json sha256 MISMATCH" >&2
    echo "    got      $GOT_BENCH" >&2
    echo "    expected $EXPECT_BENCH_SHA" >&2
    FAIL=1
else
    echo "  benchmark.json sha256 OK  ($EXPECT_BENCH_SHA)"
fi
if [ "$GOT_AXIS" != "$EXPECT_AXIS_SHA" ]; then
    echo "  axis.npz sha256 MISMATCH -- the SCORING AXIS differs, so no score produced" >&2
    echo "  here is comparable to a published one." >&2
    echo "    got      $GOT_AXIS" >&2
    echo "    expected $EXPECT_AXIS_SHA" >&2
    FAIL=1
else
    echo "  axis.npz       sha256 OK  ($EXPECT_AXIS_SHA)"
fi
[ "$FAIL" = 1 ] && exit 3

echo
echo "--- the seven panels (only additive and nonadditive are scored)"
python - "$OUT/splithalf/splits_manifest.json" <<'PY'
import json, sys
man = json.load(open(sys.argv[1]))
for name, spec in sorted((man.get("panels") or {}).items()):
    n = len(spec.get("pairs") or [])
    note = {"additive": "  <- SCORED, effect-size-controlled",
            "nonadditive": "  <- SCORED, effect-size-controlled",
            "middle": "  (middle tercile, held out by design)",
            "all": "  (mixes the strata)"}.get(name, "  (globally terciled -- DO NOT QUOTE)")
    print(f"  {name:16} {n:4} pairs{note}")
PY

echo
echo "finished: $(date -Is)"
echo "next: sbatch scripts/repro/norman_train.sh   (or evaluate the published checkpoint:"
echo "      sbatch scripts/repro/norman_eval.sh)"
