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

# The reference to compare a rebuild against. Its own digests, for the record:
#   benchmark.json  8ac7da0d5adb3ae49e3a3dfe084494f4dce051286d219e346b4ea11c467cb3cf
#   axis (content)  6efa91af1af5c8b88fa5b024c683a53962ab9b7aa0c02b82173babb45966c77d
PUBLISHED="${PUBLISHED:-data/benchmarks/norman_combo}"
OUT="${OUT:-data/benchmarks/norman_combo}"
SRC="${SRC:-data/eval/pert/norman_2019_k562.h5ad}"
# The axis is an INPUT, so it is read from its own path rather than from OUT -- that way
# `OUT=/tmp/check` verifies the published artifact without anyone copying files around.
AXIS="${AXIS:-data/benchmarks/norman_combo/axis.npz}"

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
if [ ! -f "$AXIS" ]; then
    echo "$AXIS is missing." >&2
    echo "It ships inside benchmarks/norman_combo/ on the Hub. Re-deriving it needs" >&2
    echo "the raw GEO matrix: scripts/norman/norman_gene_stats.py then" >&2
    echo "scripts/norman/norman_pick_axis.py --n-genes 8000 --min-detection 0.08" >&2
    exit 1
fi

echo "--- step 1/3: split-half panel (5 splits, seed 42, min_cells 40, 20 val pairs)"
python scripts/norman/norman_build_panel.py \
    --source "$SRC" --axis "$AXIS" --out-dir "$OUT/splithalf"

# ⚠️ REQUIRED, and it used to be missing. Without this step the manifest comes out with
# `panels: []` and `scored_panels: null` -- a benchmark nobody can score the way we do,
# because the two effect-size-controlled strata every result is reported on would not
# exist. It must run BEFORE the benchmark definition, which stamps the manifest.
echo
echo "--- step 2/3: additivity strata (the additive / nonadditive panels)"
python scripts/norman/norman_stratify_panels.py --panel "$OUT/splithalf"

echo
echo "--- step 3/3: benchmark definition (stamps the panel manifest)"
python scripts/norman/norman_build_benchmark.py \
    --source "$SRC" --axis "$AXIS" --panel "$OUT/splithalf" --out-dir "$OUT"

echo
echo "--- verifying against the published benchmark"
# ⚠️ NOT a byte hash of benchmark.json. That file embeds `git_sha`, `built` and the
# output path, so two builds of the SAME benchmark on different days differ in bytes
# while being identical in content -- the first version of this script reported a
# MISMATCH on a rebuild that was in fact identical in every scientific quantity.
# axis.npz is worse: it is a zip and carries per-entry timestamps.
if [ "$(cd "$OUT" && pwd)" = "$(cd "$PUBLISHED" && pwd)" ]; then
    echo "  OUT is the published directory itself; nothing to compare against."
    echo "  Rebuild into a scratch dir to verify:  OUT=/tmp/bench_check bash $0"
else
    python scripts/repro/verify_benchmark.py --rebuilt "$OUT" --published "$PUBLISHED"
fi

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
