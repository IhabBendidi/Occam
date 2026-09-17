#!/usr/bin/env bash
# CellFlow on OUR rhj_fewshot data, 5 seeds. Official hyperparameters, our dataloader.
#
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-4%2 \
#       scripts/slurm/competitor_train_cellflow.sbatch
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell.
#
# This job's question: what does CellFlow score on OUR instrument -- our three h5ads, our
# frozen 6,983-gene axis, our 611-perturbation split-half cohort, our cell-eval 0.6.0?
#
# VARIED: `model.seed` (42-46). HELD FIXED: every hyperparameter from
# reproducibility/cellflow__replogle_nadig_fewshot/launch.train.sh, unchanged -- 100,000
# iterations, validation every 20,000, and `pca.n_comps = 256`.
#
# ⚠️ `pca.n_comps = 256` IS A PCA RANK, NOT A GENE COUNT. Do not "update" it to 6,983 while
# widening the axis. It is the dimensionality of the space the flow is learned in; the axis
# change affects what PCA is fitted ON, not how many components are kept.
#
# TWO DELIBERATE DEVIATIONS, recorded in docs/rhj_competitors.md:
#   1. `hooke_bridge/occam.py` -- a sibling of their `load.py` with the same
#      `-> ad.AnnData` signature, reading OUR h5ads through
#      competitors/shared/occam_data.py. `apply_cellflow_gene_schema` is untouched.
#   2. Our 6,983-gene axis instead of their 2,000.
#
# ⚠️ CHECKPOINT SELECTION IS IN-MEMORY HERE. Unlike the other three arms, CellFlow keeps the
# best parameter set in memory and restores it before saving the pickle, rather than writing a
# `best_*.ckpt`. `run_meta.json`'s `selection` block is the record of what was chosen; the
# tail of this script prints it. If that block is missing, the pickle holds the FINAL state and
# the run is NOT selected-on-validation -- say so beside the number rather than assuming.
#
# ⚠️ EMISSION SPACE. It decodes from a 256-component PCA basis, so its output space is
# structurally different from a direct gene-space head. Its Phase-A adapter declares the map
# explicitly and assert_same_space REFUSES rather than warns. The map is applied ONCE and
# checked -- not iterated on until a number looks good. (Re-emitting our own decoder's
# predictions as integer counts cost 0.034 pearson_delta, and rescaling the library size back
# recovered none of it.)
#
# WHAT EACH OUTCOME MEANS
#   * above `transfer` 0.2323 -> earning its parameters here
#   * between `meanpert` 0.1324 and `transfer` -> perturbation-specific but weaker than copying
#     a measured neighbouring context
#   * at or below `meanpert` with `discrimination_score_l1` near 0.5 -> a
#     perturbation-independent mean
#   * far outside 0.0-0.4 -> suspect the PORT, not the model. Their own half-split
#     reproducibility ceiling was 0.386 on their panel; our `experimental` arm is
#     0.3722 on ours.
#
# Its seed noise on their panel was ±0.021, second largest of the four arms.
#
# SIZING: upstream 8 h at 256 G for 100,000 iterations on a 2,000-gene panel. Our axis is 3.5x
# wider, and the PCA is fitted on it; 12 h is the transcribed budget plus margin.

#SBATCH --job-name=cmp_cflow
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/cmp_cflow_%A_%a.out
#SBATCH --error=runs/slurm/cmp_cflow_%A_%a.err

source "$(dirname "${BASH_SOURCE[0]}")/_competitor_common.sh" 2>/dev/null || \
    source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/_competitor_common.sh"

HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

ARM=cellflow
SEED=$(( ${SEED_BASE:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))
ITERATIONS="${ITERATIONS:-100000}"
VALID_FREQ="${VALID_FREQ:-20000}"
TAG="occam_s${SEED}${TAG_SUFFIX:-}"
OUT_ROOT="$HERE/runs/competitors/cellflow"
CELLFLOW_ROOT="$HERE/competitors/CellFlow"
CF_PKG="$CELLFLOW_ROOT/cellflow"

PY="$(competitor_venv_python "$ARM" "$HERE")" || exit 1
competitor_banner "$ARM" train "$SEED" "$HERE"
echo "  iterations $ITERATIONS · valid_freq $VALID_FREQ · out $OUT_ROOT/$TAG"

mkdir -p "$OUT_ROOT"
# Their package is versioned from VCS metadata that our copy does not carry, so both pretend
# variables are required or the import fails at runtime, not at build time.
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
export HATCH_VCS_PRETEND_VERSION=0.1.0
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

rm -f "$OUT_ROOT/$TAG/celleval_manifest.json"

# Keep every comment ABOVE this invocation. See the warning in _competitor_common.sh.
cd "$CF_PKG"
"$PY" -u ../train_cellflow.py \
    --config "$CELLFLOW_ROOT/configs/occam_rhj_fewshot.yaml" \
    --output-dir "$OUT_ROOT" \
    --run-name "$TAG" \
    --num-iterations "$ITERATIONS" \
    --set "train.valid_freq=$VALID_FREQ" \
    --set "model.seed=$SEED" \
    --set "data.occam_bench_dir=$HERE/data/benchmarks/rhj_fewshot"

echo "=== TRAIN DONE ==="
RUN_DIR="$OUT_ROOT/$TAG"
"$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

meta_path = Path(sys.argv[1]) / "run_meta.json"
if not meta_path.is_file():
    print(f"WARNING: no run_meta.json at {meta_path}; selection is UNRECORDED",
          file=sys.stderr)
    raise SystemExit(0)
meta = json.loads(meta_path.read_text())
sel = meta.get("selection") or {}
best = sel.get("best")
if not best:
    print("WARNING: no selection record; the pickle holds the FINAL state, so this run is "
          "NOT selected-on-validation. Say so beside its number.", file=sys.stderr)
else:
    print(f"selected: {best['tag']} selection_score={best['selection_score']:.5f} "
          f"(of {sel['n_scored_states']} scored)")
    for h in sel.get("history", []):
        print(f"    {h['tag']:<8} {h['selection_score']:.5f}")
PY
ls -la "$RUN_DIR"
echo "next: ARM=cellflow SEED=$SEED sbatch --array=0-4 scripts/slurm/competitor_export.sbatch"
