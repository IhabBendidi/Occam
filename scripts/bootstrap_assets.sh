#!/usr/bin/env bash
# Copy the binary assets Occam needs out of the research tree.
#
# This is the ONE documented touchpoint back into scMAE-main. Everything it
# copies is gitignored, so a fresh clone runs this once. After it completes,
# nothing in src/ or configs/ references the old tree.
#
#     ./scripts/bootstrap_assets.sh SOURCE_REPO
#
# SOURCE_REPO is required: no absolute path is hardcoded anywhere here.
#
# Copies are real bytes, not symlinks: GPFS filesets block hardlinks between
# different filesets, and a symlinked repo is not
# self-contained for a HuggingFace upload. ~5.9 GB total.
#
# Provenance note: the checkpoint paths below are NOT what
# experiments/NEURIPS_CHAIN_A_XCELL_44K_OFFICIAL.md originally documented.
# Two SLURM run directories were written by colliding array tasks. See
# docs/reproducing_chain_a.md. The recipes in configs/ are transcribed from the
# checkpoints' own embedded configs, which are the only primary records.

set -euo pipefail

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
    echo "usage: $0 SOURCE_REPO" >&2
    echo "  SOURCE_REPO is the research checkout holding the checkpoints and" >&2
    echo "  bmdb_database/. No default: this repository contains no absolute paths." >&2
    exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$SRC/external_decoder.py" ]]; then
    echo "ERROR: $SRC does not look like scMAE-main (no external_decoder.py)." >&2
    echo "Usage: $0 [SOURCE_REPO]" >&2
    exit 2
fi

echo "source : $SRC"
echo "target : $HERE"
echo

# ---------------------------------------------------------------------------
# BMDB gene-relationship databases (6 MB, carries its own LICENSE)
# ---------------------------------------------------------------------------
echo "== BMDB databases =="
mkdir -p "$HERE/assets/bmdb_database"
cp -f "$SRC/bmdb_database"/*.txt \
      "$SRC/bmdb_database"/*.tsv \
      "$SRC/bmdb_database"/*.csv \
      "$SRC/bmdb_database/LICENSE" \
      "$HERE/assets/bmdb_database/"
ls -1 "$HERE/assets/bmdb_database" | sed 's/^/   /'
echo

# ---------------------------------------------------------------------------
# Chain A checkpoints
# ---------------------------------------------------------------------------
# name|relative source path|destination
CKPTS=(
  "stage1_pretrain|outputs/experiments/exp_045_training_2026-04-28_10-22-41/checkpoints/final_model.pth|checkpoints/reference/occam_small_stage1_pretrain.pth"
  "stage2_finetune|outputs/experiments/exp_062_finetuning_Finetuningfromfinal_model.pth_2026-04-30_17-54-02/finetuned_final_model.pth|checkpoints/reference/occam_small_stage2_finetune.pth"
  "stage3_release|decoder_results/ihabnet_decoder_20260501_172850/checkpoints/best_decoder.pth|checkpoints/release/occam_small_encoder_decoder.pth"
)

echo "== checkpoints =="
for entry in "${CKPTS[@]}"; do
    IFS='|' read -r name rel dst <<< "$entry"
    src_path="$SRC/$rel"
    dst_path="$HERE/$dst"
    if [[ ! -f "$src_path" ]]; then
        echo "ERROR: missing $name: $src_path" >&2
        exit 3
    fi
    mkdir -p "$(dirname "$dst_path")"
    if [[ -f "$dst_path" ]] && [[ "$(stat -c%s "$src_path")" == "$(stat -c%s "$dst_path")" ]]; then
        echo "   $name: already present, same size -- skipping"
        continue
    fi
    echo "   $name: copying $(numfmt --to=iec "$(stat -c%s "$src_path")")..."
    cp -f "$src_path" "$dst_path"
done
echo

# ---------------------------------------------------------------------------
# The one surviving stage-3 config record.
# ---------------------------------------------------------------------------
# Copied for the archaeology, NOT as a source of truth: it belongs to a
# different array task (lr 1e-5) than the weights (lr 3e-6). The `.UNTRUSTED`
# suffix is deliberate -- see docs/reproducing_chain_a.md.
cp -f "$SRC/decoder_results/ihabnet_decoder_20260501_172850/run_config.json" \
      "$HERE/checkpoints/release/occam_small_encoder_decoder.run_config.json.UNTRUSTED"

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
echo "== manifest (sha256 over ~5.9 GB, a few minutes) =="
python "$HERE/scripts/write_manifest.py" --repo "$HERE" --source-repo "$SRC"

echo
echo "Bootstrap complete. Next:"
echo "  cp configs/paths.example.yaml configs/paths.yaml   # and edit"
echo "  python scripts/verify_checkpoint.py"
