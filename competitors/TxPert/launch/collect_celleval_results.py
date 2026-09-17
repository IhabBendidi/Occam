"""Summarise the cell-eval CSVs from a TxPert LR sweep.

Reads every ``outputs/celleval/<tag>/<celltype>_agg_results.csv`` written by
``export_for_celleval`` and prints a per-LR table (mean +/- sd across seeds), plus the
per-seed detail so an outlier is visible rather than averaged away.

    uv run python launch/collect_celleval_results.py
    uv run python launch/collect_celleval_results.py --outputs outputs/celleval --csv summary.csv

Tags are expected to look like ``lr0.0003-seed1`` (what launch/txpert_zarr_lr_sweep.sh
writes); anything else is reported under its own name with no grouping.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# The metrics worth eyeballing. pearson_delta is the headline the other arms are ranked on.
HEADLINE = [
    "pearson_delta",
    "mae",
    "mse",
    "de_direction_match",
    "de_sig_genes_recall",
    "de_spearman_sig",
    "overlap_at_N",
    "precision_at_100",
    "discrimination_score_l1",
    "discrimination_score_cosine",
]

TAG_RE = re.compile(r"^lr(?P<lr>[0-9.eE+-]+)-seed(?P<seed>\d+)$")


def _tag_for(agg: Path, root: Path) -> str:
    """Recover the run tag from a CSV path.

    ``main.py`` appends ``wandb.run.id`` to the configured output_dir, so the layout is
    ``<root>/<tag>/<wandb-id>/hepg2_agg_results.csv``. Walk up to the first ancestor that
    looks like a sweep tag, else fall back to the directory directly under root.
    """
    parts = agg.relative_to(root).parts[:-1]  # drop the filename
    for part in parts:
        if TAG_RE.match(part):
            return part
    return parts[0] if parts else agg.parent.name


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for agg in sorted(root.rglob("*_agg_results.csv")):
        tag = _tag_for(agg, root)
        stats = pd.read_csv(agg).set_index("statistic")
        if "mean" not in stats.index:
            print(f"  ! {tag}: no 'mean' row, skipping")
            continue
        m = TAG_RE.match(tag)
        row = {
            "tag": tag,
            "lr": m.group("lr") if m else tag,
            "seed": int(m.group("seed")) if m else -1,
            "n_perts": int(stats.loc["count"].iloc[0]),
        }
        row.update({k: float(stats.loc["mean"][k]) for k in HEADLINE if k in stats.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(Path(__file__).resolve().parents[1] / "outputs" / "celleval"))
    ap.add_argument("--csv", default=None, help="also write the per-run table here")
    args = ap.parse_args()

    root = Path(args.outputs)
    runs = load_runs(root)
    if runs.empty:
        raise SystemExit(f"no *_agg_results.csv found under {root}")

    # A run scored on fewer than the full holdout is not comparable to the other arms.
    odd = runs[runs["n_perts"] != 380]
    if not odd.empty:
        print("WARNING: runs not scored on 380 perturbations:")
        print(odd[["tag", "n_perts"]].to_string(index=False), "\n")

    print(f"=== per-run ({len(runs)} runs) ===")
    print(runs.sort_values(["lr", "seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    grouped = runs.groupby("lr")
    print("\n=== per-LR, mean +/- sd across seeds ===")
    summary = pd.DataFrame({
        "n_seeds": grouped.size(),
        **{
            k: grouped[k].mean().round(4).astype(str) + " +/- " + grouped[k].std().round(4).fillna(0).astype(str)
            for k in HEADLINE if k in runs.columns
        },
    })
    print(summary.to_string())

    if "pearson_delta" in runs.columns:
        best = grouped["pearson_delta"].mean().idxmax()
        print(f"\nbest LR by mean pearson_delta: {best}")
        print("reference: TxPert's own sweep optimum was 0.4779 (its internal metric, 50 epochs)")

    if args.csv:
        runs.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
