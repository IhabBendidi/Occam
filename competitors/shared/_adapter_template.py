"""The shape every arm's `export_occam.py` takes, and the contract it must satisfy.

NOT IMPORTED BY ANYTHING. This file is the reference an arm's adapter is written against, so
that four adapters written in four different environments agree on the one thing that must be
identical: what `predict` receives, what it returns, and who converts the space.

THE CONTRACT
------------
    predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray

* `ctrl` is `(n, 6983)` float32 in **the arm's own training space** -- by default
  `log1p(target_sum=10_000)`, the space `occam_data` hands the arm during training. One real
  half-B HepG2 control cell per row.
* `pert_names` is the perturbation to apply to each row: same length, same order, gene SYMBOLS.
* the return is `(n, 6983)` in the arm's **emission** space, whatever that is.

`export_split` then converts that emission into our evaluation space ONCE, at the boundary,
and REFUSES anything that does not land in it. The arm never needs to know about our space --
that is the point. Its internal preprocessing (PCA, z-scoring, whatever) is its own business;
the interface is ours in both directions.

DECLARE THE EMISSION SPACE. DO NOT LET IT BE GUESSED.
-----------------------------------------------------
`export_split(..., from_target_sum=..., from_log1p=...)` is a DECLARATION about what the arm
returns, and `assert_same_space` checks it. Get it wrong and you get a plausible number from a
meaningless comparison: cell-eval cannot tell, because `guess_is_lognorm` reads 500 UNSEEDED
cells and asks only whether the row sums have fractional parts.

    CellFlow     decodes from a 256-component PCA basis  -> declare what the decode returns
    PerturbDiff  emits gene space through `output_fn: relu` -> non-negative, not necessarily log1p
    STATE        `output_space: gene`                     -> the space it was trained on
    TxPert       gene space                               -> the space it was trained on

⚠️ Apply the map ONCE and check it. Do not iterate on it until a number looks good: the repo
has already measured that this class of change is not free -- re-emitting our own decoder's
predictions as integer counts cost 0.034 `pearson_delta`, and rescaling the library size back
recovered none of it.

ROW ORDER IS A CONTRACT
-----------------------
The rows are the half-A perturbed cells of a split, ordered by
`np.argsort(labels, kind="stable")`. `load_panel_split` produces them; `labels.json` records
them; `run_split` re-checks them. An adapter must not reorder, deduplicate or batch across
that order.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np


def build_parser(arm: str) -> argparse.ArgumentParser:
    """The CLI every adapter exposes, because `competitor_export.sbatch` calls them all."""
    ap = argparse.ArgumentParser(description=f"Phase A for {arm}")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--checkpoint", default="",
                    help="default: the arm's own selection marker for this seed")
    ap.add_argument("--batch-size", type=int, default=2048)
    return ap


def example(argv=None) -> int:
    """What an adapter's `main` looks like. The two arm-specific parts are marked."""
    arm = "example"
    args = build_parser(arm).parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from export_predictions import TRAIN_TARGET_SUM, export_split, load_panel_split

    split = load_panel_split(args.panel_dir, args.split, target_sum=TRAIN_TARGET_SUM)

    # ---- ARM-SPECIFIC 1: load the checkpoint the arm selected on VALIDATION ----
    model = None  # e.g. Arm.load_from_checkpoint(args.checkpoint).eval()

    # ---- ARM-SPECIFIC 2: one control cell in, one predicted cell out ----
    def predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray:
        del model
        raise NotImplementedError(f"{arm}: implement predict")

    export_split(predict, split, args.out_dir, arm=arm,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=args.batch_size,
                 extra_manifest={"checkpoint": args.checkpoint, "seed": args.seed})
    return 0
