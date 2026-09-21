#!/usr/bin/env python
"""Pass 2 of the Norman-2019 rebuild: choose the 8,000-gene SCORING AXIS, with reasons.

WHY THIS EXISTS
    The axis decides what every metric sees. `pearson_delta` is a correlation over it, `mae` a
    mean over it, and `overlap_at_N` ranks DE genes inside it — so a gene that is zero in most
    cells does not merely add nothing, it adds noise to three of the four metrics and can never
    be called DE. The axis therefore has to be chosen on measured gene quality, and the rule has
    to be written down next to the artifact.

THE RULE (in order)
    1. **In Occam's vocabulary.** 29,187 of the 33,694 GEO genes are in the encoder's
       44,728-gene Ensembl vocabulary. A gene the encoder cannot represent is not a fair target
       for it or for any competitor reading the same axis.
    2. **Detected in > 8% of the benchmark's own QC cells.** 8,279 candidates survive — the
       threshold was chosen because it is the highest round value that still leaves more than
       8,000, so the arbitrary step (ranking) trims only 279 genes (3.4%) instead of doing the
       work. Detection is measured on the 91,168 QC cells the benchmark scores, never on the
       111,668 raw barcodes.
    3. ⚠️ **Force-include every perturbation target that the annotation resolves.** Norman is
       CRISPR**a**: the perturbed genes are transcription factors that are OFF at baseline and
       get switched ON, so a baseline-detection filter excludes exactly the genes being
       perturbed — **41 of 102 targets fail step 2**, including `FOXO4` (8.1%), `SNAI1` (5.4%)
       and `PRDM1` (2.9%). A benchmark that cannot see the activated target gene is not
       measuring the perturbation. This is a deliberate, documented exception to step 2.
    4. Fill the remainder to exactly 8,000 by descending variance of the log1p-normalised
       expression (the space the model trains and is scored in).

⚠️ THREE TARGETS ARE UNRESOLVABLE on this annotation: `C19orf26`, `C3orf72`, `KIAA1804` are
absent from GSE133344's symbol list (a GRCh37-era CellRanger reference). They stay usable as
perturbation LABELS — Occam's `pert2id` reaches them through a 2-entry alias map — but their own
expression cannot be scored. Their ENSG IDs are NOT guessed here; guessing one produced
`ENSG00000143674 -> RP5-862P8.2`, which is why it is not done.

    python scripts/norman/norman_pick_axis.py --n-genes 8000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))


def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default="data/external/GSE133344/gene_stats.npz")
    ap.add_argument("--qc-cells", default="data/external/GSE133344/qc_cells.npz")
    ap.add_argument("--checkpoint", default="checkpoints/release/occam_small_encoder_decoder.pth")
    ap.add_argument("--n-genes", type=int, default=8000)
    ap.add_argument("--min-detection", type=float, default=0.08)
    ap.add_argument("--out", default="data/benchmarks/norman_combo/axis.npz")
    a = ap.parse_args(argv)

    z = np.load(a.stats, allow_pickle=True)
    ens = np.array([_dec(x) for x in z["ensembl"]])
    sym = np.array([_dec(x) for x in z["symbol"]])
    det, var, mean = z["detection_rate"], z["var_log1p"], z["mean_log1p"]
    n_cells = int(z["n_cells"])

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    vocab = set(_dec(x) for x in ck["gene_names"])
    in_vocab = np.array([e in vocab for e in ens])

    q = np.load(a.qc_cells, allow_pickle=True)
    labels = [_dec(x) for x in q["label"]]
    targets = set()
    for l in set(labels):
        if l == "non-targeting":
            continue
        targets.update(l.split("+"))
    first = {}
    for i, s in enumerate(sym):
        first.setdefault(s, i)
    target_rows = np.array(sorted(first[t] for t in targets if t in first), dtype=np.int64)
    unresolved = sorted(t for t in targets if t not in first)

    passes = in_vocab & (det > a.min_detection)
    forced = np.zeros(len(ens), dtype=bool)
    forced[target_rows] = True
    # a forced target must still be representable by the encoder, or nothing can predict it
    forced &= in_vocab
    n_forced_below = int((forced & ~passes).sum())

    keep = forced.copy()
    room = a.n_genes - int(keep.sum())
    if room < 0:
        raise SystemExit(f"{int(keep.sum())} forced targets exceed --n-genes {a.n_genes}")
    pool = np.flatnonzero(passes & ~keep)
    order = pool[np.argsort(-var[pool])]
    keep[order[:room]] = True
    sel = np.flatnonzero(keep)
    assert len(sel) == a.n_genes, f"selected {len(sel)}, wanted {a.n_genes}"

    # order the axis by Ensembl id: a stable, annotation-independent order, like the rhj panel
    sel = sel[np.argsort(ens[sel])]
    digest = hashlib.sha256("\n".join(ens[sel]).encode()).hexdigest()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, row=sel.astype(np.int64), axis_ensembl=ens[sel], axis_symbols=sym[sel],
             detection_rate=det[sel], var_log1p=var[sel], mean_log1p=mean[sel],
             is_pert_target=forced[sel], n_qc_cells=np.int64(n_cells),
             min_detection=np.float64(a.min_detection), axis_sha256=digest)
    meta = {
        "axis_sha256": digest, "n_genes": int(len(sel)), "source_genes": int(len(ens)),
        "in_vocab": int(in_vocab.sum()), "min_detection": a.min_detection,
        "candidates_passing_filters": int(passes.sum()),
        "perturbation_targets": len(targets),
        "targets_resolved_on_annotation": int(len(target_rows)),
        "targets_force_included": int(forced.sum()),
        "targets_force_included_below_threshold": n_forced_below,
        "targets_unresolved": unresolved,
        "qc_cells": n_cells,
        "detection_of_selected": {
            "min": float(det[sel].min()), "median": float(np.median(det[sel])),
            "max": float(det[sel].max()),
            "below_threshold": int((det[sel] <= a.min_detection).sum()),
        },
    }
    json.dump(meta, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(f"[axis] {len(sel):,} genes -> {a.out}   sha256 {digest[:16]}…")
    print(f"  in vocab {in_vocab.sum():,} of {len(ens):,} · detection > {a.min_detection:.0%} "
          f"on {n_cells:,} QC cells -> {passes.sum():,} candidates")
    print(f"  perturbation targets: {len(targets)} total, {len(target_rows)} resolved, "
          f"{int(forced.sum())} force-included ({n_forced_below} of them BELOW threshold)")
    print(f"  unresolved targets (label-only, not scorable): {unresolved}")
    print(f"  selected detection: min {det[sel].min():.2%} median {np.median(det[sel]):.1%} "
          f"max {det[sel].max():.1%}; {int((det[sel] <= a.min_detection).sum())} below threshold "
          f"(the forced targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
