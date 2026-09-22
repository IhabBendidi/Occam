#!/usr/bin/env python
"""Rebuild PerturbDiff's GenePT pickle at its REAL coverage of our perturbations.

WHAT THIS ANSWERS
-----------------
Overall goal: the best model on SEEN perturbations in an UNSEEN cell line.

This script's question: is PerturbDiff's genept arm handicapped by the MODEL, or by an asset
that was truncated to someone else's gene panel?

THE PROBLEM
-----------
`replogle_gene_emb_dict_perturbation_emb_dict.pkl`, which their fewshot recipe passes as
`replogle_gene_encoding=genept`, holds **2,024 keys** -- their 2,000-HVG panel's vocabulary.
Against `rhj_fewshot` that leaves:

    the 611 scored perturbations : 547 covered (89.5%),  64 with a ZERO embedding
    all 2,381 symbols            : 2,021 covered (84.9%), 360 with a ZERO embedding

64 of the scored perturbations sharing one all-zero representation means the model cannot tell
them apart -- and that is a property of the pickle, not of PerturbDiff.

WHY REBUILDING IS LEGITIMATE, AND NOT MANUFACTURING SIGNAL
----------------------------------------------------------
The full GenePT v2 release holds **133,736 keys at the same 3,072 dimensions**, and their
pickle is a strict subset of it. MEASURED 2026-09-18, and this is the check that licenses the
rebuild:

    their keys present in the full release : 2,009 / 2,024
    max abs value difference on 200 shared keys : **0.000e+00** (200/200 identical)

Same embedding model, same dimensionality, byte-identical values. So this does not invent an
embedding for anything -- it stops truncating an asset that already covers our genes. Per the
project's "not biology is not the same as cheating" rule, this is using an asset at its real
coverage, which makes the arm a more faithful representation of the model.

⚠️ THE UNION, NOT A REPLACEMENT. Their pickle has **15 keys the full release lacks**
(`AC118549.1`, `C12orf45`, `C7orf26`, ...) -- aliases they added. Dropping those would make the
genept arm worse on exactly the genes they cared about, so the output is
`full release ∪ theirs`, with THEIR value kept wherever both have a key (identical anyway, so
the rule only matters for the 15).

WHAT IT BUYS, MEASURED
----------------------
    the 611 scored perturbations : 89.5% -> **98.7%**  (64 zeros -> 8)
    all 2,381 symbols            : 84.9% -> **99.2%**  (360 zeros -> 19)

⚠️ **8 of the 611 are still absent** (`C7orf26`, `C9orf16`, `FAM102B`, `INTS2`, `PHB`,
`SPATA5L1`, ...) and keep a zero embedding. 1.3% is not zero and goes in the record.

⚠️ THIS MAKES THE ARM NON-COMPARABLE TO THEIR PUBLISHED 0.40306, which was measured with the
truncated pickle on their own panel. Say so beside the number.

    python scripts/build_genept_for_occam.py --dry-run
    python scripts/build_genept_for_occam.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "competitors", "shared"))

FULL = ("/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/GenePT/genept_embeddings/"
        "extracted/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.")
THEIRS = ("/rxrx/scratch/sandbox/ihab.bendidi/perturbdiff_out/assets_dl/perturb_data/"
          "gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl")
OUT = os.path.join(REPO, "data", "assets", "genept_occam_rhj_fewshot.pkl")
DIM = 3072


def _sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--theirs", default=THEIRS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--bench-dir",
                    default=os.path.join(REPO, "data", "benchmarks", "rhj_fewshot"))
    ap.add_argument("--all-keys", action="store_true",
                    help="keep every key in the union (133k, ~3 GB) instead of restricting to "
                         "our symbols. Not needed: the model only looks up its pert_dict.")
    ap.add_argument("--compose-pairs", default="mean", choices=["none", "mean", "sum"],
                    help="how a multi-gene condition `A+B` gets an embedding. A GenePT "
                         "table is per GENE, so without this every pair is a zero vector "
                         "(131 of 236 conditions on norman_combo, which PerturbDiff "
                         "refuses). `mean` keeps the norm in the range the conditioning "
                         "MLP saw on singles and is symmetric, as a pair must be. OUR "
                         "choice, recorded in the manifest.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    import occam_data as od

    full = pickle.load(open(args.full, "rb"))
    theirs = pickle.load(open(args.theirs, "rb"))
    print(f"[genept] full release : {len(full):,} keys")
    print(f"[genept] theirs       : {len(theirs):,} keys")

    # ⛔ The gate. If the values ever diverge, their pickle is NOT the same asset and merging
    # would silently change the arm rather than un-truncate it.
    shared = sorted(set(map(str, theirs)) & set(map(str, full)))
    rng = np.random.default_rng(0)
    sample = [shared[i] for i in rng.choice(len(shared), min(300, len(shared)), replace=False)]
    worst = max(float(np.abs(np.asarray(theirs[k], dtype=np.float64)
                             - np.asarray(full[k], dtype=np.float64)).max()) for k in sample)
    print(f"[genept] value check on {len(sample)} shared keys: max abs diff {worst:.3e}")
    if worst > 1e-9:
        raise SystemExit(
            f"their pickle and the full release DISAGREE on shared keys (max {worst:.3e}). "
            f"They are not the same asset, so merging would change the arm rather than "
            f"un-truncate it. Refusing.")

    bench = od.load_occam_benchmark(args.bench_dir, verify_sources=False)
    wanted = set()
    for k in ("train", "val", "test_unseen_pert"):
        wanted.update(str(p) for p in (bench.split.get(k) or []))
    wanted.discard(od.CONTROL_LABEL)
    with open(os.path.join(args.bench_dir, "splithalf", "cohort.json")) as fh:
        cohort = [str(p) for p in json.load(fh)["perts"]]

    # union, THEIR value winning on a shared key (identical anyway; matters for their 15 extras)
    merged = {}
    keys = set(map(str, full)) | set(map(str, theirs))
    if not args.all_keys:
        keys &= wanted
    for k in sorted(keys):
        v = theirs.get(k, full.get(k))
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.shape[0] != DIM:
            raise SystemExit(f"{k}: embedding is {arr.shape[0]}-dim, expected {DIM}")
        # their format is a plain list of floats; match it exactly
        merged[k] = arr.tolist()

    def cov(ps, d):
        n = sum(1 for p in ps if p in d)
        return n, len(ps) - n, 100.0 * n / len(ps)

    print()
    for name, ps in (("the 611 scored cohort", cohort), ("all 2,381 symbols", sorted(wanted))):
        bt = cov(ps, theirs)
        bm = cov(ps, merged)
        print(f"{name} ({len(ps):,})")
        print(f"  theirs : {bt[0]:,} covered ({bt[2]:.1f}%), {bt[1]} zero")
        print(f"  MERGED : {bm[0]:,} covered ({bm[2]:.1f}%), {bm[1]} zero")
        still = sorted(set(ps) - set(merged))
        if still:
            print(f"  ⚠️ still zero: {len(still)} -> {still[:8]}")

    # ─── composed embeddings for MULTI-GENE conditions ───────────────────────────────
    # A GenePT table is per GENE, so a combinatorial cohort leaves every `A+B` condition
    # with a zero vector: on norman_combo that is 131 of 236 conditions, and PerturbDiff
    # refuses above 50% zero-filled -- correctly, since a mostly-zero table trains
    # without complaint and destroys the conditioning.
    #
    # ⚠️ THIS IS OUR CHOICE, not GenePT's and not PerturbDiff's. The pair's embedding is
    # the MEAN of its components':
    #   * MEAN rather than SUM so the vector's norm stays in the range the conditioning
    #     MLP saw during training, which only ever sees singles (norman_combo trains on
    #     105 single conditions);
    #   * it is the same assumption PerturbDiff's composition rule already makes at
    #     inference -- `decode_seq_mean` averages the two orders -- so the conditioning
    #     and the emission agree rather than pulling in different directions;
    #   * and it is symmetric, which a pair must be: `A+B` and `B+A` are one condition.
    # Recorded in the manifest as `composed_pairs` so a reader can see which keys are
    # measured and which are derived.
    composed = []
    if args.compose_pairs != "none":
        import numpy as _np
        for cond in sorted(set(wanted)):
            if "+" not in str(cond) or str(cond) in merged:
                continue
            parts = [p for p in str(cond).split("+") if p]
            vecs = [merged[p] for p in parts if p in merged]
            if len(vecs) != len(parts):
                continue          # a component we do not have; leave it zero, honestly
            stack = _np.stack(vecs, axis=0)
            merged[str(cond)] = (stack.mean(axis=0) if args.compose_pairs == "mean"
                                 else stack.sum(axis=0)).astype(stack.dtype)
            composed.append(str(cond))
        if composed:
            print(f"[genept] composed {len(composed):,} multi-gene key(s) as the "
                  f"{args.compose_pairs.upper()} of their components "
                  f"(e.g. {composed[:3]})")

    print(f"\n[genept] merged: {len(merged):,} keys x {DIM} dims")
    if args.dry_run:
        print("[genept] --dry-run: nothing written")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(merged, fh, protocol=4)
    man = {
        "built": "scripts/build_genept_for_occam.py",
        "n_keys": len(merged), "dim": DIM,
        "compose_pairs": args.compose_pairs,
        "composed_pairs": composed,
        "restricted_to_benchmark_symbols": not args.all_keys,
        "sources": {
            "full_release": {"path": args.full, "n_keys": len(full),
                             "sha256": _sha256(args.full)},
            "perturbdiff_pickle": {"path": args.theirs, "n_keys": len(theirs),
                                   "sha256": _sha256(args.theirs)},
        },
        "value_check_max_abs_diff": worst,
        "merge_rule": "union; their value kept on a shared key",
        "coverage": {
            "cohort_611": {"theirs": cov(cohort, theirs)[0], "merged": cov(cohort, merged)[0],
                           "still_zero": sorted(set(cohort) - set(merged))},
            "all_symbols": {"theirs": cov(sorted(wanted), theirs)[0],
                            "merged": cov(sorted(wanted), merged)[0]},
        },
        "caveat": ("NOT comparable to PerturbDiff's published 0.40306, which used the "
                   "truncated 2,024-key pickle on their own 2,000-gene panel."),
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    print(f"[genept] wrote {os.path.relpath(args.out, REPO)} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB) + .json manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
