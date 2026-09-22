#!/usr/bin/env python
"""Emit the `norman_combo` BENCHMARK DEFINITION the vcell training stage loads.

WHY THIS EXISTS
    `occam.vcell.benchmark.load_benchmark` is the gate between a training run and a frozen
    task: it refuses to start if the definition moved, if a source file's sha256 changed, or
    if an axis gene is outside the encoder's vocabulary. Every one of those failures would
    otherwise produce a plausible number rather than an error. The split-half PANEL
    (`splithalf/`) is the evaluation artifact; THIS is the training-side one, and the stage
    cannot run without it.

WHAT IT WRITES  (data/benchmarks/norman_combo/)
    benchmark.json    ids, encoder provenance, gene-axis block, source checksums
    gene_axis.csv     the 8,000 (ensembl, symbol) pairs, hashed
    splits.json       train = the 105 SINGLE genes · val = 20 pairs · test = 111 pairs
    exclusions.json   the 131 double-perturbation labels, held out of training BY CELL

THE TASK, ENCODED IN THE SPLIT
    train  105 single-gene conditions (47,269 cells) + the shared controls
    val     20 double pairs   -- held out for checkpoint selection
    test   111 double pairs   -- never seen in any form

⚠️ DOUBLES ARE HELD OUT TWICE OVER, on purpose. They are absent from `splits.json:train`
(which is the name set the streaming datasets filter on) AND listed in `exclusions.json`
(which drops their cells per (file, perturbation)). Either alone would be enough; the pair of
them means a change to one mechanism cannot silently leak the test set into training.

⚠️ `all_perts_train: false`. Every rhj_fewshot checkpoint carried `all_perts_train: true`,
which is what made "unseen perturbation" a post-hoc reconstruction there rather than a
property of the run. Here the held-out axis is real and recorded.

    python scripts/norman/norman_build_benchmark.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

CONTROL = "non-targeting"


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/eval/pert/norman_2019_k562.h5ad")
    ap.add_argument("--axis", default="data/benchmarks/norman_combo/axis.npz")
    ap.add_argument("--panel", default="data/benchmarks/norman_combo/splithalf")
    ap.add_argument("--checkpoint", default="checkpoints/release/occam_small_encoder_decoder.pth")
    ap.add_argument("--out-dir", default="data/benchmarks/norman_combo")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    import h5py
    import torch

    az = np.load(a.axis, allow_pickle=True)
    axis_ens = [_dec(x) for x in az["axis_ensembl"]]
    axis_sym = [_dec(x) for x in az["axis_symbols"]]

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    enc_names = [_dec(x) for x in ck["gene_names"]]
    outside = sorted(set(axis_ens) - set(enc_names))
    if outside:
        raise SystemExit(f"{len(outside)} axis genes outside the encoder vocabulary "
                         f"(e.g. {outside[:5]}); the axis and checkpoint disagree")

    with h5py.File(a.source, "r") as f:
        o = f["obs"]["gene_name"]
        if isinstance(o, h5py.Group):
            cats = [_dec(c) for c in o["categories"][:]]
            labels = np.asarray(cats)[o["codes"][:]]
        else:
            labels = np.array([_dec(x) for x in o[:]])
        n_obs = len(labels)
        n_vars = f["var"]["gene_id"].shape[0]
    singles = sorted({l for l in set(labels) if l != CONTROL and "+" not in l})
    doubles = sorted({l for l in set(labels) if "+" in l})
    counts = {l: int((labels == l).sum()) for l in set(labels)}

    with open(os.path.join(a.panel, "splits_manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(a.panel, "cohort.json")) as fh:
        cohort = json.load(fh)
    val = sorted(man["val_pairs"])
    test = sorted(cohort["perts"])
    if set(val) & set(test):
        raise SystemExit("the val pairs and the test cohort overlap; the panel is inconsistent")
    unaccounted = sorted(set(doubles) - set(val) - set(test))

    os.makedirs(a.out_dir, exist_ok=True)

    # ---- gene_axis.csv ---------------------------------------------------------------
    axis_csv = os.path.join(a.out_dir, "gene_axis.csv")
    with open(axis_csv, "w") as fh:
        fh.write("ensembl_gene_id,gene_name\n")
        for e, s in zip(axis_ens, axis_sym):
            fh.write(f"{e},{s}\n")
    axis_sha = hashlib.sha256("\n".join(axis_ens).encode()).hexdigest()

    # ---- splits.json ----------------------------------------------------------------
    splits = {
        "_note": ("`train` is the SINGLE-gene set only: the 131 double pairs are the task and "
                  "appear in NO training label. exclusions.json drops their cells as well, so "
                  "neither mechanism alone is load-bearing."),
        "benchmark_id": "norman_combo",
        "seed": a.seed,
        "train": singles,
        "val": val,
        "test_unseen_pert": test,
        "all_perts_train": False,
        "holdout_cell_line": None,
        # ⚠️ LISTS, not counts. `PerturbationSplitter.load` calls len() on
        # `test_unseen_cell`, so an int here fails with
        # `TypeError: object of type 'int' has no len()` after the benchmark has already
        # loaded -- 45 seconds into three jobs. rhj_fewshot stores cell-line NAMES and empty
        # lists; the earlier diagnostic that printed `len(v)` for lists is what made these
        # look like integers.
        "cell_lines_train": ["K562"],
        "cell_lines_test": [],
        "test_unseen_cell": [],
        "train_fraction": 1.0, "val_fraction": 0.0, "test_fraction": 0.0,
    }
    with open(os.path.join(a.out_dir, "splits.json"), "w") as fh:
        json.dump(splits, fh, indent=2, sort_keys=True)

    # ---- exclusions.json ------------------------------------------------------------
    src_name = os.path.basename(a.source)
    ex = {
        "_note": ("Every double-perturbation label, held out of training BY CELL. The task is "
                  "to predict these from the single-gene arms alone."),
        "by_file": {src_name: sorted(doubles)},
        "eval_cell_lines": {src_name: ["K562"]},
        "n_excluded_labels": len(doubles),
        "n_excluded_cells": int(sum(counts[d] for d in doubles)),
    }
    with open(os.path.join(a.out_dir, "exclusions.json"), "w") as fh:
        json.dump(ex, fh, indent=2, sort_keys=True)

    # ---- benchmark.json -------------------------------------------------------------
    bench = {
        "schema": 1,
        "benchmark_id": "norman_combo",
        "builder_version": "norman_combo/v1.0.0",
        "built": str(date.today()),
        "command": "scripts/norman/norman_build_benchmark.py",
        "task": ("train on single-gene perturbations, predict DOUBLE-gene pairs; both genes of "
                 "every test pair are seen individually and the pair never is"),
        "cell_line": "K562",
        "cell_line_note": ("Norman et al. 2019 (Science 365:786), GSE133344. The upstream "
                           "norman_2016_a549_preprocessed.h5ad mislabels these cells A549; see "
                           "docs/norman_benchmark.md for the three lines of evidence."),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip() or "unknown",
        "encoder_checkpoint": a.checkpoint,
        "encoder_sha256": _sha256_file(a.checkpoint),
        "encoder_n_genes": len(enc_names),
        "encoder_target_sum": 2000.0,
        "n_graph_nodes": int(len(ck["pert2id"])) if "pert2id" in ck else None,
        "gene_axis": {
            "key": "var.gene_id",
            "n": len(axis_ens),
            "sha256": axis_sha,
            "symbols_sha256": hashlib.sha256("\n".join(axis_sym).encode()).hexdigest(),
            "selection": ("in the encoder's Ensembl vocabulary AND detected in >8% of the "
                          "91,168 QC cells, PLUS all 102 resolvable perturbation targets "
                          "force-included (41 of them fail the detection filter because "
                          "Norman is CRISPRa and the targets are off at baseline)"),
            "n_source_genes": int(n_vars),
            "rejected_keys": {
                "var.gene_name": ("only 3 of 5,045 symbols are in the encoder vocabulary; the "
                                  "encoder axis is 99.6% Ensembl"),
            },
        },
        "counts": {
            "cells": int(n_obs),
            "single_conditions": len(singles),
            "double_pairs": len(doubles),
            "val_pairs": len(val),
            "test_pairs": len(test),
            "doubles_in_neither_val_nor_test": unaccounted,
            "controls": int(counts.get(CONTROL, 0)),
            "train_cells": int(sum(counts[s] for s in singles)),
            "excluded_cells": int(sum(counts[d] for d in doubles)),
        },
        "sources": {
            src_name: {
                "bytes": os.path.getsize(a.source),
                "cell_line": "K562",
                "n_obs": int(n_obs),
                "n_vars": int(n_vars),
                "sha256": _sha256_file(a.source),
                "upstream": "GSE133344 filtered_matrix.mtx.gz",
                "qc": "good_coverage == True AND number_of_cells == 1 (Norman's own filter)",
            }
        },
        "panel": {
            "dir": os.path.relpath(a.panel, REPO),
            "benchmark_sha256": man.get("benchmark_sha256"),
            "cohort_sha256": cohort.get("cohort_sha256"),
            "scored_panels": man.get("scored_panels"),
        },
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "artifact_sha256": {
            n: _sha256_file(os.path.join(a.out_dir, n))
            for n in ("gene_axis.csv", "splits.json", "exclusions.json")
        },
    }
    bpath = os.path.join(a.out_dir, "benchmark.json")
    with open(bpath, "w") as fh:
        json.dump(bench, fh, indent=2, sort_keys=True)
    bsha = _sha256_file(bpath)

    # ---- write the definition hash back into the PANEL manifest ---------------------
    # `eval_splithalf.assert_checkpoint_matches` compares the checkpoint's recorded
    # `benchmark_sha256` against the PANEL's, and a mismatch is a hard refusal -- correctly,
    # because it is the guard against scoring a checkpoint on a benchmark it was not trained
    # on. The panel is built first (the benchmark reads its cohort), so it cannot know this
    # hash at its own build time; it is stamped here. My first version left the panel holding
    # a DIFFERENT quantity under the same name -- sha256(axis|cohort|n_splits|seed) -- and
    # 247 of 360 eval tasks refused before doing any work.
    man_path = os.path.join(a.panel, "splits_manifest.json")
    with open(man_path) as fh:
        pman = json.load(fh)
    pman["panel_identity_sha256"] = pman.get("benchmark_sha256")
    pman["panel_identity_recipe"] = pman.get("benchmark_sha256_recipe")
    pman["benchmark_sha256"] = bsha
    pman["benchmark_sha256_recipe"] = ("sha256(benchmark.json) -- the DEFINITION hash the "
                                       "training stage stamps into every checkpoint")
    with open(man_path, "w") as fh:
        json.dump(pman, fh, indent=2, sort_keys=True, default=str)
    print(f"[bench] stamped benchmark_sha256 into {man_path}")

    print(f"[bench] wrote {a.out_dir}")
    print(f"  benchmark.json sha256 {bsha}   <-- put this in the config's benchmark.sha256")
    print(f"  train {len(singles)} single genes ({bench['counts']['train_cells']:,} cells) · "
          f"val {len(val)} pairs · test {len(test)} pairs")
    print(f"  excluded from training: {len(doubles)} double labels, "
          f"{bench['counts']['excluded_cells']:,} cells")
    print(f"  axis {len(axis_ens):,} genes sha256 {axis_sha[:16]}…")
    if unaccounted:
        print(f"  ⚠️ {len(unaccounted)} double pair(s) are in neither val nor test "
              f"(below min_cells): {unaccounted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
