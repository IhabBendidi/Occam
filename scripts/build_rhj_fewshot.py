#!/usr/bin/env python
"""Build the frozen RPE1 / HepG2 / Jurkat few-shot benchmark definition.

    python scripts/build_rhj_fewshot.py --dry-run     # report, write nothing
    python scripts/build_rhj_fewshot.py

WHAT THIS IS FOR
----------------
A benchmark has to be a benchmark, not a configuration. This writes a small, all-text,
checked-in definition -- gene axis, perturbation split, per-file exclusions, eligibility
table -- and hashes every piece, so that two runs six months apart are comparable and so
that "we changed the axis" cannot happen quietly.

THE DESIGN, IN FOUR FACTS
-------------------------
1. **The gene axis is the 3-way `var['ensembl_gene_id']` intersection.** Ensembl is the only
   viable key: `gene_name` matches essentially nothing in the encoder vocabulary and carries a
   duplicate (`HSPA14`), and `var.index` holds gene symbols in one file and positional integer
   strings in the other two, so its "intersection" is a coincidence rather than a gene match.
   Symbols whose Ensembl mapping is ambiguous are dropped entirely, not deduplicated.

2. **RPE1 and Jurkat contribute every perturbation to training. HepG2 is split.** The named
   validation and test perturbations are withheld from HepG2 *only*.

3. ⚠️ **`splits.json` deliberately has `train ∩ val != {}`.** `train` is the global name set
   the streaming datasets filter on, and every held-out symbol legitimately trains from Jurkat
   and RPE1. The per-(file, perturbation) exclusion in `exclusions.json` is what makes the
   held-out *cells* held out. This reads as leakage if you have not read the design; it is not.

4. **Eligibility is checked against the perturbation graph, and a failure is fatal.** On the
   streaming path an unresolvable perturbation symbol is dropped *silently* (a bare `continue`
   in the v1 loader; `pert_ok &= pid >= 0` in v2). If a held-out symbol has no graph node the
   model cannot represent it at all, the held-out set shrinks invisibly, and the benchmark
   quietly becomes a different benchmark. So the builder resolves every symbol up front and
   refuses to write unless the counts match what is declared below.

Measured on the 2026-09 data: 4 of the 728 requested held-out symbols have no graph node --
`CCDC144NL` (val) and `ALG1L`, `C14orf178`, `NEDD8-MDP1` (test). They are dropped, giving 103
val + 621 test. `EXPECT_VAL`/`EXPECT_TEST` pin those numbers so a graph-vocabulary update
cannot redefine the benchmark without someone editing this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

BUILDER_VERSION = "1.0.0"
BENCHMARK_ID = "rhj_fewshot"

# The frozen split sizes, after graph-eligibility filtering. The builder refuses to write if
# it computes anything else -- that is the whole point of pinning them.
# The arithmetic, so these are readable rather than magic:
#   2,394 perturbation labels per file (incl. the control label)
#   2,391 non-control labels present in ALL THREE files  (2 are missing from at least one)
#   2,381 of those also resolve to a perturbation-graph node   <- "eligible"
#     724 eligible labels held out of HepG2 (103 val + 621 test)
#   1,657 eligible labels remain as HepG2 training perturbations
EXPECT_VAL = 103
EXPECT_TEST = 621
EXPECT_HEPG2_TRAIN = 1657


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# h5py readers -- deliberately not anndata. The eval bundle is written by a
# newer anndata than the training environment can read.
# --------------------------------------------------------------------------

def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _read_column(grp, name: str) -> Optional[List[str]]:
    """Read one obs/var column as strings, resolving anndata's categorical encoding.

    Categoricals appear two ways depending on the writer's anndata version: as a subgroup
    holding `codes` + `categories`, or as a plain dataset of codes with the categories in a
    sibling `__categories` group. Both are handled, because these files were written by a
    newer anndata than the training environment can read with anndata itself.
    """
    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, type(grp)):                       # modern categorical subgroup
        codes = node["codes"][:]
        cats = _decode(node["categories"][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]
    if "__categories" in grp and name in grp["__categories"]:   # legacy encoding
        cats = _decode(grp["__categories"][name][:])
        return [cats[c] if 0 <= c < len(cats) else "" for c in node[:]]
    return _decode(node[:])


def _read_obs_column(f, name: str) -> Optional[List[str]]:
    return _read_column(f["obs"], name)


def _read_var_column(f, name: str) -> List[str]:
    out = _read_column(f["var"], name)
    if out is None:
        raise KeyError(f"var has no column {name!r}; available: {sorted(f['var'].keys())}")
    return out


def _shape(f) -> Tuple[int, int]:
    x = f["X"]
    if hasattr(x, "shape") and getattr(x, "shape", None) is not None and len(x.shape) == 2:
        return int(x.shape[0]), int(x.shape[1])
    return tuple(int(v) for v in x.attrs["shape"])  # CSR group


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------

def read_file(path: str, gene_col: str, pert_col: str, control_label: str) -> Dict:
    """Everything the builder needs from one h5ad, without touching X."""
    import h5py

    with h5py.File(path, "r") as f:
        n_obs, n_vars = _shape(f)
        ens = _read_var_column(f, gene_col)
        sym = _read_var_column(f, "gene_name")
        perts = _read_obs_column(f, pert_col)
        lines = _read_obs_column(f, "cell_line")
        batches = _read_obs_column(f, "dataset_batch_num")

    if perts is None:
        raise ValueError(f"{path}: obs has no {pert_col!r} column")
    if lines is None:
        raise ValueError(f"{path}: obs has no 'cell_line' column")

    uniq_lines = sorted(set(lines))
    if len(uniq_lines) != 1:
        raise ValueError(
            f"{path}: expected exactly one cell line, found {len(uniq_lines)}: "
            f"{uniq_lines[:5]}. The per-file exclusion keys on the file, so a multi-line "
            f"file would need the finer (file, line, perturbation) key this benchmark "
            f"deliberately does not use.")

    dup = len(ens) - len(set(ens))
    if dup:
        # The alignment loop appends to dst_indices, so a duplicated ID would produce two
        # destination columns and the last write would silently win.
        raise ValueError(f"{path}: {dup} duplicated {gene_col!r} values; alignment would be "
                         f"ambiguous")

    per_pert: Dict[str, int] = {}
    for p in perts:
        per_pert[p] = per_pert.get(p, 0) + 1

    return {
        "path": path,
        "name": Path(path).name,
        "sha256": _sha256(path),
        "bytes": os.path.getsize(path),
        "mtime": os.path.getmtime(path),
        "n_obs": n_obs,
        "n_vars": n_vars,
        "cell_line": uniq_lines[0],
        "ensembl": ens,
        "symbols": sym,
        "ens_to_sym": dict(zip(ens, sym)),
        "n_controls": int(per_pert.get(control_label, 0)),
        "n_perturbed": int(n_obs - per_pert.get(control_label, 0)),
        "per_pert_cells": {k: int(v) for k, v in per_pert.items() if k != control_label},
        "n_batches": len(set(batches)) if batches else 0,
        "batch_ids": sorted(set(batches)) if batches else [],
    }


def build_axis(files: Sequence[Dict], encoder_genes: Sequence[str]) -> Tuple[List[str], Dict]:
    """The 3-way Ensembl intersection, minus ambiguous symbols, all inside the encoder vocab."""
    vocab = set(encoder_genes)
    sets = [set(f["ensembl"]) for f in files]
    inter = set.intersection(*sets)
    n_inter = len(inter)

    outside = sorted(inter - vocab)
    inter &= vocab

    # Drop BOTH members of any symbol that maps from more than one Ensembl ID on this axis.
    # Keeping one would leave a symbol whose Ensembl mapping is ambiguous, and since var.index
    # must carry symbols for the scorer's self-gene exclusion to work, that ambiguity would
    # travel into the scored files.
    sym_of = files[0]["ens_to_sym"]
    by_sym: Dict[str, List[str]] = {}
    for e in inter:
        by_sym.setdefault(sym_of.get(e, e), []).append(e)
    ambiguous = {s: sorted(v) for s, v in by_sym.items() if len(v) > 1}
    dropped = sorted(e for v in ambiguous.values() for e in v)
    axis = sorted(inter - set(dropped))

    info = {
        "key": f"var.ensembl_gene_id",
        "n_3way_intersection": n_inter,
        "n_outside_encoder_vocab": len(outside),
        "outside_encoder_vocab": outside[:20],
        "ambiguous_symbols_dropped": ambiguous,
        "n": len(axis),
        "sha256": _sha256_text("\n".join(axis)),
        "symbols_sha256": _sha256_text("\n".join(sym_of.get(e, e) for e in axis)),
        "genes_dropped_per_file": {
            f["name"]: int(f["n_vars"] - len(axis)) for f in files
        },
        "rejected_keys": {
            "var.gene_name": "only a handful are in the encoder vocabulary, and HSPA14 is "
                             "duplicated in all three files",
            "var.index": "gene symbols in the HepG2 file, positional integer strings in the "
                         "other two -- any intersection is a coincidence, not a gene match",
        },
    }
    return axis, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=[
        "data/eval/pert/nadig_2024_hepg2.h5ad",
        "data/eval/pert/nadig_2024_jurkat.h5ad",
        "data/eval/pert/replogle_2022_rpe1_e.h5ad",
    ])
    ap.add_argument("--holdout-file", default="nadig_2024_hepg2.h5ad",
                    help="the file whose perturbations are split into train/val/test")
    ap.add_argument("--splits-source", default="data/benchmarks/rhj_fewshot/REQUESTED.json",
                    help="JSON with {'val': [...], 'test': [...]} as originally specified")
    ap.add_argument("--encoder-checkpoint",
                    default="checkpoints/release/occam_small_encoder_decoder.pth")
    ap.add_argument("--master-gene-set", default="data/graphs/master_gene_set_sorted.csv")
    ap.add_argument("--gene-col", default="ensembl_gene_id")
    ap.add_argument("--pert-col", default="gene_name")
    ap.add_argument("--control-label", default="non-targeting")
    ap.add_argument("--out-dir", default="data/benchmarks/rhj_fewshot")
    ap.add_argument("--benchmark-id", default=BENCHMARK_ID)
    # The declared counts are overridable ONLY so the smoke corpus can go through this same
    # builder. For the real benchmark they stay at the module constants, which is what makes
    # a silent redefinition impossible.
    ap.add_argument("--expect-val", type=int, default=None)
    ap.add_argument("--expect-test", type=int, default=None)
    ap.add_argument("--expect-train", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(REPO, p)

    sources = [_abs(p) for p in args.sources]
    out_dir = _abs(args.out_dir)
    for p in sources + [_abs(args.encoder_checkpoint), _abs(args.master_gene_set),
                        _abs(args.splits_source)]:
        if not os.path.exists(p):
            ap.error(f"missing input: {p}")

    print(f"[{BENCHMARK_ID}] builder {BUILDER_VERSION}\n")

    # ---- pass 1: metadata, no X -----------------------------------------
    print("[1] reading metadata")
    files = [read_file(p, args.gene_col, args.pert_col, args.control_label) for p in sources]
    for f in files:
        print(f"  {f['name']:32s} {f['cell_line']:8s} {f['n_obs']:>8,} cells  "
              f"{f['n_vars']:>6,} genes  {len(f['per_pert_cells']):>5,} perts  "
              f"{f['n_controls']:>6,} controls  {f['n_batches']:>3} batches")

    holdout = next((f for f in files if f["name"] == args.holdout_file), None)
    if holdout is None:
        ap.error(f"--holdout-file {args.holdout_file} is not among the sources")

    # ---- pass 2: the gene axis ------------------------------------------
    print("\n[2] gene axis")
    import torch
    ck = torch.load(_abs(args.encoder_checkpoint), map_location="cpu", weights_only=False)
    encoder_genes = [str(g) for g in ck["gene_names"]]
    target_sum = float(ck.get("target_sum", 0.0))
    enc_sha = _sha256(_abs(args.encoder_checkpoint))
    del ck

    axis, axis_info = build_axis(files, encoder_genes)
    print(f"  3-way intersection      {axis_info['n_3way_intersection']:,}")
    print(f"  outside encoder vocab   {axis_info['n_outside_encoder_vocab']:,}")
    print(f"  ambiguous symbols       {sorted(axis_info['ambiguous_symbols_dropped'])}")
    print(f"  FROZEN AXIS             {len(axis):,} genes  sha {axis_info['sha256'][:16]}")
    for name, n in axis_info["genes_dropped_per_file"].items():
        print(f"    {name:32s} drops {n:,} of its own genes")

    # ---- pass 3: eligibility against the perturbation graph -------------
    print("\n[3] perturbation eligibility")
    from occam.vcell.graph import load_pert2id
    from occam.vcell.dataset import resolve_pert_indices

    pert2id, _ = load_pert2id(_abs(args.master_gene_set))
    shared = set(files[0]["per_pert_cells"])
    for f in files[1:]:
        shared &= set(f["per_pert_cells"])

    eligible, unresolvable = [], []
    for p in sorted(shared):
        (eligible if resolve_pert_indices(p, pert2id) is not None
         else unresolvable).append(p)
    print(f"  present in all three files  {len(shared):,}")
    print(f"  resolvable in the graph     {len(eligible):,}")
    print(f"  UNRESOLVABLE               {len(unresolvable)}: {unresolvable}")

    # ---- pass 4: the split ---------------------------------------------
    print("\n[4] split")
    with open(_abs(args.splits_source)) as fh:
        requested = json.load(fh)
    req_val, req_test = list(requested["val"]), list(requested["test"])
    if set(req_val) & set(req_test):
        raise ValueError("requested val and test overlap")

    unres = set(unresolvable)
    val = [p for p in req_val if p not in unres]
    test = [p for p in req_test if p not in unres]
    dropped_val = sorted(set(req_val) & unres)
    dropped_test = sorted(set(req_test) & unres)

    missing = sorted((set(req_val) | set(req_test)) - set(holdout["per_pert_cells"]))
    if missing:
        raise ValueError(f"{len(missing)} requested held-out symbols are absent from "
                         f"{holdout['name']}: {missing[:10]}")

    held_out = set(val) | set(test)
    hepg2_train = sorted(set(holdout["per_pert_cells"]) & set(eligible) - held_out)

    print(f"  val    {len(req_val):>4} requested -> {len(val):>4}  dropped {dropped_val}")
    print(f"  test   {len(req_test):>4} requested -> {len(test):>4}  dropped {dropped_test}")
    print(f"  HepG2 train                     {len(hepg2_train):>4}")
    print(f"  excluded from HepG2 training    {len(held_out):>4}")

    exp_val = EXPECT_VAL if args.expect_val is None else args.expect_val
    exp_test = EXPECT_TEST if args.expect_test is None else args.expect_test
    exp_train = EXPECT_HEPG2_TRAIN if args.expect_train is None else args.expect_train
    for label, got, want in (("val", len(val), exp_val),
                             ("test", len(test), exp_test),
                             ("hepg2 train", len(hepg2_train), exp_train)):
        if got != want:
            raise ValueError(
                f"{label}: computed {got}, this builder declares {want}. Either the data or "
                f"the graph vocabulary changed. That may be fine, but it redefines the "
                f"benchmark, so edit EXPECT_* deliberately rather than letting the numbers "
                f"drift.")

    # every held-out symbol must survive in the other two contexts, or the exclusion is a
    # global one in disguise
    for f in files:
        if f["name"] == holdout["name"]:
            continue
        absent = sorted(held_out - set(f["per_pert_cells"]))
        if absent:
            raise ValueError(f"{f['name']}: {len(absent)} held-out symbols absent here "
                             f"({absent[:5]}) -- they could not train from this context")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # ---- pass 5: write --------------------------------------------------
    print("\n[5] writing")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "files"), exist_ok=True)
    sym_of = files[0]["ens_to_sym"]

    with open(os.path.join(out_dir, "gene_axis.csv"), "w") as fh:
        fh.write("ensembl_gene_id,gene_name\n")
        for e in axis:
            fh.write(f"{e},{sym_of.get(e, e)}\n")

    splits = {
        "train": sorted(eligible),          # the GLOBAL name set the streaming path filters on
        "val": sorted(val),
        "test_unseen_pert": sorted(test),
        "test_unseen_cell": [],
        "holdout_cell_line": None,
        "cell_lines_train": sorted(f["cell_line"] for f in files),
        "cell_lines_test": [],
        "seed": 42,
        "train_fraction": 1.0, "val_fraction": 0.0, "test_fraction": 0.0,
        "all_perts_train": False,
        "benchmark_id": args.benchmark_id,
        "_note": (
            "`train` OVERLAPS `val` and `test_unseen_pert` ON PURPOSE. It is the global "
            "perturbation-name set the streaming datasets filter on, and every held-out "
            "symbol legitimately trains from Jurkat and RPE1. exclusions.json is what makes "
            "the held-out CELLS held out, per (file, perturbation)."),
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as fh:
        json.dump(splits, fh, indent=2, sort_keys=True)

    exclusions = {
        "schema": 1,
        "benchmark_id": args.benchmark_id,
        "semantics": ("per (FILE, perturbation); TRAINING only. Never applied by the eval "
                      "builders -- reading exactly these rows back is what makes them "
                      "val/test."),
        "by_file": {holdout["name"]: sorted(held_out)},
        "eval_cell_lines": {"val": [holdout["cell_line"]],
                            "test_unseen_pert": [holdout["cell_line"]]},
        "counts": {holdout["name"]: {
            "perts": len(held_out),
            "cells": int(sum(holdout["per_pert_cells"].get(p, 0) for p in held_out)),
        }},
    }
    with open(os.path.join(out_dir, "exclusions.json"), "w") as fh:
        json.dump(exclusions, fh, indent=2, sort_keys=True)

    split_of = {p: "train" for p in eligible}
    split_of.update({p: "val" for p in val})
    split_of.update({p: "test" for p in test})
    with open(os.path.join(out_dir, "perturbations.csv"), "w") as fh:
        fh.write("gene_name,split,eligible,pert2id," +
                 ",".join(f"n_cells_{f['cell_line']}" for f in files) + "\n")
        for p in sorted(shared):
            ids = resolve_pert_indices(p, pert2id)
            fh.write(f"{p},{split_of.get(p, 'ineligible')},{int(ids is not None)},"
                     f"{ids[0] if ids else -1}," +
                     ",".join(str(f['per_pert_cells'].get(p, 0)) for f in files) + "\n")

    for f in files:
        rec = {k: v for k, v in f.items()
               if k not in ("ensembl", "symbols", "ens_to_sym")}
        rec["n_genes_on_axis"] = len(axis)
        rec["n_genes_dropped"] = int(f["n_vars"] - len(axis))
        with open(os.path.join(out_dir, "files", f["name"].replace(".h5ad", ".json")),
                  "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)

    benchmark = {
        "schema": 1,
        "benchmark_id": args.benchmark_id,
        "builder_version": BUILDER_VERSION,
        "built": date.today().isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "encoder_checkpoint": os.path.relpath(_abs(args.encoder_checkpoint), REPO),
        "encoder_sha256": enc_sha,
        "encoder_n_genes": len(encoder_genes),
        "encoder_target_sum": target_sum,
        "master_gene_set": os.path.relpath(_abs(args.master_gene_set), REPO),
        "master_gene_set_sha256": _sha256(_abs(args.master_gene_set)),
        "n_graph_nodes": len(pert2id),
        "gene_axis": axis_info,
        "sources": {f["name"]: {"sha256": f["sha256"], "bytes": f["bytes"],
                                "n_obs": f["n_obs"], "n_vars": f["n_vars"],
                                "cell_line": f["cell_line"]} for f in files},
        "counts": {
            "perts_in_all_three": len(shared),
            "eligible": len(eligible),
            "unresolvable": unresolvable,
            "val": len(val), "test": len(test),
            "hepg2_train": len(hepg2_train),
            "held_out_of_hepg2": len(held_out),
            "dropped_from_val": dropped_val,
            "dropped_from_test": dropped_test,
        },
        "artifact_sha256": {},
    }
    for rel in ("gene_axis.csv", "splits.json", "exclusions.json", "perturbations.csv"):
        benchmark["artifact_sha256"][rel] = _sha256(os.path.join(out_dir, rel))
    bench_path = os.path.join(out_dir, "benchmark.json")
    with open(bench_path, "w") as fh:
        json.dump(benchmark, fh, indent=2, sort_keys=True)
    benchmark_sha = _sha256(bench_path)
    print(f"  benchmark.json  sha {benchmark_sha[:16]}")

    # ---- pass 6: re-read our own output and check it --------------------
    print("\n[6] self-verification")
    with open(os.path.join(out_dir, "splits.json")) as fh:
        sp = json.load(fh)
    with open(os.path.join(out_dir, "exclusions.json")) as fh:
        ex = json.load(fh)
    axis_back = [l.split(",")[0] for l in
                 open(os.path.join(out_dir, "gene_axis.csv")).read().splitlines()[1:]]

    checks = [
        ("gene_axis re-hashes", _sha256_text("\n".join(axis_back)) == axis_info["sha256"]),
        ("axis length", len(axis_back) == len(axis)),
        ("val size", len(sp["val"]) == exp_val),
        ("test size", len(sp["test_unseen_pert"]) == exp_test),
        ("val and test disjoint", not set(sp["val"]) & set(sp["test_unseen_pert"])),
        ("exclusion == val + test",
         set(ex["by_file"][holdout["name"]]) == set(sp["val"]) | set(sp["test_unseen_pert"])),
        ("every excluded symbol is in `train`",
         set(ex["by_file"][holdout["name"]]) <= set(sp["train"])),
        ("every excluded symbol trains elsewhere",
         all(set(ex["by_file"][holdout["name"]]) <= set(f["per_pert_cells"])
             for f in files if f["name"] != holdout["name"])),
        ("every axis gene is in the encoder vocab", set(axis_back) <= set(encoder_genes)),
    ]
    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if bad:
        print(f"\nFAILED: {bad}", file=sys.stderr)
        return 1

    print(f"\nOK  {out_dir}")
    print(f"    axis {len(axis):,} genes · val {len(val)} · test {len(test)} · "
          f"HepG2 train {len(hepg2_train)} · excluded {len(held_out)}")
    print(f"    benchmark_sha256 {benchmark_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
