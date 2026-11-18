#!/usr/bin/env python
"""Freeze one arm's complete, reportable result into `reproducibility/<benchmark>/<arm>_s<seed>/`.

WHY THIS EXISTS
    `runs/` is a working directory. Sweeps overwrite it, `best-*.ckpt` is re-selected at every
    validation, and arm directories get pruned. A number in a doc whose weights and
    predictions have been overwritten is not reproducible -- it is folklore. This copies the
    artifacts needed to re-derive a number, and records the provenance needed to interpret it.

WHAT IT FREEZES, and why each piece

    checkpoints/        ONLY the checkpoint the reported number was produced from, named by
                        `--selected`. The others are deliberately NOT kept: a run writes
                        several ("best" under val loss, "last", "final"), they are ~530 MB
                        each, and keeping a checkpoint that was never evaluated invites
                        someone to score it later and report the result as this arm's. One
                        checkpoint, one number.
    artifacts/          the run's own condition-space objects (one-hot maps, var dims, the
                        resolved config). These matter more than they look: our export used to
                        RE-DERIVE the perturbation and batch vocabularies instead of reading
                        them, which is an unpinned invariant -- a rebuild that came out in a
                        different order would point every covariate at the wrong column with
                        no error.
    predictions/        the per-split prediction h5ad -- the ONE artifact that cannot be
                        recovered from anything else here. The matching TARGET h5ad is NOT
                        copied: it is a deterministic function of the frozen panel and the
                        split seed, so it is reproducible, and it doubles the footprint.
    celleval/           the per-split scores as cell-eval wrote them (`agg_results.csv`,
                        `results.csv`, `metrics.json`). `pred_de.csv` is NOT copied: ~95 MB
                        per split and fully derivable from the prediction h5ad.
    MANIFEST.json       sha256 of every file, the selection rule, the seed, the per-split and
                        aggregate scores, the anchors to read them against, and -- the
                        load-bearing field -- `deliberate_deviations`.

⚠️ `deliberate_deviations` IS NOT OPTIONAL. Every arm here was retrained on OUR data, OUR
gene axis and OUR splits, so none of these numbers reproduces the method's published figure,
and comparing them to one is wrong. The manifest has to say so per entry.

USAGE
    python scripts/freeze_for_reproducibility.py --arm state --seed 42 \
        --benchmark rhj_fewshot --run-dir runs/competitors/state/occam_s42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Read these against nothing else. A scaled score with no anchor beside it is unreadable a
# day later.
ANCHORS = {
    "ctrlcell": -0.0018, "ctrlmean": 0.0016, "meanpert": 0.1324,
    "transfer": 0.2323, "experimental": 0.3722, "occam_best": 0.3415,
}
METRICS = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
N_SPLITS = 5


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _rel(path: Path) -> str:
    """`path` relative to the repo when it is inside it, else absolute.

    Not `Path.is_relative_to`: that is Python 3.9+, and the scoring interpreter is 3.8.
    """
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def copy_in(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"bytes": dst.stat().st_size, "sha256": sha256(dst)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--benchmark", default="rhj_fewshot")
    ap.add_argument("--run-dir", required=True, help="the training run directory")
    ap.add_argument("--scored-dir", default="",
                    help="runs/rhj_splithalf/compet_<arm>_s<seed>; derived if omitted")
    ap.add_argument("--selected", default="",
                    help="basename of the checkpoint the reported number came from")
    ap.add_argument("--note", default="", help="one line for the manifest")
    ap.add_argument("--deviation", action="append", default=[], metavar="TEXT",
                    help="one deliberate deviation from the method's published pipeline; "
                         "repeat per deviation. REQUIRED -- a manifest whose "
                         "`deliberate_deviations` is a placeholder is exactly the folklore "
                         "this script exists to prevent, so the freeze refuses without it.")
    a = ap.parse_args(argv)
    if not a.deviation:
        print("REFUSED: pass at least one --deviation. Every arm here was retrained on OUR "
              "data, OUR gene axis and OUR splits, so none of these numbers reproduces the "
              "method's published figure -- the manifest has to say so.", file=sys.stderr)
        return 2

    run = Path(a.run_dir)
    if not run.is_dir():
        raise SystemExit(f"no run directory at {run}")
    scored = Path(a.scored_dir) if a.scored_dir else (
        REPO / "runs" / "rhj_splithalf" / f"compet_{a.arm}_s{a.seed}")
    out = REPO / "reproducibility" / a.benchmark / f"{a.arm}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}

    # ---- the ONE evaluated checkpoint
    if not a.selected:
        raise SystemExit(
            "--selected is required: name the checkpoint the reported number came from. "
            "Freezing every checkpoint a run wrote would preserve weights that were never "
            "evaluated, and someone would eventually score one and report it as this arm's.")
    cands = list((run / "checkpoints").glob(a.selected)) + list(run.glob(a.selected))
    cands = [c for c in cands if c.is_file()]
    if len(cands) != 1:
        raise SystemExit(
            f"--selected {a.selected!r} matched {len(cands)} file(s) under {run}: "
            f"{[c.name for c in cands]}. It must identify exactly one checkpoint.")
    ck = cands[0]
    files[f"checkpoints/{ck.name}"] = copy_in(ck, out / "checkpoints" / ck.name)
    others = sorted(x.name for x in (run / "checkpoints").glob("*.ckpt") if x.name != ck.name)
    print(f"[freeze] selected {ck.name}; NOT freezing {others} (never evaluated)")

    # ---- the run's own condition-space artifacts + resolved config
    # ⚠️ EXCLUDE the selected checkpoint. The `*.pkl` / `*.pt` globs below are meant to catch
    # small condition-space objects (one-hot maps, var dims), but for an arm whose checkpoint
    # IS a .pkl -- CellFlow's `CellFlow.pkl` -- the glob also matched the checkpoint and froze
    # a second 18 GB copy beside the one in checkpoints/. Same bytes, same sha256, twice.
    for pat in ("*.torch", "*.pt", "*.pkl", "config.yaml", "*.txt"):
        for f in sorted(run.glob(pat)):
            if f.is_file() and f.resolve() != ck.resolve():
                files[f"artifacts/{f.name}"] = copy_in(f, out / "artifacts" / f.name)
    for csv in sorted(run.glob("version_*/metrics.csv")):
        files[f"artifacts/{csv.parent.name}_metrics.csv"] = copy_in(
            csv, out / "artifacts" / f"{csv.parent.name}_metrics.csv")

    # ---- predictions + scores, per split
    per_split: dict[str, dict] = {}
    for s in range(N_SPLITS):
        sd = scored / f"split_{s:02d}"
        if not sd.is_dir():
            continue
        for h5 in sorted(sd.glob("*pred*.h5ad")):
            files[f"predictions/split_{s}/{h5.name}"] = copy_in(
                h5, out / "predictions" / f"split_{s}" / h5.name)
        ce = sd / "celleval_model_nocal"
        for name in ("agg_results.csv", "results.csv"):
            if (ce / name).is_file():
                files[f"celleval/split_{s}/{name}"] = copy_in(
                    ce / name, out / "celleval" / f"split_{s}" / name)
        if (sd / "metrics.json").is_file():
            files[f"celleval/split_{s}/metrics.json"] = copy_in(
                sd / "metrics.json", out / "celleval" / f"split_{s}" / "metrics.json")
        if (ce / "agg_results.csv").is_file():
            from occam.vcell.panel.celleval_v1 import _read_agg
            m = _read_agg(str(ce / "agg_results.csv"))
            if m:
                per_split[str(s)] = {k: m[k] for k in METRICS
                                     if isinstance(m.get(k), (int, float))}

    import numpy as np
    agg = ({k: float(np.mean([v[k] for v in per_split.values()])) for k in METRICS}
           if per_split else {})
    sd_ = ({k: float(np.std([v[k] for v in per_split.values()], ddof=1)) for k in METRICS}
           if len(per_split) > 1 else {})

    man = {
        "arm": a.arm, "seed": a.seed, "benchmark": a.benchmark,
        "note": a.note,
        "selected_checkpoint": ck.name,
        "checkpoints_not_frozen": {
            "names": others,
            "why": "written by the same run but NEVER evaluated -- 'best' under a different "
                   "criterion, plus last/final. Excluded so no one can score one later and "
                   "report it as this arm's result.",
        },
        # `Path.is_relative_to` is 3.9+; the scoring interpreter is 3.8
        "source_run_dir": _rel(run),
        "source_scored_dir": _rel(scored),
        "n_splits_scored": len(per_split),
        "test_scores_model_nocal": {"per_split": per_split, "mean": agg, "sd": sd_},
        "anchors_for_reading": ANCHORS,
        "scored_by": "cell-eval 0.6.0 / pdex 0.1.24, uncalibrated (model_nocal), one drawn "
                     "half-B control cell per predicted row",
        "not_copied": {
            "target_h5ad": "a deterministic function of the frozen panel and the split seed; "
                           "reproducible, and it would double the footprint",
            "pred_de.csv": "~95 MB per split, fully derivable from the prediction h5ad",
        },
        "deliberate_deviations": list(a.deviation),
        "files": files,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")

    total = sum(f["bytes"] for f in files.values())
    print(f"[freeze] {_rel(out)}")
    print(f"[freeze]   {len(files)} file(s), {total / 2**30:.2f} GiB")
    print(f"[freeze]   {len(per_split)}/{N_SPLITS} splits scored"
          + (f" · pearson_delta {agg['pearson_delta']:.4f}" if agg else ""))
    print(f"[freeze]   {len(a.deviation)} deliberate deviation(s) recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
