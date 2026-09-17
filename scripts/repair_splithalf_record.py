#!/usr/bin/env python
"""Rewrite a split's `metrics.json` from the cell-eval outputs already on disk.

WHY THIS EXISTS
---------------
Four reference splits completed and scored all five arms, but `run_split` wrote them to
`metrics_probe.json` with `probe: true` -- because `probe` and `score_only` were one flag
until 2026-09-17, and the reference pass legitimately uses `score_only` to skip the
checkpoint-dependent arm. `paired_report.py` correctly refuses probe records, so four
complete splits were unreadable.

The cell-eval results themselves are fine: `celleval_<arm>/agg_results.csv` is on disk for
every scored arm. This re-reads them and rewrites the record with `probe: false`, rather
than paying 20 more cell-eval runs (11 min each) to recompute identical numbers.

It changes ONLY the record. Every metric value is re-parsed from the same
`agg_results.csv` the original run parsed, and the provenance fields are carried across
unchanged, so the result is what `run_split` would have written had the flag been right.

    python scripts/repair_splithalf_record.py --run-dir runs/rhj_splithalf/reference
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from occam.vcell.panel.celleval_v1 import _read_agg

    found = sorted(glob.glob(os.path.join(args.run_dir, "split_*",
                                          "metrics_probe.json")))
    if not found:
        raise SystemExit(f"no metrics_probe.json under {args.run_dir}/split_*/")

    n_written = 0
    for path in found:
        d = os.path.dirname(path)
        with open(path) as fh:
            rec = json.load(fh)
        if not rec.get("probe"):
            print(f"  {os.path.basename(d)}: already probe=false, skipping")
            continue

        rescored, missing = {}, []
        for arm_dir in sorted(glob.glob(os.path.join(d, "celleval_*"))):
            tag = os.path.basename(arm_dir).replace("celleval_", "")
            agg = os.path.join(arm_dir, "agg_results.csv")
            if not os.path.isfile(agg):
                missing.append(tag)
                continue
            m = _read_agg(agg)
            if not m:
                missing.append(tag)
                continue
            # the ctrlmean pearson_delta convention is preserved exactly as run_split has it
            rescored[tag] = m

        if not rescored:
            print(f"  {os.path.basename(d)}: no agg_results.csv found, leaving as-is")
            continue

        # arms that were never scored in this pass keep their status line
        for tag, val in rec.get("arms", {}).items():
            if tag not in rescored and isinstance(val, dict) and "status" in val:
                rescored[tag] = {"status": "not scored in this pass"}

        out = dict(rec)
        out["probe"] = False
        out["arms"] = rescored
        out["repaired_from"] = os.path.basename(path)
        out["repaired_note"] = (
            "metrics re-parsed from the same celleval_<arm>/agg_results.csv the original "
            "run parsed; the record was mis-stamped probe:true because `probe` and "
            "`score_only` were one flag before 2026-09-17")

        target = os.path.join(d, "metrics.json")
        print(f"  {os.path.basename(d)}: {len(rescored)} arm(s) "
              f"({', '.join(sorted(rescored))}) -> {os.path.basename(target)}"
              + (f"  [missing: {missing}]" if missing else ""))
        if not args.dry_run:
            with open(target, "w") as fh:
                json.dump(out, fh, indent=2, sort_keys=True, default=str)
            n_written += 1

    print(f"\n[repair] {'would write' if args.dry_run else 'wrote'} {n_written} "
          f"metrics.json file(s)")
    if not args.dry_run and n_written:
        print(f"[repair] now readable by: python scripts/paired_report.py "
              f"--run-dir {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
