#!/usr/bin/env python
"""Compare a rebuilt benchmark against the published one -- SEMANTICALLY, not byte-wise.

WHY NOT JUST HASH THE FILE
    Because it can never match, and a check that always fails is worse than no check.
    `benchmark.json` embeds the commit that built it (`git_sha`), the date (`built`) and
    the output path (`panel.dir` / `command`). Two builds of the SAME benchmark on
    different days therefore differ in bytes while being identical in content. The first
    version of the rebuild scripts hashed the file and reported a MISMATCH on a rebuild
    that was in fact bit-identical in every scientific quantity -- found on 2026-09-22 by
    running it. `axis.npz` is worse: it is a zip, so it carries per-entry timestamps.

WHAT IS COMPARED, and why each item is the right one
    gene_axis sha256      the SCORING AXIS. Different axis -> no score is comparable.
    artifact_sha256       the builder's own per-file content digests (gene_axis.csv,
                          splits.json, exclusions.json, ...), which are computed over
                          content and ARE stable across machines.
    counts                how many perturbations are in train / val / test, and which
                          were dropped and why.
    sources[*].sha256     the input .h5ad files. A mismatch here explains every other
                          mismatch, so it is reported first.
    panel.cohort_sha256   the frozen cohort: which perturbations, with how many cells.
    panel.benchmark_sha256  the panel's stamp of the definition it was built against.
    panels / scored_panels  (norman_combo) the membership of every stratum, pair by pair.

WHAT IS DELIBERATELY IGNORED
    git_sha, built, command, panel.dir, numpy, python, builder_version -- provenance
    about the build, not about the benchmark.

USAGE
    python scripts/repro/verify_benchmark.py --rebuilt /tmp/check \\
        --published data/benchmarks/norman_combo

Exit 0 = the rebuild is the published benchmark. Exit 3 = it is not, with every
differing field named.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

IGNORE = {"git_sha", "built", "command", "numpy", "python", "builder_version",
          "encoder_checkpoint", "master_gene_set", "source", "schema"}
# Nested keys ignored inside `panel`, for the same reason.
IGNORE_PANEL = {"dir"}


def _load(d, name):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuilt", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--panel-subdir", default="splithalf")
    args = ap.parse_args()

    a = _load(args.published, "benchmark.json")
    b = _load(args.rebuilt, "benchmark.json")
    if a is None or b is None:
        print(f"benchmark.json missing in "
              f"{args.published if a is None else args.rebuilt}", file=sys.stderr)
        return 1

    bad = []

    # Sources first: a source mismatch explains every downstream one.
    sa, sb = a.get("sources") or {}, b.get("sources") or {}
    if set(sa) != set(sb):
        bad.append(("sources", sorted(set(sa) ^ set(sb)), "different input files"))
    for k in sorted(set(sa) & set(sb)):
        if sa[k].get("sha256") != sb[k].get("sha256"):
            bad.append((f"sources[{k}].sha256", sb[k].get("sha256"),
                        sa[k].get("sha256")))
    if not bad:
        print(f"  sources                     OK ({len(sa)} file(s), sha256 match)")

    for key in sorted((set(a) | set(b)) - IGNORE - {"sources", "panel"}):
        va, vb = a.get(key), b.get(key)
        if key == "gene_axis" and isinstance(va, dict) and isinstance(vb, dict):
            # only the axis CONTENT digest and size are identity; the per-file drop
            # counts are derived from the same inputs and follow.
            for sub in ("sha256", "n_genes", "n_axis_genes"):
                if sub in va or sub in vb:
                    if va.get(sub) != vb.get(sub):
                        bad.append((f"gene_axis.{sub}", vb.get(sub), va.get(sub)))
            continue
        if va != vb:
            bad.append((key, str(vb)[:120], str(va)[:120]))
        elif key in ("artifact_sha256", "counts", "gene_axis_sha256", "cell_line",
                     "benchmark_id", "task", "encoder_sha256", "encoder_target_sum",
                     "n_axis_genes", "axis_scope"):
            print(f"  {key:27} OK")

    pa, pb = a.get("panel") or {}, b.get("panel") or {}
    for key in sorted((set(pa) | set(pb)) - IGNORE_PANEL):
        if pa.get(key) != pb.get(key):
            bad.append((f"panel.{key}", str(pb.get(key))[:120], str(pa.get(key))[:120]))
        else:
            print(f"  panel.{key:21} OK")

    # The strata, pair by pair. This is the part a consumer actually scores on.
    ma = _load(os.path.join(args.published, args.panel_subdir), "splits_manifest.json")
    mb = _load(os.path.join(args.rebuilt, args.panel_subdir), "splits_manifest.json")
    if ma and mb:
        ka, kb = ma.get("panels") or {}, mb.get("panels") or {}
        if set(ka) != set(kb):
            bad.append(("panels", sorted(kb), sorted(ka)))
        for name in sorted(set(ka) & set(kb)):
            if set(map(str, ka[name]["pairs"])) != set(map(str, kb[name]["pairs"])):
                bad.append((f"panels[{name}]", f"{len(kb[name]['pairs'])} pairs",
                            f"{len(ka[name]['pairs'])} pairs"))
            else:
                print(f"  panels[{name:16}] OK ({len(ka[name]['pairs'])} pairs)")
        if ma.get("scored_panels") != mb.get("scored_panels"):
            bad.append(("scored_panels", mb.get("scored_panels"),
                        ma.get("scored_panels")))
        elif ma.get("scored_panels"):
            print(f"  scored_panels               OK "
                  f"({', '.join(ma['scored_panels'])})")
        # ⚠️ `benchmark_sha256` is NOT compared, and this is a real limitation rather
        # than a convenience. It is literally `sha256(benchmark.json)` -- a byte hash of
        # a file that embeds `git_sha`, `built` and `panel.dir` -- so a rebuild can
        # NEVER reproduce it even when every scientific quantity matches, as this run
        # demonstrates. It is reported as provenance below.
        #
        # CONSEQUENCE, stated plainly: `eval_splithalf.assert_checkpoint_matches`
        # refuses when a checkpoint's recorded `benchmark_sha256` differs from the
        # panel's -- correctly, since that guard is what stops a checkpoint being
        # scored on a benchmark it was not trained on. A REBUILT panel therefore
        # cannot be used to score a PUBLISHED checkpoint: verify with the rebuild,
        # then score against the downloaded artifact.
        for key in ("cohort_sha256", "gene_axis_sha256",
                    "panel_identity_sha256", "min_cells", "n_splits", "base_seed",
                    "val_pairs"):
            if key in ma or key in mb:
                if ma.get(key) != mb.get(key):
                    bad.append((f"manifest.{key}", str(mb.get(key))[:100],
                                str(ma.get(key))[:100]))
                else:
                    print(f"  manifest.{key:18} OK")

    if ma and mb and ma.get("benchmark_sha256") != mb.get("benchmark_sha256"):
        print(f"\n  benchmark_sha256 (provenance, NOT compared)")
        print(f"    rebuilt   {mb.get('benchmark_sha256')}")
        print(f"    published {ma.get('benchmark_sha256')}")
        print("    This is sha256(benchmark.json), which embeds git_sha, the build date")
        print("    and the output path, so it CANNOT match across machines or days. It")
        print("    is the quantity eval_splithalf refuses on, so score the published")
        print("    checkpoint against the DOWNLOADED panel, not against this rebuild.")

    if bad:
        print(f"\n{len(bad)} DIFFERENCE(S) -- the rebuild is NOT the published "
              f"benchmark:", file=sys.stderr)
        for field, got, want in bad:
            print(f"  {field}\n    rebuilt   {got}\n    published {want}",
                  file=sys.stderr)
        print("\nCheck the `sources[*].sha256` lines first: a different input .h5ad "
              "explains every other difference. Everything else means the build "
              "parameters differ -- compare the two splits_manifest.json.",
              file=sys.stderr)
        return 3

    print("\nREPRODUCED: every scientific quantity matches the published benchmark.")
    print("(git_sha, built, command and panel.dir differ by construction and are "
          "ignored -- they describe the build, not the benchmark.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
