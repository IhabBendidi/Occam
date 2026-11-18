#!/usr/bin/env python
"""Refuse to publish a dataset until it is actually publishable.

    python scripts/check_upload_gate.py                 # every release
    python scripts/check_upload_gate.py --release eval  # just one

Two releases are prepared in this repository:

    diverserna   the pretraining corpus (data/diverserna/)
    eval         the evaluation bundle  (data/eval/)

Exits 0 only when every condition in data/diverserna/UPLOAD_GATE.md holds.

Nothing in this repository uploads anything -- there is no publish script and no
call to the Hub API. This is preparation only. When an upload script is
eventually written it should call this first and refuse on non-zero.

This gate is expected to FAIL today. Seven of nine blocks are unattributed and
the two that are identified carry redistribution conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "diverserna")
BANNED = re.compile(r"ihabnet|scmae", re.IGNORECASE)
PLACEHOLDERS = ("TBD", "TODO", "UNRESOLVED", "XXX", "FIXME")


class Gate:
    def __init__(self) -> None:
        self.failures = []
        self.passed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failures.append((label, detail))
            print(f"  FAIL  {label}")
            for line in (detail or "").splitlines():
                print(f"          {line}")
        return ok


def check_eval_bundle(data: str) -> Gate:
    """Gate the evaluation bundle.

    Lighter than the corpus gate -- there is no block structure to attribute --
    but the licensing condition is the same and just as unmet: these are
    third-party datasets redistributed for convenience, each with its own terms.
    """
    print("Evaluation bundle\n")
    g = Gate()

    manifest_path = os.path.join(data, "manifest.json")
    if not g.check(os.path.isfile(manifest_path), "export manifest present",
                   f"missing {manifest_path}\nRun: python scripts/export_eval_data.py"):
        return g
    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)
    ds = man.get("datasets", [])

    g.check(not man.get("partial"), "export is complete",
            "manifest records partial=true; it was written with --limit-rows")

    missing = [d["name"] for d in ds
               if not os.path.isfile(os.path.join(REPO, d["file"]))]
    g.check(not missing, "every exported file present", f"missing: {missing}")

    stale = []
    for d in ds:
        path = os.path.join(REPO, d["file"])
        if os.path.isfile(path) and d.get("sha256"):
            import hashlib
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 24), b""):
                    h.update(block)
            if h.hexdigest() != d["sha256"]:
                stale.append(d["name"])
    g.check(not stale, "every file matches its recorded checksum",
            f"changed since export: {stale}")

    g.check(all(d.get("obs_columns") and d.get("var_columns") for d in ds),
            "every dataset records its columns",
            "manifest entries are missing obs_columns/var_columns")

    dropped_only_obsm = all(set(d.get("dropped", {})) <= {"obsm"} for d in ds)
    g.check(dropped_only_obsm, "only obsm was dropped",
            "a dataset records something other than obsm as dropped")

    card = os.path.join(data, "DATASET_CARD.md")
    if g.check(os.path.isfile(card), "dataset card present",
               f"missing {card}\nRun: python scripts/write_eval_card.py"):
        with open(card, encoding="utf-8") as fh:
            text = fh.read()
        head, _, _ = text.partition("## Provenance and licensing")
        offenders = sorted({p for p in PLACEHOLDERS if p in head})
        g.check(not offenders, "no placeholders outside the licensing section",
                f"found {offenders}")
        g.check("TBD" not in text.split("| licence |", 1)[-1][:4000]
                if "| licence |" in text else False,
                "every dataset has a licence and citation",
                "the licensing table still reads TBD. These are third-party "
                "datasets;\neach needs its upstream source, licence and citation "
                "resolved before\nany of them can be redistributed.")

    review = os.path.join(data, "LICENSE_REVIEW.md")
    if os.path.isfile(review):
        with open(review, encoding="utf-8") as fh:
            text = fh.read()
        ok = bool(re.search(r"^\s*approver\s*:\s*\S+", text, re.IGNORECASE | re.MULTILINE)) \
            and bool(re.search(r"^\s*date\s*:\s*\d{4}-\d{2}-\d{2}", text,
                               re.IGNORECASE | re.MULTILINE))
        g.check(ok, "licence review signed",
                "LICENSE_REVIEW.md needs 'Approver: <name>' and 'Date: YYYY-MM-DD'")
    else:
        g.check(False, "licence review signed",
                f"missing {review}\nA named human must sign off before publication.")
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--release", choices=["diverserna", "eval", "all"], default="all")
    args = ap.parse_args()
    data = args.data_dir

    gates = {}
    if args.release in ("eval", "all"):
        gates["eval"] = check_eval_bundle(os.path.join(REPO, "data", "eval"))
        print()
    if args.release == "eval":
        return _report(gates)

    print("DiverseRNA upload gate\n")
    g = Gate()

    # 1 + 2: attribution and licensing
    status_path = os.path.join(data, "sources", "status.json")
    if not g.check(os.path.isfile(status_path), "source status present",
                   f"missing {status_path}\nRun: python scripts/identify_sources.py"):
        status = {}
    else:
        with open(status_path, encoding="utf-8") as fh:
            status = json.load(fh)

    waivers = {}
    waiver_path = os.path.join(data, "sources", "waivers.json")
    if os.path.isfile(waiver_path):
        with open(waiver_path, encoding="utf-8") as fh:
            for w in json.load(fh):
                if all(w.get(k) for k in ("block", "waived_by", "date", "rationale")):
                    waivers[str(w["block"])] = w

    if status:
        unresolved = [k for k, v in status.items()
                      if v.get("status") != "CONFIRMED" and k not in waivers]
        g.check(not unresolved, "every block confirmed or waived",
                f"{len(unresolved)} block(s) unattributed: {sorted(unresolved, key=int)}\n"
                f"See data/diverserna/UPLOAD_GATE.md for how to resolve them.")

        unlicensed = [k for k, v in status.items()
                      if not (v.get("license") or waivers.get(k, {}).get("license"))]
        g.check(not unlicensed, "every block has a licence",
                f"{len(unlicensed)} block(s) with no licence: "
                f"{sorted(unlicensed, key=int)}\n"
                f"GTEx and TCGA both carry redistribution conditions.")

        uncited = [k for k, v in status.items() if not v.get("citation")]
        g.check(not uncited, "every block has a citation",
                f"{len(uncited)} block(s) with no citation: {sorted(uncited, key=int)}")

    # 3: shard integrity
    shards_path = os.path.join(data, "shards.json")
    if os.path.isfile(shards_path):
        with open(shards_path, encoding="utf-8") as fh:
            shards = json.load(fh)
        missing = [s["file"] for s in shards.get("shards", [])
                   if not os.path.isfile(os.path.join(data, s["file"]))]
        g.check(not missing, "every shard present", f"missing: {missing[:5]}")
        totals = shards.get("totals", {})
        g.check(bool(totals.get("nnz")) and bool(totals.get("sum")),
                "shard totals recorded (nnz and sum vs the source)",
                "shards.json carries no totals; the export cannot be verified "
                "against the source")
    else:
        g.check(False, "shards exported",
                f"missing {shards_path}\nRun: python scripts/export_dataset_hf.py")

    # 4: no placeholders outside the designated section
    card = os.path.join(data, "DATASET_CARD.md")
    if g.check(os.path.isfile(card), "dataset card present", f"missing {card}"):
        with open(card, encoding="utf-8") as fh:
            text = fh.read()
        head, _, _ = text.partition("## Known issues")
        offenders = sorted({p for p in PLACEHOLDERS if p in head})
        g.check(not offenders, "no placeholders outside '## Known issues'",
                f"found {offenders} in the body of the card")

    # 5: a named human
    review = os.path.join(data, "LICENSE_REVIEW.md")
    if os.path.isfile(review):
        with open(review, encoding="utf-8") as fh:
            text = fh.read()
        has_approver = bool(re.search(r"^\s*approver\s*:\s*\S+", text,
                                      re.IGNORECASE | re.MULTILINE))
        has_date = bool(re.search(r"^\s*date\s*:\s*\d{4}-\d{2}-\d{2}", text,
                                  re.IGNORECASE | re.MULTILINE))
        g.check(has_approver and has_date, "licence review signed",
                "LICENSE_REVIEW.md needs 'Approver: <name>' and 'Date: YYYY-MM-DD'")
    else:
        g.check(False, "licence review signed",
                f"missing {review}\nA named human must sign off before publication.")

    # 6: banned tokens
    offenders = []
    for dirpath, dirnames, filenames in os.walk(data):
        dirnames[:] = [d for d in dirnames if d != "data"]   # skip the shards
        for fn in filenames:
            if not fn.endswith((".md", ".json", ".yaml", ".yml", ".txt")):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if BANNED.search(line):
                        offenders.append(f"{os.path.relpath(p, REPO)}:{i}")
    g.check(not offenders, "no legacy project names in the release",
            f"{len(offenders)} line(s): {offenders[:5]}")

    gates["diverserna"] = g
    return _report(gates)


def _report(gates) -> int:
    print()
    closed = {k: v for k, v in gates.items() if v.failures}
    for name, g in gates.items():
        state = f"CLOSED — {len(g.failures)} unmet, {g.passed} met" if g.failures \
            else f"OPEN — all {g.passed} conditions met"
        print(f"  {name:12s} GATE {state}")
    if closed:
        print("\nThis is the expected state until provenance and licensing are resolved.")
        print("Nothing in this repository can upload anything: there is no publish")
        print("script and no call to the Hub API. See data/diverserna/UPLOAD_GATE.md.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
