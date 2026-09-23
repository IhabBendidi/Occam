#!/usr/bin/env python
"""Move stale checkpoints, results, benchmarks and docs OUT of the live tree into archive/.

WHY (user, 2026-09-23): stale material must be completely separate so that nobody -- human or
agent -- picks up a withdrawn number, a superseded panel or a pre-recipe checkpoint by mistake.

WHERE: a top-level `archive/` (same filesystem, so every move is a rename, nothing is copied or
deleted). It is OUTSIDE runs/ and data/ on purpose: ten scripts glob `runs/**`, and an archive
under runs/ would be picked up by all of them. Every move is appended to archive/MANIFEST.tsv
(old path, new path, reason), so any move can be undone with `mv new old`.

WHAT STAYS LIVE (the keep rule)
  rhj_fewshot   published Y06; the bases Q02, M06, M03; arms from the CURRENT recipe on
                (Q02 base + 150 ep / eta_min 3e-4, adopted 2026-09-22): Y*, AA*, RB*, S*_Y06
  norman_combo  published NQ02; bases NM03, NM06; the recipe sweep NA..NF (and later N[A-Z])
  competitors   the headline variants (txpert_single, state, cellflow, perturbdiff_genept),
                the perturbdiff onehot arm, every *__norman_combo run, and the checkpoint
                stores they read
  benchmarks    rhj_fewshot, norman_combo, h1_seen (80), k562_seen, k562_seen_h1set, shared

WHAT IS ARCHIVED -- see RULES below, each with its reason.

USAGE   python scripts/archive_stale.py            # dry run: prints every move
        python scripts/archive_stale.py --apply
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(REPO, "archive")

RHJ_KEEP_EXACT = {"Q02", "M06", "M03"}
RHJ_KEEP_RE = re.compile(r"^(Y\d+|AA\d+|RB\d+|S\d+_Y06)$")
NORMAN_STALE = {"NQ02s", "NM03s", "NM06s"}          # the 30-epoch, pre-recipe variants


def arm_of(name: str, prefix: str) -> str:
    return name[len(prefix):].split("__", 1)[0]


def rules():
    """(path, reason) for every item to archive. Paths are repo-relative."""
    out = []
    add = lambda p, why: out.append((p, why))

    # --- rhj_fewshot training runs --------------------------------------------------
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "rhj_fewshot_*"))):
        name = os.path.basename(d)
        arm = arm_of(name, "rhj_fewshot_")
        real = os.path.isdir(os.path.join(d, "vcell"))
        if not real:
            add(f"runs/{name}", "marker-only dir (slurm_job.json, no vcell/): not a run")
        elif arm in RHJ_KEEP_EXACT or RHJ_KEEP_RE.match(arm):
            continue
        else:
            add(f"runs/{name}", "pre-recipe rhj_fewshot sweep checkpoint (before the "
                                "2026-09-22 Q02/150ep/3e-4 default)")
    # --- their split-half score dirs --------------------------------------------------
    keep_roots = {"eval", "reference", "probe", "_de_real_shared"}
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "rhj_splithalf", "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name in keep_roots or name.startswith("reference_s"):
            continue
        if name.startswith(("CKSWEEP_", "_DEFECTIVE_")):
            add(f"runs/rhj_splithalf/{name}",
                "checkpoint-sweep scores of an archived arm, or an export recorded as DEFECTIVE "
                "(STATE mixed-sentence export)")
            continue
        if name.startswith("compet_"):
            if re.match(r"^compet_(txpert|txpert_mean|txpert_ep20|_gate_ctrlcell)_s\d+$", name):
                add(f"runs/rhj_splithalf/{name}",
                    "competitor score WITHDRAWN (txpert 0.3077: mean-control training, "
                    "single-cell inference, missing control token) or A/B diagnostic / "
                    "infrastructure gate, not a method")
            continue
        arm = name.split("__", 1)[0]
        if "__" in name and not (arm in RHJ_KEEP_EXACT or RHJ_KEEP_RE.match(arm)):
            add(f"runs/rhj_splithalf/{name}", "scores of an archived pre-recipe rhj_fewshot arm")
    # --- norman_combo -----------------------------------------------------------------
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "norman_combo_*"))):
        name = os.path.basename(d)
        if arm_of(name, "norman_combo_") in NORMAN_STALE:
            add(f"runs/{name}", "30-epoch pre-recipe Norman base (superseded by NQ02/NM03/NM06)")
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "norman_combo", "eval", "*"))):
        name = os.path.basename(d)
        if name.split("__", 1)[0] in NORMAN_STALE:
            add(f"runs/norman_combo/eval/{name}", "scores of an archived 30-epoch Norman base")
    # --- misc runs ------------------------------------------------------------------------
    for pat, why in [
        ("runs/rhj_adjmap_s*", "marker-only dir written by an old launcher: looks like a run, is not"),
        ("runs/rhj_loss_*", "marker-only dir written by an old launcher: looks like a run, is not"),
        ("runs/aborted_3h_limit", "runs killed at a 3 h walltime; incomplete"),
        ("runs/_hung_NF32_*", "NF32 run that hung at epoch 92; relaunched"),
        ("runs/rhj_smoke_wiretest_*", "smoke test"),
        ("runs/h1_seen_99", "scores on the SUPERSEDED 99-perturbation h1_seen panel"),
    ]:
        for d in sorted(glob.glob(os.path.join(REPO, pat))):
            add(os.path.relpath(d, REPO), why)
    # --- competitors ------------------------------------------------------------------------
    C = "runs/competitors"
    for pat, why in [
        (f"{C}/txpert_mean", "A/B diagnostic: TxPert on (cell_type, batch) MEAN controls"),
        (f"{C}/txpert_ep20", "A/B diagnostic: TxPert epoch-20 checkpoint"),
        (f"{C}/_gate_ctrlcell", "infrastructure gate, not a method"),
        (f"{C}/cellflow_preflight", "CellFlow preflight runs"),
        (f"{C}/*__h1_seen_99", "exports on the SUPERSEDED 99-perturbation h1_seen panel"),
        (f"{C}/txpert/s42", "WITHDRAWN TxPert exports (0.3077)"),
        (f"{C}/txpert/ckpt/occam_s42", "WITHDRAWN TxPert checkpoint (mean-control training)"),
        (f"{C}/txpert/ckpt/occam_s42_mean", "TxPert mean-control diagnostic checkpoint"),
        (f"{C}/txpert/ckpt/_killed_by_pdex_hang_2026-09-23", "partial runs killed by the pdex hang"),
        (f"{C}/txpert/ckpt_frozen", "TxPert epoch-19 diagnostic checkpoint"),
    ]:
        for d in sorted(glob.glob(os.path.join(REPO, pat))):
            add(os.path.relpath(d, REPO), why)
    # --- benchmarks -------------------------------------------------------------------------
    if os.path.isdir(os.path.join(REPO, "data/benchmarks/h1_seen_99")):
        add("data/benchmarks/h1_seen_99", "SUPERSEDED 99-perturbation h1_seen panel (now 80)")
    # --- docs and the scripts that only regenerate them ---------------------------------------
    for p, why in [
        ("docs/rhj_fewshot_results.md", "27-arm FiLM/GNN validation grid (pre split-half)"),
        ("docs/rhj_all_results.md", "table of every measured arm, almost all now archived; "
                                    "live arms are in docs/rhj_scoreboard.md"),
        ("docs/rhj_adjmap_eval.md", "adjacency-map sweep, calibrated-era evals"),
        ("docs/rhj_adjmap_tables.md", "adjacency-map sweep, calibrated-era evals"),
        ("scripts/collect_adjmap_eval.py", "only regenerates the adjmap docs"),
    ]:
        if os.path.exists(os.path.join(REPO, p)):
            add(p, why)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    items = rules()
    by_reason = {}
    for p, why in items:
        by_reason.setdefault(why, []).append(p)
    for why, ps in by_reason.items():
        print(f"\n[{len(ps):4d}] {why}")
        for p in ps[:4]:
            print(f"       {p}")
        if len(ps) > 4:
            print(f"       ... +{len(ps) - 4} more")
    print(f"\n{len(items)} item(s) -> archive/ ({'APPLY' if args.apply else 'DRY RUN'})")
    if not args.apply:
        return 0
    os.makedirs(ARCH, exist_ok=True)
    man = os.path.join(ARCH, "MANIFEST.tsv")
    new = not os.path.exists(man)
    with open(man, "a") as fh:
        if new:
            fh.write("date\told_path\tnew_path\treason\n")
        for p, why in items:
            src, dst = os.path.join(REPO, p), os.path.join(ARCH, p)
            if os.path.exists(dst):
                raise SystemExit(f"refusing to overwrite {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            fh.write(f"{date.today()}\t{p}\tarchive/{p}\t{why}\n")
    print(f"moved; log appended to {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
