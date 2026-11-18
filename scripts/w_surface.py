#!/usr/bin/env python
"""Emit the F-sched2 W-grid block (ranked arms + the 4 per-metric epochs x eta_min surfaces).

WHY THIS EXISTS
    The W grid is 24 arms x 5 splits x 4 metrics scored by an array job that fills in over
    hours, so the tables in docs/rhj_splithalf.md go stale every time a task lands and were
    being hand-patched row by row -- which is how a 2/5 mean got reported as if it were final.
    This regenerates the whole block from disk, and `--splice` drops it into the doc between
    the two headings that bound it.

    Directions: pearson_delta UP · mae DOWN (it is an ERROR) · overlap_at_N UP · disc_l1 UP.
    Anchors to read the cells against: transfer 0.2323 · experimental 0.3722 · Q02 base
    (30 ep) 0.3783 · E3 (Q02 60 ep, eta_min 1e-5) 0.3933 · W23 (Q02 90 ep, 1e-5) 0.3959.

    ⚠️ `W19` is CONFIGURATION-IDENTICAL to `E3`, and `W13` to the `Q02` base -- when they land
    they are determinism checks, not new arms.

    python scripts/w_surface.py                 # print
    python scripts/w_surface.py --splice        # rewrite the block in docs/rhj_splithalf.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Importable without the package installed, exactly like scripts/scoreboard.py: the repo's
# `src/` goes on the path relative to THIS file, so the script runs from any cwd.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

DOC = "docs/rhj_splithalf.md"
START = "### Every arm scored so far"
END = "### Reading the surface"
M = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
LAB = {"pearson_delta": "`pearson_delta` ↑", "mae": "`mae` ↓ (ERROR)",
       "overlap_at_N": "`overlap_at_N` ↑", "discrimination_score_l1": "`disc_l1` ↑ (chance 0.5)"}
EM = ("1e-7", "1e-6", "1e-5", "1e-4")
DUP = {"W13": "= `Q02` base", "W19": "= `E3`"}


def _mean(x):
    return sum(x) / len(x)


def _sd(x):
    if len(x) < 2:
        return float("nan")
    m = _mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def scores(arm):
    from occam.vcell.panel.celleval_v1 import _read_agg
    d = sorted(glob.glob(f"runs/rhj_splithalf/{arm}__*"))
    if not d:
        return None
    per = {}
    for k in range(5):
        f = os.path.join(d[0], f"split_{k:02d}", "celleval_model_nocal", "agg_results.csv")
        if os.path.isfile(f):
            m = _read_agg(f)
            if m:
                per[k] = m
    return per or None


def grid():
    out, i = {}, 1
    for base in ("M03", "Q02"):
        for ep in (30, 60, 90):
            for em in EM:
                out[(base, ep, em)] = f"W{i:02d}"
                i += 1
    return out


def build(progress=""):
    W = grid()
    P = {k: scores(v) for k, v in W.items()}
    L = [f"### Every arm scored so far, ranked{progress}", ""]
    L += ["| arm | base | epochs | `eta_min` | n | `pearson_delta` ↑ | `mae` ↓ | `ovl@N` ↑ | `disc_l1` ↑ |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = [(_mean([v["pearson_delta"] for v in p.values()]), k, W[k], p) for k, p in P.items() if p]
    for pd_, k, a, p in sorted(rows, reverse=True):
        n = len(p)
        ep = f"**{k[1]}**" if k[1] == 90 else str(k[1])
        note = f" {DUP[a]}" if a in DUP else ""
        # n=1 has no spread: print an em dash, never "nan" -- a nan in a results table reads
        # as a broken metric rather than as "one split so far".
        _s = _sd([v["pearson_delta"] for v in p.values()])
        _spread = "—" if _s != _s else f"± {_s:.4f}"
        L.append(f"| `{a}`{note} | {k[0]} | {ep} | {k[2]} | {n}/5{'' if n == 5 else ' ⚠️'} | "
                 f"**{pd_:.4f}** {_spread} | "
                 f"{_mean([v['mae'] for v in p.values()]):.4f} | "
                 f"{_mean([v['overlap_at_N'] for v in p.values()]):.4f} | "
                 f"{_mean([v['discrimination_score_l1'] for v in p.values()]):.4f} |")
    miss = sorted(W[k] for k, p in P.items() if not p)
    L += ["", (f"**Not yet scored ({len(miss)} of 24):** " + ", ".join(f"`{a}`" for a in miss)
               if miss else "✅ **All 24 arms scored.**")
          + "  ⚠️ Rows below 5/5 are PARTIAL (F-infra1 pdex race); their means may move.", ""]
    L += ["### The full surface, all four metrics (`mae` is an ERROR: LOWER is better)", "",
          "Cell = mean over the splits scored so far, `(n)` beside it. `—` = not yet scored.", ""]
    for m in M:
        L += [f"**{LAB[m]}**", "",
              "| base | epochs | 1e-7 | 1e-6 | 1e-5 | 1e-4 | row spread |", "|---|---|---|---|---|---|---|"]
        for base in ("M03", "Q02"):
            for ep in (30, 60, 90):
                cells, vals = [], []
                for em in EM:
                    p = P[(base, ep, em)]
                    if p is None:
                        cells.append("—")
                    else:
                        v = _mean([x[m] for x in p.values()])
                        vals.append(v)
                        cells.append(f"{v:.4f} ({len(p)})")
                spread = f"{max(vals) - min(vals):.4f}" if len(vals) > 1 else "—"
                L.append(f"| {base} | {'**' + str(ep) + '**' if ep == 90 else ep} | "
                         + " | ".join(cells) + f" | {spread} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splice", action="store_true", help=f"rewrite the block in {DOC}")
    ap.add_argument("--progress", default="", help="e.g. ' (array 18249983 at 71/100)'")
    a = ap.parse_args(argv)
    block = build(a.progress)
    if not a.splice:
        print(block)
        return 0
    s = open(DOC).read()
    i, j = s.index(START), s.index(END)
    open(DOC, "w").write(s[:i] + block + "\n" + s[j:])
    print(f"spliced {len(block.splitlines())} lines into {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
