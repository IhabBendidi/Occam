#!/usr/bin/env python
"""Join each adjacency-map arm's VALIDATION metric against its CELL-EVAL scores.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This table's question: **does the validation metric used for checkpoint selection predict
the benchmark result at all?** `best_metric` is group-mean `pearson_delta` on 103 HepG2
validation perturbations; the benchmark is cell-eval on 611 held-out perturbations with the
split-half A/B control asymmetry. One GNN arm already showed val 0.3368 against cell-eval
0.0020 (`docs/rhj_splithalf.md` F11), so the two are not interchangeable — the question is
whether they are merely loosely related or actively misleading.

Comparison metrics (decision 2026-09-17): `pearson_delta`, `mae`, `overlap_at_N`,
`discrimination_score_l1`. Reported for the calibrated AND uncalibrated arm, because they
win different metrics of the four (F12).

Prints Spearman rank correlation between the validation metric and each cell-eval metric:

  * |rho| high and positive on pearson_delta -> validation is a usable proxy; selecting on it
    is defensible.
  * rho near zero -> validation is uninformative about the benchmark. Arms must be chosen on
    cell-eval, and the 30-epoch selection inside each run is choosing on noise.
  * rho negative -> validation is ANTI-correlated and actively harmful for selection.

    python scripts/collect_adjmap_eval.py
    python scripts/collect_adjmap_eval.py --out docs/rhj_adjmap_eval.md
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
COMPARISON = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
LOWER_IS_BETTER = ("mae",)
MODEL_ARMS = ("model_nocal", "model_cal")

# 5 splits, all baselines, from docs/rhj_baseline_results.md
BASELINES = {
    "ctrlcell":     (-0.0018, 0.0296, 0.0000, 0.5038),
    "ctrlmean":     (0.0016, 0.0231, 0.0081, 0.5014),
    "meanpert":     (0.1324, 0.0290, 0.0108, 0.5037),
    "transfer":     (0.2323, 0.0299, 0.0351, 0.6404),
    "experimental": (0.3722, 0.0257, 0.1942, 0.7821),
}


def _spearman(xs, ys):
    """Rank correlation, no scipy. Ties get average ranks."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="configs/rhj_adjmap_sweep.tsv")
    ap.add_argument("--eval-root", default="runs/rhj_splithalf")
    ap.add_argument("--prefix", default="adjmap_")
    ap.add_argument("--out", default="docs/rhj_adjmap_eval.md")
    args = ap.parse_args(argv)

    from occam.vcell.panel.celleval_v1 import _read_agg

    tsv, cols = {}, None
    for line in open(os.path.join(REPO, args.grid)):
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if cols is None:
            cols = parts
            continue
        tsv[parts[0]] = dict(zip(cols, parts))

    recs = []
    for arm, row in sorted(tsv.items()):
        run = glob.glob(os.path.join(REPO, "runs", f"rhj_fewshot_{arm}__*"))
        val, epochs = None, None
        if run:
            led = os.path.join(run[0], "ledger.json")
            if os.path.isfile(led):
                st = (json.load(open(led)).get("stages") or {}).get("vcell") or {}
                if st.get("status") == "completed":
                    val = (st.get("metrics") or {}).get("best_metric")
                    epochs = (st.get("metrics") or {}).get("epochs")
        ev = glob.glob(os.path.join(REPO, args.eval_root, f"{args.prefix}{arm}__*"))
        scores = {a: {} for a in MODEL_ARMS}
        nsplits = {a: 0 for a in MODEL_ARMS}
        if ev:
            for a in MODEL_ARMS:
                acc = {}
                for s in range(5):
                    f = os.path.join(ev[0], f"split_{s:02d}", f"celleval_{a}",
                                     "agg_results.csv")
                    if os.path.isfile(f):
                        m = _read_agg(f)
                        if m:
                            nsplits[a] += 1
                            for k in COMPARISON:
                                if isinstance(m.get(k), (int, float)):
                                    acc.setdefault(k, []).append(m[k])
                scores[a] = {k: sum(v) / len(v) for k, v in acc.items()}
        if val is not None or any(scores.values()):
            recs.append({"arm": arm, "row": row, "val": val, "epochs": epochs,
                         "scores": scores, "nsplits": nsplits})

    scored = [r for r in recs if r["scores"]["model_nocal"]]
    print(f"{len(recs)} arm(s) with a validation metric; "
          f"{len(scored)} also cell-eval'd")

    lines = []
    W = lines.append
    W("# `rhj_fewshot` adjacency-map sweep — validation vs cell-eval\n")
    W("**The question:** does the validation metric used for checkpoint selection predict "
      "the benchmark result? One GNN arm showed val 0.3368 against cell-eval 0.0020 "
      "(`docs/rhj_splithalf.md` F11), so they are not interchangeable.\n")
    W("Comparison metrics (decision 2026-09-17): `pearson_delta`, `mae` ↓, `overlap_at_N`, "
      "`discrimination_score_l1` *(chance 0.5)*. Both calibration settings are shown "
      "because they win different metrics of the four (F12).\n")
    W(f"`val` is group-mean `pearson_delta` on the 103 HepG2 **validation** perturbations "
      f"— the checkpoint-selection metric, read at the epoch that scored best. The "
      f"cell-eval columns are the 611 held-out **test** perturbations under the split-half "
      f"protocol.\n")
    W(f"{len(scored)} of {len(recs)} completed arms scored so far.\n")

    W("## Baselines, all 5 splits\n")
    W("| arm | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` |")
    W("|---|---|---|---|---|")
    for name, v in BASELINES.items():
        W(f"| `{name}` | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} |")
    W("")

    if scored:
        W("## Rank correlation: validation vs cell-eval\n")
        W("Spearman rho over the arms scored so far. **Positive and high** = validation is "
          "a usable proxy. **Near zero** = validation is uninformative and arms must be "
          "chosen on cell-eval. **Negative** = validation is actively harmful.\n")
        W("| cell-eval metric | arm | Spearman rho vs val | n |")
        W("|---|---|---|---|")
        for a in MODEL_ARMS:
            sub = [r for r in scored if r["val"] is not None and r["scores"][a]]
            for k in COMPARISON:
                pairs = [(r["val"], r["scores"][a][k]) for r in sub
                         if k in r["scores"][a]]
                if len(pairs) < 3:
                    W(f"| `{k}` | `{a}` | — | {len(pairs)} |")
                    continue
                rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
                note = " *(lower is better, so a NEGATIVE rho is the good sign)*" \
                    if k in LOWER_IS_BETTER else ""
                W(f"| `{k}` | `{a}` | **{rho:+.3f}**{note} | {len(pairs)} |")
        W("")

    for a in MODEL_ARMS:
        sub = sorted((r for r in scored if r["scores"][a]),
                     key=lambda r: -r["scores"][a].get("pearson_delta", -9))
        if not sub:
            continue
        W(f"## Arms by cell-eval `pearson_delta` — `{a}`\n")
        W("| arm | val | `pearson_delta` | `mae` ↓ | `overlap_at_N` | `disc_l1` | splits "
          "| lr | dlr | cos | dpo | film_mode | fh | fmf |")
        W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            sc, row = r["scores"][a], r["row"]
            vv = "—" if r["val"] is None else f"{r['val']:.4f}"
            W(f"| `{r['arm']}` | {vv} | "
              + " | ".join(f"{sc.get(k, float('nan')):.4f}" for k in COMPARISON)
              + f" | {r['nsplits'][a]} | {row['lr']} | {row['decoder_lr']} "
              f"| {row['cosine_loss_weight']} | {row['data_perts_only']} "
              f"| {row['film_mode']} | {row['film_hidden_dim']} "
              f"| {row['film_mean_free']} |")
        W("")

    body = "\n".join(lines) + "\n"
    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(body)
    print(f"[adjmap-eval] wrote {args.out}")
    if scored:
        for a in MODEL_ARMS:
            sub = [r for r in scored if r["val"] is not None and r["scores"][a]
                   and "pearson_delta" in r["scores"][a]]
            if len(sub) >= 3:
                rho = _spearman([r["val"] for r in sub],
                                [r["scores"][a]["pearson_delta"] for r in sub])
                print(f"[adjmap-eval] {a:12s} Spearman(val, cell-eval pearson_delta) = "
                      f"{rho:+.3f}  (n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
