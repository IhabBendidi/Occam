#!/usr/bin/env python
"""Write `docs/rhj_baseline_results.md`: every baseline, every split, every cell-eval metric.

Baselines only. **No model arm is ever written by this script** -- it filters to the five
model-independent reference arms by name, so a model score cannot appear here by accident.

Regenerate freely; it reads whatever `celleval_<arm>/agg_results.csv` files exist and
reports per-arm coverage, so a partially finished reference run produces a correct partial
document rather than a wrong complete-looking one.

    python scripts/write_baseline_results.py
    python scripts/write_baseline_results.py --run-dir runs/rhj_splithalf/reference
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

# The five model-INDEPENDENT arms, in ladder order. Anything not in this list is not
# written, which is what keeps model scores out of this document.
ARMS = ("ctrlmean", "ctrlcell", "meanpert", "transfer", "experimental")

# ⛔ THE COMPARISON SET, by decision 2026-09-17. Reported first and flagged in the doc;
# every other cell-eval metric still appears, but comparisons are made on these.
COMPARISON = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")

LOWER_IS_BETTER = ("mae", "mse", "mae_delta", "mse_delta", "de_nsig_counts_pred")
CHANCE_IS_HALF = ("discrimination_score_l1", "discrimination_score_l2",
                  "discrimination_score_cosine", "de_direction_match", "roc_auc")

ROLE = {
    "ctrlmean": (
        "Zero-signal anchor — a constant",
        "The mean of all 2,488 half-B control cells, tiled identically across every one of "
        "the 18,575 predicted rows. Every perturbation gets the same vector.",
        "It carries no biology by construction, so whatever it scores is what the metric "
        "reads from the emission format alone. Its within-group variance is **zero**, which "
        "makes the Wilcoxon DE tests over-call massively — it is the upper bracket on "
        "`de_nsig_counts_pred`."),
    "ctrlcell": (
        "Zero-signal anchor — the floor",
        "The drawn half-B control cell itself, unchanged. Row *i*'s prediction is exactly "
        "the control cell that the model would have been fed for row *i*.",
        "**The load-bearing floor.** Its delta is the mean of *m* drawn half-B controls "
        "minus the full half-B control mean, i.e. pure control-sampling noise at *m* cells, "
        "which scales correctly with cell count. Unlike `ctrlmean` it has *natural* "
        "single-cell variance, so it calls **zero** genes significant — the lower bracket "
        "on the DE metrics."),
    "meanpert": (
        "Baseline — \"it learned the average knockdown\"",
        "`ctrl_used + (training grand mean − half-B control mean)`. The grand mean is the "
        "equal-weight-per-perturbation average of the 1,657 HepG2 **training** "
        "perturbations' mean profiles (95,527 cells). One single delta, added to every row "
        "regardless of which perturbation it is.",
        "It has no per-perturbation information at all. Beating it is the minimum bar for "
        "claiming a model learned anything perturbation-specific — and its "
        "`discrimination_score_l2` should sit at chance, which is the check that it is "
        "constructed correctly. Averaged per *perturbation*, not per cell: per-cell would "
        "be dominated by TFAM's 1,213 cells."),
    "transfer": (
        "⭐ Baseline — the non-learned version of the task itself",
        "`ctrl_used + delta[this row's own perturbation]`, where `delta[p]` is the "
        "count-weighted mean of *p*'s (perturbed mean − that file's control mean) over "
        "**Jurkat and RPE1 only** — median 158 source cells per perturbation. HepG2 is "
        "deliberately excluded: it is the held-out context and using it would read the "
        "answer.",
        "This is what the model is asked to do, done by averaging instead of learning: "
        "take the same perturbation's measured effect in two other cell lines and add it to "
        "a HepG2 control cell. **It is the bar.** A model that cannot beat it is not earning "
        "its parameters. On the *validation* side its analogue beat all 24 grid arms by "
        "0.20 (`docs/rhj_fewshot.md` F5)."),
    "experimental": (
        "Reference — replicate agreement, NOT a ceiling",
        "The real half-B **perturbed** cells, as the prediction. Same perturbation, same "
        "cell line, different cells — scored against the half-A reference exactly as any "
        "other arm is.",
        "⚠️ **Not an upper bound.** At median *m* = 22 both this arm and the half-A "
        "reference are noisy pseudobulks, while a low-variance predictor is attenuated on "
        "one side only — so a model can beat it from noise alone on perturbations whose "
        "true effect is small relative to σ/√m. Read it as *\"agreement between two "
        "22-cell replicates\"* and check the count-stratified gap."),
}

METRIC_NOTES = {
    "pearson_delta": "Field convention. Pearson r between (pred − pred-file control mean) "
                     "and (real − real-file control mean), per perturbation, averaged. The "
                     "A/B asymmetry lives here: the target file carries half-A controls, "
                     "the prediction file half-B.",
    "discrimination_score_l2": "1 − rank/n_perts over the 611-perturbation cohort, L2 "
                               "distance. Excludes each perturbation's own gene by SYMBOL.",
    "discrimination_score_l1": "Same, L1 distance.",
    "discrimination_score_cosine": "Same, cosine distance.",
    "overlap_at_N": "Top-N DE gene overlap, N = the number of FDR-significant genes on the "
                    "REAL side. Exactly 0 when the arm calls nothing significant.",
    "overlap_at_50": "Same at fixed N = 50.", "overlap_at_100": "Same at fixed N = 100.",
    "overlap_at_200": "Same at fixed N = 200.", "overlap_at_500": "Same at fixed N = 500.",
    "precision_at_N": "Precision form of `overlap_at_N`.",
    "precision_at_50": "Precision at fixed N = 50.",
    "precision_at_100": "Precision at fixed N = 100.",
    "precision_at_200": "Precision at fixed N = 200.",
    "precision_at_500": "Precision at fixed N = 500.",
    "de_direction_match": "Fraction of DE genes where prediction and truth agree on "
                          "up/down.",
    "de_spearman_sig": "Spearman of effect sizes over significant genes.",
    "de_spearman_lfc_sig": "Spearman of log fold changes over significant genes.",
    "de_sig_genes_recall": "Fraction of the real significant genes the arm also calls "
                           "significant. Trivially raised by calling everything.",
    "de_nsig_counts_real": "Genes FDR-significant on the real side — the reference count.",
    "de_nsig_counts_pred": "Genes FDR-significant on the predicted side. Read against BOTH "
                           "anchors: `ctrlcell` 0 and `ctrlmean` ~3,000 bracket it.",
    "mae": "Control-INDEPENDENT mean absolute error against the real pseudobulk.",
    "mse": "Control-independent squared error.",
    "mae_delta": "Control-DEPENDENT: error on the delta. The headline error metric here.",
    "mse_delta": "Control-dependent squared error on the delta.",
    "pr_auc": "Precision-recall AUC over the DE calls.",
    "roc_auc": "ROC AUC over the DE calls.",
    "clustering_agreement": "Panel-level, not per perturbation. Not re-aggregable.",
    "pearson_edistance": "Panel-level energy-distance correlation. Not re-aggregable.",
}


def _mean(xs):
    return sum(xs) / len(xs)


def _std1(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/rhj_splithalf/reference")
    ap.add_argument("--panel-dir",
                    default="data/benchmarks/rhj_fewshot/splithalf")
    ap.add_argument("--out", default="docs/rhj_baseline_results.md")
    ap.add_argument("--expect-splits", type=int, default=5)
    args = ap.parse_args(argv)

    from occam.vcell.panel.celleval_v1 import _read_agg

    run_dir = os.path.join(REPO, args.run_dir)
    data = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "split_*"))):
        sp = int(os.path.basename(d).split("_")[1])
        for arm in ARMS:
            agg = os.path.join(d, f"celleval_{arm}", "agg_results.csv")
            if os.path.isfile(agg):
                m = _read_agg(agg)
                if m:
                    data[(sp, arm)] = m
    if not data:
        raise SystemExit(f"no agg_results.csv found under {run_dir}/split_*/celleval_*/")

    splits = sorted({s for s, _ in data})
    metrics = sorted({k for v in data.values() for k in v})
    coverage = {a: sorted(s for s, x in data if x == a) for a in ARMS}

    with open(os.path.join(REPO, args.panel_dir, "splits_manifest.json")) as fh:
        man = json.load(fh)

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        W = fh.write
        W("# `rhj_fewshot` split-half benchmark — BASELINE results\n\n")
        W("**Baselines only.** No model score appears in this file: "
          "`scripts/write_baseline_results.py` filters to the five model-independent "
          "reference arms by name. Regenerate with that script.\n\n")

        complete = [a for a in ARMS if len(coverage[a]) == args.expect_splits]
        partial = [a for a in ARMS if 0 < len(coverage[a]) < args.expect_splits]
        if partial:
            W(f"> ⚠️ **PARTIAL.** {len(complete)} of {len(ARMS)} arms have all "
              f"{args.expect_splits} splits. Per-arm coverage is stated in every table; "
              f"`n` is the number of splits behind each mean.\n>\n")
            for a in partial:
                miss = sorted(set(range(args.expect_splits)) - set(coverage[a]))
                W(f"> * `{a}`: splits {coverage[a]} — missing {miss}\n")
            W("\n")

        W("## What this benchmark is\n\n")
        W("**Objective.** Predict a perturbation's effect in a context where that "
          "perturbation was never measured, from a single control cell.\n\n")
        W("**The protocol** (a reviewer's request, implemented verbatim). Within each "
          "perturbation, split its HepG2 cells into balanced halves A and B. Half A is the "
          "**reference**: the scored target file holds half-A perturbed cells plus half-A "
          "controls. Each arm's prediction is written with **half-B controls**. Because "
          "cell-eval's `pearson_delta` uses *each file's own* control pseudobulk, that one "
          "substitution makes the reference delta A-relative and every predicted delta "
          "B-relative — so reference and prediction share no cells.\n\n")
        W("**What is varied:** which cells land in half A and half B — "
          f"{args.expect_splits} independent, batch-stratified splits. **Held fixed:** the "
          f"cohort ({man['n_cohort_perts']} perturbations, `{man['cohort_sha256'][:16]}…`), "
          f"the {man['n_axis_genes']:,}-gene axis, and the per-perturbation row counts. "
          f"`m(p) = floor(n_p/2)` depends only on `n_p`, so every arm in every split emits "
          f"exactly the same rows for the same perturbation.\n\n")

        W("| | value |\n|---|---|\n")
        W(f"| source | `{man['source']}`, sha256 `{man['source_sha256'][:16]}…` |\n")
        W(f"| cohort | **{man['n_cohort_perts']}** of 621 test perturbations "
          f"(`min_cells` {man['min_cells']}) |\n")
        W(f"| half size `m` | median **{man['half_size_strata']['m_median']}**; "
          f"`m≥5` {man['half_size_strata']['m>=5']} · "
          f"`m≥10` {man['half_size_strata']['m>=10']} · "
          f"`m≥25` {man['half_size_strata']['m>=25']} |\n")
        W(f"| rows per arm | {man['splits'][0]['n_pert_per_half']:,} |\n")
        W(f"| controls per half | {man['splits'][0]['n_ctrl_per_half']:,} |\n")
        W(f"| gene axis | {man['n_axis_genes']:,}, sha256 "
          f"`{man['gene_axis_sha256'][:16]}…` |\n")
        W(f"| normalisation | {man['normalisation']} |\n\n")
        W("Scored with ARC cell-eval v1, profile `full`, `--allow-discrete`, "
          "`--batch-size 2000`, 16 threads.\n\n")
        W("`↓` marks metrics where **lower is better**. **Chance is 0.5**, not 0, for "
          "`discrimination_score_*`, `de_direction_match` and `roc_auc`.\n\n")
        W("> ⛔ **The comparison set** (decision 2026-09-17): every comparison between arms "
          "is made on **`pearson_delta`, `mae`, `overlap_at_N` and "
          "`discrimination_score_l1`** — note **l1**, not l2. They are listed first in "
          "every table below. The other 24 metrics are reported for completeness, not for "
          "ranking.\n\n")

        W("---\n\n## The five baselines\n\n")
        for arm in ARMS:
            head, how, why = ROLE[arm]
            W(f"### `{arm}` — {head}\n\n")
            W(f"**What it predicts.** {how}\n\n")
            W(f"**Why it is here.** {why}\n\n")
            if not coverage[arm]:
                W("*Not yet scored.*\n\n")
                continue
            W(f"**Coverage:** splits {coverage[arm]} "
              f"({len(coverage[arm])} of {args.expect_splits}).\n\n")
            W("| metric | " + " | ".join(f"split {s}" for s in splits)
              + " | mean | std | n |\n")
            W("|---|" + "---|" * (len(splits) + 3) + "\n")
            for k in [m for m in COMPARISON if m in metrics] + \
                     [m for m in metrics if m not in COMPARISON]:
                vals, cells = [], []
                for s in splits:
                    v = data.get((s, arm), {}).get(k)
                    if isinstance(v, (int, float)) and v == v:
                        vals.append(v)
                        cells.append(_fmt(v))
                    else:
                        cells.append("—")
                arrow = " ↓" if k in LOWER_IS_BETTER else (
                    " *(chance 0.5)*" if k in CHANCE_IS_HALF else "")
                mu = _fmt(_mean(vals)) if vals else "—"
                sd = _fmt(_std1(vals)) if len(vals) > 1 else "—"
                W(f"| `{k}`{arrow} | " + " | ".join(cells)
                  + f" | **{mu}** | {sd} | {len(vals)} |\n")
            W("\n")

        W("---\n\n## All five baselines side by side\n\n")
        W("Mean ± std across the splits each arm has, `ddof=1`. `n` differs per arm where "
          "coverage does — see above.\n\n")
        for k in metrics:
            arrow = " ↓" if k in LOWER_IS_BETTER else (
                " *(chance 0.5)*" if k in CHANCE_IS_HALF else "")
            W(f"### `{k}`{arrow}\n\n")
            note = METRIC_NOTES.get(k)
            if note:
                W(f"{note}\n\n")
            W("| arm | " + " | ".join(f"split {s}" for s in splits)
              + " | mean ± std | n |\n")
            W("|---|" + "---|" * (len(splits) + 2) + "\n")
            for arm in ARMS:
                vals, cells = [], []
                for s in splits:
                    v = data.get((s, arm), {}).get(k)
                    if isinstance(v, (int, float)) and v == v:
                        vals.append(v)
                        cells.append(_fmt(v))
                    else:
                        cells.append("—")
                if not vals:
                    W(f"| `{arm}` | " + " | ".join(cells) + " | — | 0 |\n")
                    continue
                sd = _fmt(_std1(vals)) if len(vals) > 1 else "—"
                W(f"| `{arm}` | " + " | ".join(cells)
                  + f" | **{_fmt(_mean(vals))}** ± {sd} | {len(vals)} |\n")
            W("\n")

        W("---\n\n## How to read a model score against these\n\n")
        W("1. **`ctrlcell` is the floor** for both metric families — `pearson_delta` ~0 and "
          "`overlap_at_N` exactly 0. A model at the floor has learned nothing.\n")
        W("2. **`meanpert` is the minimum bar for perturbation-specificity.** It predicts "
          "one delta for everything, so its `discrimination_score_l2` sits at chance. A "
          "model above it on `pearson_delta` but at chance on discrimination has learned "
          "the average response, not the perturbations.\n")
        W("3. **`transfer` is the bar that matters.** It is this task done by averaging two "
          "other cell lines instead of learning. A model below it is not earning its "
          "parameters.\n")
        W("4. **`experimental` is not a ceiling.** Beating it is possible from noise alone; "
          "check whether the gap shrinks as `m` grows.\n")
        W("5. **The split-to-split std is the resolution limit.** It is 0.001–0.002 on "
          "`pearson_delta` here, so a model-vs-baseline difference above ~0.005 is real "
          "and anything smaller is not.\n\n")
        W("Panel definition: `data/benchmarks/rhj_fewshot/splithalf/CARD.md`. "
          "Protocol findings: `docs/rhj_splithalf.md`.\n")

    print(f"[baselines] {len(data)} (split, arm) results over {len(metrics)} metrics")
    for a in ARMS:
        print(f"  {a:14s} splits {coverage[a]}")
    print(f"[baselines] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
