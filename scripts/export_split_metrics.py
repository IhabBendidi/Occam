#!/usr/bin/env python
"""Export EVERY cell-eval metric, for every arm, for every split -- not just the four.

WHY THIS EXISTS
    Reports quote four metrics (`pearson_delta`, `mae`, `overlap_at_N`,
    `discrimination_score_l1`) and their mean over 5 splits. cell-eval 0.6.0 computes
    ~30, and the per-split values are what a reader needs to check a claim: a mean of
    five numbers hides whether the effect is consistent across splits or carried by one.
    This script writes all of them, in a tidy long table plus wide per-panel tables.

WHAT IT READS
    Every `<eval root>/split_NN/metrics.json` under `runs/`. That file records
    `arms.<arm>.<metric>` for every arm scored in that pass, over the WHOLE panel.
    For a benchmark whose `splits_manifest.json` declares PANELS (norman_combo's
    `additive` / `nonadditive`), the per-panel aggregates are recomputed from the arm's
    own `celleval_*/results.csv` with the same `_aggregate_subset` the scorer uses, so
    the numbers are identical to the reported ones by construction rather than by
    transcription.

WHAT IT WRITES  (into --out, default data/release/metrics)
    metrics_long.csv                  one row per (benchmark, arm, mode, split, panel, metric)
    metrics_wide__<benchmark>__<panel>.csv   arms x metrics, one file per panel, split-mean
    per_split__<benchmark>__<panel>.csv      arms x (metric, split), every split visible
    per_perturbation/<benchmark>/<arm>/split_NN.csv   the per-perturbation results.csv,
                                      copied for --per-pert-arms only (they are 48 KB each
                                      and there are 1,700+ arm-splits)
    README.md                         the schema and which direction is good per metric

USAGE
    python scripts/export_split_metrics.py                       # everything
    python scripts/export_split_metrics.py --benchmark norman_combo
    python scripts/export_split_metrics.py --calibration nocal   # the default

⚠️ `--calibration nocal` is the default and matches what is reported. Calibrated arms
   (`model_cal`) exist on disk for older RHJ passes and are exported only with
   `--calibration all`, tagged `calibration=cal`, so they cannot be mistaken for the
   headline numbers.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# Which benchmark an eval root belongs to. The path is the only record -- metrics.json
# does not carry a benchmark id -- so `n_perts` is exported alongside as a cross-check.
BENCHMARK_BY_PREFIX = (
    ("runs/norman_combo/", "norman_combo"),
    ("runs/rhj_splithalf/", "rhj_fewshot"),
    ("runs/rhj_fewshot/", "rhj_fewshot"),
    ("runs/competitors/", "rhj_fewshot"),
    ("runs/emit_eval/", "rhj_fewshot"),
)
PANEL_DIR = {
    "norman_combo": "data/benchmarks/norman_combo/splithalf",
    "rhj_fewshot": "data/benchmarks/rhj_fewshot/splithalf",
}
EXPECTED_N_PERTS = {"norman_combo": 111, "rhj_fewshot": 611}

# ERRORS: lower is better. Everything else here is "higher is better". Metrics whose
# direction is genuinely ambiguous are listed as "context" and must not be ranked on.
LOWER_IS_BETTER = ("mae", "mse", "rmse", "nmae", "wasserstein", "energy_distance",
                   "de_nsig_counts_pred", "de_nsig_counts_real", "clipped_fraction")
CONTEXT_ONLY = ("overlap_at_N", "precision_at_N", "de_nsig_counts_pred",
                "de_nsig_counts_real", "clipped_fraction")

HEADLINE = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")


def benchmark_of(root: str) -> str:
    rel = os.path.relpath(root, REPO).replace(os.sep, "/")
    for prefix, name in BENCHMARK_BY_PREFIX:
        if rel.startswith(prefix):
            return name
    return "unknown"


def arm_label(root: str, benchmark: str) -> tuple:
    """(arm, compose_mode, config_tag) from the eval root's directory name.

    Both benchmarks name their eval roots `<arm id>__<tail>`, but the tail means
    different things and must not share a column:

        norman_combo   NQ02__decode_seq_mean            tail = the COMPOSITION MODE,
                       because 12 modes share one checkpoint and without the mode in
                       the path they would overwrite each other's scores.
        rhj_fewshot    Y06__smooth_l1_nbw0_..._s42      tail = the CONFIG TAG, the
                       run's full identity on disk (every varied axis is in it).

    Splitting on `__` is what makes `--per-pert-arms Y06` work without anyone pasting
    a 60-character loss tag.
    """
    name = os.path.basename(root)
    arm, sep, tail = name.partition("__")
    if not sep:
        return name, "", ""
    if benchmark == "norman_combo":
        return arm, tail, ""
    return arm, "", tail


def panels_for(benchmark: str):
    """{panel name: set of perturbation labels}, or {} when the panel declares none."""
    d = PANEL_DIR.get(benchmark)
    if not d:
        return {}
    p = os.path.join(REPO, d, "splits_manifest.json")
    if not os.path.isfile(p):
        return {}
    man = json.load(open(p))
    out = {}
    for name, spec in (man.get("panels") or {}).items():
        # `all` is skipped deliberately: metrics.json already carries cell-eval's own
        # whole-panel aggregate under panel="all", and emitting the manifest's `all`
        # too gave every row twice, which the wide tables then averaged. The two agree
        # to 4 decimals, so the duplicate was silent.
        if name == "all":
            continue
        pairs = spec.get("pairs") or spec.get("perts") or []
        if pairs:
            out[name] = set(pairs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/release/metrics")
    ap.add_argument("--benchmark", default="all",
                    choices=["all", "norman_combo", "rhj_fewshot"])
    ap.add_argument("--calibration", default="nocal", choices=["nocal", "all"])
    ap.add_argument("--per-pert-arms", default="NQ02__decode_seq_mean,Y06",
                    help="comma-separated eval-root names whose per-perturbation "
                         "results.csv is copied verbatim; reference arms are always "
                         "included")
    args = ap.parse_args()

    from occam.vcell.panel.run import _aggregate_subset, _read_results_csv

    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)

    roots = sorted({os.path.dirname(p) for p in
                    glob.glob(os.path.join(REPO, "runs/**/split_*/metrics.json"),
                              recursive=True)})
    roots = sorted({os.path.dirname(r) for r in roots})
    rows = []
    panel_cache = {}
    n_skipped = 0
    per_pert_want = {s.strip() for s in args.per_pert_arms.split(",") if s.strip()}
    copied = 0

    for root in roots:
        bench = benchmark_of(root)
        if bench == "unknown":
            n_skipped += 1
            continue
        if args.benchmark != "all" and bench != args.benchmark:
            continue
        arm, mode, cfg_tag = arm_label(root, bench)
        if bench not in panel_cache:
            panel_cache[bench] = panels_for(bench)
        panels = panel_cache[bench]

        for mpath in sorted(glob.glob(os.path.join(root, "split_*", "metrics.json"))):
            split_dir = os.path.dirname(mpath)
            split = int(os.path.basename(split_dir).split("_")[1])
            try:
                mj = json.load(open(mpath))
            except (json.JSONDecodeError, OSError):
                n_skipped += 1
                continue
            n_perts = mj.get("n_perts")
            for scored_arm, metrics in (mj.get("arms") or {}).items():
                if not isinstance(metrics, dict) or "status" in metrics:
                    continue  # "not scored in this pass"
                cal = ("cal" if scored_arm.endswith("_cal")
                       else "nocal" if scored_arm.endswith("_nocal")
                       else "n/a")
                if args.calibration == "nocal" and cal == "cal":
                    continue
                kind = "model" if scored_arm.startswith("model") else "reference"
                for metric, value in metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    rows.append(dict(
                        benchmark=bench, eval_root=os.path.relpath(root, REPO),
                        arm=arm, compose_mode=mode, config_tag=cfg_tag,
                        scored_arm=scored_arm,
                        arm_kind=kind, calibration=cal, split=split, panel="all",
                        n_perts=n_perts, metric=metric, value=value))

                csv_path = os.path.join(split_dir, f"celleval_{scored_arm}",
                                        "results.csv")

                # Per-perturbation detail, only for the arms worth 48 KB x 5 splits.
                # ⚠️ This runs BEFORE the panel gate. It used to sit after it, behind
                # `if not panels: continue`, so on a benchmark that declares no
                # sub-panels (rhj_fewshot) nothing was ever copied -- silently.
                base = os.path.basename(root)
                if (base in per_pert_want or arm in per_pert_want
                        or f"{arm}__{mode}" in per_pert_want or kind == "reference"):
                    if os.path.isfile(csv_path):
                        dst_dir = os.path.join(out_dir, "per_perturbation", bench,
                                               f"{base}__{scored_arm}")
                        os.makedirs(dst_dir, exist_ok=True)
                        dst = os.path.join(dst_dir, f"split_{split:02d}.csv")
                        if not os.path.exists(dst):
                            shutil.copy2(csv_path, dst)
                            copied += 1

                # Per-panel aggregates, recomputed with the scorer's own function.
                if not panels or not os.path.isfile(csv_path):
                    continue
                per = _read_results_csv(csv_path)
                if not per:
                    continue
                for pname, members in panels.items():
                    agg = _aggregate_subset(per, members)
                    for metric, value in agg.items():
                        if not isinstance(value, (int, float)):
                            continue
                        rows.append(dict(
                            benchmark=bench, eval_root=os.path.relpath(root, REPO),
                            arm=arm, compose_mode=mode, config_tag=cfg_tag,
                            scored_arm=scored_arm,
                            arm_kind=kind, calibration=cal, split=split, panel=pname,
                            n_perts=len(members), metric=metric, value=value))

            if n_perts is not None and bench in EXPECTED_N_PERTS \
                    and n_perts != EXPECTED_N_PERTS[bench]:
                print(f"  NOTE {os.path.relpath(split_dir, REPO)}: n_perts {n_perts} "
                      f"!= {EXPECTED_N_PERTS[bench]} expected for {bench}")

    if not rows:
        print("no metrics found; has anything been evaluated?", file=sys.stderr)
        return 1

    long_path = os.path.join(out_dir, "metrics_long.csv")
    fields = ["benchmark", "eval_root", "arm", "compose_mode", "config_tag",
              "scored_arm", "arm_kind", "calibration", "split", "panel", "n_perts",
              "metric", "value"]
    with open(long_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Wide tables: split-mean per (arm, metric), and the every-split view beside it.
    import statistics as st
    groups = {}
    for r in rows:
        key = (r["benchmark"], r["panel"])
        aid = (r["arm"], r["compose_mode"], r["scored_arm"])
        groups.setdefault(key, {}).setdefault(aid, {}).setdefault(
            r["metric"], {})[r["split"]] = r["value"]

    n_wide = 0
    for (bench, panel), arms in sorted(groups.items()):
        metrics = sorted({m for v in arms.values() for m in v})
        ordered = [m for m in HEADLINE if m in metrics] + \
                  [m for m in metrics if m not in HEADLINE]
        tag = f"{bench}__{panel}"
        with open(os.path.join(out_dir, f"metrics_wide__{tag}.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "compose_mode", "scored_arm", "n_splits"]
                       + [f"{m}_mean" for m in ordered]
                       + [f"{m}_sem" for m in ordered])
            for aid in sorted(arms):
                per_m = arms[aid]
                n = max((len(v) for v in per_m.values()), default=0)
                means, sems = [], []
                for m in ordered:
                    vals = list(per_m.get(m, {}).values())
                    means.append(f"{st.fmean(vals):.6f}" if vals else "")
                    sems.append(f"{st.stdev(vals) / len(vals) ** 0.5:.6f}"
                                if len(vals) > 1 else "")
                w.writerow(list(aid) + [n] + means + sems)
        splits = sorted({s for v in arms.values() for m in v.values() for s in m})
        with open(os.path.join(out_dir, f"per_split__{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "compose_mode", "scored_arm", "metric"]
                       + [f"split_{s:02d}" for s in splits])
            for aid in sorted(arms):
                for m in ordered:
                    per_s = arms[aid].get(m, {})
                    if not per_s:
                        continue
                    w.writerow(list(aid) + [m]
                               + [f"{per_s[s]:.6f}" if s in per_s else ""
                                  for s in splits])
        n_wide += 2

    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(_readme(rows, groups))

    print(f"[metrics] {len(rows):,} rows from {len(roots):,} eval root(s)")
    print(f"[metrics] {long_path}")
    print(f"[metrics] {n_wide} wide/per-split table(s) over "
          f"{len(groups)} (benchmark, panel) combination(s)")
    print(f"[metrics] {copied} per-perturbation file(s) copied")
    if n_skipped:
        print(f"[metrics] {n_skipped} path(s) skipped (unmapped root or unreadable json)")
    return 0


def _readme(rows, groups) -> str:
    benches = sorted({r["benchmark"] for r in rows})
    metrics = sorted({r["metric"] for r in rows})
    arms = len({(r["benchmark"], r["eval_root"], r["scored_arm"]) for r in rows})
    lines = [
        "# Per-split cell-eval metrics — every metric, every arm, every split",
        "",
        "## What this is",
        "",
        "The complete metric output behind Occam's reported numbers. Reports quote four "
        "metrics and their mean over 5 splits; this is all "
        f"**{len(metrics)}** metrics cell-eval 0.6.0 produced, for "
        f"**{arms}** arm-scorings, split by split, so any claim can be re-derived rather "
        "than trusted.",
        "",
        f"Benchmarks: {', '.join(benches)}. Generated by "
        "`scripts/export_split_metrics.py`.",
        "",
        "## Files",
        "",
        "| file | contents |",
        "|---|---|",
        "| `metrics_long.csv` | tidy: one row per (benchmark, arm, compose_mode, "
        "scored_arm, split, panel, metric) |",
        "| `metrics_wide__<benchmark>__<panel>.csv` | arms × metrics, split **mean** and "
        "**SEM** |",
        "| `per_split__<benchmark>__<panel>.csv` | arms × metrics with **one column per "
        "split** |",
        "| `per_perturbation/<benchmark>/<arm>/split_NN.csv` | cell-eval's own "
        "per-perturbation rows, for the published arms and every reference arm |",
        "",
        "## Columns",
        "",
        "| column | meaning |",
        "|---|---|",
        "| `benchmark` | `rhj_fewshot` (unseen perturbation) or `norman_combo` (unseen "
        "pair) |",
        "| `arm` | the trained arm, e.g. `Y06`, `NQ02` |",
        "| `compose_mode` | multi-perturbation composition mode; empty on "
        "single-perturbation benchmarks. **Part of the result's identity** — 12 modes "
        "share one checkpoint |",
        "| `config_tag` | `rhj_fewshot` only: the run's full identity on disk, every "
        "varied axis included (loss terms, epochs, eta_min, seed) |",
        "| `scored_arm` | what was scored: `model_nocal`, or a reference arm "
        "(`meanpert`, `transfer`, `sum_of_singles`, `experimental`, `ctrlmean`, "
        "`ctrlcell`) |",
        "| `arm_kind` | `model` or `reference` |",
        "| `calibration` | `nocal` throughout. `cal` appears only with "
        "`--calibration all` and is **not** what is reported |",
        "| `panel` | `all`, or a declared sub-panel (`additive`, `nonadditive` on "
        "`norman_combo`) |",
        "| `n_perts` | perturbations in that panel, a cross-check on the row |",
        "",
        "## Direction — which way is good",
        "",
        "**Lower is better** (these are ERRORS): "
        + ", ".join(f"`{m}`" for m in LOWER_IS_BETTER if m in metrics) + ".",
        "",
        "**Higher is better** for everything else. `discrimination_score_*` has chance "
        "at **0.5**, not 0.",
        "",
        "⚠️ **Do not rank on these**, in either direction — they are context, not "
        "quality: "
        + ", ".join(f"`{m}`" for m in CONTEXT_ONLY if m in metrics)
        + ". On `rhj_fewshot` a zero-signal predictor beats a real model on "
        "`overlap_at_N` (the emission-floor artifact), and `de_nsig_counts_pred` vs "
        "`_real` differ by ~5x because a low-variance prediction inflates the Wilcoxon "
        "test's significance calls.",
        "",
        "## Anchors — a score is unreadable without them",
        "",
        "`rhj_fewshot`, `pearson_delta`: `ctrlcell` −0.0018 · `ctrlmean` 0.0016 · "
        "`meanpert` 0.1324 · `transfer` 0.2323 · `experimental` 0.3722. Published arm "
        "**Y06 = 0.4043**.",
        "",
        "`norman_combo`, `pearson_delta` (additive / nonadditive): `meanpert` "
        "0.4634 / 0.4016 · **`sum_of_singles` 0.8824 / 0.7757 ← the bar** · "
        "`experimental` 0.9165 / 0.8377. Published arm **NQ02 × decode_seq_mean = "
        "0.8169 / 0.7380**.",
        "",
        "⚠️ `experimental` is replicate agreement at the panel's own cell counts, **not "
        "an upper bound**: a model predicting the group mean can beat a noisy empirical "
        "half, and on `rhj_fewshot` Y06 does.",
        "",
        "⚠️ A `pearson_delta` is a rank within its own panel. Do not compare a "
        "`norman_combo` number to an `rhj_fewshot` one.",
        "",
        "## Which `norman_combo` panel is the reported one",
        "",
        "**`additive` and `nonadditive` only.** Both are 36 pairs and are "
        "**effect-size-controlled**: the additivity residual "
        "`‖Δ_A + Δ_B − Δ_real‖ / ‖Δ_real‖` is terciled *within* effect-size terciles, so "
        "the two panels are matched on how large the true effect is. The `*_raw` panels "
        "tercile the residual globally and are therefore confounded with effect size — "
        "they are exported for completeness and **must not be quoted**. `middle` is the "
        "middle tercile, held out of the comparison by design. `all` is cell-eval's own "
        "aggregate over all 111 test pairs and mixes the strata in whatever ratio the "
        "panel happens to have.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
