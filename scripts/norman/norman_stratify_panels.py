#!/usr/bin/env python
"""Define norman_combo's ADDITIVITY STRATA -- the two panels every result is reported on.

WHY THIS FILE EXISTS
    The 111 test pairs are not scored as one number. They are split by how well
    `sum_of_singles` (d_A + d_B, measured from the singles) already explains the pair,
    because a model that beats additivity on pairs additivity already handles has proved
    nothing. The two SCORED panels are `additive` and `nonadditive`.

    ⚠️ This step had no script. `additivity.npz` and the `panels` block of
    `splits_manifest.json` were produced by an ad-hoc pass during the campaign and were
    therefore IRREPRODUCIBLE from the repo: rebuilding the benchmark with
    scripts/repro/build_benchmark_norman.sh yielded `panels: []` and
    `scored_panels: null`, i.e. a benchmark a consumer could not score the way we do.
    Found on 2026-09-22 by actually running that script. The recipe below was recovered
    from the published artifacts and reproduces every quantity BIT-FOR-BIT
    (max|diff| = 0.0 on residual, effect_norm, magnitude_ratio and the cell counts), and
    all seven panel memberships exactly. `--verify` re-checks that against a reference.

THE QUANTITIES, per test pair
    d_real  mean of the pair's panel cells  -  mean of the panel's controls
            (8,000-gene axis, the panel's own eval space, ALL cells not a half)
    d_sum   the pair's `sum_of_singles` delta
    residual        ||d_sum - d_real|| / ||d_real||     <- how NON-additive the pair is
    effect_norm     ||d_real||                          <- how LARGE the true effect is
    magnitude_ratio ||d_sum|| / ||d_real||
    pearson_r       corr(d_sum, d_real)

THE CUT, and why there are two of them
    `*_raw`   terciles of `residual` over all 111 pairs -> 37 / 37 / 37.
              ⚠️ CONFOUNDED: residual is a RATIO, so a pair with a tiny true effect has
              an inflated one. The raw strata therefore differ in effect size as well as
              in additivity, and a model that is simply better on large effects looks
              like a model that handles non-additivity. Kept for completeness. DO NOT
              QUOTE.
    OFFICIAL  `tercile_effect_controlled`: residual terciles taken WITHIN effect-size
              terciles -> 36 / 39 / 36. The strata are then matched on effect size
              (median ||d_real|| 3.61 / 3.59 / 3.66 against the raw cut's spread), so a
              difference between them is about additivity and not about magnitude.

⚠️ SELECTION CAVEAT, carried into the manifest verbatim: the strata are defined by
   `sum_of_singles`'s OWN residual on the SAME cells, so that baseline's per-stratum
   score is not an independent measurement -- it is low on `nonadditive` BY
   CONSTRUCTION. Model-vs-model comparisons across strata are unaffected;
   baseline-vs-model on `nonadditive` is biased against the baseline.

USAGE
    python scripts/norman/norman_stratify_panels.py            # write into the panel
    python scripts/norman/norman_stratify_panels.py --verify    # check, write nothing
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

BUILDER_VERSION = "norman_stratify_panels/1"
SCORED = ("additive", "nonadditive")
# index -> stratum name, for both cuts. 0 = lowest residual = most additive.
NAMES = {0: "additive", 1: "middle", 2: "nonadditive"}


def _terciles(values):
    """(labels 0/1/2, q_lo, q_hi) by the 1/3 and 2/3 quantiles.

    ⚠️ THE BOUNDARY CONVENTION IS ASYMMETRIC AND LOAD-BEARING: `>=` at the lower edge,
    `>` at the upper. It was recovered by matching the published labels, and only ONE of
    the four combinations reproduces them (36 / 39 / 36 on the official cut; the other
    three give 39/36/36, 39/33/39 or 36/36/39).

    It matters because of the sizes involved. With 111 values the 1/3 quantile falls
    between elements -- index (111-1)/3 = 36.67 -- so every convention agrees and the raw
    cut is 37/37/37 either way. Within an effect tercile there are 37 values, index
    (37-1)/3 = 12.0 lands EXACTLY on an element, and the convention then decides which
    side that element falls. `np.quantile` default (linear) interpolation throughout.
    """
    import numpy as np

    q_lo, q_hi = np.quantile(values, [1 / 3, 2 / 3])
    lab = np.zeros(len(values), dtype=np.int8)
    lab[values >= q_lo] = 1
    lab[values > q_hi] = 2
    return lab, float(q_lo), float(q_hi)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="data/benchmarks/norman_combo/splithalf")
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against what is already on disk "
                         "(or --reference); write nothing, exit 3 on any difference")
    ap.add_argument("--reference", default=None,
                    help="panel dir to compare against; default = --panel itself")
    args = ap.parse_args()

    import numpy as np

    panel = args.panel
    man_path = os.path.join(panel, "splits_manifest.json")
    for need in ("panel.npz", "sum_of_singles.npz", "splits_manifest.json"):
        if not os.path.isfile(os.path.join(panel, need)):
            print(f"missing {os.path.join(panel, need)}; build the panel first with "
                  f"scripts/norman/norman_build_panel.py", file=sys.stderr)
            return 1

    p = np.load(os.path.join(panel, "panel.npz"), allow_pickle=True)
    sos = np.load(os.path.join(panel, "sum_of_singles.npz"), allow_pickle=True)
    man = json.load(open(man_path))

    X = p["X"]
    labels = np.asarray([str(x) for x in p["labels"]])
    sos_map = {str(q): sos["deltas"][i] for i, q in enumerate(sos["perts"])}
    # The cohort pair list comes from cohort.json (`cells_per_pert`), sorted, so the
    # arrays have a deterministic order independent of panel row order.
    # ⚠️ NOT from the manifest's `cohort_pairs` -- that key is a COUNT (111), not a list.
    cohort_path = os.path.join(panel, "cohort.json")
    if os.path.isfile(cohort_path):
        cohort = json.load(open(cohort_path))
        pairs = sorted(str(q) for q in (cohort.get("cells_per_pert") or {})
                       if "+" in str(q))
    else:
        pairs = sorted(str(q) for q in sos["perts"])
    if not pairs:
        print(f"no multi-gene pairs found in {cohort_path}", file=sys.stderr)
        return 1
    missing = [q for q in pairs if q not in sos_map]
    if missing:
        print(f"{len(missing)} cohort pair(s) have no sum_of_singles delta, e.g. "
              f"{missing[:5]}; the stratification is defined by that baseline and "
              f"cannot be computed without it", file=sys.stderr)
        return 1

    cmean = X[labels == "non-targeting"].mean(0)
    residual, effect_norm, mag_ratio, pear, ncells = [], [], [], [], []
    for q in pairs:
        m = labels == q
        d_real = X[m].mean(0) - cmean
        d_sum = sos_map[q]
        n_real = np.linalg.norm(d_real)
        residual.append(np.linalg.norm(d_sum - d_real) / n_real)
        effect_norm.append(n_real)
        mag_ratio.append(np.linalg.norm(d_sum) / n_real)
        pear.append(float(np.corrcoef(d_sum, d_real)[0, 1]))
        ncells.append(int(m.sum()))
    residual = np.asarray(residual, np.float32)
    effect_norm = np.asarray(effect_norm, np.float32)
    mag_ratio = np.asarray(mag_ratio, np.float32)
    pear = np.asarray(pear, np.float64)
    ncells = np.asarray(ncells, np.int64)

    # The RAW cut: residual terciles over all pairs. Confounded with effect size.
    tercile, q_lo, q_hi = _terciles(residual)
    # The OFFICIAL cut: residual terciles taken WITHIN effect-size terciles.
    effect_tercile, _, _ = _terciles(effect_norm)
    controlled = np.zeros(len(pairs), dtype=np.int8)
    for t in (0, 1, 2):
        idx = np.flatnonzero(effect_tercile == t)
        sub, _, _ = _terciles(residual[idx])
        controlled[idx] = sub

    def _members(lab, suffix=""):
        return {NAMES[t] + suffix: sorted(pairs[i] for i in np.flatnonzero(lab == t))
                for t in (0, 1, 2)}

    panels = {"all": sorted(pairs)}
    panels.update(_members(controlled))            # additive / middle / nonadditive
    panels.update(_members(tercile, "_raw"))       # *_raw
    counts = {k: len(v) for k, v in sorted(panels.items())}
    print(f"[strat] {len(pairs)} cohort pairs")
    print(f"[strat] residual terciles q_lo={q_lo:.6f} q_hi={q_hi:.6f}")
    for k in sorted(panels):
        med = float(np.median(effect_norm[[pairs.index(q) for q in panels[k]]]))
        mark = "  <- SCORED" if k in SCORED else (
            "  <- DO NOT QUOTE (confounded with effect size)" if k.endswith("_raw")
            else "")
        print(f"[strat]   {k:16} {counts[k]:4} pairs   median ||d_real|| {med:6.3f}{mark}")

    ref = args.reference or panel
    if args.verify:
        rc = 0
        ref_npz = os.path.join(ref, "additivity.npz")
        if os.path.isfile(ref_npz):
            z = np.load(ref_npz, allow_pickle=True)
            order = [pairs.index(str(q)) for q in z["perts"]]
            for name, got in (("residual", residual), ("effect_norm", effect_norm),
                              ("magnitude_ratio", mag_ratio), ("n_cells", ncells),
                              ("tercile", tercile), ("effect_tercile", effect_tercile),
                              ("tercile_effect_controlled", controlled)):
                if name not in z:
                    continue
                d = np.abs(np.asarray(got)[order] - z[name]).max()
                print(f"[verify] {name:26} max|diff| = {d}")
                if d != 0:
                    rc = 3
        ref_man = json.load(open(os.path.join(ref, "splits_manifest.json")))
        for k, v in sorted((ref_man.get("panels") or {}).items()):
            got = set(panels.get(k, []))
            want = set(str(x) for x in v["pairs"])
            same = got == want
            print(f"[verify] panel {k:16} {'OK' if same else 'DIFFERS'} "
                  f"({len(got)} vs {len(want)} pairs)")
            if not same:
                rc = 3
        print("[verify] " + ("REPRODUCED exactly" if rc == 0 else "DIFFERENCES FOUND"))
        return rc

    np.savez(
        os.path.join(panel, "additivity.npz"),
        perts=np.asarray(pairs, dtype=object), pearson_r=pear, residual=residual,
        magnitude_ratio=mag_ratio, n_cells=ncells, tercile=tercile,
        q_lo=np.asarray(q_lo), q_hi=np.asarray(q_hi), effect_norm=effect_norm,
        effect_tercile=effect_tercile, tercile_effect_controlled=controlled,
    )

    def _stats(lab, names):
        out = {}
        for t, nm in names.items():
            i = np.flatnonzero(lab == t)
            out[nm] = {"n": int(i.size),
                       "residual_median": float(np.median(residual[i])),
                       "effect_norm_median": float(np.median(effect_norm[i])),
                       "magnitude_ratio_median": float(np.median(mag_ratio[i])),
                       "pearson_r_median": float(np.median(pear[i]))}
        i = np.arange(len(pairs))
        out["all"] = {"n": int(i.size),
                      "residual_median": float(np.median(residual)),
                      "effect_norm_median": float(np.median(effect_norm)),
                      "magnitude_ratio_median": float(np.median(mag_ratio)),
                      "pearson_r_median": float(np.median(pear))}
        return out

    man["panels"] = {k: {"n_pairs": len(v), "pairs": v} for k, v in sorted(panels.items())}
    man["scored_panels"] = list(SCORED)
    man["additivity_stratification"] = {
        "OFFICIAL_CUT": "tercile_effect_controlled — residual terciles taken "
                        "WITHIN effect-size terciles",
        "SELECTION_CAVEAT":
            "These strata are defined by the sum_of_singles baseline's own residual on "
            "the same cells, so that baseline's score PER STRATUM is not an independent "
            "measurement: it is low on `nonadditive` by construction. Model-vs-model "
            "comparisons across strata are unaffected; baseline-vs-model on "
            "`nonadditive` is biased against the baseline.",
        "RAW_CUT_CAVEAT":
            "The *_raw panels tercile the residual globally. Residual is a ratio, so a "
            "pair with a tiny true effect has an inflated one and the raw strata differ "
            "in effect size as well as in additivity. Do not quote them.",
        "residual": "||d_sum - d_real|| / ||d_real||, d_real = mean(pair cells) - "
                    "mean(panel controls) on the scoring axis, d_sum from "
                    "sum_of_singles.npz",
        "cut": f"terciles of the residual over the {len(pairs)} cohort pairs -> "
               + " / ".join(str(counts[NAMES[t] + '_raw']) for t in (0, 1, 2)),
        "official_cut": "residual terciles within effect-size terciles -> "
                        + " / ".join(str(counts[NAMES[t]]) for t in (0, 1, 2)),
        "q_lo": q_lo, "q_hi": q_hi,
        "per_stratum": _stats(tercile, NAMES),
        "per_stratum_official": _stats(controlled, NAMES),
        "builder_version": BUILDER_VERSION,
        "built": _dt.date.today().isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
                   or "unknown",
    }
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    print(f"[strat] wrote {os.path.join(panel, 'additivity.npz')}")
    print(f"[strat] stamped panels / scored_panels / additivity_stratification into "
          f"{man_path}")
    print(f"[strat] SCORED panels: {', '.join(SCORED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
