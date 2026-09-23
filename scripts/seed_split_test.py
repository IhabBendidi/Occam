#!/usr/bin/env python
"""Compare two arms on a 5-seed x 5-split design, with the right unit of replication.

WHY THIS FILE EXISTS
    The `norman_combo` design is NESTED: 5 training seeds, and within each seed the SAME
    5 seeded split-halves. Those two levels are not interchangeable, and treating them as
    one pool is the mistake this script exists to prevent.

    Splits are NOT independent replicates of a model. They re-slice the same cells from
    the same trained checkpoint, so their variance measures panel sampling, not model
    variance. Pooling 25 numbers and running a 25-sample t-test inflates n by 5x and
    shrinks the standard error by ~2.2x -- it will call almost anything significant.
    Measured on this benchmark: per-split SEMs are 0.0003-0.0034 while the effects being
    judged are 0.02-0.06, so a split-level test declares everything a win.

    ⚠️ THE SEED IS THE UNIT OF REPLICATION, because the seed is what re-draws the thing
    under test (the trained model). So: average over splits WITHIN each seed, then test
    across the 5 seed-level means.

WHAT IT REPORTS, and why each
    paired t-test on the 5 seed means   the primary test. Paired because both arms see
                                        the SAME seeds and the SAME splits, which removes
                                        seed-level difficulty as a nuisance term.
    Wilcoxon signed-rank                the distribution-free companion. With n=5 its
                                        smallest attainable two-sided p is 0.0625, so it
                                        CANNOT reach 0.05 -- reported as a consistency
                                        check on the sign, never as the verdict.
    sign test (k/5 seeds)               the most robust statement available at this n, and
                                        usually the one worth quoting: "5/5 seeds agree".
    Cohen's d_z and a 95% CI            the effect size and its range. At n=5 the CI is
                                        wide; that width IS the result and must be shown.

⚠️ n=5 IS SMALL. A paired t-test on 5 pairs has ~80% power only for d_z >= 1.6. Read a
   non-significant p as "underpowered", not as "no difference", and read a significant one
   with the CI beside it. This is why the sign test is reported: it makes no distributional
   claim and 5/5 is interpretable without one.

USAGE
    python scripts/seed_split_test.py --a "NQ02:decode_seq_mean" --b txpert \\
        --panel nonadditive --metric pearson_delta
    python scripts/seed_split_test.py --list          # what arms are available
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO, "data", "release", "metrics_all", "metrics_long.csv")
LOWER_IS_BETTER = ("mae", "mse", "rmse", "nmae")


def load(csv_path, benchmark="norman_combo"):
    """{(arm_key, panel, metric): {(seed, split): value}}, arm_key = 'arm[:mode]'."""
    out = defaultdict(dict)
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["benchmark"] != benchmark:
                continue
            kind = r["arm_kind"]
            if kind == "reference":
                key, seed = r["scored_arm"], 42
            elif kind == "competitor" and r["scored_arm"] == "model_nocal":
                # `compet_<arm>__<bench>_s<seed>` -> arm, seed. The seed is in the
                # eval_root, not in a column, because a competitor's identity on disk is
                # its run directory.
                base = os.path.basename(r["eval_root"])
                seed = int(base.rsplit("_s", 1)[1]) if "_s" in base else 42
                key = r["arm"].split("__")[0]
            elif kind == "occam" and r["scored_arm"] == "model_nocal":
                key = r["arm"] + (f":{r['compose_mode']}" if r["compose_mode"] else "")
                tag = r["config_tag"] or ""
                seed = int(tag.rsplit("_s", 1)[1]) if "_s" in tag else 42
            else:
                continue
            out[(key, r["panel"], r["metric"])][(seed, int(r["split"]))] = float(r["value"])
    return out


def seed_means(cell):
    """{seed: mean over that seed's splits} -- the seed IS the unit of replication."""
    by = defaultdict(list)
    for (seed, _split), v in cell.items():
        by[seed].append(v)
    return {s: sum(v) / len(v) for s, v in by.items()}, {s: len(v) for s, v in by.items()}


def _t_sf(t, df):
    """Two-sided p for Student's t, via the regularised incomplete beta (no scipy)."""
    x = df / (df + t * t)
    # Edge guards: t = 0 gives x = 1 and log(1 - x) is a domain error; t -> inf gives
    # x = 0. Both are reachable -- t is exactly 0 whenever two arms tie on every seed.
    if x >= 1.0:
        return 1.0
    if x <= 0.0:
        return 0.0
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, it=200, eps=3e-16):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m in range(1, it + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    ib = math.exp(lb) * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else \
        1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                       + b * math.log(1.0 - x) + a * math.log(x)) * \
        betacf(b, a, 1.0 - x) / b
    return max(0.0, min(1.0, ib))


# Exact two-sided Wilcoxon signed-rank p for small n, by enumeration.
def wilcoxon_p(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    ranks = {}
    for i, (_, idx) in enumerate(sorted((abs(x), i) for i, x in enumerate(d))):
        ranks[idx] = i + 1
    w_plus = sum(ranks[i] for i, x in enumerate(d) if x > 0)
    total = n * (n + 1) / 2
    w = min(w_plus, total - w_plus)
    count = 0
    for mask in range(1 << n):
        s = sum(ranks[i] for i in range(n) if mask >> i & 1)
        if min(s, total - s) <= w:
            count += 1
    return min(1.0, count / (1 << n)), w


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--a", help="arm A, e.g. 'NQ02:decode_seq_mean' or 'txpert'")
    ap.add_argument("--b", help="arm B")
    ap.add_argument("--panel", default="nonadditive")
    ap.add_argument("--metric", default="pearson_delta")
    ap.add_argument("--benchmark", default="norman_combo")
    ap.add_argument("--list", action="store_true", help="list available arms and exit")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"no metric export at {args.csv}; run scripts/export_split_metrics.py "
              f"--scope all first", file=sys.stderr)
        return 1
    data = load(args.csv, args.benchmark)

    if args.list or not (args.a and args.b):
        arms = sorted({k for k, p, m in data
                       if p == args.panel and m == args.metric})
        print(f"arms on {args.benchmark} / {args.panel} / {args.metric}:")
        for a in arms:
            sm, ns = seed_means(data[(a, args.panel, args.metric)])
            print(f"  {a:34} seeds {sorted(sm)} (splits per seed: "
                  f"{sorted(set(ns.values()))})")
        if not (args.a and args.b):
            print("\npass --a and --b to test a pair")
        return 0

    ca = data.get((args.a, args.panel, args.metric))
    cb = data.get((args.b, args.panel, args.metric))
    for nm, c in ((args.a, ca), (args.b, cb)):
        if not c:
            print(f"no data for {nm!r} on {args.panel}/{args.metric}. --list to see "
                  f"what is available.", file=sys.stderr)
            return 1

    ma, na = seed_means(ca)
    mb, nb = seed_means(cb)
    seeds = sorted(set(ma) & set(mb))
    if not seeds:
        print(f"no shared seeds: {args.a} has {sorted(ma)}, {args.b} has {sorted(mb)}",
              file=sys.stderr)
        return 1

    lower = args.metric in LOWER_IS_BETTER
    print(f"\n{args.benchmark} · {args.panel} · {args.metric} "
          f"({'LOWER is better' if lower else 'higher is better'})")
    print(f"unit of replication: the SEED. Splits are averaged within each seed first.\n")
    print(f"{'seed':>6} {'splits':>7} {args.a[:16]:>17} {args.b[:16]:>17} {'A-B':>10}")
    diffs = []
    for s in seeds:
        d = ma[s] - mb[s]
        diffs.append(d)
        print(f"{s:>6} {na[s]:>3}/{nb[s]:<3} {ma[s]:>17.4f} {mb[s]:>17.4f} {d:>+10.4f}")

    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        print(f"\nonly {n} shared seed(s): no test is possible. The per-seed difference "
              f"is {mean:+.4f}; report it as a single observation.")
        return 0
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    better = sum(1 for x in diffs if (x < 0 if lower else x > 0))
    print(f"\n  mean(A-B)        {mean:+.4f}   (sd {sd:.4f}, se {se:.4f}, n={n} seeds)")
    if se > 0:
        t = mean / se
        p = _t_sf(abs(t), n - 1)
        tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        print(f"  paired t-test    t({n-1}) = {t:+.3f}, p = {p:.4f}   <- PRIMARY")
        print(f"  95% CI           [{lo:+.4f}, {hi:+.4f}]")
        print(f"  Cohen's d_z      {mean / sd:+.3f}")
    else:
        print("  paired t-test    sd is 0; every seed gives the identical difference")
    wp, w = wilcoxon_p(diffs)
    floor = 2.0 / (2 ** n)
    print(f"  Wilcoxon         W = {w:g}, exact two-sided p = {wp:.4f} "
          f"(floor at n={n} is {floor:.4f}; cannot reach 0.05 below n=6)")
    print(f"  sign test        {better}/{n} seeds favour "
          f"{args.a if better * 2 > n else args.b}")
    print(f"\n  ⚠️ n={n} seeds. A paired t-test here has ~80% power only for |d_z| >= 1.6, "
          f"so a large p means UNDERPOWERED, not 'no difference'. Quote the "
          f"{better}/{n} sign result and the CI alongside any p.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
