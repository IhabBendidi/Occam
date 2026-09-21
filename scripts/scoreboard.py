#!/usr/bin/env python
"""Every scored arm on the rhj_fewshot split-half panel, ALL FOUR metrics, one table.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a cellular context where that perturbation was
never measured, from a single control cell.

This table's question: across our loss/DCOS arms, the four competitor ports, the post-hoc
emission treatments and the model-independent baselines -- what does each actually score, on
the same instrument, with no metric omitted?

WHY ALL FOUR, ALWAYS. `pearson_delta` alone has ranked arms in the opposite order to
`overlap_at_N` (F-emit6) and hidden a +0.065 `disc_l1` gain (F-loss4). A one-metric table is
how both of those were nearly missed.

DIRECTIONS: `pearson_delta` ↑, `mae` ↓ (it is an ERROR), `overlap_at_N` ↑,
`discrimination_score_l1` ↑ (chance 0.5).

⚠️ `n` IS LOAD-BEARING. A pdex shared-memory race (F-infra1) made cell-eval return nothing for
some (arm, split, mode) triples while the SLURM task still exited 0, so several means are over
fewer than 5 splits. Any row with n < 5 is PARTIAL and is marked.

    python scripts/scoreboard.py
    python scripts/scoreboard.py --out docs/rhj_scoreboard.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

M = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
N_SPLITS = 5
REF = ("runs/rhj_splithalf/adjmap_s313__lr1e-3_dlr1e-3_cos0.5_dpo0_"
       "fmode-mult_fh256_fmf1_s42")
BASE_DIR = "runs/rhj_splithalf/reference"


def _mean(x):
    return sum(x) / len(x)


def _sd(x):
    if len(x) < 2:
        return float("nan")
    m = _mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def agg(run_dir, sub):
    from occam.vcell.panel.celleval_v1 import _read_agg
    per = {}
    for k in range(N_SPLITS):
        f = os.path.join(run_dir, f"split_{k:02d}", sub, "agg_results.csv")
        if os.path.isfile(f):
            m = _read_agg(f)
            if m:
                per[k] = m
    if not per:
        return None
    return len(per), {x: [per[k][x] for k in sorted(per) if x in per[k]] for x in M}


def collect():
    out = {"baselines": [], "ours_nocal": [], "ours_cal": [], "competitors": [], "emission": [],
           "campaign_sched": [], "campaign_reg": [], "campaign_loss": []}
    for b in ("ctrlcell", "ctrlmean", "meanpert", "transfer", "experimental"):
        out["baselines"].append((b, agg(BASE_DIR, f"celleval_{b}"), ""))
    # ⚠️ THE ORIGINAL GLOB ONLY SAW `L*` AND `D0*`, so this file silently omitted every arm
    # of the schedule, VICReg, latent, recon-split and dispersion campaigns -- including the
    # campaign BEST. A scoreboard that is missing the best arm is worse than no scoreboard,
    # so the families are enumerated explicitly and each one names the question it answers.
    for key, sub in (("ours_nocal", "celleval_model_nocal"), ("ours_cal", "celleval_model_cal")):
        rows = []
        for d in sorted(glob.glob("runs/rhj_splithalf/L*__*")) + sorted(glob.glob("runs/rhj_splithalf/D0*__*")):
            arm, tag = os.path.basename(d).split("__", 1)
            r = agg(d, sub)
            if r:
                rows.append((f"`{arm}` {tag.replace('_s42','')}", r, _mean(r[1]["pearson_delta"])))
        out[key] = [(a, r, "") for a, r, _ in sorted(rows, key=lambda x: -x[2])]
    for lbl, dirn, note in [
            ("**TxPert** (single-cell controls)", "compet_txpert_single_s42", "headline"),
            ("TxPert mean-controls", "compet_txpert_mean_s42", "A/B diagnostic"),
            ("TxPert epoch-20", "compet_txpert_ep20_s42", "A/B diagnostic"),
            ("~~TxPert original~~", "compet_txpert_s42", "⛔ WITHDRAWN (C28)"),
            ("**PerturbDiff** (genept)", "compet_perturbdiff_genept_s42", "headline"),
            ("**STATE**", "compet_state_s42", "headline"),
            ("**CellFlow**", "compet_cellflow_s42", "headline")]:
        out["competitors"].append((lbl, agg(f"runs/rhj_splithalf/{dirn}", "celleval_model_nocal"), note))
    # --- the rest of the campaign, by family. Each arm is labelled with what it VARIES,
    # because `W22` says nothing and "Q02 base, 90 ep, eta_min 1e-6" says everything.
    FAMILIES = [
        ("campaign_sched", "W", "epochs x eta_min (F-sched2)"),
        ("campaign_sched", "Y", "extended schedule (F-sched3)"),
        ("campaign_sched", "E", "60-epoch schedule (F-sched1)"),
        ("campaign_reg", "V", "VICReg latent terms (F-vicreg2)"),
        ("campaign_reg", "X", "VICReg x schedule compounding (F-vicreg3)"),
        ("campaign_reg", "Z", "dispersion terms (F-disp1)"),
        ("campaign_loss", "Q", "latent losses on M06 (F-latent4/6)"),
        ("campaign_loss", "M", "dcos/delta_mag scale-match (F-loss7/8)"),
        ("campaign_loss", "G", "split reconstruction (F-recon3/4)"),
        ("campaign_loss", "T", "latent losses at 0.01x (F-latent5)"),
    ]
    for key, prefix, what in FAMILIES:
        rows = []
        for d in sorted(glob.glob(f"runs/rhj_splithalf/{prefix}[0-9][0-9]__*")):
            arm, tag = os.path.basename(d).split("__", 1)
            r = agg(d, "celleval_model_nocal")
            if r:
                rows.append((f"`{arm}` {tag.replace('_s42', '')}", r, what,
                             _mean(r[1]["pearson_delta"])))
        out.setdefault(key, []).extend((a_, r, n) for a_, r, n, _ in
                                       sorted(rows, key=lambda x: -x[3]))
    for key in ("campaign_sched", "campaign_reg", "campaign_loss"):
        out[key] = sorted(out.get(key, []),
                          key=lambda t: -_mean(t[1][1]["pearson_delta"]))
    out["emission"].append(("none (`model_nocal`)", agg(REF, "celleval_model_nocal"), "the default"))
    out["emission"].append(("calibration only (`model_cal`)", agg(REF, "celleval_model_cal"), "control-fit quantile"))
    from occam.vcell.panel.celleval_v1 import _read_agg
    for sub, lbl, note in [
            ("s313_fair_lib17500/split_00/nb_xcolsum", "NB emission + `exact_colsum`", "fair theta, lib 17,500"),
            ("s313_fair_lib17500/split_00/nb", "NB emission alone", "fair theta, lib 17,500"),
            ("s313/split_00/nb_xcolsum", "NB emission + `exact_colsum`", "⛔ BROKEN theta/lib (F-emit4)"),
            ("s313/split_00/nb", "NB emission alone", "⛔ BROKEN theta/lib (F-emit4)")]:
        f = f"runs/emit_eval/{sub}/celleval/agg_results.csv"
        if os.path.isfile(f):
            m = _read_agg(f)
            if m:
                out["emission"].append((lbl, (1, {x: [m[x]] for x in M if x in m}), note))
    return out


def fmt(v, nd=4):
    return "—" if v is None or v != v else f"{v:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    data = collect()

    W = []
    W.append("# `rhj_fewshot` split-half — the scoreboard\n")
    W.append("**Objective.** Predict a perturbation's effect in a cellular context where that "
             "perturbation was never measured, from a single control cell.\n")
    W.append("**This table's question.** Across our loss/DCOS arms, the four competitor ports, "
             "the post-hoc emission treatments and the model-independent baselines — what does "
             "each actually score, on the same instrument, with no metric omitted?\n")
    W.append("**Varied:** the arm. **Held fixed:** the 611-perturbation cohort, the 6,983-gene "
             "axis, the seeded per-split half-B control draw, cell-eval 0.6.0 / pdex 0.1.24.\n")
    W.append("**Directions:** `pearson_delta` ↑ · `mae` **↓ (it is an ERROR)** · "
             "`overlap_at_N` ↑ · `discrimination_score_l1` ↑ (chance 0.5).\n")
    W.append("⚠️ **`n` is load-bearing.** A pdex shared-memory race (F-infra1) made cell-eval "
             "return nothing for some (arm, split, mode) triples while the SLURM task still "
             "exited 0. **Any row with n < 5 is PARTIAL**, marked ⚠️, and its mean may move.\n")
    W.append("⚠️ **Never compare across the two calibration tables.** Calibration costs "
             "0.008–0.068 `pearson_delta` and quadruples `mae` (F-loss3); the uncalibrated "
             "table is the reportable one.\n")

    for key, title, note in [
            ("baselines", "Baselines — model-independent anchors", "Read every model number against these, never against 0."),
            ("competitors", "Competitors — retrained on our data, our axis, our splits", "⚠️ None of these reproduces the method's published figure; see `docs/rhj_competitors_results.md` for each arm's deliberate deviations."),
            ("ours_nocal", "Our arms — UNCALIBRATED (`model_nocal`), the reportable table", "Loss composition and delta-cosine sweep, seed 42."),
            ("campaign_sched", "Schedule arms — epochs x `eta_min`",
             "F-sched1/2/3. `epochs` is the strongest lever in the campaign; `eta_min` is noise "
             "inside 1e-7…1e-4 and a real gain at 3e-4. The campaign BEST lives here."),
            ("campaign_reg", "Regulariser arms — VICReg and the dispersion terms",
             "F-vicreg2/3, F-disp1. VICReg helps at 30 epochs and hurts at 60–90; the "
             "dispersion terms hold per-cell spread at parity (mechanism verified in training)."),
            ("campaign_loss", "Loss-composition arms — latent, scale-match, split reconstruction",
             "F-latent4/5/6, F-loss7/8, F-recon3/4. These produced the three bases M03/M06/Q02."),
            ("ours_cal", "Our arms — CALIBRATED (`model_cal`), same checkpoints", "Shown because it was measured, NOT because it should be used."),
            ("emission", "Post-hoc emission treatments on `s313`", "⚠️ split 00 only for the NB rows (n=1)."),
    ]:
        rows = data[key]
        if not any(r for _, r, _ in rows):
            continue
        W.append(f"## {title}\n")
        W.append(f"{note}\n")
        W.append("| arm | n | `pearson_delta` ↑ | `mae` ↓ | `overlap_at_N` ↑ | `disc_l1` ↑ | note |")
        W.append("|---|---|---|---|---|---|---|")
        for lbl, r, extra in rows:
            if r is None:
                W.append(f"| {lbl} | 0 | — | — | — | — | not scored yet |")
                continue
            n, v = r
            p = v["pearson_delta"]
            pv = f"**{_mean(p):.4f}**" + (f" ± {_sd(p):.4f}" if n > 1 else "")
            warn = " ⚠️ PARTIAL" if n < N_SPLITS else ""
            W.append(f"| {lbl} | {n} | {pv} | {fmt(_mean(v['mae']))} | "
                     f"{fmt(_mean(v['overlap_at_N']))} | "
                     f"{fmt(_mean(v['discrimination_score_l1']))} | {extra}{warn} |")
        W.append("")

    text = "\n".join(W) + "\n"
    if a.out:
        open(a.out, "w").write(text)
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
