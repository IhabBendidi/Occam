#!/usr/bin/env python
"""Occam ablation grid: build, launch, register for scoring, and tabulate -- from ONE arm spec.

WHY. Which parts of the official rhj_fewshot model (Y06) carry its performance? Every arm is Y06
with exactly ONE thing changed (arms: scripts/ablation/arms.tsv), trained on rhj_fewshot at seeds
42-46 and scored at 5 split-halves on rhj_fewshot, h1_seen, k562_seen, k562_seen_h1set (nocal).
Plan, arms, what each outcome means, results: docs/ABLATION_PLAN.md.

SUBCOMMANDS (run from the repo root, any python with pandas; `launch` needs sbatch)
  grids     write scripts/grids/ablation/<arm>.tsv: the Y06 grid row with the arm's overrides,
            one row per seed. Deterministic; re-running reproduces the files byte-for-byte.
  launch    sbatch scripts/slurm/rhj_loss_sweep.sbatch once per arm (array = seeds), passing the
            arm's COS_W and EXTRA_SETS. Refuses `needs-code` arms and arms whose run dirs already
            hold a final_model.pth. Every launch is appended to runs/_ledger/ablation_launches.tsv.
  register  add the arm's score units (4 panels x 5 seeds) to configs/score_registry.tsv, so the
            score ledger (scripts/score_ledger.py --daemon) scores them when training lands.
  table     from data/release/final/scores_by_seed_split.csv (scripts/final_table.py), write
            data/release/ablation/scores_by_seed_split.csv -- one row per (panel, arm, seed,
            split), every cell-eval metric, Y06 included as the reference -- and summary.csv
            (mean, std, n seeds, n splits per panel x arm for the headline metrics).
  status    per arm and seed: grid written? launched (job)? trained? scored splits per panel.

EXAMPLES
  python scripts/ablation/ablate.py grids
  python scripts/ablation/ablate.py launch --arms ABL_GO,ABL_STR --dry-run
  python scripts/ablation/ablate.py launch --arms ABL_GO --seeds 42,43
  python scripts/ablation/ablate.py register --arms ABL_GO
  python scripts/export_split_metrics.py --scope all --out data/release/metrics_all && \\
      python scripts/final_table.py && python scripts/ablation/ablate.py table

ADDING AN ARM: one row in arms.tsv (grid overrides and/or EXTRA_SETS), then grids -> launch ->
register. EXTRA_SETS do not enter the run-dir name, so two arms must differ by arm id.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = os.path.join(REPO, "scripts/ablation/arms.tsv")
BASE_GRID = os.path.join(REPO, "scripts/grids/rhj_seeds.tsv")
BASE_ROW = "S43_Y06"                       # Y06 exactly; only arm/base/seed are rewritten
GRID_DIR = os.path.join(REPO, "scripts/grids/ablation")
LAUNCHER = "scripts/slurm/rhj_loss_sweep.sbatch"
LAUNCH_LOG = os.path.join(REPO, "runs/_ledger/ablation_launches.tsv")
REGISTRY = os.path.join(REPO, "configs/score_registry.tsv")
SEEDS = (42, 43, 44, 45, 46)
PANELS = ("rhj_fewshot", "h1_seen", "k562_seen", "k562_seen_h1set")
HEADLINE = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
# The five Y06 runs = the reference arm of every axis (seed 42 is the published run).
Y06_MODELS = {"Y06"}


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"))


def load_arms(select=None):
    arms = read_tsv(ARMS)
    if select:
        want = select.split(",")
        unknown = set(want) - {a["arm"] for a in arms}
        if unknown:
            sys.exit(f"unknown arm(s): {sorted(unknown)} (see {ARMS})")
        arms = [a for a in arms if a["arm"] in want]
    return arms


def base_row():
    with open(BASE_GRID) as fh:
        lines = [ln.rstrip("\n") for ln in fh if not ln.startswith("#")]
    header = lines[0].split("\t")
    for ln in lines[1:]:
        f = ln.split("\t")
        if f[0] == BASE_ROW:
            return header, dict(zip(header, f))
    sys.exit(f"{BASE_ROW} not found in {BASE_GRID}")


def arm_rows(arm):
    header, base = base_row()
    over = {}
    if arm["grid"] != "-":
        for kv in arm["grid"].split(","):
            k, v = kv.split("=", 1)
            if k not in header or k in ("arm", "base", "seed"):
                sys.exit(f"{arm['arm']}: bad grid column {k!r}")
            over[k] = v
    rows = []
    for s in SEEDS:
        r = dict(base, **over)
        r.update(arm=f"S{s}_{arm['arm']}", base=f"abl_{arm['arm']}_s{s}", seed=str(s))
        rows.append(r)
    return header, rows


def is_off(v):
    return v in ("0", "0.0", "", "none")


def run_tag(r):
    """The run-dir tag rhj_loss_sweep.sbatch derives (SUFFIX minus '<arm>__'), for Y06-family rows.

    Only the axes Y06 arms can vary are rendered; any other axis switched on is refused rather
    than guessed, so a mismatch with the launcher cannot pass silently.
    """
    for k, off in (("latent_cos_weight", "0.0"), ("vicreg_var_weight", "0.0"),
                   ("vicreg_cov_weight", "0.0"), ("recon_split", "false"),
                   ("dcos_conc_weight", "0.0"), ("radial_spread_weight", "0.0"),
                   ("gene_var_weight", "0.0"), ("batch_size", "2048"), ("lr", "1e-3")):
        if r[k] != off:
            sys.exit(f"{r['arm']}: run_tag does not render {k}={r[k]}; extend run_tag first")
    t = f"{r['loss_type']}_nbw{r['nb_weight']}_nbs{r['nb_scale']}_dcos{r['delta_cos_weight']}"
    if not is_off(r["delta_mag_weight"]):
        t += f"_mag{r['delta_mag_weight']}"
    if not is_off(r["latent_loss_weight"]):
        t += f"_lmse{r['latent_loss_weight']}"
    return t + f"_e{r['epochs']}_em{r['eta_min']}_s{r['seed']}"


def grid_path(arm):
    return os.path.join(GRID_DIR, f"{arm['arm']}.tsv")


def cmd_grids(args):
    os.makedirs(GRID_DIR, exist_ok=True)
    for arm in load_arms(args.arms):
        header, rows = arm_rows(arm)
        with open(grid_path(arm), "w") as fh:
            fh.write(f"# ABLATION {arm['arm']} (axis {arm['axis']}): Y06 with ONE change -- "
                     f"{arm['change']}.\n# Generated by scripts/ablation/ablate.py grids from "
                     f"{os.path.relpath(BASE_GRID, REPO)}:{BASE_ROW}; do not edit by hand.\n"
                     f"# Launcher env: COS_W={arm['cos_w']} EXTRA_SETS={arm['extra_sets']}\n")
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write("\t".join(r[h] for h in header) + "\n")
        print(f"wrote {os.path.relpath(grid_path(arm), REPO)} ({len(rows)} seeds)")


def marker(r):
    return os.path.join(REPO, f"runs/rhj_fewshot_{r['arm']}__{run_tag(r)}/vcell/checkpoints/"
                              "final_model.pth")


def cmd_launch(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    for arm in load_arms(args.arms):
        if arm["status"] != "ready" or arm["extra_sets"] == "TBD":
            print(f"SKIP {arm['arm']}: status {arm['status']} (not launchable yet)")
            continue
        if not os.path.isfile(grid_path(arm)):
            sys.exit(f"{arm['arm']}: no grid; run `ablate.py grids` first")
        _, rows = arm_rows(arm)
        idx = [i for i, r in enumerate(rows) if int(r["seed"]) in seeds]
        done = [rows[i]["arm"] for i in idx if os.path.isfile(marker(rows[i]))]
        if done:
            sys.exit(f"{arm['arm']}: already trained: {done}; drop those seeds")
        env = dict(os.environ, PYENV_VERSION=os.environ.get("PYENV_VERSION", "scmae"),
                   GRID=os.path.relpath(grid_path(arm), REPO), COS_W=arm["cos_w"],
                   EXTRA_SETS="" if arm["extra_sets"] == "-" else arm["extra_sets"])
        cmd = ["sbatch", "--parsable", f"--qos={args.qos}", f"--wckey={args.qos}",
               f"--time={arm['time']}", f"--array={','.join(map(str, idx))}",
               "--export=ALL", LAUNCHER]
        print(f"{arm['arm']}: COS_W={env['COS_W']} EXTRA_SETS={env['EXTRA_SETS'] or '-'}\n  "
              + " ".join(cmd))
        if args.dry_run:
            continue
        out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        jid = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if not jid.isdigit():
            sys.exit(f"sbatch failed: {out.stderr.strip()}")
        log_launch(arm, jid, [rows[i]["seed"] for i in idx], env)
        print(f"  -> job {jid}")


def log_launch(arm, jid, seeds, env):
    new = not os.path.isfile(LAUNCH_LOG)
    os.makedirs(os.path.dirname(LAUNCH_LOG), exist_ok=True)
    with open(LAUNCH_LOG, "a") as fh:
        if new:
            fh.write("time\tarm\tjob\tseeds\tcos_w\textra_sets\tgrid\n")
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{arm['arm']}\t{jid}\t"
                 f"{','.join(seeds)}\t{env['COS_W']}\t{env['EXTRA_SETS'] or '-'}\t"
                 f"{env['GRID']}\n")


def launch_jobs():
    """arm -> {seed: job id} from the launch log (for registry seed_jobs)."""
    out = {}
    if os.path.isfile(LAUNCH_LOG):
        for r in read_tsv(LAUNCH_LOG):
            for s in r["seeds"].split(","):
                out.setdefault(r["arm"], {})[s] = r["job"]
    return out


def cmd_register(args):
    have = {tuple(r[k] for k in ("kind", "bench", "arm", "seed")) for r in read_tsv(REGISTRY)}
    add = []
    for arm in load_arms(args.arms):
        if arm["status"] != "ready":
            print(f"SKIP {arm['arm']}: status {arm['status']}")
            continue
        _, rows = arm_rows(arm)
        for r in rows:
            for p in PANELS:
                key = ("occam", p, r["arm"], r["seed"])
                if key in have:
                    continue
                # scoring jobs are submitted by the ledger itself; seed_jobs stays empty
                add.append("\t".join((*key, run_tag(r), "")))
    if not add:
        print("nothing to add")
        return
    with open(REGISTRY, "a") as fh:
        fh.write("\n".join(add) + "\n")
    print(f"added {len(add)} score units to {os.path.relpath(REGISTRY, REPO)}")


def ablation_models():
    return {a["arm"] for a in load_arms()} | Y06_MODELS


def cmd_table(args):
    import pandas as pd
    src = os.path.join(REPO, "data/release/final/scores_by_seed_split.csv")
    df = pd.read_csv(src)
    df = df[(df["model_kind"] == "occam") & df["model"].isin(ablation_models())
            & df["benchmark"].isin(PANELS)].copy()
    spec = {a["arm"]: a for a in load_arms()}
    df.insert(3, "axis", df["model"].map(lambda m: spec[m]["axis"] if m in spec else "ref"))
    df.insert(4, "change", df["model"].map(lambda m: spec[m]["change"] if m in spec
                                          else "Y06 (reference for every axis)"))
    out_dir = os.path.join(REPO, "data/release/ablation")
    os.makedirs(out_dir, exist_ok=True)
    df = df.sort_values(["benchmark", "panel", "axis", "model", "seed", "split"])
    df.to_csv(os.path.join(out_dir, "scores_by_seed_split.csv"), index=False)
    g = df.groupby(["benchmark", "panel", "axis", "model", "change"])
    summ = g.agg(n_seeds=("seed", "nunique"), n_rows=("split", "size"))
    for m in HEADLINE:
        # mean over seeds of the per-seed split-mean; std across seeds
        per_seed = df.groupby(["benchmark", "panel", "axis", "model", "change", "seed"])[m].mean()
        lvl = ["benchmark", "panel", "axis", "model", "change"]
        summ[f"{m}_mean"] = per_seed.groupby(level=lvl).mean()
        summ[f"{m}_std_seeds"] = per_seed.groupby(level=lvl).std()
    summ.reset_index().to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"wrote {out_dir}/scores_by_seed_split.csv ({len(df)} rows) and summary.csv")


def cmd_status(args):
    import glob
    for arm in load_arms(args.arms):
        _, rows = arm_rows(arm)
        jobs = launch_jobs().get(arm["arm"], {})
        print(f"{arm['arm']:8s} [{arm['axis']}] {arm['status']:10s} {arm['change']}")
        for r in rows:
            tag = run_tag(r)
            sc = []
            for p in PANELS:
                root = {"rhj_fewshot": "runs/rhj_splithalf"}.get(p, f"runs/{p}")
                n = len(glob.glob(os.path.join(REPO, root, f"{r['arm']}__{tag}",
                                               "split_*", "metrics.json")))
                sc.append(f"{p}:{n}/5")
            print(f"   s{r['seed']} job {jobs.get(r['seed'], '-'):>9s} "
                  f"trained={'yes' if os.path.isfile(marker(r)) else 'no ':3s} " + " ".join(sc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grids", "launch", "register", "table", "status"):
        p = sub.add_parser(name)
        p.add_argument("--arms", default=None, help="comma-separated arm ids (default: all)")
        if name == "launch":
            p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
            p.add_argument("--qos", default="hooke-predict")
            p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"grids": cmd_grids, "launch": cmd_launch, "register": cmd_register,
     "table": cmd_table, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
