#!/usr/bin/env python
"""Stage the trained competitor checkpoints (5 seeds each) for the public Hub repo.

WHY (user, 2026-09-25). Others should be able to re-evaluate our competitor baselines without
retraining them. Layout (no README, no cards -- user decision):

    <competitor>/<benchmark>/seed_<N>/<checkpoint + the files its loader reads>

competitor  txpert | state | perturbdiff | cellflow
benchmark   rhj_fewshot | norman_combo   (h1_seen / k562_seen / k562_seen_h1set are scored with the
                                         rhj_fewshot checkpoints -- no separate training)

The checkpoint of each (arm, seed) is read from the export manifest
(runs/competitors/<arm>/s<seed>/split_0/manifest.json: `checkpoint`, or `run_dir` for CellFlow),
i.e. exactly the file that produced the reported scores (validation-selected, never `last`).
Staged as symlinks; the uploader follows them.

Per folder, besides the checkpoint:
  txpert       config-txpert-occam.yaml
  state        config.yaml, pert_onehot_map.pt, var_dims.pkl, data_module.torch,
               cell_type_onehot_map.torch, batch_onehot_map.torch
  perturbdiff  rawdata_diffusion_sampling.yaml, manifest occam_<bench>.toml, the GenePT table
  cellflow     CellFlow.pkl, pca/, genes.txt, config.yaml, run_meta.json  (--include-cellflow;
               ⚠️ the pickle embeds the training AnnData, 18.9 GB per rhj seed)

USAGE  python scripts/stage_competitor_ckpts.py [--include-cellflow] [--out data/release/competitor_ckpts]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (42, 43, 44, 45, 46)
# (competitor, benchmark) -> export arm dir under runs/competitors
ARMS = {("txpert", "rhj_fewshot"): "txpert_single", ("txpert", "norman_combo"): "txpert__norman_combo",
        ("state", "rhj_fewshot"): "state", ("state", "norman_combo"): "state__norman_combo",
        ("perturbdiff", "rhj_fewshot"): "perturbdiff_genept",
        ("perturbdiff", "norman_combo"): "perturbdiff__norman_combo",
        ("cellflow", "rhj_fewshot"): "cellflow", ("cellflow", "norman_combo"): "cellflow__norman_combo"}
GENEPT = {"rhj_fewshot": "data/assets/genept_occam_rhj_fewshot.pkl",
          "norman_combo": "data/assets/genept_occam_norman_combo.pkl"}


def manifest(arm, seed):
    for sp in ("split_0", "split_00"):
        p = os.path.join(REPO, "runs/competitors", arm, f"s{seed}", sp, "manifest.json")
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
    raise FileNotFoundError(f"no export manifest for {arm} s{seed}")


def files_for(comp, bench, m):
    """[(source path, name in the folder)] for one checkpoint folder."""
    if comp == "cellflow":
        rd = m["run_dir"]
        return [(os.path.join(rd, n), n) for n in ("CellFlow.pkl", "pca", "genes.txt",
                                                   "config.yaml", "run_meta.json")]
    ck = m["checkpoint"]
    out = [(ck, "checkpoint.ckpt")]
    if comp == "txpert":
        out.append((os.path.join(REPO, "competitors/TxPert/configs/config-txpert-occam.yaml"),
                    "config-txpert-occam.yaml"))
    elif comp == "state":
        rd = os.path.dirname(os.path.dirname(ck))
        out += [(os.path.join(rd, n), n) for n in (
            "config.yaml", "pert_onehot_map.pt", "var_dims.pkl", "data_module.torch",
            "cell_type_onehot_map.torch", "batch_onehot_map.torch")]
    elif comp == "perturbdiff":
        out += [(os.path.join(REPO, "competitors/PerturbDiff/configs/rawdata_diffusion_sampling.yaml"),
                 "rawdata_diffusion_sampling.yaml"),
                (os.path.join(REPO, f"competitors/PerturbDiff/configs/manifests/occam_{bench}.toml"),
                 f"occam_{bench}.toml"),
                (os.path.join(REPO, GENEPT[bench]), os.path.basename(GENEPT[bench]))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "data/release/competitor_ckpts"))
    ap.add_argument("--include-cellflow", action="store_true")
    args = ap.parse_args()
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    total = 0
    for (comp, bench), arm in ARMS.items():
        if comp == "cellflow" and not args.include_cellflow:
            continue
        for s in SEEDS:
            m = manifest(arm, s)
            if "__h1_seen" in json.dumps(m) or "/archive/" in json.dumps(m):
                raise RuntimeError(f"{arm} s{s}: manifest points outside the live runs: {m}")
            d = os.path.join(args.out, comp, bench, f"seed_{s}")
            os.makedirs(d)
            for src, name in files_for(comp, bench, m):
                if not os.path.exists(src):
                    raise FileNotFoundError(f"{comp}/{bench}/seed_{s}: missing {src}")
                os.symlink(src, os.path.join(d, name))
                if os.path.isfile(src):
                    total += os.path.getsize(src)
                else:
                    total += sum(os.path.getsize(os.path.join(r, f))
                                 for r, _, fs in os.walk(src) for f in fs)
    # STATE's weights are derived from Arc's code and fall under its non-commercial terms: its
    # upstream licence files go at the top of the state/ folder (user, 2026-09-25).
    lic = os.path.join(REPO, "competitors/STATE/licenses")
    for n in ("LICENSE", "MODEL_LICENSE.md", "MODEL_ACCEPTABLE_USE_POLICY.md"):
        os.symlink(os.path.join(lic, n), os.path.join(args.out, "state", n))
    print(f"staged -> {os.path.relpath(args.out, REPO)}: {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
