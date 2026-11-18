#!/usr/bin/env python
"""Per-(cell line, perturbation) DISPERSION statistics, in model input space, on the encoder axis.

WHY THIS EXISTS
    F-disp1 measured that our predicted cells carry 2.6% of the real within-perturbation
    dispersion (0.0026 vs 0.1009 mean per-gene std, 568/568 perturbations), which inflates
    FDR-significant gene calls 30.8x. Three training terms are built to correct that, and
    each needs a REAL target to match -- which is what this table is:

      * `gene_var_weight`      per-gene within-group variance      -> `gvar`
      * `radial_spread_weight` spread of ||x - ctrl_mean||          -> `rad_mean`, `rad_std`
      * `dcos_conc_weight`     concentration of effect DIRECTIONS   -> `cos_mean`, `cos_std`

    It is the sibling of `scripts/pert_group_means.py` and DEPENDS on its output: the means
    and control means define the reference each dispersion statistic is measured around, and
    the group key order is copied from it so `load_group_dispersion` can align row-for-row
    with `load_group_means` without re-deriving anything.

WHAT IT WRITES
    cache/pert_disp/<source>.npz
        keys       (G,)         "<cell_line>|<pert>" -- SAME ORDER as cache/pert_means/<source>.npz
        gvar       (G, n_enc)   per-gene within-group variance, log1p model space, float32
        rad_mean   (G,)         mean over cells of ||(x - ctrl_mean) * gene_mask||
        rad_std    (G,)         std  over cells of the same
        cos_mean   (G,)         mean over cells of cos(x - ctrl_mean, group_mean - ctrl_mean)
        cos_std    (G,)         std  over cells of the same
        n_cells    (G,)
        gene_mask  (n_enc,)

⚠️ THE REFERENCE MATTERS. `gvar` is the spread around the group's OWN mean (what a variance
term should match). `rad_*` and `cos_*` are around the LINE CONTROL MEAN, because that is the
reference `delta_mag` and `dcos` are already defined against -- mixing the two references would
make the new terms incomparable with the existing ones.

⚠️ float32 and the same `target_sum` as training, for the reason pert_group_means.py gives: the
per-gene effect is ~5e-4 and fp16's step at log1p~4 is ~2e-3.

⚠️ GROUPS WITH n_cells < 2 have no defined dispersion. They are kept as rows (so the indexing
stays aligned) with their statistics set to NaN, and the loss MUST drop them by n_cells.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CHUNK = 8192


def build_one(path: str, genes: list, target_sum: float, means_npz: str, out_npz: str,
              *, gene_col: str, pert_col: str, batch_col: str,
              limit_chunks: int = 0) -> dict:
    from occam.vcell.dataset import _load_file_metadata
    from occam.vcell.stream_v2 import densify_csr, load_chunk_sparse, make_col_map

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(means_npz, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    gmeans = torch.from_numpy(np.asarray(z["means"], dtype=np.float32)).to(dev)
    cmeans = torch.from_numpy(np.asarray(z["ctrl_means"], dtype=np.float32)).to(dev)
    gline = torch.from_numpy(np.asarray(z["group_line"], dtype=np.int64)).to(dev)
    gmask = torch.from_numpy(np.asarray(z["gene_mask"], dtype=bool)).to(dev)
    key_pos = {k: i for i, k in enumerate(keys)}
    G, n_genes = len(keys), len(genes)

    g2i = {g: i for i, g in enumerate(genes)}
    fm = _load_file_metadata(path, genes, g2i, gene_name_col=gene_col,
                             pert_gene_col=pert_col, batch_col=batch_col)
    is_c = np.asarray(fm.is_control, dtype=bool)
    lines = np.asarray(fm.obs_cell_lines, dtype=str)
    perts = np.asarray(fm.pert_gene_names, dtype=str)
    row_key = np.char.add(np.char.add(lines, "|"), perts)
    # -1 for control rows and for any row whose group the means table dropped (n_cells == 0)
    row_gid = np.array([-1 if c else key_pos.get(k, -1) for k, c in zip(row_key, is_c)],
                       dtype=np.int64)
    if (row_gid >= 0).sum() == 0:
        raise SystemExit(f"{path}: no perturbed row maps into {means_npz}; the key convention "
                         f"or the axis must have changed. Refusing to write an empty table.")

    col_map = make_col_map(fm.n_vars, fm.src_indices, fm.dst_indices)
    sumsq = torch.zeros(G, n_genes, dtype=torch.float32, device=dev)
    cnt = torch.zeros(G, dtype=torch.float64, device=dev)
    r_s = torch.zeros(G, dtype=torch.float64, device=dev)
    r_ss = torch.zeros(G, dtype=torch.float64, device=dev)
    c_s = torch.zeros(G, dtype=torch.float64, device=dev)
    c_ss = torch.zeros(G, dtype=torch.float64, device=dev)
    gid_t = torch.from_numpy(row_gid)
    gmf = gmask.float()
    # the true group delta, masked -- the direction every cell's own delta is projected onto
    dt = (gmeans - cmeans[gline]) * gmf.unsqueeze(0)
    dt_n = dt.norm(dim=1).clamp(min=1e-8)

    n_chunks = (fm.n_obs + CHUNK - 1) // CHUNK
    todo = range(n_chunks if limit_chunks <= 0 else min(n_chunks, limit_chunks))
    for ci in todo:
        s, e = ci * CHUNK, min(fm.n_obs, (ci + 1) * CHUNK)
        g = gid_t[s:e]
        keep = g >= 0
        if not bool(keep.any()):
            continue
        v, c, rp, rs = load_chunk_sparse(path, s, e, fm.n_vars, np.arange(e - s), col_map)
        x = densify_csr(torch.from_numpy(v).to(dev), torch.from_numpy(c).to(dev),
                        torch.from_numpy(rp).to(dev), torch.from_numpy(rs).to(dev),
                        n_genes, target_sum, False, False)
        gk = g[keep].to(dev)
        xk = x[keep.to(dev)]
        # per-gene second moment about the group mean
        d_mu = xk - gmeans[gk]
        sumsq.index_add_(0, gk, d_mu * d_mu)
        # radial and directional statistics about the LINE CONTROL MEAN
        dcell = (xk - cmeans[gline[gk]]) * gmf.unsqueeze(0)
        r = dcell.norm(dim=1)
        cos = (dcell * dt[gk]).sum(dim=1) / (r.clamp(min=1e-8) * dt_n[gk])
        ones = torch.ones(gk.numel(), dtype=torch.float64, device=dev)
        cnt.index_add_(0, gk, ones)
        r_s.index_add_(0, gk, r.double())
        r_ss.index_add_(0, gk, (r * r).double())
        c_s.index_add_(0, gk, cos.double())
        c_ss.index_add_(0, gk, (cos * cos).double())
        del x, xk, d_mu, dcell, r, cos

    n = cnt.clamp(min=1)
    # population variance about the stored mean; clamp the fp32 cancellation floor at 0
    gvar = (sumsq / n.unsqueeze(1).float()).clamp(min=0.0)
    r_mean = (r_s / n)
    r_var = (r_ss / n - r_mean ** 2).clamp(min=0.0)
    c_mean = (c_s / n)
    c_var = (c_ss / n - c_mean ** 2).clamp(min=0.0)
    nc = cnt.cpu().numpy().astype(np.int64)
    # n < 2 -> no dispersion is defined. NaN, not 0: a 0 would read as "this group is a point
    # mass", which is exactly the claim the terms are built to falsify.
    bad = nc < 2
    out = {k: v.cpu().numpy() for k, v in
           {"gvar": gvar, "rad_mean": r_mean.float(), "rad_std": r_var.sqrt().float(),
            "cos_mean": c_mean.float(), "cos_std": c_var.sqrt().float()}.items()}
    for k in ("rad_mean", "rad_std", "cos_mean", "cos_std"):
        out[k] = out[k].astype(np.float32)
        out[k][bad] = np.nan
    out["gvar"] = out["gvar"].astype(np.float32)
    out["gvar"][bad] = np.nan

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, keys=np.asarray(keys), n_cells=nc,
             gene_mask=np.asarray(z["gene_mask"]).astype(bool), **out)
    ok = ~bad
    rec = {"path": path, "groups": int(G), "groups_with_dispersion": int(ok.sum()),
           "cells": int(nc.sum()), "target_sum": target_sum,
           "mean_per_gene_std": float(np.sqrt(out["gvar"][ok][:, np.asarray(z["gene_mask"], dtype=bool)]).mean()),
           "mean_rad": float(np.nanmean(out["rad_mean"][ok])),
           "mean_rad_std": float(np.nanmean(out["rad_std"][ok])),
           "mean_cos": float(np.nanmean(out["cos_mean"][ok])),
           "mean_cos_std": float(np.nanmean(out["cos_std"][ok])),
           "seconds": round(time.time() - t0, 1)}
    json.dump(rec, open(out_npz + ".done.json", "w"), indent=1)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rhj_fewshot.yaml")
    ap.add_argument("--means-dir", default="cache/pert_means")
    ap.add_argument("--out-dir", default="cache/pert_disp")
    ap.add_argument("--limit-chunks", type=int, default=0)
    a = ap.parse_args(argv)

    from occam.config.loader import load_config
    from occam.vcell.stream_v2 import source_name_of

    cfg = load_config(a.config).raw
    vc = cfg["stages"]["vcell"]
    data = vc["data"]
    paths = [str(p) for p in data["paths"]]
    target_sum = float(data["target_sum"])
    ck_path = str(vc["init_from"])
    if not os.path.isabs(ck_path):
        ck_path = str(REPO / ck_path)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    genes = [str(g) for g in ck["gene_names"]]
    print(f"[disp] encoder axis: {len(genes)} genes from {os.path.basename(ck_path)}")
    print(f"[disp] target_sum {target_sum:g} (must match training and the means table)")

    for p in paths:
        name = source_name_of(p)
        means = os.path.join(a.means_dir, f"{name}.npz")
        if not os.path.isfile(means):
            raise SystemExit(f"missing {means} -- run scripts/pert_group_means.py first; this "
                             f"table is defined relative to those means.")
        dest = os.path.join(a.out_dir, f"{name}.npz")
        print(f"[disp] {name}: {p}")
        rec = build_one(p, genes, target_sum, means, dest,
                        gene_col=str(data["gene_col"]), pert_col=str(data["pert_col"]),
                        batch_col=str(data.get("batch_col") or "") or None,
                        limit_chunks=a.limit_chunks)
        print(f"[disp]   {rec['groups_with_dispersion']}/{rec['groups']} groups with n>=2 · "
              f"{rec['cells']:,} cells · {rec['seconds']}s -> {dest}")
        print(f"[disp]   mean per-gene std {rec['mean_per_gene_std']:.4f} · "
              f"||delta|| {rec['mean_rad']:.3f} ± {rec['mean_rad_std']:.3f} · "
              f"cos(cell delta, group delta) {rec['mean_cos']:.4f} ± {rec['mean_cos_std']:.4f}")
    print(f"[disp] wrote tables to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
