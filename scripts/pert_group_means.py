#!/usr/bin/env python
"""Per-(cell line, perturbation) MEAN profiles, in model input space, on the encoder axis.

WHY THIS EXISTS
    Three training terms read a group-mean table and CANNOT RUN WITHOUT IT:
      * `delta_mag_weight` -- the magnitude of the predicted effect against the group truth
      * `delta_cos_weight` (DCOS) -- the DIRECTION of the effect
      * `target_pb_k != 1`  -- pseudobulk targets
    `config.pert_mean_dir` and `stream_v2.load_group_means` both name this script, but it was
    never ported into this repo. The consequence was not a crash but SILENCE: the loader is
    gated, so a term whose inputs were missing evaluated to a constant zero and the run looked
    successful. Three DCOS arms at weights 0.25/0.5/1.0 produced bit-identical curves before
    this was noticed (F-loss1). `pipeline.py` now refuses to train in that situation.

WHAT IT WRITES
    cache/pert_means/<source>.npz
        keys        (G,)          "<cell_line>|<pert>"
        means       (G, n_enc)    the group's mean profile, log1p model space, float32
        n_cells     (G,)
        group_line  (G,)          index into ctrl_keys -- this group's control mean
        ctrl_keys   (L,)          cell lines
        ctrl_means  (L, n_enc)    that line's control mean
        gene_mask   (n_enc,)      genes present in this file

⚠️ float32 DELIBERATELY. The perturbation delta is ~5e-4 per gene and fp16's step at log1p~4
is ~2e-3, which would swamp the very quantity these terms are built to supervise.

⚠️ THE SPACE MUST MATCH TRAINING. `target_sum` is read from the config (2000.0 for
`rhj_fewshot`, pinned to the checkpoint), and transforms are applied the same way the streaming
loader applies them. A table built at a different target_sum yields a delta the model can never
match, and nothing downstream would report the mismatch.
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


def build_one(path: str, genes: list, target_sum: float, out_npz: str,
              *, gene_col: str, pert_col: str, batch_col: str,
              limit_chunks: int = 0) -> dict:
    from occam.vcell.dataset import _load_file_metadata
    from occam.vcell.stream_v2 import densify_csr, load_chunk_sparse, make_col_map

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g2i = {g: i for i, g in enumerate(genes)}
    fm = _load_file_metadata(path, genes, g2i, gene_name_col=gene_col,
                             pert_gene_col=pert_col, batch_col=batch_col)

    is_c = np.asarray(fm.is_control, dtype=bool)
    lines = np.asarray(fm.obs_cell_lines, dtype=str)
    perts = np.asarray(fm.pert_gene_names, dtype=str)
    # controls collapse into one group per line; that group becomes the line's control mean
    key = np.where(is_c,
                   np.char.add(lines, "|__CTRL__"),
                   np.char.add(np.char.add(lines, "|"), perts))
    uniq, gid = np.unique(key, return_inverse=True)
    G, n_genes = len(uniq), len(genes)

    col_map = make_col_map(fm.n_vars, fm.src_indices, fm.dst_indices)
    sums = torch.zeros(G, n_genes, dtype=torch.float32, device=dev)
    counts = torch.zeros(G, dtype=torch.float64, device=dev)
    gid_t = torch.from_numpy(gid.astype(np.int64))

    n_chunks = (fm.n_obs + CHUNK - 1) // CHUNK
    todo = range(n_chunks if limit_chunks <= 0 else min(n_chunks, limit_chunks))
    for ci in todo:
        s, e = ci * CHUNK, min(fm.n_obs, (ci + 1) * CHUNK)
        v, c, rp, rs = load_chunk_sparse(path, s, e, fm.n_vars, np.arange(e - s), col_map)
        x = densify_csr(torch.from_numpy(v).to(dev), torch.from_numpy(c).to(dev),
                        torch.from_numpy(rp).to(dev), torch.from_numpy(rs).to(dev),
                        n_genes, target_sum, False, False)
        g = gid_t[s:e].to(dev)
        sums.index_add_(0, g, x)
        counts.index_add_(0, g, torch.ones(e - s, dtype=torch.float64, device=dev))
        del x

    cnt = counts.cpu().numpy()
    means = (sums / counts.clamp(min=1).unsqueeze(1).float()).cpu().numpy()
    line_of = np.array([u.split("|")[0] for u in uniq])
    pert_of = np.array([u.split("|", 1)[1] for u in uniq])
    is_ctrl_grp = pert_of == "__CTRL__"
    ctrl_keys, ctrl_means = line_of[is_ctrl_grp], means[is_ctrl_grp]
    keep = (~is_ctrl_grp) & (cnt > 0)

    ck_pos = {k: i for i, k in enumerate(ctrl_keys)}
    grp_line = np.array([ck_pos.get(l, -1) for l in line_of[keep]], dtype=np.int64)
    if (grp_line < 0).any():
        # a perturbed group whose line has no controls has no reference to take a delta
        # against; the loss guard would silently skip those rows instead.
        bad = sorted(set(line_of[keep][grp_line < 0]))
        raise SystemExit(
            f"{path}: cell line(s) {bad} have perturbed cells but NO control cells, so their "
            f"delta has no reference. Every delta term would silently drop them.")

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz,
             keys=np.char.add(np.char.add(line_of[keep], "|"), pert_of[keep]),
             means=means[keep].astype(np.float32),
             n_cells=cnt[keep].astype(np.int64),
             group_line=grp_line,
             ctrl_keys=ctrl_keys, ctrl_means=ctrl_means.astype(np.float32),
             gene_mask=np.asarray(fm.gene_mask).astype(bool))
    rec = {"path": path, "groups": int(keep.sum()), "lines": int(is_ctrl_grp.sum()),
           "cells": int(cnt.sum()), "target_sum": target_sum,
           "seconds": round(time.time() - t0, 1)}
    json.dump(rec, open(out_npz + ".done.json", "w"), indent=1)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rhj_fewshot.yaml")
    ap.add_argument("--out-dir", default="cache/pert_means")
    ap.add_argument("--limit-chunks", type=int, default=0)
    a = ap.parse_args(argv)

    from occam.config.loader import load_config
    from occam.vcell.stream_v2 import source_name_of

    # `load_config` returns an OccamConfig; `.raw` is the resolved mapping (with !include
    # expanded), which is what we need -- the typed VC config drops the data block's columns.
    cfg = load_config(a.config).raw
    vc = cfg["stages"]["vcell"]
    data = vc["data"]
    paths = [str(p) for p in data["paths"]]
    target_sum = float(data["target_sum"])

    # The encoder axis, from the checkpoint the stage initialises from -- NOT the panel axis.
    # These means are consumed in MODEL space during training, so they must be on the model's
    # gene axis, and `load_group_means` indexes them positionally.
    ck_path = str(vc["init_from"])
    if not os.path.isabs(ck_path):
        ck_path = str(REPO / ck_path)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    genes = [str(g) for g in ck["gene_names"]]
    print(f"[means] encoder axis: {len(genes)} genes from {os.path.basename(ck_path)}")
    print(f"[means] target_sum {target_sum:g} (must match training)")

    out = []
    for p in paths:
        name = source_name_of(p)
        dest = os.path.join(a.out_dir, f"{name}.npz")
        print(f"[means] {name}: {p}")
        rec = build_one(p, genes, target_sum, dest,
                        gene_col=str(data["gene_col"]), pert_col=str(data["pert_col"]),
                        batch_col=str(data.get("batch_col") or "") or None,
                        limit_chunks=a.limit_chunks)
        print(f"[means]   {rec['groups']} groups · {rec['lines']} line(s) · "
              f"{rec['cells']:,} cells · {rec['seconds']}s -> {dest}")
        out.append(rec)
    print(f"[means] wrote {len(out)} table(s) to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
