"""Reading a panel off disk, and asserting its two control pools are disjoint.

Ported from the research tree's held-out-context evaluation module. Behaviour
is preserved; what changed is that the second, count-space scorer and its
emission machinery are gone -- Occam scores with cell-eval v1 only -- and the
names are no longer dataset-specific, because the panel builder is
corpus-agnostic. See docs/evaluation.md.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONTROL = "non-targeting"

import torch

class Panel:
    """The H1 held-out panel, aligned onto the encoder's gene vocabulary.

    Attributes
    ----------
    real_X_model : (n_real, n_keep) float32
        `panel_real.h5ad` preprocessed exactly as model input. The cell-eval v1 TARGET.
    real_labels : (n_real,) str
        Perturbation per real cell, `non-targeting` for controls.
    ctrl_X_model : (n_ctrl, n_keep) float32
        Model-visible controls, preprocessed as model input. The only thing the model sees.
    enc_cols : (n_keep,) int64
        Encoder-vocabulary column index of each kept gene.
    panel_cols : (n_keep,) int64
        Position of each kept gene on the panel's own 18,080-symbol axis.
    panel_genes : list[str]
        The panel's full gene axis, before restriction to the encoder-reachable genes.
    """

    def __init__(self, real_X_model, real_labels, real_counts, ctrl_X_model, ctrl_counts,
                 ctrl_batches, enc_cols, panel_cols, panel_genes, n_enc):
        self.real_X_model = real_X_model
        self.real_labels = real_labels
        self.real_counts = real_counts
        self.ctrl_X_model = ctrl_X_model
        self.ctrl_counts = ctrl_counts
        self.ctrl_batches = ctrl_batches
        self.enc_cols = enc_cols
        self.panel_cols = panel_cols
        self.panel_genes = panel_genes
        self.n_enc = n_enc

    @property
    def n_keep(self) -> int:
        return len(self.enc_cols)


def _read_h5ad_raw(path, obs_cols: Optional[List[str]] = None, rows=None,
                   obs_only: bool = False):
    """Read an .h5ad with h5py directly, bypassing anndata entirely.

    The H1 panel is produced by the VCCv2 venv (anndata 0.13.2) while training runs in the
    `occam` env (anndata 0.9.2), and 0.9.2 cannot read 0.13's obs `_index` encoding:

        AnnDataReadError: ... while reading key '/obs/_index'

    Rather than pin versions across two environments that exist precisely because they
    cannot be unified, read the parts we need from the HDF5 directly. The on-disk layout is
    stable and simple: CSR triplet under /X, categorical columns as {categories, codes}.

    Args:
        path: .h5ad file.
        obs_cols: obs columns to decode (categorical or plain), plus always `_index`.
        rows: optional sorted row indices of X to read (all rows when None).
        obs_only: skip X entirely and return the FULL obs. Use this to pick rows before
            paying for the counts; passing `rows=[]` would empty obs as well.

    Returns:
        (X dense float32, {col: np.ndarray of str}, var_names list[str])
    """
    import h5py

    with h5py.File(str(path), "r") as h:
        xg = h["X"]
        if obs_only:
            n_var = (int(xg.attrs["shape"][1]) if isinstance(xg, h5py.Group)
                     else int(xg.shape[1]))
            X = np.zeros((0, n_var), dtype=np.float32)
        elif isinstance(xg, h5py.Group):                     # sparse CSR
            shape = tuple(int(v) for v in xg.attrs["shape"])
            enc = xg.attrs.get("encoding-type", "csr_matrix")
            if enc != "csr_matrix":
                raise ValueError(f"{path}: unsupported X encoding {enc!r}")
            indptr = xg["indptr"][:]
            sel = np.arange(shape[0]) if rows is None else np.asarray(rows, dtype=np.int64)
            X = np.zeros((len(sel), shape[1]), dtype=np.float32)
            data, indices = xg["data"], xg["indices"]
            for i, r in enumerate(sel):
                s, e = int(indptr[r]), int(indptr[r + 1])
                if e > s:
                    X[i, indices[s:e]] = data[s:e]
        else:                                                # dense
            X = (xg[:] if rows is None else xg[np.asarray(rows)]).astype(np.float32)

        obs_rows = None if obs_only else rows
        _SENTINEL = object()

        def _as_str(a):
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])

        def _col(grp, name, idx=_SENTINEL):
            """Decode one obs/var column, dispatching on anndata's encoding-type attribute.

            Group-vs-Dataset is not enough to tell these apart: anndata 0.13 writes the
            index as a `nullable-string-array` GROUP of {values, mask}, which has no
            `categories` member. That encoding is precisely what anndata 0.9.2 cannot read,
            so it has to be handled explicitly rather than assumed categorical.
            """
            d = grp[name]
            enc = d.attrs.get("encoding-type")
            enc = enc.decode() if isinstance(enc, bytes) else enc
            if idx is _SENTINEL:
                idx = None if obs_rows is None else np.asarray(obs_rows, dtype=np.int64)

            if enc == "categorical":
                cats = _as_str(d["categories"][:]).astype(object)
                codes = d["codes"][:]
                if idx is not None:
                    codes = codes[idx]
                return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "").astype(str)

            if isinstance(d, h5py.Group):
                # nullable-{string,integer,boolean}-array: {values, mask}; mask marks NA
                if "values" not in d:
                    raise ValueError(f"{path}: obs/{name} has encoding {enc!r} with "
                                     f"members {list(d.keys())}, which is not supported")
                v = d["values"][:]
                out = _as_str(v if idx is None else v[idx])
                if "mask" in d:
                    m = d["mask"][:]
                    out = np.where(m if idx is None else m[idx], "", out)
                return out

            v = d[:] if idx is None else d[idx]
            return _as_str(v)

        obs = {}
        for name in set(list(obs_cols or []) + ["_index"]):
            if name in h["obs"]:
                obs[name] = _col(h["obs"], name)
        vkey = h["var"].attrs.get("_index", "_index")
        vkey = vkey.decode() if isinstance(vkey, bytes) else vkey
        var_names = _col(h["var"], vkey, idx=None).tolist()
    return X, obs, var_names


def _preprocess(raw: np.ndarray, target_sum: float,
                skip_normalize: bool, skip_log1p: bool) -> np.ndarray:
    """NormalizePerCell -> Log1p, honouring the encoder checkpoint's skip flags."""
    x = np.asarray(raw, dtype=np.float32)
    if not skip_normalize:
        s = x.sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        x = x / s * target_sum
    if not skip_log1p:
        x = np.log1p(x)
    return x.astype(np.float32, copy=False)


def assert_disjoint_controls(real_obs, ctrl_obs, real_counts, ctrl_counts) -> str:
    """Prove the scoring controls and the model-visible controls share no cells.

    This is the one leak that cannot be detected downstream: if the model sees
    the same control cells that the scoring reference is built from, every
    control-derived arm is scored against cells it was constructed from, and
    the result looks good for the wrong reason.

    ⚠️ The research tree's check here was **vacuous**. It compared
    ``set(real_obs["_index"]) & set(ctrl_obs["_index"])``, but the two files were
    written by different scripts using different index prefixes -- one wrote
    ``H1_{i}`` and the other ``H1_modelctrl_{i}`` -- so the intersection was
    empty regardless of whether the same cells were in both. It could not fail.
    (The panel it guarded was in fact clean, verified independently by row
    hashing: 18,400 unique hashes per side, intersection 0. But that was a
    property of how it happened to be built, not something the loader checked.)

    Three tiers, hard-failing on the first that is available:

    1. **Provenance.** If both files carry ``source_file`` + ``source_row``,
       compare those pairs. Authoritative, and O(n). ``build_panel.py`` always
       writes them.
    2. **Row hash.** blake2b over each control row's raw values. Catches a
       genuine duplicate whatever the obs index says. Sound only because
       ``build_panel.py`` applies depth thinning ONCE, before the split -- if
       the two halves were thinned with different RNG streams, a leaked cell
       would hash differently on each side.
    3. **Refuse.** Never fall back to comparing obs indices.

    Returns the method used, for the record.
    """
    import hashlib

    n_real, n_ctrl = len(real_obs.get("_index", ())), len(ctrl_obs.get("_index", ()))

    # Tier 1: provenance
    if "source_file" in real_obs and "source_row" in real_obs \
            and "source_file" in ctrl_obs and "source_row" in ctrl_obs:
        real_ids = set(zip(real_obs["source_file"], real_obs["source_row"]))
        ctrl_ids = set(zip(ctrl_obs["source_file"], ctrl_obs["source_row"]))
        shared = real_ids & ctrl_ids
        if shared:
            raise ValueError(
                f"{len(shared)} source cells appear in BOTH the scoring panel and the "
                f"model-visible control pool. Those pools must be disjoint. "
                f"Rebuild with scripts/build_panel.py, which splits controls 2x-then-half."
            )
        print(f"[panel] control pools disjoint: {n_real} scoring vs {n_ctrl} model-visible "
              f"(verified by source provenance)")
        return "provenance"

    # Tier 2: row hash
    def hashes(counts):
        out = set()
        for i in range(counts.shape[0]):
            row = counts[i]
            if hasattr(row, "toarray"):
                row = row.toarray().ravel()
            row = np.asarray(row).ravel()
            nz = np.nonzero(row)[0]
            h = hashlib.blake2b(digest_size=16)
            h.update(nz.astype(np.int32).tobytes())
            h.update(row[nz].astype(np.float32).tobytes())
            out.add(h.digest())
        return out

    real_ctrl_mask = np.asarray(
        [str(v) == CONTROL for v in real_obs.get("target_gene", [])], dtype=bool)
    if real_ctrl_mask.any():
        real_h = hashes(real_counts[real_ctrl_mask])
        ctrl_h = hashes(ctrl_counts)
        shared = real_h & ctrl_h
        if shared:
            raise ValueError(
                f"{len(shared)} control rows are byte-identical between the scoring panel "
                f"and the model-visible pool. Rebuild with scripts/build_panel.py."
            )
        print(f"[panel] control pools disjoint: {len(real_h)} scoring vs {len(ctrl_h)} "
              f"model-visible (verified by row hash)")
        return "rowhash"

    raise ValueError(
        "cannot prove the two control pools are disjoint: the panel carries neither "
        "source provenance (source_file/source_row) nor any control rows to hash.\n"
        "Rebuild it with scripts/build_panel.py, which records provenance.\n"
        "Comparing obs indices is NOT an acceptable fallback -- panels written by "
        "different scripts use different index prefixes, so that comparison can never "
        "fail and proves nothing."
    )


def load_panel(config, encoder_gene_names: List[str],
                  skip_normalize: bool, skip_log1p: bool) -> Panel:
    """Read the panel and map its gene symbols onto the encoder's Ensembl vocabulary.

    The encoder vocabulary is 44,728 entries, 44,560 of them Ensembl IDs; the panel axis is
    18,080 HGNC symbols. Raw string overlap is 19 genes, so the symbol->Ensembl map is
    mandatory — 18,066 of the 18,080 land in the vocabulary through it (18,068 counting the
    direct hits). Genes that do not map are dropped from scoring and reported.
    """
    panel_dir = Path(config.panel_dir)
    real_path = Path(config.panel_real_path) if config.panel_real_path else panel_dir / "panel_real.h5ad"
    ctrl_path = (Path(config.panel_model_controls_path) if config.panel_model_controls_path
                 else panel_dir / "panel_controls_model.h5ad")
    for p in (real_path, ctrl_path):
        if not p.exists():
            raise FileNotFoundError(
                f"H1 panel file missing: {p}\n"
                f"Build it with:  VCCv2/.venv/bin/python VCCv2/scripts/build_h1_panel.py "
                f"--out-dir {panel_dir} --thin-to-vcc\n"
                f"then:           VCCv2/.venv/bin/python VCCv2/scripts/make_model_controls.py "
                f"--out-dir {panel_dir}")

    print(f"[panel] real (scoring)      : {real_path}")
    print(f"[panel] controls (model-visible): {ctrl_path}")

    real_counts, real_obs, panel_genes = _read_h5ad_raw(
        real_path, [config.panel_pert_col])
    ctrl_counts, ctrl_obs, ctrl_genes = _read_h5ad_raw(
        ctrl_path, [config.panel_pert_col, "dataset_batch_num", "batch"])
    if ctrl_genes != panel_genes:
        raise ValueError("control pool and panel_real disagree on the gene axis")

    assert_disjoint_controls(real_obs, ctrl_obs, real_counts, ctrl_counts)

    # --- gene alignment: symbol -> Ensembl -> encoder column ------------------------
    if not Path(config.panel_symbol_map).exists():
        print(f"[panel] WARNING: symbol map {config.panel_symbol_map} not found — direct matching "
              f"only, which reaches 19/18,080 genes")
    enc_cols, panel_cols = _align_to_encoder(panel_genes, encoder_gene_names, config.panel_symbol_map)
    print(f"[panel] genes: {len(enc_cols)}/{len(panel_genes)} reachable in the encoder vocabulary "
          f"({len(panel_genes) - len(enc_cols)} dropped from scoring)")

    # Preprocessing is done on the FULL panel axis (library size must be the cell's real
    # total), then the kept columns are taken. Subsetting first would renormalise against a
    # truncated library and silently change every value.
    real_X = _preprocess(real_counts, config.target_sum, skip_normalize, skip_log1p)[:, panel_cols]
    ctrl_X = _preprocess(ctrl_counts, config.target_sum, skip_normalize, skip_log1p)[:, panel_cols]

    labels = real_obs[config.panel_pert_col].astype(str)
    labels = np.where(labels == config.panel_control_label, CONTROL, labels)

    bcol = next((c for c in ("dataset_batch_num", "batch", "gem_group")
                 if c in ctrl_obs), None)
    ctrl_batches = (ctrl_obs[bcol].astype(str) if bcol
                    else np.array(["_"] * ctrl_counts.shape[0]))
    if bcol:
        print(f"[panel] control batch column: {bcol} ({len(set(ctrl_batches))} batches)")

    n_pert_cells = int((labels != CONTROL).sum())
    print(f"[panel] panel {real_counts.shape}: {n_pert_cells} perturbed + "
          f"{int((labels == CONTROL).sum())} scoring controls; "
          f"{ctrl_X.shape[0]} model-visible controls")

    return Panel(real_X, labels, real_counts, ctrl_X, ctrl_counts, ctrl_batches,
                   enc_cols, panel_cols, panel_genes, len(encoder_gene_names))


def _align_to_encoder(panel_genes: List[str], encoder_gene_names: List[str],
                      symbol_map_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Map panel gene symbols onto encoder-vocabulary columns via the symbol->Ensembl map."""
    import pandas as pd

    enc_idx = {g: i for i, g in enumerate(encoder_gene_names)}
    sym2ens: Dict[str, str] = {}
    # An empty or blank setting means "no map", not Path("") -- which is `.`, and
    # `Path(".").exists()` is True, so the next line would read_csv a directory.
    p = Path(str(symbol_map_path).strip()) if str(symbol_map_path).strip() else None
    if p is not None and p.is_file():
        m = pd.read_csv(p)
        sym2ens = dict(zip(m.gene_symbol.astype(str), m.ensembl_gene_id.astype(str)))
    enc_cols, panel_cols = [], []
    for j, g in enumerate(panel_genes):
        col = enc_idx.get(g)
        if col is None:
            col = enc_idx.get(sym2ens.get(g, "\0"))
        if col is not None:
            enc_cols.append(col)
            panel_cols.append(j)
    if len(enc_cols) != len(set(enc_cols)):
        raise ValueError("symbol->Ensembl map is not 1:1 over this panel")
    return np.asarray(enc_cols, np.int64), np.asarray(panel_cols, np.int64)
