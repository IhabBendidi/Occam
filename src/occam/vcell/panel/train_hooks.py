"""Training-time hooks: control-identity samples and the panel validation set.

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

from occam.vcell.panel.panel_io import Panel, _align_to_encoder, _preprocess, _read_h5ad_raw

CONTROL = "non-targeting"

import torch

def build_control_identity_samples(config, encoder_gene_names: List[str],
                              skip_normalize: bool, skip_log1p: bool,
                              mode: str = "exact", n: int = 0, seed: int = 42):
    """H1 control-in / control-out training samples, from the MODEL-VISIBLE pool only.

    This is the one and only way H1 enters training. No H1 perturbation is ever a training
    target; the model sees H1's control state and is asked to reproduce it under the null
    perturbation token. That mirrors the real task exactly — VCC 2026 hands you the unseen
    cell line's non-targeting controls and nothing else — so it is a legitimate use of the
    data rather than a concession.

    Pool discipline (`VCCv2/CLAUDE.md` §0):
      * TRAIN + inference input + calibration -> `panel_controls_model.h5ad`  (pool A)
      * cell-eval injection / scoring         -> the controls inside `panel_real.h5ad` (pool B)
    The two are disjoint by construction and the disjointness is asserted at load time. If
    pool B were trained on, every control-derived arm would be scored against cells it was
    built from.

    Returns `(ctrl, target, names, indices, cell_types, mask)` shaped for PseudobulkDataset,
    on the full 44,728-gene encoder axis with a presence mask over H1's 18,068 genes.
    """
    panel_dir = Path(config.panel_dir)
    ctrl_path = (Path(config.panel_model_controls_path) if config.panel_model_controls_path
                 else panel_dir / "panel_controls_model.h5ad")
    if not ctrl_path.exists():
        raise FileNotFoundError(f"H1 model-visible control pool missing: {ctrl_path}")

    raw, obs, var_names = _read_h5ad_raw(
        ctrl_path, [config.panel_pert_col, "dataset_batch_num"])
    labels = (obs[config.panel_pert_col].astype(str) if config.panel_pert_col in obs
              else np.array([CONTROL] * raw.shape[0]))
    if not np.all(np.isin(labels, [CONTROL, config.panel_control_label])):
        raise ValueError(
            f"{ctrl_path} contains non-control cells; H1 perturbations must never be "
            f"training targets. Labels found: {sorted(set(labels))[:5]}")

    enc_cols, panel_cols = _align_to_encoder(var_names, encoder_gene_names,
                                          config.panel_symbol_map)
    Xm = _preprocess(raw, config.target_sum, skip_normalize, skip_log1p)
    batches = (obs["dataset_batch_num"].astype(str) if "dataset_batch_num" in obs
               else np.array(["_"] * raw.shape[0]))
    del raw

    rng = np.random.default_rng(seed)
    n_pool = Xm.shape[0]
    take = np.arange(n_pool) if n <= 0 or n >= n_pool else np.sort(
        rng.choice(n_pool, size=n, replace=False))

    if mode == "exact":
        src_rows = take
    elif mode == "paired":
        from collections import defaultdict
        by_b = defaultdict(list)
        for i, b in enumerate(batches):
            by_b[b].append(i)
        by_b = {b: np.asarray(v, np.int64) for b, v in by_b.items()}
        src_rows = np.empty_like(take)
        for k, t in enumerate(take):
            pool = by_b.get(batches[t], np.arange(n_pool))
            cand = pool[pool != t]
            src_rows[k] = cand[rng.integers(len(cand))] if len(cand) else t
    else:
        raise ValueError(f"unknown control identity mode {mode!r}")

    n_take, n_enc = len(take), len(encoder_gene_names)
    ctrl = torch.zeros(n_take, n_enc, dtype=torch.float32)
    tgt = torch.zeros(n_take, n_enc, dtype=torch.float32)
    ctrl[:, enc_cols] = torch.from_numpy(Xm[src_rows][:, panel_cols])
    tgt[:, enc_cols] = torch.from_numpy(Xm[take][:, panel_cols])

    gene_mask = torch.zeros(n_enc, dtype=torch.bool)
    gene_mask[torch.from_numpy(enc_cols)] = True
    mask = gene_mask.unsqueeze(0).expand(n_take, -1)

    print(f"  [panel] {n_take} H1 control identity samples (mode={mode}) from {ctrl_path.name}; "
          f"{len(enc_cols)} genes present, null token [-1]")
    return (ctrl, tgt, ["non-targeting"] * n_take, [[-1]] * n_take,
            ["H1"] * n_take, mask)


def build_panel_val_dataset(config, encoder_gene_names: List[str], pert2id: Dict[str, int],
                         skip_normalize: bool, skip_log1p: bool, seed: int = 42):
    """Validation set drawn from H1 itself: real H1 perturbed cells as targets.

    Every other cell line is now entirely in training, so there is no in-domain validation
    split left — and that is deliberate, because in-domain validation loss is not what this
    experiment is trying to maximise. Selecting on H1 selects on the actual objective,
    cross-cell-line transfer.

    ⚠ **This is selection on the evaluation distribution.** By default
    (`h1_val_in_panels=True`) these perturbations REMAIN in the scored panels, so the panel
    stays the full 255 — identical to the one VCCv2 measured both reference ends of, which
    is what keeps the headline comparable to `VCCv2/RUNS.md` and across arms without
    recomputing anchors. The price is an early-stopping bias on ~15% of the scored
    perturbations; it is logged as `h1/coverage/val_perts_in_panel` and must be stated
    wherever the number is quoted. Set `h1_val_in_panels=False` to score the disjoint
    complement instead.

    Input is a pool-A (model-visible) control cell, exactly as at inference. Target is a
    real H1 perturbed cell. One predicted cell per real cell; nothing is averaged.

    Returns `(dataset, val_perts)`.
    """
    from occam.vcell.dataset import PseudobulkDataset

    panel_dir = Path(config.panel_dir)
    real_path = (Path(config.panel_real_path) if config.panel_real_path
                 else panel_dir / "panel_real.h5ad")
    ctrl_path = (Path(config.panel_model_controls_path) if config.panel_model_controls_path
                 else panel_dir / "panel_controls_model.h5ad")

    # obs only first: the label column is all we need to choose rows, and reading 120k x
    # 18k of counts to then discard 99% of it would be wasteful.
    _, real_obs, panel_genes = _read_h5ad_raw(real_path, [config.panel_pert_col], obs_only=True)
    labels = real_obs[config.panel_pert_col].astype(str)
    labels = np.where(labels == config.panel_control_label, CONTROL, labels)
    enc_cols, panel_cols = _align_to_encoder(panel_genes, encoder_gene_names, config.panel_symbol_map)

    perts = sorted({p for p in set(labels) if p != CONTROL and p in pert2id})
    rng = np.random.default_rng(seed)
    frac = float(getattr(config, "panel_val_frac", 0.15))
    n_val = max(1, int(round(len(perts) * frac)))
    val_perts = sorted(rng.choice(perts, size=min(n_val, len(perts)), replace=False).tolist())

    per_pert = int(getattr(config, "panel_val_cells_per_pert", 32))
    rows, tgt_labels = [], []
    for p in val_perts:
        idx = np.where(labels == p)[0]
        take = rng.choice(idx, size=min(per_pert, len(idx)), replace=False)
        rows.append(np.sort(take))
        tgt_labels.extend([p] * len(take))
    rows = np.concatenate(rows)

    tgt_raw, _, _ = _read_h5ad_raw(real_path, rows=rows)
    tgt_full = _preprocess(tgt_raw, config.target_sum, skip_normalize, skip_log1p)
    del tgt_raw

    _, ctrl_obs, _ = _read_h5ad_raw(ctrl_path, obs_only=True)
    n_ctrl = len(ctrl_obs["_index"])
    pick = rng.integers(0, n_ctrl, size=len(rows))
    uniq = np.unique(pick)
    ctrl_raw, _, _ = _read_h5ad_raw(ctrl_path, rows=uniq)
    ctrl_full = _preprocess(ctrl_raw, config.target_sum, skip_normalize, skip_log1p)
    # rows were read in sorted-unique order; map each pairing back to its row
    pos = {int(r): i for i, r in enumerate(uniq)}
    ctrl_full = ctrl_full[[pos[int(r)] for r in pick]]
    del ctrl_raw

    n, n_enc = len(rows), len(encoder_gene_names)
    ctrl = torch.zeros(n, n_enc, dtype=torch.float32)
    tgt = torch.zeros(n, n_enc, dtype=torch.float32)
    ctrl[:, enc_cols] = torch.from_numpy(ctrl_full[:, panel_cols])
    tgt[:, enc_cols] = torch.from_numpy(tgt_full[:, panel_cols])

    gene_mask = torch.zeros(n_enc, dtype=torch.bool)
    gene_mask[torch.from_numpy(enc_cols)] = True

    ds = PseudobulkDataset(
        control_profiles=ctrl,
        perturbed_profiles=tgt,
        pert_gene_names=list(tgt_labels),
        pert_indices=[[pert2id[p]] for p in tgt_labels],
        cell_types=["H1"] * n,
        gene_masks=gene_mask.unsqueeze(0).expand(n, -1),
        batch_ids=None,
    )
    in_panels = bool(getattr(config, "h1_val_in_panels", True))
    print(f"  [panel] validation: {n} single cells over {len(val_perts)}/{len(perts)} "
          f"H1 perturbations ({per_pert}/pert) — SELECTION ON THE EVAL DISTRIBUTION. "
          + ("These perturbations REMAIN in the scored panels (full 255), so the headline "
             "carries an early-stopping bias on ~"
             f"{100 * len(val_perts) / max(len(perts), 1):.0f}% of it."
             if in_panels else
             "These perturbations are excluded from the scored panels."))
    return ds, val_perts
