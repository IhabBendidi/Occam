"""Writing the injctrl pair and running cell-eval v1 over it.

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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from occam.vcell.panel.panel_io import Panel

CONTROL = "non-targeting"

# The eval space: log1p(counts / rowsum_FULL_PANEL * EVAL_TARGET_SUM). Everything cell-eval
# sees is in this space, targets and predictions alike, and the whole competitor bridge is
# built on that equality -- so an emitter MUST return to it.
EVAL_TARGET_SUM = 2000.0


def nb_emit(pred_log1p: np.ndarray, *, theta: np.ndarray, n_per_cell_lib: float,
            rng: np.random.Generator,
            exact_colsum: bool = False,
            labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Negative-binomial COUNT emission, returned in the eval space.

    WHY THIS EXISTS
        Occam writes its continuous log1p profile straight to the h5ad. That profile is the
        conditional MEAN; real cells are counts drawn around a mean, and the DE-based metrics
        (`overlap_at_N`, `precision_at_N`) run a Wilcoxon test whose significance depends on
        WITHIN-GROUP DISPERSION. A mean-only prediction has almost none, which inflates the
        number of genes it calls significant and dilutes the overlap -- the pseudobulk-variance
        artifact documented at length for XCellv2. Emitting counts restores that dispersion.

    ⚠️ THIS IS AN EMISSION MODEL, NOT A BETTER PREDICTION. It adds no information: the mean is
    unchanged in expectation. It changes the *shape* of the predicted distribution so the
    scorer's assumptions hold. Label it as engineering, next to the number.

    THE SPACE, WHICH IS THE PART THAT GOES WRONG
        in :  `pred_log1p`, log1p at `EVAL_TARGET_SUM` (2000) on the FULL-panel denominator
        1.   `mu_2000 = expm1(pred_log1p)`      -- expected value per 2000-count library
        2.   `mu_cell = mu_2000 * lib / 2000`   -- scaled to the emitted library size
        3.   `counts ~ NB(mu_cell, theta)`      -- per-gene dispersion
        4.   (optional) pin each perturbation's column sum -- see `exact_colsum`
        5.   `log1p(counts / counts.sum(1) * 2000)` -- BACK to the eval space
        Step 5 is not optional. Skipping it hands cell-eval a count matrix while the target is
        log1p, and every metric silently becomes a comparison between two different spaces.

    Parameters
    ----------
    theta : (n_genes,) NB dispersion, the `1/alpha` convention -- variance is
        `mu + mu^2 / theta`, so LARGE theta approaches Poisson and small theta is
        over-dispersed. Take it from an NB-trained checkpoint's `log_theta` where one exists;
        a scalar broadcast is the fallback and is a different (weaker) claim.
    n_per_cell_lib : the emitted library size in counts. 2000 keeps the emitted depth equal to
        the eval normalisation, so `mu_cell == mu_2000` and step 2 is a no-op; a larger value
        emits deeper cells with proportionally less sampling noise.
    exact_colsum : pin each perturbation's per-gene COLUMN SUM onto `round(sum_i mu_cell)`
        after sampling. ⚠️ Only meaningful here, downstream of a sampler: the scatter it
        removes is the scatter step 3 introduces. It is variance reduction, not biology --
        `labels` is required so the blocks are the right groups.
    """
    x = np.asarray(pred_log1p, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"pred_log1p must be 2-D, got {x.shape}")
    g = x.shape[1]
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    if th.size == 1:
        th = np.repeat(th, g)
    if th.size != g:
        raise ValueError(f"theta has {th.size} entries for {g} genes")
    if not np.all(th > 0):
        raise ValueError("theta must be strictly positive (it is 1/alpha, not alpha)")

    mu = np.expm1(x)                                  # per-EVAL_TARGET_SUM expectation
    np.clip(mu, 0.0, None, out=mu)
    mu_cell = mu * (float(n_per_cell_lib) / EVAL_TARGET_SUM)

    # NB as a Gamma-Poisson mixture: the standard, and it keeps theta's meaning explicit.
    #   lam ~ Gamma(shape=theta, scale=mu/theta)   ->  E[lam]=mu, Var[lam]=mu^2/theta
    #   k   ~ Poisson(lam)                         ->  Var[k]=mu + mu^2/theta
    shape = np.broadcast_to(th, mu_cell.shape)
    lam = rng.gamma(shape=shape, scale=np.divide(mu_cell, shape, where=shape > 0))
    counts = rng.poisson(lam).astype(np.int64)

    if exact_colsum:
        if labels is None:
            raise ValueError(
                "exact_colsum needs `labels`: the column sum is pinned PER PERTURBATION, so "
                "without the grouping it would pin the whole matrix as one block.")
        counts = _pin_colsum(counts, mu_cell, np.asarray(labels, dtype=str), rng)

    # step 5 -- BACK to the eval space, or the comparison is between two different spaces
    rs = counts.sum(axis=1, keepdims=True).astype(np.float64)
    np.maximum(rs, 1.0, out=rs)
    return np.log1p(counts / rs * EVAL_TARGET_SUM).astype(np.float32)


def _pin_colsum(counts: np.ndarray, mu: np.ndarray, labels: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Force each gene's PER-PERTURBATION column sum onto `round(sum_i mu[i, g])`.

    Ported from the research tree's `exact_colsum`, with the block boundaries taken from
    `labels` rather than a fixed stride -- our panel has a variable number of cells per
    perturbation, so a fixed `n_per` would straddle groups.

    The residual is added to (or removed from) randomly chosen cells, so per-cell dispersion
    survives; only the group total is pinned.
    """
    out = counts
    for lbl in np.unique(labels):
        rows = np.flatnonzero(labels == lbl)
        if rows.size == 0:
            continue
        blk = out[rows]
        tgt = np.rint(mu[rows].sum(0)).astype(np.int64)
        d = tgt - blk.sum(0, dtype=np.int64)
        for gi in np.flatnonzero(d):
            k = int(d[gi])
            if k > 0:                                  # scatter the deficit over random cells
                idx = rng.integers(0, rows.size, size=k)
                np.add.at(blk[:, gi], idx, 1)
            else:                                      # remove only where a count exists
                need = -k
                nz = np.flatnonzero(blk[:, gi] > 0)
                while need > 0 and nz.size:
                    m = min(need, int(np.minimum(blk[nz, gi], 1).sum()))
                    pick = rng.permutation(nz)[:m]
                    blk[pick, gi] -= 1
                    need -= m
                    nz = np.flatnonzero(blk[:, gi] > 0)
        out[rows] = blk
    return out


def write_celleval_v1(pred: np.ndarray, labels: np.ndarray, panel: Panel,
                      out_dir: str, tag: str,
                      pred_ctrl: Optional[np.ndarray] = None,
                      emit: Optional[Dict] = None) -> Tuple[str, str]:
    """Write the decoder-convention `injctrl` pair: model-space target and prediction.

    Targets are the real panel preprocessed as model input; predictions are the raw model
    output with the REAL controls appended (the model is never run for controls). Scoring is
    on the panel's own genes only — the 44,728-gene encoder axis is 59.6% structural zero
    padding for H1, and padding ties distort Spearman badly enough to reverse head
    comparisons (§12.15).
    """
    import anndata as ad
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    var = pd.DataFrame(index=[panel.panel_genes[j] for j in panel.panel_cols])

    ctrl_mask = panel.real_labels == CONTROL
    # `pred_ctrl` replaces the controls written into the PREDICTION file only; the target
    # file always keeps the panel's own. Default None = the real controls at the same
    # indices, which is the `injctrl` convention and every caller's behaviour to date.
    # The split-half protocol passes the B controls here, and that single substitution is
    # what makes the reference (A) and the prediction (B) independent: `pearson_delta` uses
    # each FILE's own control pseudobulk, so the deltas become A-relative and B-relative.
    ctrl_block = (panel.real_X_model[ctrl_mask] if pred_ctrl is None
                  else np.asarray(pred_ctrl, dtype=panel.real_X_model.dtype))
    if ctrl_block.ndim != 2 or ctrl_block.shape[1] != pred.shape[1]:
        raise ValueError(
            f"pred_ctrl has shape {ctrl_block.shape}, expected (n_ctrl, {pred.shape[1]}) "
            f"on the panel's kept-gene axis")
    # ─── optional NB count emission ─────────────────────────────────────────────────────
    # ⚠️ THE PERTURBED ROWS ONLY. The control block is REAL DATA -- real half-B cells that
    # already carry real dispersion -- so re-emitting it would replace measurements with
    # samples and break the one thing that makes `pearson_delta` a paired comparison.
    # `emit=None` (the default) is the behaviour every number to date was produced with.
    pred_emitted = pred
    if emit:
        theta = emit.get("theta")
        if theta is None:
            raise ValueError(
                "emit requires `theta`. Take it from an NB-trained checkpoint's log_theta "
                "(`theta = exp(log_theta)`); a scalar broadcast is a weaker, different claim "
                "and must be recorded as one.")
        rng = np.random.default_rng(int(emit.get("seed", 0)))
        pred_emitted = nb_emit(
            pred, theta=theta,
            n_per_cell_lib=float(emit.get("lib", EVAL_TARGET_SUM)),
            rng=rng,
            exact_colsum=bool(emit.get("exact_colsum", False)),
            labels=labels,
        )
        print(f"[panel] {tag}: NB emission lib={emit.get('lib', EVAL_TARGET_SUM):g} "
              f"exact_colsum={bool(emit.get('exact_colsum', False))} "
              f"theta={'per-gene' if np.asarray(theta).size > 1 else float(np.asarray(theta))} "
              f"| zero-frac {float((pred_emitted == 0).mean()):.4f} "
              f"(was {float((pred == 0).mean()):.4f}, target "
              f"{float((panel.real_X_model == 0).mean()):.4f})")

    pred_X = np.concatenate([pred_emitted, ctrl_block], axis=0)
    pred_lab = np.concatenate([labels, np.array([CONTROL] * ctrl_block.shape[0])])

    tpath = os.path.join(out_dir, f"h1_{tag}_target.h5ad")
    ppath = os.path.join(out_dir, f"h1_{tag}_pred_injctrl.h5ad")
    ad.AnnData(X=panel.real_X_model,
               obs=pd.DataFrame({"target_gene": panel.real_labels},
                                index=np.arange(panel.real_X_model.shape[0]).astype(str)),
               var=var.copy()).write_h5ad(tpath)
    ad.AnnData(X=pred_X,
               obs=pd.DataFrame({"target_gene": pred_lab},
                                index=np.arange(pred_X.shape[0]).astype(str)),
               var=var.copy()).write_h5ad(ppath)
    which = "real" if pred_ctrl is None else "override"
    print(f"[panel] {tag}: target {panel.real_X_model.shape} + pred {pred_X.shape} "
          f"({ctrl_block.shape[0]} injected {which} controls)")
    return ppath, tpath


def run_celleval_v1(pred_path: str, target_path: str, out_dir: str,
                    num_threads: int = 16, timeout: int = 43200,
                    de_batch_size: Optional[int] = 2000,
                    de_real: Optional[str] = None,
                    allow_discrete: bool = True,
                    profile: str = "full") -> Dict[str, float]:
    """cell-eval v1, parsed from agg_results.csv.

    A thin adapter onto `occam.decoder_train.metrics.run_cell_eval`, which is the proven
    runner: it already tails rather than heads stderr (cell-eval logs INFO there, so the
    first 500 chars are banner and the traceback is at the end), and it already carries
    `de_batch_size` and `de_real`.

    BUG 4 was here. `run.py` set `kwargs["de_real"]` and splatted it into this function,
    which had no such parameter -- so the advertised ground-truth-DE reuse was dead and
    every arm recomputed the dominant cost. Maintaining a second subprocess wrapper is what
    let the two drift apart, so there is now only one.

    `de_batch_size` defaults to 2000, not cell-eval's 100: the DE work is perturbations x
    genes, and at 100 a 2,394 x 8,873 panel becomes ~212,000 pool tasks, which tripped a
    CPython shared-memory race. `allow_discrete` defaults True so cell-eval does not decide
    whether to renormalise our already-log1p input from an unseeded 100-cell heuristic.
    """
    from occam.decoder_train.metrics import run_cell_eval

    return run_cell_eval(
        pred_path, target_path, out_dir,
        num_threads=num_threads, profile=profile, timeout=timeout,
        de_batch_size=de_batch_size, de_real=de_real, allow_discrete=allow_discrete,
    )


def _read_agg(agg_path: str) -> Dict[str, float]:
    """Parse cell-eval's `agg_results.csv`.

    Both tools write the same shape: WIDE, one column per metric, with a `statistic` column
    indexing the rows (`count`, `null_count`, `mean`, `std`, `min`, `max`, `median`). The
    wanted row is `mean` — taking `.iloc[0]` instead silently returns the perturbation
    COUNT, which is a large plausible-looking number and would be reported as the metric.
    Panel-level members such as `expr_mse_unbiased_capped_norm` carry their single value on
    the `mean` row too, with NaN elsewhere.
    """
    import csv as _csv

    if not os.path.exists(agg_path):
        return {}
    try:
        with open(agg_path) as fh:
            rows = {r["statistic"]: r for r in _csv.DictReader(fh)}
    except Exception as exc:                                   # noqa: BLE001
        print(f"[panel] could not parse {agg_path}: {exc}")
        return {}
    out: Dict[str, float] = {}
    for k, v in (rows.get("mean") or {}).items():
        if k == "statistic":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out[k] = f
    return out
