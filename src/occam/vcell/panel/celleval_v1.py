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


def write_celleval_v1(pred: np.ndarray, labels: np.ndarray, panel: Panel,
                      out_dir: str, tag: str) -> Tuple[str, str]:
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
    pred_X = np.concatenate([pred, panel.real_X_model[ctrl_mask]], axis=0)
    pred_lab = np.concatenate([labels, np.array([CONTROL] * int(ctrl_mask.sum()))])

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
    print(f"[panel] {tag}: target {panel.real_X_model.shape} + pred {pred_X.shape} "
          f"({int(ctrl_mask.sum())} injected real controls)")
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
