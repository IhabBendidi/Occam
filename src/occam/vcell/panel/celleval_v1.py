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
                    num_threads: int = 16, timeout: int = 43200) -> Dict[str, float]:
    """cell-eval v1 through the `jupyter` pyenv, parsed from agg_results.csv."""
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    cmd = ["cell-eval", "run", "-ap", pred_path, "-ar", target_path, "-o", out_dir,
           "--num-threads", str(num_threads), "--profile", "full"]
    env = os.environ.copy()
    env["PYENV_VERSION"] = "jupyter"
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"[panel] cell-eval v1 failed (exit {r.returncode}): {r.stderr[-800:]}")
            return {}
    except subprocess.TimeoutExpired:
        print(f"[panel] cell-eval v1 timed out after {timeout}s")
        return {}
    except FileNotFoundError:
        print("[panel] cell-eval not found (pyenv jupyter unavailable)")
        return {}

    return _read_agg(os.path.join(out_dir, "agg_results.csv"))


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
