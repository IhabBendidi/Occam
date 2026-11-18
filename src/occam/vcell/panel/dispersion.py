"""F-disp1 dispersion diagnostics, reported NEXT TO the four scored metrics.

WHY THIS EXISTS
    All four scored metrics are PSEUDOBULK (verified in cell-eval 0.6.0: `pearson_delta`,
    `mae` and `discrimination_score_l1` go through `iter_bulk_arrays`; `overlap_at_N` takes N
    from the REAL significant-gene count and ranks by |fold_change| of the means). So a score
    alone cannot tell you whether a dispersion term did anything at all -- the exact shape of
    failure that let `delta_cos_weight` run inert for three arms (F-loss1).

    Two numbers close that gap, and they are printed with every arm:

      std_pred / std_real   mean per-gene WITHIN-perturbation std, prediction over truth.
                            1.0 = matched. MEASURED 0.026 on W23 before any dispersion term
                            existed: our predicted cells were 38x under-dispersed.
      nsig_pred / nsig_real FDR-significant genes per perturbation, prediction over truth.
                            1.0 = matched. MEASURED 30.8x on W23 (3,475 vs 113 of 6,983),
                            because near-zero within-group variance makes the Wilcoxon test
                            call half the transcriptome significant.

⚠️ THESE ARE DIAGNOSTICS, NOT SCORES. Do not rank arms on them. They say whether a term took
effect; the four metrics say whether that was worth anything.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

CTRL = "non-targeting"
MIN_CELLS = 10        # a per-gene std over fewer cells is noise, not dispersion
FDR = 0.05


def _per_gene_std(X: np.ndarray, groups: np.ndarray) -> Optional[float]:
    """Mean over (perturbation, gene) of the within-perturbation std."""
    out, n = 0.0, 0
    for g in np.unique(groups):
        if g == CTRL:
            continue
        m = groups == g
        if int(m.sum()) < MIN_CELLS:
            continue
        out += float(X[m].std(axis=0).mean())
        n += 1
    return (out / n) if n else None


def dispersion_from_h5ads(pred_path: str, target_path: str,
                          pert_col: str = "target_gene") -> Dict[str, Optional[float]]:
    """The scored files are the source of truth: read exactly what cell-eval read."""
    import anndata as ad

    P = ad.read_h5ad(pred_path)
    T = ad.read_h5ad(target_path)
    gp = np.asarray(P.obs[pert_col].values).astype(str)
    gt = np.asarray(T.obs[pert_col].values).astype(str)
    if P.shape != T.shape or not (gp == gt).all():
        # row-for-row alignment is what makes this a paired comparison; without it the two
        # stds are computed over different cohorts and the ratio means nothing.
        return {"std_pred": None, "std_real": None, "std_ratio": None,
                "note": "pred/target rows are not aligned"}
    Xp = np.asarray(P.X.todense() if hasattr(P.X, "todense") else P.X, dtype=np.float32)
    Xt = np.asarray(T.X.todense() if hasattr(T.X, "todense") else T.X, dtype=np.float32)
    sp = _per_gene_std(Xp, gp)
    st = _per_gene_std(Xt, gt)
    return {"std_pred": sp, "std_real": st,
            "std_ratio": (sp / st) if (sp is not None and st) else None}


def nsig_from_de(pred_de: str, real_de: Optional[str]) -> Dict[str, Optional[float]]:
    """FDR-significant genes per perturbation, from the DE tables cell-eval already wrote."""
    import pandas as pd

    out: Dict[str, Optional[float]] = {"nsig_pred": None, "nsig_real": None,
                                       "nsig_ratio": None}
    if not os.path.isfile(pred_de):
        return out
    dp = pd.read_csv(pred_de, usecols=["target", "fdr"])
    sp = dp.groupby("target")["fdr"].apply(lambda s: int((s < FDR).sum()))
    out["nsig_pred"] = float(sp.mean())
    if real_de and os.path.isfile(real_de):
        dr = pd.read_csv(real_de, usecols=["target", "fdr"])
        sr = dr.groupby("target")["fdr"].apply(lambda s: int((s < FDR).sum()))
        ks = sorted(set(sp.index) & set(sr.index))
        if ks:
            out["nsig_pred"] = float(sp[ks].mean())
            out["nsig_real"] = float(sr[ks].mean())
            out["nsig_ratio"] = (out["nsig_pred"] / out["nsig_real"]
                                 if out["nsig_real"] else None)
    return out


def report(tag: str, pred_path: str, target_path: str, arm_dir: str,
           real_de: Optional[str] = None, pert_col: str = "target_gene") -> Dict:
    """Compute both diagnostics, PRINT them, and return them for metrics.json.

    Failures here must never fail an arm: a diagnostic that can break a scored run is worse
    than no diagnostic. Every exception is caught and reported as such.
    """
    rec: Dict = {}
    try:
        rec.update(dispersion_from_h5ads(pred_path, target_path, pert_col))
    except Exception as e:                                        # noqa: BLE001
        rec["std_error"] = f"{type(e).__name__}: {e}"
    try:
        rec.update(nsig_from_de(os.path.join(arm_dir, "pred_de.csv"), real_de))
    except Exception as e:                                        # noqa: BLE001
        rec["nsig_error"] = f"{type(e).__name__}: {e}"
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"          # noqa: E731
    print(f"[disp] {tag}: per-gene within-pert std  pred {f(rec.get('std_pred'), 4)}  "
          f"real {f(rec.get('std_real'), 4)}  RATIO {f(rec.get('std_ratio'), 4)}"
          f"   |   FDR-significant genes/pert  pred {f(rec.get('nsig_pred'), 0)}  "
          f"real {f(rec.get('nsig_real'), 0)}  RATIO {f(rec.get('nsig_ratio'), 2)}",
          flush=True)
    print(f"[disp] {tag}: 1.0 = matched. W23 baseline: std ratio 0.026, nsig ratio 30.8. "
          f"DIAGNOSTIC ONLY -- every scored metric is pseudobulk (F-disp1); do not rank on it.",
          flush=True)
    for k in ("std_error", "nsig_error"):
        if k in rec:
            print(f"[disp] ⚠️ {tag}: {k} {rec[k]}", flush=True)
    return rec
