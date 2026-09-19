#!/usr/bin/env python
"""Score our best arm three ways: mean-only, NB emission, NB + exact_colsum.

WHAT THIS ANSWERS
    Our prediction is the conditional MEAN, written straight to the h5ad: measured zero
    fraction 0.0000 against real data's ~0.52, and almost no within-group dispersion. The
    DE-based metrics (`overlap_at_N`, `precision_at_N`) run a Wilcoxon test that reads exactly
    that dispersion, so it has been calling significance on a distribution no real data has.
    Does restoring the dispersion raise `overlap_at_N`?

    ⚠️ PRE-REGISTERED EXPECTATION, so the conclusion is not written after seeing the number.
    VCC measured exact-colsum at +0.038-0.040 on `nmae` with `jac` at -0.002...-0.004, and
    `overlap_at_N` is in the same Wilcoxon family as `jac`. So the mechanism predicts:
        `pearson_delta`/`mae`  -- small gain from colsum (group-sum pseudobulk)
        `overlap_at_N`         -- LITTLE OR NOTHING, possibly negative
    A large `overlap_at_N` gain would be a surprise and should be treated as one.

WHY POST-HOC, NOT A RE-RUN
    The emitter's input is the saved prediction h5ad -- already log1p at 2000 on the full-panel
    denominator, which is exactly `nb_emit`'s input contract. So no model forward pass is
    needed and the mean-only arm is byte-identical to the published 0.3415.

⚠️ WHERE `theta` COMES FROM, AND THE LEAK IT WOULD BE EASY TO CREATE
    `s313` is a `smooth_l1` arm and carries NO dispersion tensor, so theta is estimated from
    data by method of moments: `Var = mu + mu^2/theta  ->  theta = mu^2 / (Var - mu)`.

    It is estimated from the PREDICTION file's control rows -- the half-B controls, which are
    the MODEL'S OWN INPUT -- and never from the target file's half-A controls, which are the
    scoring reference. Using the latter is the E2 control leak, which this project has
    reintroduced once already and which cost 0.0052.

    A model-learned per-gene `log_theta` is the better source and is not available yet; the NB
    arms (L01-L09) are training. Label any number from here as data-estimated theta.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

CONTROL = "non-targeting"
EVAL_TARGET_SUM = 2000.0


def estimate_theta(ctrl_log1p: np.ndarray, floor: float = 0.05,
                   cap: float = 1e4) -> np.ndarray:
    """Per-gene NB dispersion by method of moments, in COUNT space.

    `theta` is the 1/alpha convention: Var = mu + mu^2/theta, so LARGE theta -> Poisson.
    Genes whose sample variance does not exceed the mean carry no evidence of
    over-dispersion, so they get `cap` (effectively Poisson) rather than a negative or
    infinite estimate.
    """
    x = np.expm1(np.asarray(ctrl_log1p, dtype=np.float64))
    mu = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)
    excess = var - mu
    theta = np.full(mu.shape, cap, dtype=np.float64)
    ok = (excess > 0) & (mu > 0)
    theta[ok] = (mu[ok] ** 2) / excess[ok]
    return np.clip(theta, floor, cap)


def theta_from_checkpoint(ckpt_path: str, panel_genes) -> np.ndarray:
    """Per-gene NB dispersion from a MODEL-LEARNED `log_theta`, aligned to the panel axis.

    ⚠️ THIS IS THE PRINCIPLED SOURCE, and it exists for the smooth_l1 arms too. The released
    encoder+decoder checkpoint carries `log_theta` over the full 44,728-gene encoder axis,
    trained during decoder pretraining -- which is exactly what `vc_load_dispersion: true`
    warm-starts for the NB virtual-cell arms. `exp(log_theta)` has median ~1.9 (strongly
    over-dispersed), against the 10,000 CAP that method-of-moments on library-NORMALISED data
    returned for every gene, which silently degenerated NB into Poisson (F-emit4).

    Aligned BY GENE NAME, never positionally: the checkpoint's axis is 44,728 and the panel's
    is 6,983, so a positional read would be gibberish that still has the right shape.
    """
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lt = None
    for key in ("dispersion", "loss_state_dict"):
        blk = ck.get(key)
        if isinstance(blk, dict) and "log_theta" in blk:
            lt = blk["log_theta"]
            break
    if lt is None:
        raise SystemExit(
            f"{ckpt_path} carries no log_theta under 'dispersion' or 'loss_state_dict'. "
            f"A smooth_l1 VC checkpoint does not have one; use the RELEASED "
            f"encoder+decoder checkpoint, which does.")
    theta_full = np.exp(np.asarray(lt, dtype=np.float64).reshape(-1))
    names = [str(g) for g in ck.get("gene_names", [])]
    if len(names) != theta_full.size:
        raise SystemExit(
            f"gene_names has {len(names)} entries for {theta_full.size} theta values")
    idx = {g: i for i, g in enumerate(names)}
    missing = [g for g in panel_genes if g not in idx]
    if missing:
        raise SystemExit(
            f"{len(missing)} panel genes are absent from the checkpoint axis, e.g. "
            f"{missing[:5]}. Refusing to fill them with a default: a wrong dispersion on a "
            f"real gene is invisible in the output.")
    theta = theta_full[[idx[g] for g in panel_genes]]
    print(f"[emit] model-learned theta from {os.path.basename(ckpt_path)}: "
          f"{theta.size} genes on the panel axis, median {np.median(theta):.3f} "
          f"p10 {np.percentile(theta, 10):.3f} p90 {np.percentile(theta, 90):.3f}")
    return theta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-dir", required=True, help="runs/rhj_splithalf/<arm>")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits", default="0,1,2,3,4")
    ap.add_argument("--lib", type=float, default=EVAL_TARGET_SUM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--panel-npz",
                    default="data/benchmarks/rhj_fewshot/splithalf/panel.npz",
                    help="carries axis_ensembl/axis_symbols in the same order -- needed "
                         "because the checkpoint's theta is on the ENSEMBL axis")
    ap.add_argument("--theta-ckpt", default="",
                    help="checkpoint carrying a MODEL-LEARNED log_theta. Empty = estimate "
                         "from the model-visible controls, which on library-normalised data "
                         "collapses to Poisson (F-emit4) -- prefer a checkpoint.")
    ap.add_argument("--arms", default="nb,nb_xcolsum",
                    help="comma list from: nb, nb_xcolsum, cal (calibration only)")
    a = ap.parse_args(argv)

    import anndata as ad
    import pandas as pd
    from occam.vcell.panel.celleval_v1 import _read_agg, nb_emit, run_celleval_v1

    METRICS = ("pearson_delta", "mae", "overlap_at_N", "discrimination_score_l1")
    rows: dict[str, dict[int, dict]] = {"nb": {}, "nb_xcolsum": {}}

    for s in [int(x) for x in a.splits.split(",") if x.strip()]:
        sd = os.path.join(a.arm_dir, f"split_{s:02d}")
        ppath = os.path.join(sd, "h1_model_nocal_pred_injctrl.h5ad")
        tpath = os.path.join(sd, "h1_model_nocal_target.h5ad")
        if not (os.path.isfile(ppath) and os.path.isfile(tpath)):
            print(f"[emit] split {s}: missing pair, skipping")
            continue

        P = ad.read_h5ad(ppath)
        lab = np.asarray(P.obs["target_gene"], dtype=str)
        is_ctrl = lab == CONTROL
        X = np.asarray(P.X.todense() if hasattr(P.X, "todense") else P.X, dtype=np.float32)

        if a.theta_ckpt:
            # ⚠️ ALIGN ON ENSEMBL, NOT SYMBOLS. The prediction h5ad's var index is SYMBOLS
            # (cell-eval needs those -- it excludes each perturbation's own target gene by
            # symbol), but the released checkpoint's axis is ENSEMBL: 44,560 ENSG ids plus
            # 168 unmapped symbol-like leftovers. Matching by symbol found 2 of 6,983 genes.
            # The panel carries both axes in the same order, so the ensembl one is the key.
            zz = np.load(a.panel_npz, allow_pickle=True)
            ens = [str(g) for g in zz["axis_ensembl"]]
            sym = [str(g) for g in zz["axis_symbols"]]
            if sym != [str(g) for g in P.var.index]:
                raise SystemExit(
                    "the prediction h5ad's gene order does not match the panel's symbol "
                    "axis; the ensembl alignment would silently be off by position.")
            theta = theta_from_checkpoint(a.theta_ckpt, ens)
        else:
            # theta from the MODEL-VISIBLE half-B controls, never the target's half-A ones.
            # ⚠️ On library-normalised data this collapses to the cap -- see F-emit4.
            theta = estimate_theta(X[is_ctrl])
            print(f"[emit] split {s}: {int(is_ctrl.sum())} model-visible controls -> theta "
                  f"median {np.median(theta):.3f} p10 {np.percentile(theta,10):.3f} "
                  f"p90 {np.percentile(theta,90):.3f}")

        want = [t.strip() for t in a.arms.split(",") if t.strip()]
        for tag, xc in [("nb", False), ("nb_xcolsum", True)]:
            if tag not in want:
                continue
            out_dir = os.path.join(a.out_root, f"split_{s:02d}", tag)
            os.makedirs(out_dir, exist_ok=True)
            # emit the PERTURBED rows only; the control block is real data
            emitted = X.copy()
            emitted[~is_ctrl] = nb_emit(
                X[~is_ctrl], theta=theta, n_per_cell_lib=a.lib,
                rng=np.random.default_rng(a.seed + 1000 * s),
                exact_colsum=xc, labels=lab[~is_ctrl])
            np_path = os.path.join(out_dir, "pred.h5ad")
            ad.AnnData(X=emitted,
                       obs=pd.DataFrame({"target_gene": lab},
                                        index=np.arange(len(lab)).astype(str)),
                       var=P.var.copy()).write_h5ad(np_path)
            print(f"[emit] split {s} {tag}: zero-frac {float((emitted[~is_ctrl]==0).mean()):.4f} "
                  f"(mean-only {float((X[~is_ctrl]==0).mean()):.4f})")
            m = run_celleval_v1(np_path, tpath, os.path.join(out_dir, "celleval"),
                                num_threads=a.num_threads, profile="full")
            if m:
                rows[tag][s] = {k: m.get(k) for k in METRICS}

    print(f"\n{'arm':14s} " + " ".join(f"{k:>22}" for k in METRICS))
    for tag in ("nb", "nb_xcolsum"):
        r = rows[tag]
        if r:
            print(f"{tag:14s} " + " ".join(
                f"{np.mean([v[k] for v in r.values()]):>22.4f}" for k in METRICS)
                + f"   (n={len(r)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
