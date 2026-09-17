#!/usr/bin/env python
"""Predict + CellEval for a trained CellFlow run (Phase 3).

Example::

    cd competitors/CellFlow/cellflow
    unset VIRTUAL_ENV
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
    uv run --frozen --no-sync python ../predict_cellflow.py \\
      --run-dir /rxrx/scratch/sandbox/$USER/cellflow/norman/norman_fold0_smoke \\
      --split-key test --max-control-cells 256 --profile minimal
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
HOOKE_TX_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hooke_bridge import (  # noqa: E402
    apply_cellflow_gene_schema,
    control_row_indices,
    load_fold,
    load_hooke_anndata,
    load_pca_artifacts,
    read_obs_columns,
    reconstruct_from_pca,
    write_scorer_pca_state,
)

CELL_EVAL_GENE_MAX = 14.0


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be a mapping, got {type(cfg)}")
    selected = cfg.get("data", {}).get("selected_genes_path")
    if selected and not Path(selected).is_absolute():
        cfg["data"]["selected_genes_path"] = str(HOOKE_TX_ROOT / selected)
    return cfg


def _subset_by_indices(adata: ad.AnnData, indices: list[int], *, name: str) -> ad.AnnData:
    if not indices:
        raise ValueError(f"split {name!r} is empty")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.min() < 0 or idx.max() >= adata.n_obs:
        raise IndexError(
            f"split {name!r} indices out of range for adata with n_obs={adata.n_obs} "
            f"(min={idx.min()}, max={idx.max()})"
        )
    return adata[idx].copy()


def _union_indices(*parts: list[int] | np.ndarray) -> np.ndarray:
    arrays = [np.asarray(p, dtype=np.int64) for p in parts if len(p)]
    if not arrays:
        raise ValueError("no indices to load")
    return np.unique(np.concatenate(arrays))


def _remap_indices(original: list[int] | np.ndarray, loaded_rows: np.ndarray) -> list[int]:
    pos = {int(i): j for j, i in enumerate(loaded_rows)}
    try:
        return [pos[int(i)] for i in original]
    except KeyError as exc:
        raise KeyError(f"index {exc.args[0]} missing from loaded row_indices") from exc


def _eval_source_covariates(data_dir: Path | str, columns: Sequence[str]) -> dict[str, str]:
    """Read the constant value each covariate takes in the evaluated dump.

    Each Replogle-Nadig dump holds exactly one cell line, so the split covariates
    identify which control population predictions should flow from.
    """
    if not columns:
        return {}
    obs = read_obs_columns(data_dir, list(columns))
    out: dict[str, str] = {}
    for col in columns:
        values = obs[col].astype(str).unique()
        if len(values) != 1:
            raise ValueError(
                f"{data_dir}: split covariate {col!r} must be constant within a dump, "
                f"found {len(values)} values"
            )
        out[col] = str(values[0])
    return out


def _mean_metrics_from_agg(agg_results) -> dict[str, float]:
    """Extract the ``mean`` row from cell-eval aggregate output (polars DataFrame)."""
    import polars as pl

    if isinstance(agg_results, pd.DataFrame):
        # Older cell-eval versions returned pandas.
        if "mean" in agg_results.index:
            row = agg_results.loc["mean"]
            return {str(k): float(v) for k, v in row.items() if v is not None and np.isfinite(float(v))}
        agg_results = pl.from_pandas(agg_results.reset_index())

    first_col = agg_results.columns[0]
    mean_row = agg_results.filter(pl.col(first_col) == "mean")
    if mean_row.height == 0:
        return {}
    record = mean_row.to_dicts()[0]
    out: dict[str, float] = {}
    for key, value in record.items():
        if key == first_col or value is None:
            continue
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _run_celleval(
    adata_pred: ad.AnnData,
    adata_real: ad.AnnData,
    *,
    pert_col: str,
    control_pert: str,
    profile: str,
    outdir: Path,
    num_threads: int,
    skip_metrics: list[str] | None,
) -> dict[str, float]:
    from cell_eval import MetricsEvaluator

    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=control_pert,
        pert_col=pert_col,
        num_threads=num_threads,
        outdir=str(outdir),
    )
    _, agg = evaluator.compute(
        profile=profile,
        skip_metrics=list(skip_metrics) if skip_metrics else None,
        write_csv=True,
    )
    return _mean_metrics_from_agg(agg)


def run_predict(
    *,
    run_dir: Path,
    split_key: str = "test",
    max_control_cells: int | None = 512,
    profile: str = "full",
    num_threads: int = 8,
    skip_eval: bool = False,
    skip_metrics: list[str] | None = None,
    seed: int = 0,
    artifact_dir: Path | str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = _load_config(cfg_path)
    data_cfg = cfg["data"]
    genes_path = run_dir / "genes.txt"
    if not genes_path.exists():
        raise FileNotFoundError(genes_path)
    genes = [line.strip() for line in genes_path.read_text().splitlines() if line.strip()]
    pca = load_pca_artifacts(run_dir / "pca")

    print(f"[predict_cellflow] run_dir={run_dir} split_key={split_key}")
    t0 = time.perf_counter()

    from cellflow.model import CellFlow

    cf = CellFlow.load(str(run_dir))
    schema = cf.adata.uns.get("cellflow_schema") or {}
    control_key = schema.get("control_key", "is_control")
    slot_names = list(schema.get("perturbation_covariates", {}).get("gene", ("gene_1",)))
    sample_covariates = list(schema.get("sample_covariates") or ())
    split_covariates = list(schema.get("split_covariates") or ())
    print(
        f"[predict_cellflow] loaded CellFlow train_n={cf.adata.n_obs} "
        f"n_vars={cf.adata.n_vars} wall_s={time.perf_counter() - t0:.1f}"
    )

    # The evaluated dump holds one cell line; predictions must flow from its own controls.
    eval_covariates = _eval_source_covariates(
        data_cfg["data_dir"], sorted(set(split_covariates) | set(sample_covariates))
    )
    if eval_covariates:
        print(f"[predict_cellflow] eval covariates {eval_covariates}")

    # Control sources: from the pickled train AnnData (already has X_pca + schema).
    ctrl_mask = np.array(cf.adata.obs[control_key].astype(bool), dtype=bool)
    for col, value in eval_covariates.items():
        ctrl_mask &= np.array(cf.adata.obs[col].astype(str) == value, dtype=bool)
    if not ctrl_mask.any():
        raise ValueError(
            f"no {control_key!r} cells in trained CellFlow.adata matching {eval_covariates}"
        )
    adata_ctrl = cf.adata[ctrl_mask].copy()
    if max_control_cells is not None and adata_ctrl.n_obs > max_control_cells:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(adata_ctrl.n_obs, size=max_control_cells, replace=False))
        adata_ctrl = adata_ctrl[take].copy()
    print(f"[predict_cellflow] control sources n={adata_ctrl.n_obs}")

    fold = load_fold(data_cfg["split_path"], fold=int(data_cfg.get("fold", 0)))
    if split_key not in fold:
        raise KeyError(f"split_key={split_key!r} not in fold keys {sorted(fold)}")
    train_key = data_cfg.get("train_key", "finetune")

    # CellEval GT controls: if the eval split already holds them (STATE), use those
    # as-is — do not also inject from another split key (that duplicated rows).
    # If the eval split has none (Norman), inject from control_from_split_key.
    split_rows = np.asarray(fold[split_key], dtype=np.int64)
    n_ctrl_in_split = int(
        read_obs_columns(data_cfg["data_dir"], ["negative_control"])
        .iloc[split_rows]["negative_control"]
        .astype(bool)
        .sum()
    )
    inject_ctrl_rows: np.ndarray | None = None
    if n_ctrl_in_split == 0:
        control_split_key = data_cfg.get("control_from_split_key", train_key)
        if control_split_key not in fold:
            raise KeyError(
                f"control_from_split_key={control_split_key!r} not in fold keys {sorted(fold)}"
            )
        inject_ctrl_rows = control_row_indices(
            data_cfg["data_dir"],
            within=fold[control_split_key],
            max_cells=max_control_cells,
            seed=seed + 1,
        )
        loaded_rows = _union_indices(fold[split_key], inject_ctrl_rows)
        print(
            f"[predict_cellflow] eval split has no controls; injecting "
            f"{len(inject_ctrl_rows)} from {control_split_key!r}"
        )
    else:
        loaded_rows = np.unique(split_rows)
        print(
            f"[predict_cellflow] eval split already has {n_ctrl_in_split} controls; "
            "using them for CellEval (no inject)"
        )

    # Ground-truth dump for eval split (+ injected controls when needed), fold rows only.
    adata_loaded = load_hooke_anndata(
        data_cfg["data_dir"],
        selected_genes=genes,
        feature=data_cfg.get("feature", "auto"),
        row_indices=loaded_rows,
        target_sum=data_cfg.get("target_sum"),
        log1p=bool(data_cfg.get("log1p", False)),
        on_missing_genes="error",
        seed=seed,
    )
    apply_cellflow_gene_schema(
        adata_loaded,
        max_combination_length=int(data_cfg.get("max_combination_length", 1)),
        sample_covariates=sample_covariates,
        split_covariates=split_covariates,
    )
    adata_eval = _subset_by_indices(
        adata_loaded, _remap_indices(fold[split_key], loaded_rows), name=split_key
    )
    eval_ctrl_mask = np.array(adata_eval.obs[control_key].astype(bool), dtype=bool)
    if inject_ctrl_rows is None:
        # STATE: controls already in the eval split.
        ctrl_gt = adata_eval[eval_ctrl_mask].copy()
        adata_real_pert = adata_eval[~eval_ctrl_mask].copy()
    else:
        # Norman: eval split is pert-only; inject controls from another fold key.
        ctrl_gt = _subset_by_indices(
            adata_loaded, _remap_indices(inject_ctrl_rows, loaded_rows), name="injected_controls"
        )
        adata_real_pert = adata_eval.copy()
    if ctrl_gt.n_obs == 0:
        raise ValueError(f"no ground-truth controls available for CellEval on split {split_key!r}")
    print(
        f"[predict_cellflow] gt loaded n={adata_loaded.n_obs} "
        f"eval_pert={adata_real_pert.n_obs} real_controls={ctrl_gt.n_obs}"
    )

    cov_cols = [*slot_names, *sample_covariates, *split_covariates, "pert_str", control_key]
    cov_cols = list(dict.fromkeys(cov_cols))
    cov = (
        adata_eval.obs.loc[~adata_eval.obs[control_key].astype(bool), cov_cols]
        .drop_duplicates(subset=["pert_str"])
        .copy()
    )
    if cov.empty:
        raise ValueError(f"no non-control conditions in split {split_key!r}")
    cov[control_key] = False
    print(f"[predict_cellflow] conditions to predict n={len(cov)}")

    t1 = time.perf_counter()
    preds = cf.predict(
        adata_ctrl,
        covariate_data=cov,
        sample_rep="X_pca",
        condition_id_key="pert_str",
        key_added_prefix=None,
    )
    if not preds:
        raise RuntimeError("CellFlow.predict returned empty predictions")
    print(
        f"[predict_cellflow] predict_done n_conditions={len(preds)} "
        f"wall_s={time.perf_counter() - t1:.1f}"
    )

    # Reconstruct gene space and stack predicted cells.
    x_blocks: list[np.ndarray] = []
    pert_labels: list[str] = []
    for pert_id, x_pca in preds.items():
        x_pca = np.asarray(x_pca)
        if x_pca.ndim == 1:
            x_pca = x_pca[None, :]
        pert_id = str(pert_id)
        x_gene = reconstruct_from_pca(
            x_pca,
            means=pca["means"],
            pcs=pca["pcs"],
            clip_min=0.0,
            clip_max=CELL_EVAL_GENE_MAX,
        )
        x_blocks.append(x_gene)
        pert_labels.extend([pert_id] * x_gene.shape[0])

    x_pred = np.concatenate(x_blocks, axis=0)
    obs_pred = pd.DataFrame({"pert_str": pert_labels})
    var = pd.DataFrame(index=pd.Index(genes, name="ensembl_gene_id"))
    adata_pred_pert = ad.AnnData(X=x_pred, obs=obs_pred, var=var.copy())

    # Keep the cell-type column. The obs reduction below used to drop everything but `pert_str`,
    # which makes the artifacts unreadable by the shared scorer -- `infer_artifact_spec` requires
    # a cell-type column in BOTH objects and uses it to slice per cell type. It was lossy anyway:
    # the model conditions on cell_type (`sample_covariates`/`split_covariates`).
    #
    # One value by construction: the evaluated dump is a single line, which is what
    # `_eval_source_covariates` pins.
    ct_col = (sorted(set(split_covariates) | set(sample_covariates)) or [None])[0]
    ct_value = eval_covariates.get(ct_col) if ct_col else None
    keep = ["pert_str"] if ct_col is None else ["pert_str", ct_col]

    def _reduce_obs(frame):
        out = frame[[c for c in keep if c in frame.columns]].copy()
        if ct_col is not None and ct_col not in out.columns:
            out[ct_col] = ct_value
        return out

    if ct_col is not None:
        adata_pred_pert.obs[ct_col] = ct_value
    # Real controls → both AnnDatas (same set, no duplicates).
    ctrl_gt.obs = _reduce_obs(ctrl_gt.obs)
    adata_pred = ad.concat([adata_pred_pert, ctrl_gt], axis=0, join="outer", index_unique="-")
    adata_real_pert.obs = _reduce_obs(adata_real_pert.obs)
    adata_real = ad.concat([adata_real_pert, ctrl_gt], axis=0, join="outer", index_unique="-")

    # Align var order explicitly.
    adata_pred = adata_pred[:, genes].copy()
    adata_real = adata_real[:, genes].copy()
    if ct_col is not None:
        for label, obj in (("pred", adata_pred), ("real", adata_real)):
            obj.obs[ct_col] = obj.obs[ct_col].astype(str)
            n_ct = obj.obs[ct_col].nunique()
            if n_ct != 1:
                raise ValueError(f"{label} spans {n_ct} cell types; expected exactly 1")

    out_pred = run_dir / f"pred_{split_key}.h5ad"
    out_real = run_dir / f"real_{split_key}.h5ad"
    adata_pred.write_h5ad(out_pred)
    adata_real.write_h5ad(out_real)

    # A second copy under the shared scorer's fixed filenames, in its own directory, so one
    # prediction pass can be scored under several predicted-control conventions.
    #
    # `pred_{split}.h5ad` in run_dir cannot serve: the scorer's names are fixed
    # (PREDICTIONS_NAME / GROUND_TRUTH_NAME) and it writes a manifest into the artifact dir,
    # which does not belong beside CellFlow.pkl. The PCA basis is converted alongside because the
    # `decoded` arm needs it to decode the pred-side controls.
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        adata_pred.write_h5ad(artifact_dir / "adata_pred.h5ad")
        adata_real.write_h5ad(artifact_dir / "adata_real.h5ad")
        write_scorer_pca_state(
            {"pcs": pca["pcs"], "means": pca["means"]}, artifact_dir / "pca_state.npz"
        )
        print(
            f"[predict_cellflow] artifacts -> {artifact_dir} "
            f"(adata_pred.h5ad, adata_real.h5ad, pca_state.npz)"
        )
    print(
        f"[predict_cellflow] wrote {out_pred.name} n={adata_pred.n_obs} "
        f"{out_real.name} n={adata_real.n_obs} n_vars={adata_pred.n_vars}"
    )

    metrics: dict[str, float] = {}
    if not skip_eval:
        metrics_dir = run_dir / f"celleval_{split_key}"
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)
        metrics = _run_celleval(
            adata_pred,
            adata_real,
            pert_col="pert_str",
            control_pert="empty",
            profile=profile,
            outdir=metrics_dir,
            num_threads=num_threads,
            skip_metrics=skip_metrics,
        )
        metrics_path = run_dir / f"celleval_{split_key}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[predict_cellflow] celleval profile={profile} -> {metrics_path}")
        for key in sorted(metrics)[:12]:
            print(f"  {key}: {metrics[key]:.4f}")
        if len(metrics) > 12:
            print(f"  ... ({len(metrics)} metrics total)")

    meta = {
        "run_dir": str(run_dir),
        "split_key": split_key,
        "n_control_sources": int(adata_ctrl.n_obs),
        "n_conditions": int(len(cov)),
        "n_pred_cells": int(adata_pred_pert.n_obs),
        "n_real_eval_cells": int(adata_real_pert.n_obs),
        "n_real_controls": int(ctrl_gt.n_obs),
        "n_vars": int(len(genes)),
        "profile": None if skip_eval else profile,
        "metrics": metrics,
        "wall_s": time.perf_counter() - t0,
    }
    (run_dir / f"predict_{split_key}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[predict_cellflow] done total_wall_s={meta['wall_s']:.1f}")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Train run directory with CellFlow.pkl")
    parser.add_argument("--split-key", type=str, default="test", help="Fold key to score (test|validation)")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="also write adata_pred.h5ad / adata_real.h5ad / pca_state.npz here, for the shared "
             "scorer (hooke_tx/eval/celleval_artifacts.py). Required for a two-convention report.",
    )
    parser.add_argument(
        "--max-control-cells",
        type=int,
        default=512,
        help="Subsample control sources / injected real controls (None via -1 for all)",
    )
    parser.add_argument("--profile", type=str, default="full", help="cell-eval profile")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true", help="Only write pred/real h5ad")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-run CellEval on existing pred/real h5ad (skip model load/predict)",
    )
    parser.add_argument(
        "--skip-metrics",
        type=str,
        default="pearson_edistance,clustering_agreement",
        help="Comma-separated cell-eval metrics to skip (empty string = skip none)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    skip = [m.strip() for m in args.skip_metrics.split(",") if m.strip()] if args.skip_metrics else None
    if args.eval_only:
        run_dir = Path(args.run_dir)
        out_pred = run_dir / f"pred_{args.split_key}.h5ad"
        out_real = run_dir / f"real_{args.split_key}.h5ad"
        if not out_pred.exists() or not out_real.exists():
            raise FileNotFoundError(f"need {out_pred} and {out_real} for --eval-only")
        adata_pred = ad.read_h5ad(out_pred)
        adata_real = ad.read_h5ad(out_real)
        metrics_dir = run_dir / f"celleval_{args.split_key}"
        metrics = _run_celleval(
            adata_pred,
            adata_real,
            pert_col="pert_str",
            control_pert="empty",
            profile=args.profile,
            outdir=metrics_dir,
            num_threads=args.num_threads,
            skip_metrics=skip,
        )
        metrics_path = run_dir / f"celleval_{args.split_key}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[predict_cellflow] eval-only -> {metrics_path}")
        for key in sorted(metrics):
            print(f"  {key}: {metrics[key]:.4f}")
        return 0

    max_ctrl = None if args.max_control_cells is not None and args.max_control_cells < 0 else args.max_control_cells
    run_predict(
        run_dir=args.run_dir,
        split_key=args.split_key,
        max_control_cells=max_ctrl,
        profile=args.profile,
        num_threads=args.num_threads,
        skip_eval=args.skip_eval,
        skip_metrics=skip,
        seed=args.seed,
            artifact_dir=args.artifact_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
