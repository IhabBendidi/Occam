#!/usr/bin/env python
"""Train CellFlow on a Upstream-Tx dump (Phase 2).

Example::

    cd competitors/CellFlow/cellflow
    unset VIRTUAL_ENV
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
    uv run --frozen --no-sync python ../train_cellflow.py --config ../configs/norman.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from occam_bridge.occam import (  # noqa: E402
    load_occam_sources,
    load_occam_validation,
    occam_enabled,
    occam_pert_tokens,
)
from occam_bridge import (  # noqa: E402
    EMBEDDING_KEY,
    apply_cellflow_gene_schema,
    control_row_indices,
    dataset_pert_tokens,
    fit_and_project_pca,
    load_fold,
    load_dump_anndata,
    load_dump_sources,
    save_pca_artifacts,
)


def _set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.c`` on a nested mapping, creating dicts as needed."""
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError("empty --set key")
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _parse_set_item(item: str) -> tuple[str, Any]:
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"--set expects key=value, got {item!r}")
    return key.strip(), yaml.safe_load(raw)


def _expand_env_vars(value: Any) -> Any:
    """Expand environment variables recursively in loaded YAML values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        cfg = _expand_env_vars(yaml.safe_load(handle))
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be a mapping, got {type(cfg)}")
    selected = cfg.get("data", {}).get("selected_genes_path")
    if selected and not Path(selected).is_absolute():
        cfg["data"]["selected_genes_path"] = str(REPO_ROOT / selected)
    return cfg


def _subset_by_indices(adata, indices: list[int], *, name: str):
    if not indices:
        raise ValueError(f"split {name!r} is empty")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.min() < 0 or idx.max() >= adata.n_obs:
        raise IndexError(
            f"split {name!r} indices out of range for adata with n_obs={adata.n_obs} "
            f"(min={idx.min()}, max={idx.max()})"
        )
    return adata[idx].copy()


def _union_indices(*parts: list[int]) -> np.ndarray:
    arrays = [np.asarray(p, dtype=np.int64) for p in parts if p]
    if not arrays:
        raise ValueError("no indices to load")
    return np.unique(np.concatenate(arrays))


def _remap_indices(original: list[int], loaded_rows: np.ndarray) -> list[int]:
    pos = {int(i): j for j, i in enumerate(loaded_rows)}
    try:
        return [pos[int(i)] for i in original]
    except KeyError as exc:
        raise KeyError(f"fold index {exc.args[0]} missing from loaded row_indices") from exc


def _build_multi_source_specs(
    data_cfg: dict[str, Any],
    fold: dict[str, list[int]],
    *,
    train_key: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Rows to load per dump for the cross-cell-line few-shot setup.

    The evaluated line (``data_dir``) contributes only its ``train_key`` rows; the
    other lines contribute everything, which is what gives held-out perturbations a
    trained embedding. When the split holds all controls of the evaluated line in
    ``test`` (STATE few-shot), ``control_from_split_key: test`` pulls **only** those
    DMSO rows back in as OT sources — without them that line has no control
    population to flow from. Test perts stay out of training.
    """
    eval_dir = data_cfg["data_dir"]
    rows = np.asarray(fold[train_key], dtype=np.int64)

    control_key = data_cfg.get("control_from_split_key")
    if control_key:
        if control_key not in fold:
            raise KeyError(f"control_from_split_key={control_key!r} not in fold keys {sorted(fold)}")
        # Collect controls from the split key.
        ctrl_rows = control_row_indices(
            eval_dir,
            within=fold[control_key],
            max_cells=data_cfg.get("max_control_cells"),
            seed=seed,
        )
        rows = np.unique(np.concatenate([rows, ctrl_rows]))
        print(
            f"[train_cellflow] pulled {len(ctrl_rows)} controls from split "
            f"{control_key!r} into training sources"
        )

    specs: list[dict[str, Any]] = [{"data_dir": eval_dir, "row_indices": rows}]
    for entry in data_cfg.get("train_sources") or []:
        specs.append({"data_dir": entry["data_dir"], "row_indices": None, **entry})
    return specs


def _build_optimizer(learning_rate: float = 5e-5, multi_steps: int = 20):
    import optax

    return optax.MultiSteps(optax.adam(learning_rate), multi_steps)


def _ensure_validation_controls(
    adata_valid,
    adata_train,
    *,
    control_key: str,
    max_injected: int | None = None,
    cell_type_col: str | None = None,
):
    """CellFlow validation requires control cells as OT sources.

    Upstream/cell-eval folds put *all* controls in ``finetune`` and none in
    ``validation``/``test``. Inject train controls (already PCA-projected) into
    the validation AnnData so ``prepare_validation_data`` succeeds.

    ``cell_type_col`` restricts the injected controls to the validation's own cell type. That
    matters on the multi-source path: the training AnnData holds the controls of all four lines
    (39,165 of them), so injecting indiscriminately gave a validation object spanning four cell
    types, whose controls mostly belong to lines the validation cells are not from. With
    ``split_covariates: [cell_type]`` a perturbation is meant to flow from controls of its own
    line, so that is wrong on its own terms, and it also makes the object unscoreable per cell
    type. Single-source runs are unaffected -- there the filter matches everything.
    """
    import anndata as ad

    n_ctrl = int(adata_valid.obs[control_key].astype(bool).sum())
    if n_ctrl > 0:
        return adata_valid, 0

    # `copy=True`: pandas hands back a read-only view of the underlying block for a boolean
    # column, and the in-place narrowing below then fails with "output array is read-only".
    ctrl_mask = adata_train.obs[control_key].astype(bool).to_numpy(copy=True)
    if cell_type_col is not None:
        if cell_type_col not in adata_valid.obs.columns:
            raise KeyError(f"validation obs has no {cell_type_col!r} column")
        want = set(adata_valid.obs[cell_type_col].astype(str).unique())
        if len(want) != 1:
            raise ValueError(
                f"validation spans {len(want)} cell types before control injection: "
                f"{sorted(want)}"
            )
        ctrl_mask &= adata_train.obs[cell_type_col].astype(str).isin(want).to_numpy()
        if not ctrl_mask.any():
            raise ValueError(
                f"no training controls for cell type {sorted(want)[0]!r}; validation would have "
                "no OT source population of its own line"
            )
    if not ctrl_mask.any():
        raise ValueError(
            f"validation has no {control_key!r} cells and train has none to inject"
        )
    ctrl = adata_train[ctrl_mask].copy()
    if max_injected is not None and ctrl.n_obs > max_injected:
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(ctrl.n_obs, size=max_injected, replace=False))
        ctrl = ctrl[take].copy()

    n_injected = int(ctrl.n_obs)
    combined = ad.concat([ctrl, adata_valid], axis=0, join="outer", index_unique="-")
    # Preserve embeddings / schema (concat can drop or shallow-merge uns).
    for key, value in adata_train.uns.items():
        combined.uns[key] = value
    return combined, n_injected


def _gene_space_truth(adata_valid, schema) -> dict[str, np.ndarray]:
    """{perturbation label -> its RAW gene-space cells} from the validation AnnData.

    The selection callback receives validation tensors in `sample_rep` (PCA) space only, so
    scoring against an untouched target needs the original rows. Keyed by the same label
    `_condition_label` produces, i.e. the bare perturbation symbol.

    Controls are excluded: the callback appends its own shared control block, and a control
    row appearing twice would distort cell-eval's baseline.
    """
    import numpy as _np

    if adata_valid is None:
        return {}
    # `occam_target_gene` is the raw `target_gene` SYMBOL (`occam_bridge/occam.py:118`),
    # which is exactly what `_condition_label` returns as parts[0] -- so these keys line up
    # with the labels the callback builds. The schema dict has no perturbation-column entry,
    # so name the column here rather than relying on a silent `.get()` fallback.
    col = "occam_target_gene"
    if col not in adata_valid.obs.columns:
        raise KeyError(
            f"validation obs has no {col!r} column, so the gene-space truth cannot be keyed "
            f"by perturbation. Have {list(adata_valid.obs.columns)}")
    ctrl_key = schema["control_key"]
    labels = _np.asarray(adata_valid.obs[col], dtype=str)
    is_ctrl = _np.asarray(adata_valid.obs[ctrl_key], dtype=bool)
    X = adata_valid.X
    X = _np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=_np.float32)

    out: dict[str, _np.ndarray] = {}
    for lbl in sorted(set(labels[~is_ctrl].tolist())):
        rows = _np.flatnonzero((labels == lbl) & ~is_ctrl)
        if rows.size:
            out[str(lbl)] = X[rows]
    print(f"[train_cellflow] gene-space truth: {len(out)} perturbations, "
          f"{sum(len(v) for v in out.values()):,} cells (controls excluded)")
    return out


def run_train(cfg: dict[str, Any], *, config_path: Path | None = None) -> Path:
    data_cfg = cfg["data"]
    pca_cfg = cfg.get("pca", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})

    run_dir = Path(output_cfg.get("dir", ".")) / str(output_cfg.get("run_name", "cellflow_run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    # Predict reads this file; dump the resolved mapping (includes --set).
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.source.yaml")

    print(f"[train_cellflow] run_dir={run_dir}")
    t0 = time.perf_counter()

    if data_cfg.get("max_cells") is not None:
        raise NotImplementedError(
            "max_cells is for bridge smokes only; train_cellflow loads fold row_indices "
            "so split indices stay aligned. Use a smaller num_iterations instead."
        )

    # Occam's `rhj_fewshot` carries its own split inside the frozen benchmark, so there is no
    # `split_path` and no row-index JSON to keep aligned -- occam_bridge/occam.py reads it live
    # and asserts its invariants. The fold dict is unused on that path.
    occam = occam_enabled(data_cfg)
    train_key = data_cfg.get("train_key", "finetune")
    valid_key = data_cfg.get("valid_key", "validation")
    if occam:
        fold = {}
        if data_cfg.get("split_path"):
            raise ValueError(
                "both data.occam_bench_dir and data.split_path are set. The benchmark owns "
                "the split; a second split file could only disagree with it silently.")
    else:
        fold = load_fold(data_cfg["split_path"], fold=int(data_cfg.get("fold", 0)))
        if train_key not in fold:
            raise KeyError(f"train_key={train_key!r} not in fold keys {sorted(fold)}")

    seed = int(model_cfg.get("seed", 0))
    selected = data_cfg.get("selected_genes_path")
    multi_source = bool(data_cfg.get("train_sources"))
    sample_covariates = list(data_cfg.get("sample_covariates") or [])
    split_covariates = list(data_cfg.get("split_covariates") or [])

    other: dict[str, Any] = {}
    adata_valid = None

    # Vocabulary for the one-hot gene embedding, over the WHOLE dataset rather than the
    # training cells. The vocabulary fixes the model's condition-input width, so it cannot grow
    # after training: a test perturbation without a column makes `cf.predict` raise
    # "Representation for '...' not found" and produces no predictions at all. On the few-shot
    # split 8 of the 380 test perturbations are in that position -- never assayed in
    # jurkat/k562/rpe1 either. Widening to the dataset gives 2,023 tokens (+1 pad = 2,024),
    # which is exactly the size of STATE's pert_onehot_map.pt, so the two one-hot arms now span
    # the same perturbations.
    #
    # `null` disables it and restores the old train-cells-only behaviour for the other configs.
    vocab_dirs = data_cfg.get("vocabulary_data_dirs", "__unset__")
    if occam:
        # Every perturbation symbol in the benchmark, read from its own split lists rather
        # than the loaded cells, so the vocabulary is full width even on a smoke run. ⚠️ It
        # fixes the model's condition-input width and cannot grow after training: a test
        # perturbation with no column makes cf.predict raise "Representation ... not found".
        extra_tokens = occam_pert_tokens(data_cfg)
        print(f"[vocab] {len(extra_tokens)} perturbation tokens from rhj_fewshot")
    elif vocab_dirs == "__unset__":
        extra_tokens: list[str] = []
    elif vocab_dirs is None:
        extra_tokens = []
    else:
        extra_tokens = dataset_pert_tokens([str(d) for d in vocab_dirs])
        print(f"[vocab] {len(extra_tokens)} perturbation tokens from {len(vocab_dirs)} dump(s)")

    if occam:
        adata_train = load_occam_sources(data_cfg, seed=seed)
        apply_cellflow_gene_schema(
            adata_train, max_combination_length=int(data_cfg.get("max_combination_length", 1)),
            sample_covariates=sample_covariates, split_covariates=split_covariates,
            extra_tokens=extra_tokens)
        schema = adata_train.uns["cellflow_schema"]
        adata = adata_train
        loaded_rows = np.arange(adata_train.n_obs, dtype=np.int64)
        for name, n_obs in zip(adata_train.uns["occam_bridge"]["sources"],
                               adata_train.uns["occam_bridge"]["n_obs_per_source"],
                               strict=True):
            print(f"[train_cellflow]   source {name}: n_obs={n_obs}")

        adata_valid = load_occam_validation(data_cfg)
        apply_cellflow_gene_schema(
            adata_valid, max_combination_length=int(data_cfg.get("max_combination_length", 1)),
            sample_covariates=sample_covariates, split_covariates=split_covariates,
            extra_tokens=extra_tokens)
        if list(adata_valid.var_names) != list(adata_train.var_names):
            raise ValueError(
                "validation gene axis differs from training; refusing to project it through "
                "the train-fit PCA basis")

        # ⚠️ ONE VOCABULARY, NOT TWO. `apply_cellflow_gene_schema` unions the tokens PRESENT
        # in the object it is given with `extra_tokens` and then sorts, so calling it
        # separately on train and validation can yield dicts of different WIDTH and ORDER --
        # and `DataManager.get_validation_data` encodes conditions from the PASSED object's
        # `uns` (`data/_datamanager.py:186` -> `561-573`). The training vocabulary is the
        # authoritative one, because it fixes the model's condition-input width.
        #
        # This worked until now only by accident: the control-injection path below
        # (`ad.concat` + copying `adata_train.uns`) overwrote the validation `uns` with the
        # training one, and it only ran because this split happens to have zero controls.
        # Assert instead of relying on that.
        _vocab_keys = [k for k in ("cellflow_schema", EMBEDDING_KEY) if k in adata_train.uns]
        for _k in _vocab_keys:
            adata_valid.uns[_k] = adata_train.uns[_k]
        _tv = adata_train.uns[EMBEDDING_KEY] if EMBEDDING_KEY in adata_train.uns else None
        if _tv is not None:
            _n_tok = len(_tv)
            _n_seen = adata_train.obs["occam_target_gene"].nunique()
            print(f"[vocab] one shared vocabulary: {_n_tok} tokens, "
                  f"{_n_seen} distinct perturbations in the training cells")
            if _n_seen > _n_tok:
                raise ValueError(
                    f"{_n_seen} training perturbations against a {_n_tok}-token vocabulary: "
                    f"the model's condition input cannot represent them all, and the excess "
                    f"would silently take another token's column.")
        other["valid"] = adata_valid
        print(f"[train_cellflow] validation: {adata_valid.n_obs} cells from HEPG2 "
              f"({adata_valid.obs['occam_target_gene'].nunique()} perturbations)")
    elif multi_source:
        specs = _build_multi_source_specs(data_cfg, fold, train_key=train_key, seed=seed)
        adata_train = load_dump_sources(
            specs,
            selected_genes=selected,
            feature=data_cfg.get("feature", "auto"),
            target_sum=data_cfg.get("target_sum"),
            log1p=bool(data_cfg.get("log1p", False)),
            on_missing_genes=data_cfg.get("on_missing_genes", "error"),
            seed=seed,
        )
        apply_cellflow_gene_schema(
            adata_train,
            max_combination_length=int(data_cfg.get("max_combination_length", 1)),
            sample_covariates=sample_covariates,
            split_covariates=split_covariates,
            extra_tokens=extra_tokens,
        )
        schema = adata_train.uns["cellflow_schema"]
        adata = adata_train
        loaded_rows = np.arange(adata_train.n_obs, dtype=np.int64)
        for spec, n_obs in zip(specs, adata_train.uns["occam_bridge"]["n_obs_per_source"], strict=True):
            print(f"[train_cellflow]   source {Path(spec['data_dir']).name}: n_obs={n_obs}")

        # Validation for the multi-source path. This used to raise NotImplementedError, which
        # left the arm with no way to select a checkpoint without test data -- the whole point
        # of the protocol every other arm on this benchmark follows.
        #
        # It is loaded as a separate AnnData from the EVALUATED line only (data_cfg["data_dir"]),
        # because the validation split is HepG2 rows; the supporting lines contribute training
        # cells and nothing else. Everything downstream -- the train-fit PCA projection via
        # `other`, `_ensure_validation_controls`, `prepare_validation_data` -- is already
        # generic over `adata_valid`, so populating it here is all that is needed.
        if valid_key in fold and fold[valid_key]:
            adata_valid = load_dump_anndata(
                data_cfg["data_dir"],
                selected_genes=selected,
                feature=data_cfg.get("feature", "auto"),
                row_indices=np.asarray(fold[valid_key], dtype=np.int64),
                target_sum=data_cfg.get("target_sum"),
                log1p=bool(data_cfg.get("log1p", False)),
                on_missing_genes=data_cfg.get("on_missing_genes", "error"),
                seed=seed,
            )
            apply_cellflow_gene_schema(
                adata_valid,
                max_combination_length=int(data_cfg.get("max_combination_length", 1)),
                sample_covariates=sample_covariates,
                split_covariates=split_covariates,
                extra_tokens=extra_tokens,
            )
            # The gene axis must match the training AnnData exactly, or PCA projection through
            # the train-fit basis is meaningless rather than merely wrong-looking.
            if list(adata_valid.var_names) != list(adata_train.var_names):
                raise ValueError(
                    "validation gene axis differs from training; refusing to project it "
                    "through the train-fit PCA basis"
                )
            other["valid"] = adata_valid
            print(
                f"[train_cellflow] validation: {adata_valid.n_obs} cells from "
                f"{Path(str(data_cfg['data_dir'])).name}"
            )
    else:
        load_parts = [fold[train_key]]
        if valid_key in fold and fold[valid_key]:
            load_parts.append(fold[valid_key])
        loaded_rows = _union_indices(*load_parts)

        adata = load_dump_anndata(
            data_cfg["data_dir"],
            selected_genes=selected,
            feature=data_cfg.get("feature", "auto"),
            row_indices=loaded_rows,
            target_sum=data_cfg.get("target_sum"),
            log1p=bool(data_cfg.get("log1p", False)),
            on_missing_genes=data_cfg.get("on_missing_genes", "error"),
            seed=seed,
        )
        apply_cellflow_gene_schema(
            adata,
            max_combination_length=int(data_cfg.get("max_combination_length", 1)),
            sample_covariates=sample_covariates,
            split_covariates=split_covariates,
        )
        schema = adata.uns["cellflow_schema"]
        adata_train = _subset_by_indices(
            adata, _remap_indices(fold[train_key], loaded_rows), name=train_key
        )
        if valid_key in fold and fold[valid_key]:
            adata_valid = _subset_by_indices(
                adata, _remap_indices(fold[valid_key], loaded_rows), name=valid_key
            )
            other["valid"] = adata_valid

    print(
        f"[train_cellflow] loaded n_obs={adata.n_obs} n_vars={adata.n_vars} "
        f"(fold rows={len(loaded_rows)}) controls={int(adata.obs[schema['control_key']].sum())} "
        f"wall_s={time.perf_counter() - t0:.1f}"
    )

    artifacts = fit_and_project_pca(
        adata_train,
        other,
        n_comps=int(pca_cfg.get("n_comps", 50)),
        keep_centered_data=False,
    )
    save_pca_artifacts(artifacts, run_dir / "pca")
    (run_dir / "genes.txt").write_text("\n".join(map(str, adata.var_names)) + "\n")

    if adata_valid is not None and bool(data_cfg.get("inject_train_controls", True)):
        _ct_col = (list(schema.get("split_covariates") or ())
                   or list(schema.get("sample_covariates") or ())
                   or [None])[0]
        adata_valid, n_injected = _ensure_validation_controls(
            adata_valid,
            adata_train,
            control_key=schema["control_key"],
            max_injected=data_cfg.get("max_injected_controls"),
            cell_type_col=_ct_col,
        )
        if n_injected:
            print(
                f"[train_cellflow] injected {n_injected} train controls into validation "
                f"(cell-eval split has none)"
            )

    print(
        f"[train_cellflow] pca n_comps={artifacts['n_comps']} "
        f"train={adata_train.n_obs} valid={0 if adata_valid is None else adata_valid.n_obs}"
    )

    from cellflow.model import CellFlow

    cf = CellFlow(adata_train, solver=str(model_cfg.get("solver", "otfm")))
    cf.prepare_data(
        sample_rep="X_pca",  # PCA-projected data.
        control_key=schema["control_key"],
        perturbation_covariates=dict(schema["perturbation_covariates"]),
        perturbation_covariate_reps=dict(schema["perturbation_covariate_reps"]),
        sample_covariates=list(schema.get("sample_covariates") or ()) or None,
        sample_covariate_reps=dict(schema.get("sample_covariate_reps") or {}) or None,
        split_covariates=list(schema.get("split_covariates") or ()) or None,
        max_combination_length=int(schema["max_combination_length"]),
    )
    if adata_valid is not None:
        # The validation sampler draws conditions without replacement, so asking for
        # more than the split holds is a hard error (Norman fold 0 has only 23).
        n_valid_conditions = int(
            adata_valid.obs.loc[~adata_valid.obs[schema["control_key"]].astype(bool), "pert_str"].nunique()
        )
        # `null`/None means ALL conditions -- upstream's own convention
        # (`model/_cellflow.py:206-211`). `int(None)` would raise, and a NUMBER smaller than
        # the split silently makes each validation pass score a DIFFERENT random subset,
        # because `ValidationSampler` re-draws from one rng per pass; the selection scores
        # then are not comparable to each other. Cap downward only.
        def _n_conditions(key: str, default: int) -> int:
            raw = train_cfg.get(key, default)
            if raw is None:
                return n_valid_conditions
            n = min(int(raw), n_valid_conditions)
            if n < n_valid_conditions:
                print(f"[train_cellflow] ⚠️  {key}={n} of {n_valid_conditions} conditions: "
                      f"each validation pass will score a DIFFERENT random subset, so the "
                      f"selection scores are not comparable across passes. Use null for all.",
                      flush=True)
            return n

        n_log = _n_conditions("n_conditions_on_log_iteration", 8)
        n_end = _n_conditions("n_conditions_on_train_end", 32)
        print(f"[train_cellflow] valid conditions={n_valid_conditions} n_log={n_log} n_end={n_end}")
        cf.prepare_validation_data(
            adata_valid,
            name="valid",
            n_conditions_on_log_iteration=n_log,
            n_conditions_on_train_end=n_end,
        )

    prepare_kwargs: dict[str, Any] = {
        "condition_mode": model_cfg.get("condition_mode", "deterministic"),
        "pooling": model_cfg.get("pooling", "mean"),
        "condition_embedding_dim": int(model_cfg.get("condition_embedding_dim", 128)),
        "cond_output_dropout": float(model_cfg.get("cond_output_dropout", 0.0)),
        "time_freqs": int(model_cfg.get("time_freqs", 64)),
        "time_encoder_dims": tuple(model_cfg.get("time_encoder_dims", [512, 512])),
        "hidden_dims": tuple(model_cfg.get("hidden_dims", [512, 512, 512])),
        "decoder_dims": tuple(model_cfg.get("decoder_dims", [512, 512, 512])),
        "conditioning": model_cfg.get("conditioning", "concatenation"),
        "optimizer": _build_optimizer(
            float(model_cfg.get("learning_rate", 5e-5)),
            int(model_cfg.get("optimizer_multi_steps", 20)),
        ),
        "seed": int(model_cfg.get("seed", 0)),
    }
    # Optional prepare_model knobs (CellFlow tutorial encoder / probability path).
    for key in (
        "layers_before_pool",
        "layers_after_pool",
        "probability_path",
        "linear_projection_before_concatenation",
    ):
        if key in model_cfg and model_cfg[key] is not None:
            prepare_kwargs[key] = model_cfg[key]
    cf.prepare_model(**prepare_kwargs)

    n_iter = int(train_cfg.get("num_iterations", 1000))
    batch_size = int(train_cfg.get("batch_size", 1024))
    valid_freq = int(train_cfg.get("valid_freq", max(n_iter, 1)))
    print(f"[train_cellflow] train iterations={n_iter} batch_size={batch_size} valid_freq={valid_freq}")

    # Validation cell-eval + best-state selection. Without this the run keeps whatever the last
    # iteration produced, which is why the arm previously had no way to pick a checkpoint that
    # had not seen test data. Off unless a validation split exists and the config asks for it,
    # so the other dataset configs are untouched.
    select_cfg = train_cfg.get("selection") or {}
    selector = None
    if adata_valid is not None and bool(select_cfg.get("enabled", False)):
        from cellflow_selection import SelectionCallback

        ctrl_mask = adata_valid.obs[schema["control_key"]].astype(bool).to_numpy()
        ctrl_x = adata_valid.X[ctrl_mask]
        ctrl_x = np.asarray(ctrl_x.todense() if hasattr(ctrl_x, "todense") else ctrl_x)

        # ⚠️ ON `rhj_fewshot` THE VALIDATION SPLIT HAS NO CONTROLS AT ALL. Every one of the
        # 4,976 HepG2 controls is in finetune by construction -- they are the paired basal
        # pool -- so `adata_valid` holds 6,731 perturbed cells over 103 perturbations and
        # ZERO controls. `SelectionCallback.__init__` raises on an empty `control_x`, so this
        # arm would have died at startup, before a single training step.
        #
        # Fall back to the TRAINING controls of the same cell line. That is the right pool and
        # the same `injctrl` convention our own arms use: real, model-visible control cells as
        # the shared baseline, disjoint from the validation perturbations being scored.
        if ctrl_x.shape[0] == 0:
            # ⚠️ `.to_numpy()` on a pandas bool column can return a READ-ONLY view, so the
            # in-place `&=` below raises `ValueError: output array is read-only`. Copy.
            # Caught by the pre-flight, in this very fix, before it cost a job.
            tr_ctrl = np.array(
                adata_train.obs[schema["control_key"]].astype(bool).to_numpy(),
                dtype=bool, copy=True)
            _ct_col_early = (list(schema.get("split_covariates") or ())
                             or list(schema.get("sample_covariates") or ())
                             or ["cell_type"])[0]
            if _ct_col_early in adata_train.obs.columns and \
                    _ct_col_early in adata_valid.obs.columns:
                want = set(adata_valid.obs[_ct_col_early].astype(str).unique().tolist())
                tr_ctrl = tr_ctrl & adata_train.obs[_ct_col_early].astype(str).isin(
                    want).to_numpy()
            cx = adata_train.X[tr_ctrl]
            ctrl_x = np.asarray(cx.todense() if hasattr(cx, "todense") else cx)
            print(f"[train_cellflow] validation has no controls (all are in finetune by "
                  f"construction); using {ctrl_x.shape[0]:,} TRAINING controls of the same "
                  f"cell line as the shared baseline", flush=True)
            if ctrl_x.shape[0] == 0:
                raise ValueError(
                    "no control cells in either the validation or the training split; "
                    "cell-eval cannot measure a perturbation effect without a baseline")
        # The cell-type column is whatever the config conditions on; fall back to the dump's
        # own column name rather than hardcoding it in two places.
        ct_col = (list(schema.get("split_covariates") or ())
                  or list(schema.get("sample_covariates") or ())
                  or ["cell_type"])[0]
        if ct_col not in adata_valid.obs.columns:
            raise KeyError(f"validation obs has no {ct_col!r} column (have {list(adata_valid.obs.columns)})")
        cell_types = adata_valid.obs[ct_col].astype(str).unique().tolist()
        if len(cell_types) != 1:
            # The shared scorer slices per cell type; more than one here would silently score a
            # mixture under a single label.
            raise ValueError(f"validation spans {len(cell_types)} cell types: {cell_types}")
        selector = SelectionCallback(
            pcs=artifacts["pcs"],
            means=artifacts["means"],
            # ⚠️ SYMBOLS, NOT `var_names`. On the occam path `var_names` are ENSEMBL ids
            # (occam_bridge/occam.py keeps that for the positional PCA basis), but these
            # names become the var index of the h5ads cell-eval scores -- and cell-eval
            # excludes each perturbation's own target gene BY SYMBOL. Ensembl here would
            # silently stop that exclusion and inflate every selection score, with no error.
            # Same class of bug as STATE's `gene_names`, which the pre-flight caught.
            genes=[str(g) for g in (adata_train.var["gene_symbol"]
                                    if "gene_symbol" in adata_train.var.columns
                                    else adata_train.var_names)],
            control_x=ctrl_x,
            cell_type=cell_types[0],
            output_root=run_dir / "valscore",
            # OUR cell-eval 0.6.0 (see cellflow_selection.py). Both default to ours, so a
            # config naming no scorer at all is the correct one.
            **({"occam_py": select_cfg["occam_py"]} if "occam_py" in select_cfg else {}),
            **({"occam_repo": select_cfg["occam_repo"]} if "occam_repo" in select_cfg else {}),
            profile=select_cfg.get("profile", "full"),
            # ⚠️ RAW GENE-SPACE TRUTH per perturbation. Without this the selection metric
            # scored the PCA-RECONSTRUCTED target, which deletes everything the rank-256
            # basis cannot express and inflates every component -- the reported metric scores
            # raw half-A cells, so selection was optimising a different quantity. Built from
            # `adata_valid` here because the callback only receives PCA-space tensors.
            truth_by_label=_gene_space_truth(adata_valid, schema),
        )
        print(
            f"[train_cellflow] selection on: {int(ctrl_mask.sum())} control cells, "
            f"cell_type={cell_types[0]}"
        )

    t1 = time.perf_counter()
    cf.train(
        num_iterations=n_iter,
        batch_size=batch_size,
        valid_freq=valid_freq,
        callbacks=[selector.as_callback()] if selector is not None else [],
    )
    print(f"[train_cellflow] train_done wall_s={time.perf_counter() - t1:.1f}")

    # save() AFTER the callback's on_train_end restored the best params, so the single pickle
    # holds the selected state rather than the final one.
    cf.save(str(run_dir), overwrite=True)
    meta = {
        "n_obs_train": int(adata_train.n_obs),
        "n_obs_valid": 0 if adata_valid is None else int(adata_valid.n_obs),
        "n_vars": int(adata_train.n_vars),
        "n_comps": int(artifacts["n_comps"]),
        "num_iterations": n_iter,
        "batch_size": batch_size,
        "schema": schema,
        # ⚠️ The occam path has NEITHER `data_dir` NOR `split_path`: the benchmark directory
        # is both the source manifest and the split, read live (see `occam_bridge/occam.py`),
        # and `run_train` refuses a `split_path` alongside it so the split has one owner.
        # Indexing them unconditionally raised KeyError AFTER the model, the PCA basis and
        # the selection score had all been written -- a run that had done all its work and
        # then failed to describe itself.
        "source": "occam" if occam else "files",
        "occam_bench_dir": str(data_cfg.get("occam_bench_dir") or "") or None,
        "data_dir": str(data_cfg["data_dir"]) if "data_dir" in data_cfg else None,
        "train_sources": [str(s.get("data_dir")) for s in (data_cfg.get("train_sources") or [])],
        "split_path": str(data_cfg["split_path"]) if "split_path" in data_cfg else None,
        "fold": int(data_cfg.get("fold", 0)),
        # run_meta.json carried no seed and no metrics, so a finished run could not say which
        # state it had kept or why.
        "seed": int(model_cfg.get("seed", 0)),
        "valid_freq": valid_freq,
        "selection": selector.summary() if selector is not None else None,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"[train_cellflow] saved -> {run_dir} total_wall_s={time.perf_counter() - t0:.1f}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML train config")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override train.num_iterations")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output.dir")
    parser.add_argument("--run-name", type=str, default=None, help="Override output.run_name")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override, YAML-parsed value (repeatable). Example: model.pooling=attention_token",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    for item in args.set:
        key, value = _parse_set_item(item)
        _set_path(cfg, key, value)
    if args.num_iterations is not None:
        cfg.setdefault("train", {})["num_iterations"] = args.num_iterations
    if args.output_dir is not None:
        cfg.setdefault("output", {})["dir"] = str(args.output_dir)
    if args.run_name is not None:
        cfg.setdefault("output", {})["run_name"] = args.run_name

    run_train(cfg, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
