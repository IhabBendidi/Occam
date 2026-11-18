"""Map Upstream obs into CellFlow control / gene-covariate columns."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd

NULL_TOKEN = "NULL"
CONTROL_KEY = "is_control"
GENE_SLOT_PREFIX = "gene_"
EMBEDDING_KEY = "gene_embeddings"
PERT_STR_KEY = "pert_str"


def _as_tuple_list(raw) -> list[tuple[str, str, str]]:
    """Normalize parquet ``gene_pert`` cells to ``[(kind, ensembl, effect), ...]``."""
    if raw is None:
        return []
    try:
        if len(raw) == 0:
            return []
    except TypeError:
        return []

    out: list[tuple[str, str, str]] = []
    for item in raw:
        parts = list(item)
        if len(parts) < 2:
            raise ValueError(f"gene_pert entry too short: {parts!r}")
        kind = str(parts[0])
        ens = str(parts[1])
        effect = str(parts[2]) if len(parts) > 2 else ""
        out.append((kind, ens, effect))
    return out


def _encode_pert_str(perts: Sequence[tuple[str, str, str]], *, is_control: bool) -> str:
    if is_control or not perts:
        return "empty"
    ids = sorted(p[1] for p in perts if p[0] == "genetic")
    if not ids:
        raise ValueError(f"non-control cell has no genetic perts: {perts!r}")
    return "+".join(ids)


def _one_hot_embeddings(categories: Sequence[str]) -> dict[str, np.ndarray]:
    cats = list(categories)
    eye = np.eye(len(cats), dtype=np.float32)
    return {c: eye[i] for i, c in enumerate(cats)}


def apply_cellflow_gene_schema(
    adata: ad.AnnData,
    *,
    max_combination_length: int = 1,
    null_token: str = NULL_TOKEN,
    control_key: str = CONTROL_KEY,
    embedding_key: str = EMBEDDING_KEY,
    sample_covariates: Sequence[str] = (),
    split_covariates: Sequence[str] = (),
    extra_tokens: Sequence[str] = (),
    inplace: bool = True,
) -> ad.AnnData:
    """Add CellFlow gene-perturbation columns + one-hot ``uns`` embeddings.

    Sets
    ----
    - ``obs[control_key]``: bool from ``negative_control`` or empty ``gene_pert``
    - ``obs[gene_1]`` … ``obs[gene_K]``: Ensembl IDs or ``null_token`` (padded)
    - ``obs[pert_str]``: CellEval-style string (``empty`` / ``ENSG`` / ``ENSG+ENSG``)
    - ``uns[embedding_key]``: ``{token: float32 vector}`` one-hot over all tokens
    - ``uns["cellflow_schema"]``: metadata for train wrappers

    Norman uses ``max_combination_length=2``; single-KO Replogle uses ``1``.

    ``sample_covariates`` are per-cell categorical obs columns (e.g. ``cell_type``)
    that the model conditions on; each gets a one-hot ``uns[f"{col}_embeddings"]``.
    ``split_covariates`` are the obs columns CellFlow uses to split cells into
    separate control populations — for the cross-cell-line few-shot setup both are
    ``["cell_type"]``, so a perturbation is matched to controls of its own line.

    ``extra_tokens`` widens the one-hot vocabulary beyond the perturbations present in
    ``adata``. Pass the full dataset's perturbation set here when training, because the
    vocabulary determines the velocity field's condition-input width and therefore cannot grow
    afterwards: a perturbation absent from the training cells has no column, and
    ``DataManager`` then raises ``Representation for '…' not found`` at predict time, aborting
    the whole prediction.

    That is not hypothetical on the few-shot split. 8 of its 380 HepG2 test perturbations
    (CACTIN, CDC20, CMTR2, LAMB1, MRM1, NOS1AP, SLC16A5, SPAG7 — 330 of 26,878 test cells) were
    never assayed in jurkat/k562/rpe1 either, so they appear in no training cell anywhere.
    Widening the vocabulary to the dataset makes this arm behave like STATE, whose
    ``pert_onehot_map.pt`` already spans all 2,024 dataset perturbations. Note what it does and
    does not buy: those columns are never activated during training, so neither one-hot arm can
    *learn* those 8 — the point is that the scored set matches the other arms instead of the
    prediction failing outright.
    """
    if max_combination_length < 1:
        raise ValueError("max_combination_length must be >= 1")

    out = adata if inplace else adata.copy()
    if "gene_pert" not in out.obs.columns:
        raise KeyError("obs missing gene_pert")

    parsed = [_as_tuple_list(v) for v in out.obs["gene_pert"].tolist()]
    if "negative_control" in out.obs.columns:
        is_control = out.obs["negative_control"].astype(bool).to_numpy()
    else:
        is_control = np.array([len(p) == 0 for p in parsed], dtype=bool)
    # Empty gene_pert always counts as control even if the flag is wrong.
    is_control = is_control | np.array([len(p) == 0 for p in parsed], dtype=bool)

    slot_names = [f"{GENE_SLOT_PREFIX}{i + 1}" for i in range(max_combination_length)]
    slots = {name: [] for name in slot_names}
    pert_strs: list[str] = []
    all_tokens: set[str] = {null_token}

    for ctrl, perts in zip(is_control, parsed, strict=True):
        if any(p[0] != "genetic" for p in perts):
            raise NotImplementedError("CellFlow gene schema only supports genetic gene_pert entries")
        if len(perts) > max_combination_length:
            raise ValueError(
                f"cell has {len(perts)} gene perts but max_combination_length="
                f"{max_combination_length}"
            )
        ids = [p[1] for p in perts]
        # Stable order within a combo (sorted) so gene_1/gene_2 are deterministic.
        ids = sorted(ids)
        padded = ids + [null_token] * (max_combination_length - len(ids))
        if ctrl:
            padded = [null_token] * max_combination_length
        for name, token in zip(slot_names, padded, strict=True):
            slots[name].append(token)
            all_tokens.add(token)
        pert_strs.append(_encode_pert_str(perts, is_control=bool(ctrl)))

    out.obs[control_key] = is_control
    for name in slot_names:
        out.obs[name] = pd.Categorical(slots[name])
    out.obs[PERT_STR_KEY] = pert_strs

    # Union in the caller-supplied vocabulary before building the one-hot, so the width covers
    # perturbations that never appear in these rows (see `extra_tokens` in the docstring).
    all_tokens |= {str(t) for t in extra_tokens}

    tokens = sorted(all_tokens)
    # Keep NULL first so pad is easy to spot in embeddings.
    tokens = [null_token] + [t for t in tokens if t != null_token]
    out.uns[embedding_key] = _one_hot_embeddings(tokens)

    sample_covariate_reps: dict[str, str] = {}
    for col in sample_covariates:
        if col not in out.obs.columns:
            raise KeyError(f"sample covariate {col!r} not in obs")
        categories = sorted({str(v) for v in out.obs[col]})
        rep_key = f"{col}_embeddings"
        out.obs[col] = pd.Categorical(out.obs[col].astype(str))
        out.uns[rep_key] = _one_hot_embeddings(categories)
        sample_covariate_reps[col] = rep_key

    for col in split_covariates:
        if col not in out.obs.columns:
            raise KeyError(f"split covariate {col!r} not in obs")

    out.uns["cellflow_schema"] = {
        "control_key": control_key,
        "perturbation_covariates": {"gene": tuple(slot_names)},
        "perturbation_covariate_reps": {"gene": embedding_key},
        "sample_covariates": tuple(sample_covariates),
        "sample_covariate_reps": sample_covariate_reps,
        "split_covariates": tuple(split_covariates),
        "max_combination_length": max_combination_length,
        "null_token": null_token,
    }
    return out
