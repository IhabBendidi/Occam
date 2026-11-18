"""Split JSON helpers (Upstream / cell-eval ``{"folds": [...]}`` format)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fold(
    split_path: Path | str,
    *,
    fold: int = 0,
) -> dict[str, list[int]]:
    """Load one fold of cell indices from a Upstream/cell-eval split JSON.

    Expected shape::

        {"folds": [{"finetune": [...], "validation": [...], "test": [...]}, ...]}
    """
    split_path = Path(split_path)
    with split_path.open() as handle:
        doc: dict[str, Any] = json.load(handle)
    if "folds" not in doc:
        raise KeyError(f"{split_path}: missing 'folds'")
    folds = doc["folds"]
    if fold < 0 or fold >= len(folds):
        raise IndexError(f"fold {fold} out of range (n_folds={len(folds)}) for {split_path}")
    fold_doc = folds[fold]
    out: dict[str, list[int]] = {}
    skipped: list[str] = []
    for key, value in fold_doc.items():
        # Scalar entries are fold METADATA, not cell indices, and are skipped rather than
        # rejected. `split_replogle_nadig_fewshot__v1.json` carries `outer_fold` / `inner_fold`
        # ints alongside its index lists, and raising on them made the whole split -- the one
        # every other competitor arm is benchmarked on -- unloadable.
        #
        # A non-list, non-scalar value is still an error: silently dropping something
        # dict-shaped could hide a real index list behind a schema change.
        if isinstance(value, list):
            out[str(key)] = [int(i) for i in value]
        elif isinstance(value, (int, float, str)) or value is None:
            skipped.append(str(key))
        else:
            raise TypeError(
                f"{split_path} fold {fold} key {key!r}: expected a list of indices or scalar "
                f"metadata, got {type(value).__name__}"
            )
    if not out:
        raise ValueError(f"{split_path} fold {fold}: no index lists found (keys: {skipped})")
    return out
