"""Perturbation-level train/val/test splitting for Virtual Cell Mode.

Splits are on perturbation gene names (not individual cells), ensuring
all cells for a given perturbation stay in the same split.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import anndata
import numpy as np


class PerturbationSplitter:
    """Split perturbation gene names into train/val/test sets.

    Supports holdout cell line mode: when a cell line is held out,
    two test sets are produced:
      - test_unseen_pert: unseen perturbations from training cell lines
      - test_unseen_cell: ALL perturbations from the held-out cell line

    Args:
        train_fraction: Fraction of perturbations for training.
        val_fraction: Fraction for validation.
        test_fraction: Fraction for test (unseen perturbation).
        seed: Random seed for reproducibility.
        holdout_cell_line: Cell line to hold out entirely (or None).
    """

    def __init__(
        self,
        train_fraction: float = 0.75,
        val_fraction: float = 0.05,
        test_fraction: float = 0.20,
        seed: int = 42,
        holdout_cell_line: Optional[str] = None,
        all_perts_train: bool = False,
    ):
        assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.holdout_cell_line = holdout_cell_line
        self.all_perts_train = all_perts_train

    def split(
        self,
        pert_names_per_cell_line: Dict[str, Set[str]],
    ) -> Dict:
        """Create the split.

        Args:
            pert_names_per_cell_line: {cell_line: set of perturbation gene names}.

        Returns:
            Dict with keys: train, val, test_unseen_pert, test_unseen_cell,
            holdout_cell_line, seed, cell_lines_train, cell_lines_test.
        """
        rng = np.random.RandomState(self.seed)

        all_cell_lines = sorted(pert_names_per_cell_line.keys())

        # all_perts_train: every perturbation of every training cell line goes into train.
        # Used when the test set is an EXTERNAL held-out cell line (H1), so there is no
        # reason to sacrifice in-domain perturbations to an in-domain test split — doing so
        # would only shrink the training signal available for cross-cell-line transfer.
        # Validation then has to come from outside this corpus.
        if getattr(self, "all_perts_train", False):
            pool = sorted(set().union(*pert_names_per_cell_line.values())) \
                if pert_names_per_cell_line else []
            result = {
                "train": pool, "val": [], "test_unseen_pert": [], "test_unseen_cell": [],
                "holdout_cell_line": None, "seed": self.seed,
                "cell_lines_train": all_cell_lines, "cell_lines_test": [],
                "train_fraction": 1.0, "val_fraction": 0.0, "test_fraction": 0.0,
                "all_perts_train": True,
            }
            print(f"Split: ALL {len(pool)} perturbations -> train "
                  f"({len(all_cell_lines)} cell lines); val/test come from the "
                  f"external held-out cell line")
            return result

        if self.holdout_cell_line is not None:
            holdout = self.holdout_cell_line
            if holdout not in all_cell_lines:
                raise ValueError(
                    f"Holdout cell line '{holdout}' not in data. "
                    f"Available: {all_cell_lines}"
                )
            train_cell_lines = [cl for cl in all_cell_lines if cl != holdout]
            test_cell_lines = [holdout]

            # Pool perturbations from training cell lines
            train_pool = set()
            for cl in train_cell_lines:
                train_pool.update(pert_names_per_cell_line[cl])
            train_pool = sorted(train_pool)

            # All perturbations from held-out cell line
            test_unseen_cell = sorted(pert_names_per_cell_line[holdout])
        else:
            train_cell_lines = all_cell_lines
            test_cell_lines = []

            # Pool all perturbations
            train_pool = set()
            for cl in all_cell_lines:
                train_pool.update(pert_names_per_cell_line[cl])
            train_pool = sorted(train_pool)
            test_unseen_cell = []

        # Shuffle and split the training pool
        rng.shuffle(train_pool)
        n = len(train_pool)
        n_test = max(1, int(n * self.test_fraction))
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val - n_test

        train_perts = train_pool[:n_train]
        val_perts = train_pool[n_train : n_train + n_val]
        test_unseen_pert = train_pool[n_train + n_val :]

        result = {
            "train": sorted(train_perts),
            "val": sorted(val_perts),
            "test_unseen_pert": sorted(test_unseen_pert),
            "test_unseen_cell": sorted(test_unseen_cell),
            "holdout_cell_line": self.holdout_cell_line,
            "seed": self.seed,
            "cell_lines_train": sorted(train_cell_lines),
            "cell_lines_test": sorted(test_cell_lines),
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
        }

        print(
            f"Split: train={len(train_perts)}, val={len(val_perts)}, "
            f"test_unseen_pert={len(test_unseen_pert)}, "
            f"test_unseen_cell={len(test_unseen_cell)}"
        )
        return result

    @staticmethod
    def save(splits: Dict, path: str):
        """Save splits to JSON for reproducibility."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Splits saved to {path}")

    @staticmethod
    def load(path: str) -> Dict:
        """Load splits from JSON."""
        with open(path, "r") as f:
            splits = json.load(f)
        print(
            f"Loaded splits from {path}: "
            f"train={len(splits['train'])}, val={len(splits['val'])}, "
            f"test_unseen_pert={len(splits['test_unseen_pert'])}, "
            f"test_unseen_cell={len(splits['test_unseen_cell'])}"
        )
        return splits


