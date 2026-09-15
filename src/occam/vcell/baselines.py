"""Baseline models for Virtual Cell Mode evaluation.

Adapted from TxPM baselines (gspp/models/baselines.py).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from occam.vcell.dataset import VirtualCellDataset


class MeanBaseline:
    """Predicts ctrl_mean + per-perturbation delta from training data.

    For seen perturbations: uses perturbation-specific mean delta.
    For unseen perturbations: uses global mean delta across all perturbations.
    """

    def prepare(
        self,
        train_dataset: VirtualCellDataset,
        control_means: Dict[str, np.ndarray],
    ):
        """Compute per-perturbation deltas from training data.

        Args:
            train_dataset: Training VirtualCellDataset.
            control_means: {cell_type: (n_genes,) mean control expression}.
        """
        self.control_means = control_means

        # Group training perturbation data by (pert_name, cell_type)
        pert_sums: Dict[str, Dict[str, Tuple[np.ndarray, int]]] = defaultdict(dict)
        global_sums: Dict[str, List[Tuple[np.ndarray, int]]] = defaultdict(list)

        n_pert = train_dataset._n_pert  # only perturbation cells, not controls

        for i in tqdm(range(n_pert), desc="MeanBaseline: computing deltas"):
            pname = train_dataset.pert_gene_names[i]
            ct = train_dataset.cell_types[i]
            if pname == "non-targeting":
                continue

            expr = train_dataset.expression_data[i].numpy()

            if ct not in control_means:
                continue
            delta = expr - control_means[ct]

            if pname not in pert_sums[ct]:
                pert_sums[ct][pname] = (delta.copy(), 1)
            else:
                prev_sum, prev_n = pert_sums[ct][pname]
                pert_sums[ct][pname] = (prev_sum + delta, prev_n + 1)

            global_sums[ct].append((delta, 1))

        # Compute means (store count for weighted aggregation across cell types)
        self.pert_deltas: Dict[str, Dict[str, Tuple[np.ndarray, int]]] = {}
        for ct in pert_sums:
            self.pert_deltas[ct] = {}
            for pname, (s, n) in pert_sums[ct].items():
                self.pert_deltas[ct][pname] = (s / n, n)

        self.global_deltas: Dict[str, np.ndarray] = {}
        for ct, items in global_sums.items():
            total_sum = sum(d for d, _ in items)
            total_n = sum(n for _, n in items)
            self.global_deltas[ct] = total_sum / total_n

        self.train_perts = set()
        for ct_perts in pert_sums.values():
            self.train_perts.update(ct_perts.keys())

        print(
            f"MeanBaseline: {len(self.train_perts)} unique perturbations, "
            f"{len(self.global_deltas)} cell types"
        )

    def predict(
        self,
        test_dataset: VirtualCellDataset,
        test_control_means: Dict[str, np.ndarray],
    ) -> Dict:
        """Predict perturbation effects for test data.

        Args:
            test_dataset: Test VirtualCellDataset.
            test_control_means: {cell_type: control_mean} for test cell types.

        Returns:
            Dict with predictions, ground_truth, pert_names, cell_types arrays.
        """
        preds = []
        truths = []
        pert_names = []
        cell_types = []
        gene_masks = []

        n_global, n_local = 0, 0

        for i in tqdm(range(test_dataset._n_pert), desc="MeanBaseline: predicting"):
            pname = test_dataset.pert_gene_names[i]
            ct = test_dataset.cell_types[i]
            if pname == "non-targeting":
                continue

            truth = test_dataset.expression_data[i].numpy()
            ctrl_mean = test_control_means.get(ct)
            if ctrl_mean is None:
                continue

            # Check if perturbation was seen in training
            has_specific = any(
                pname in self.pert_deltas.get(train_ct, {})
                for train_ct in self.pert_deltas
            )

            if has_specific:
                # Sample-count-weighted average of per-cell-type deltas
                weighted_sum = np.zeros_like(ctrl_mean)
                total_count = 0
                for train_ct, ct_perts in self.pert_deltas.items():
                    if pname in ct_perts:
                        mean_delta, count = ct_perts[pname]
                        weighted_sum += mean_delta * count
                        total_count += count
                delta = weighted_sum / total_count
                n_local += 1
            else:
                # Use global delta (from closest or any cell type)
                if ct in self.global_deltas:
                    delta = self.global_deltas[ct]
                else:
                    delta = np.mean(
                        list(self.global_deltas.values()), axis=0
                    )
                n_global += 1

            pred = ctrl_mean + delta
            preds.append(pred)
            truths.append(truth)
            pert_names.append(pname)
            cell_types.append(ct)
            gene_masks.append(test_dataset.gene_masks[i].numpy())

        print(
            f"MeanBaseline: {n_local} specific + {n_global} global = "
            f"{n_local + n_global} predictions"
        )

        return {
            "predictions": np.stack(preds),
            "ground_truth": np.stack(truths),
            "pert_names": np.array(pert_names),
            "cell_types": np.array(cell_types),
            "gene_masks": np.stack(gene_masks),
        }


class SimpleMeanBaseline:
    """Predicts the global mean expression of all training perturbation cells.

    The same vector is used as prediction for every test perturbation.
    """

    def prepare(
        self,
        train_dataset: VirtualCellDataset,
    ):
        n_pert = train_dataset._n_pert
        running_sum = None
        count = 0

        for i in range(n_pert):
            if train_dataset.pert_gene_names[i] == "non-targeting":
                continue
            expr = train_dataset.expression_data[i].numpy()
            if running_sum is None:
                running_sum = expr.copy()
            else:
                running_sum += expr
            count += 1

        self.mean_expr = running_sum / count if count > 0 else running_sum
        print(f"SimpleMeanBaseline: computed mean from {count} perturbation cells")

    def predict(
        self,
        test_dataset: VirtualCellDataset,
    ) -> Dict:
        preds = []
        truths = []
        pert_names = []
        cell_types = []
        gene_masks = []

        for i in range(test_dataset._n_pert):
            pname = test_dataset.pert_gene_names[i]
            if pname == "non-targeting":
                continue
            preds.append(self.mean_expr)
            truths.append(test_dataset.expression_data[i].numpy())
            pert_names.append(pname)
            cell_types.append(test_dataset.cell_types[i])
            gene_masks.append(test_dataset.gene_masks[i].numpy())

        print(f"SimpleMeanBaseline: {len(preds)} predictions")

        return {
            "predictions": np.stack(preds),
            "ground_truth": np.stack(truths),
            "pert_names": np.array(pert_names),
            "cell_types": np.array(cell_types),
            "gene_masks": np.stack(gene_masks),
        }


class ControlBaseline:
    """Predicts the mean control expression of the test cell type.

    The same control mean vector is used as prediction for every test perturbation.
    """

    def predict(
        self,
        test_dataset: VirtualCellDataset,
        control_means: Dict[str, np.ndarray],
    ) -> Dict:
        preds = []
        truths = []
        pert_names = []
        cell_types = []
        gene_masks = []

        for i in range(test_dataset._n_pert):
            pname = test_dataset.pert_gene_names[i]
            ct = test_dataset.cell_types[i]
            if pname == "non-targeting":
                continue
            ctrl_mean = control_means.get(ct)
            if ctrl_mean is None:
                continue
            preds.append(ctrl_mean)
            truths.append(test_dataset.expression_data[i].numpy())
            pert_names.append(pname)
            cell_types.append(ct)
            gene_masks.append(test_dataset.gene_masks[i].numpy())

        print(f"ControlBaseline: {len(preds)} predictions")

        return {
            "predictions": np.stack(preds),
            "ground_truth": np.stack(truths),
            "pert_names": np.array(pert_names),
            "cell_types": np.array(cell_types),
            "gene_masks": np.stack(gene_masks),
        }


class ExperimentalAccuracy:
    """Upper-bound baseline: split each perturbation's cells into two halves.

    Pairs individual cells 1:1 across the two halves (no averaging).
    Estimates best achievable performance given biological noise.
    """

    def predict(
        self,
        test_dataset: VirtualCellDataset,
        seed: int = 42,
    ) -> Dict:
        """Split each perturbation's cells into two halves, paired 1:1.

        Args:
            test_dataset: Test VirtualCellDataset.
            seed: Random seed.

        Returns:
            Dict with predictions (half A), ground_truth (half B),
            pert_names, cell_types arrays. Rows are paired 1:1.
        """
        rng = np.random.RandomState(seed)

        groups: Dict[str, List[int]] = defaultdict(list)
        for i in range(test_dataset._n_pert):
            pname = test_dataset.pert_gene_names[i]
            ct = test_dataset.cell_types[i]
            if pname == "non-targeting":
                continue
            groups[f"{pname}_{ct}"].append(i)

        preds = []
        truths = []
        pert_names_out = []
        cell_types_out = []
        gene_masks_out = []

        for key, indices in tqdm(
            groups.items(), desc="ExperimentalAccuracy: splitting"
        ):
            n = len(indices)
            if n < 2:
                continue

            shuffled = rng.permutation(indices)
            half = n // 2
            half_a = shuffled[:half]
            half_b = shuffled[half:half + half]

            for ai, bi in zip(half_a, half_b):
                pname = test_dataset.pert_gene_names[ai]
                ct = test_dataset.cell_types[ai]
                preds.append(test_dataset.expression_data[ai].numpy())
                truths.append(test_dataset.expression_data[bi].numpy())
                pert_names_out.append(pname)
                cell_types_out.append(ct)
                gene_masks_out.append(test_dataset.gene_masks[ai].numpy())

        print(f"ExperimentalAccuracy: {len(preds)} paired cells")

        return {
            "predictions": np.stack(preds) if preds else np.empty((0, 0)),
            "ground_truth": np.stack(truths) if truths else np.empty((0, 0)),
            "pert_names": np.array(pert_names_out),
            "cell_types": np.array(cell_types_out),
            "gene_masks": np.stack(gene_masks_out) if gene_masks_out else np.empty((0, 0)),
        }


# ---------------------------------------------------------------------------
# Shared evaluation function (used by both pipeline.py and run_baselines.py)
# ---------------------------------------------------------------------------


def _compute_per_celltype_metrics(
    results: Dict,
    control_means: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute pearson_delta broken down by cell type.

    Returns:
        {"overall": {"pearson_delta_mean": ..., ...},
         "HepG2": {"pearson_delta_mean": ..., ...}, ...}
    """
    from occam.vcell.metrics import compute_perturbation_metrics

    predictions = results["predictions"]
    ground_truth = results["ground_truth"]
    pert_names = results["pert_names"]
    cell_types = results["cell_types"]

    if predictions.shape[0] == 0:
        return {}

    # Build per-cell-type gene masks from results
    gene_masks_per_ct = None
    if "gene_masks" in results and results["gene_masks"].size > 0:
        gene_masks_per_ct = {}
        for ct in np.unique(cell_types):
            ct_idx = np.where(cell_types == ct)[0][0]
            gene_masks_per_ct[ct] = results["gene_masks"][ct_idx].astype(bool)

    # Overall metrics
    agg, per_pert = compute_perturbation_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        pert_names=pert_names,
        cell_types=cell_types,
        control_means=control_means,
        gene_masks_per_ct=gene_masks_per_ct,
    )

    out = {"overall": agg}

    # Per cell type
    unique_cts = np.unique(cell_types)
    ct_means = []
    for ct in unique_cts:
        ct_mask = cell_types == ct
        if ct_mask.sum() == 0 or ct not in control_means:
            continue
        ct_agg, _ = compute_perturbation_metrics(
            predictions=predictions[ct_mask],
            ground_truth=ground_truth[ct_mask],
            pert_names=pert_names[ct_mask],
            cell_types=cell_types[ct_mask],
            control_means=control_means,
            gene_masks_per_ct=gene_masks_per_ct,
        )
        out[ct] = ct_agg
        ct_means.append(ct_agg["pearson_delta_mean"])

    # Geometric mean across cell types (for multi-dataset aggregate)
    if len(ct_means) > 1:
        # Shift to positive range before geometric mean (pearson_delta can be negative)
        ct_arr = np.array(ct_means)
        out["overall"]["pearson_delta_arithmetic_mean_across_ct"] = float(np.mean(ct_arr))
        # Geometric mean on clipped-positive values
        ct_clipped = np.clip(ct_arr, 1e-10, None)
        out["overall"]["pearson_delta_geometric_mean_across_ct"] = float(
            np.exp(np.mean(np.log(ct_clipped)))
        )

    return out


def run_baseline_evaluation(
    train_dataset: VirtualCellDataset,
    eval_datasets: List[Tuple[str, VirtualCellDataset]],
    control_means: Dict[str, np.ndarray],
    seed: int = 42,
) -> Dict[str, Dict[str, Dict]]:
    """Run all baselines: Control, SimpleMean, Mean, ExperimentalAccuracy.

    Args:
        train_dataset: Training VirtualCellDataset (for baseline preparation).
        eval_datasets: List of (split_name, dataset) pairs to evaluate on.
        control_means: {cell_type: (n_genes,) mean control expression}.
        seed: Random seed for ExperimentalAccuracy.

    Returns:
        {"control_baseline": ..., "simple_mean_baseline": ...,
         "mean_baseline": ..., "experimental_accuracy": ...}
        Each maps split_name -> metrics_dict with keys: "overall",
        per-cell-type, with pearson_delta_mean, median, std, etc.
    """
    results = {
        "control_baseline": {},
        "simple_mean_baseline": {},
        "mean_baseline": {},
        "experimental_accuracy": {},
    }

    # ControlBaseline
    print("Running ControlBaseline...")
    ctrl_bl = ControlBaseline()

    for split_name, ds in eval_datasets:
        bl_results = ctrl_bl.predict(ds, control_means)
        if bl_results["predictions"].shape[0] == 0:
            print(f"  ControlBaseline [{split_name}]: no predictions (skipped)")
            continue
        metrics = _compute_per_celltype_metrics(bl_results, control_means)
        results["control_baseline"][split_name] = metrics

        overall = metrics.get("overall", {})
        print(
            f"  ControlBaseline [{split_name}]: "
            f"pearson_delta_mean={overall.get('pearson_delta_mean', 0):.4f} "
            f"(median={overall.get('pearson_delta_median', 0):.4f}, "
            f"{overall.get('n_pert_celltype_pairs', 0)} pairs)"
        )
        for ct_name, ct_metrics in metrics.items():
            if ct_name == "overall":
                continue
            print(
                f"    {ct_name}: pearson_delta_mean="
                f"{ct_metrics.get('pearson_delta_mean', 0):.4f} "
                f"({ct_metrics.get('n_pert_celltype_pairs', 0)} pairs)"
            )

    # SimpleMeanBaseline
    print("Running SimpleMeanBaseline...")
    simple_bl = SimpleMeanBaseline()
    simple_bl.prepare(train_dataset)

    for split_name, ds in eval_datasets:
        bl_results = simple_bl.predict(ds)
        if bl_results["predictions"].shape[0] == 0:
            print(f"  SimpleMeanBaseline [{split_name}]: no predictions (skipped)")
            continue
        metrics = _compute_per_celltype_metrics(bl_results, control_means)
        results["simple_mean_baseline"][split_name] = metrics

        overall = metrics.get("overall", {})
        print(
            f"  SimpleMeanBaseline [{split_name}]: "
            f"pearson_delta_mean={overall.get('pearson_delta_mean', 0):.4f} "
            f"(median={overall.get('pearson_delta_median', 0):.4f}, "
            f"{overall.get('n_pert_celltype_pairs', 0)} pairs)"
        )
        for ct_name, ct_metrics in metrics.items():
            if ct_name == "overall":
                continue
            print(
                f"    {ct_name}: pearson_delta_mean="
                f"{ct_metrics.get('pearson_delta_mean', 0):.4f} "
                f"({ct_metrics.get('n_pert_celltype_pairs', 0)} pairs)"
            )

    # MeanBaseline
    print("Running MeanBaseline...")
    mean_bl = MeanBaseline()
    mean_bl.prepare(train_dataset, control_means)

    for split_name, ds in eval_datasets:
        bl_results = mean_bl.predict(ds, control_means)
        if bl_results["predictions"].shape[0] == 0:
            print(f"  MeanBaseline [{split_name}]: no predictions (skipped)")
            continue
        metrics = _compute_per_celltype_metrics(bl_results, control_means)
        results["mean_baseline"][split_name] = metrics

        overall = metrics.get("overall", {})
        print(
            f"  MeanBaseline [{split_name}]: "
            f"pearson_delta_mean={overall.get('pearson_delta_mean', 0):.4f} "
            f"(median={overall.get('pearson_delta_median', 0):.4f}, "
            f"{overall.get('n_pert_celltype_pairs', 0)} pairs)"
        )
        for ct_name, ct_metrics in metrics.items():
            if ct_name == "overall":
                continue
            print(
                f"    {ct_name}: pearson_delta_mean="
                f"{ct_metrics.get('pearson_delta_mean', 0):.4f} "
                f"({ct_metrics.get('n_pert_celltype_pairs', 0)} pairs)"
            )

    # ExperimentalAccuracy
    print("Running ExperimentalAccuracy...")
    exp_bl = ExperimentalAccuracy()

    for split_name, ds in eval_datasets:
        bl_results = exp_bl.predict(ds, seed=seed)
        if bl_results["predictions"].shape[0] == 0:
            print(f"  ExperimentalAccuracy [{split_name}]: no predictions (skipped)")
            continue
        metrics = _compute_per_celltype_metrics(bl_results, control_means)
        results["experimental_accuracy"][split_name] = metrics

        overall = metrics.get("overall", {})
        print(
            f"  ExperimentalAccuracy [{split_name}]: "
            f"pearson_delta_mean={overall.get('pearson_delta_mean', 0):.4f} "
            f"(median={overall.get('pearson_delta_median', 0):.4f}, "
            f"{overall.get('n_pert_celltype_pairs', 0)} pairs)"
        )
        for ct_name, ct_metrics in metrics.items():
            if ct_name == "overall":
                continue
            print(
                f"    {ct_name}: pearson_delta_mean="
                f"{ct_metrics.get('pearson_delta_mean', 0):.4f} "
                f"({ct_metrics.get('n_pert_celltype_pairs', 0)} pairs)"
            )

    return results
