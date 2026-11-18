import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg
from scipy.stats import hypergeom, ks_2samp
from pathlib import Path


# =========================
# TVN utilities
# =========================

def centerscale_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: Any,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    embeddings = embeddings.copy()
    if batch_col is not None and batch_col in metadata.columns:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = batch_ind & (metadata[pert_col] == control_key)
            if not np.any(batch_control_ind):
                continue
            embeddings[batch_ind] = StandardScaler().fit(embeddings[batch_control_ind]).transform(embeddings[batch_ind])
        return embeddings

    control_ind = metadata[pert_col] == control_key
    if not np.any(control_ind):
        return embeddings
    return StandardScaler().fit(embeddings[control_ind]).transform(embeddings)


def tvn_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: Any,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    print("[TVN] Copying embeddings")
    embeddings = embeddings.copy()
    print("[TVN] Center scaling on control...")
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    print("[TVN] Filtering control indices")
    ctrl_ind = metadata[pert_col] == control_key
    if not np.any(ctrl_ind):
        print("[TVN] No control rows found; skipping PCA/second scaling")
        return embeddings
    print("[TVN] Fitting PCA on control embeddings and transforming all embeddings")
    embeddings = PCA().fit(embeddings[ctrl_ind]).transform(embeddings)
    print("[TVN] Centering on control again")
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    if batch_col is not None and batch_col in metadata.columns:
        print("[TVN] Applying CORAL-like batch covariance normalization")
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = batch_ind & (metadata[pert_col] == control_key)
            if not np.any(batch_control_ind):
                continue
            source_cov = np.cov(embeddings[batch_control_ind], rowvar=False, ddof=1) + 0.5 * np.eye(embeddings.shape[1])
            source_cov_half_inv = linalg.fractional_matrix_power(source_cov, -0.5)
            embeddings[batch_ind] = np.matmul(embeddings[batch_ind], source_cov_half_inv)
    return embeddings


def v2_normalize(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: Any,
    batch_col: Optional[str] = None,
    n_components: Optional[int] = None,
) -> np.ndarray:
    embeddings = embeddings.copy()
    ctrl_mask = (metadata[pert_col] == control_key).values

    if not np.any(ctrl_mask):
        print("[v2_normalize] No control rows found; returning raw embeddings")
        return embeddings

    pca = PCA(n_components=n_components).fit(embeddings[ctrl_mask])
    embeddings = pca.transform(embeddings)

    if batch_col is not None and batch_col in metadata.columns:
        for batch in metadata[batch_col].unique():
            batch_mask = (metadata[batch_col] == batch).values
            batch_ctrl = batch_mask & ctrl_mask
            if batch_ctrl.sum() < 10:
                scaler = StandardScaler().fit(embeddings[ctrl_mask])
            else:
                scaler = StandardScaler().fit(embeddings[batch_ctrl])
            embeddings[batch_mask] = scaler.transform(embeddings[batch_mask])
    else:
        scaler = StandardScaler().fit(embeddings[ctrl_mask])
        embeddings = scaler.transform(embeddings)

    return embeddings


# =========================
# Benchmark utilities
# =========================

def filter_relationships(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sorted_entities"] = df.apply(lambda row: tuple(sorted([row.entity1, row.entity2])), axis=1)
    df["entity1"] = df.sorted_entities.apply(lambda x: x[0])
    df["entity2"] = df.sorted_entities.apply(lambda x: x[1])
    return df[["entity1", "entity2"]].query("entity1!=entity2").drop_duplicates()


def get_benchmark_relationships(benchmark_data_dir: str, src: str, filter: bool = True) -> pd.DataFrame:
    path = Path(benchmark_data_dir).joinpath(src + ".txt")
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    df = pd.read_csv(path)
    return filter_relationships(df) if filter else df


def compute_recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
    recall_threshold_pairs: list,
) -> dict:
    metrics = {}
    metrics["null_distribution_size"] = null_distribution.shape[0]
    metrics["query_distribution_size"] = query_distribution.shape[0]

    sorted_null_distribution = np.sort(null_distribution)
    query_percentage_ranks_left = np.searchsorted(sorted_null_distribution, query_distribution, side="left") / len(
        sorted_null_distribution
    )
    query_percentage_ranks_right = np.searchsorted(sorted_null_distribution, query_distribution, side="right") / len(
        sorted_null_distribution
    )
    for threshold_pair in recall_threshold_pairs:
        left_threshold, right_threshold = np.min(threshold_pair), np.max(threshold_pair)
        metrics[f"recall_{left_threshold}_{right_threshold}"] = float(
            sum((query_percentage_ranks_right <= left_threshold) | (query_percentage_ranks_left >= right_threshold))
        ) / float(len(query_distribution) if len(query_distribution) > 0 else 1)
    return metrics


def convert_metrics_to_df(metrics: dict, source: str) -> pd.DataFrame:
    metrics_dict_with_list = {key: [value] for key, value in metrics.items()}
    metrics_dict_with_list["source"] = [source]
    return pd.DataFrame.from_dict(metrics_dict_with_list)


def known_relationship_benchmark(
    map_data: Bunch,
    pert_col: str,
    benchmark_sources: list,
    recall_thr_pairs: list,
    benchmark_data_dir: str,
    min_req_entity_cnt: int = 20,
    log_stats: bool = False,
) -> pd.DataFrame:
    md = map_data.metadata
    features_df = map_data.features.set_index(md[pert_col]).rename_axis(index=None)
    if not len(features_df) == len(set(features_df.index)):
        raise ValueError("Duplicate perturbation labels in the map.")
    if not len(features_df) >= min_req_entity_cnt:
        raise ValueError("Not enough entities in the map for benchmarking.")
    if log_stats:
        print(len(features_df), "perturbations exist in the map.")

    metrics_lst = []
    cossim_matrix = pd.DataFrame(cosine_similarity(features_df, features_df), index=features_df.index, columns=features_df.index)
    cossim_values = cossim_matrix.values[np.triu_indices(cossim_matrix.shape[0], k=1)]
    for s in benchmark_sources:
        try:
            rels = get_benchmark_relationships(benchmark_data_dir, s)
        except FileNotFoundError as e:
            print(f"[Benchmark] {e} — skipping source '{s}'")
            continue
        rels = rels[rels.entity1.isin(features_df.index) & rels.entity2.isin(features_df.index)]
        if len(rels) == 0:
            if log_stats:
                print("[Benchmark] No relationships found for source", s)
            continue
        query_cossim = np.array([cossim_matrix.loc[e1, e2] for e1, e2 in rels.itertuples(index=False)])
        if log_stats:
            print(len(query_cossim), "relationships are used from the benchmark source", s)
        if len(query_cossim) > 0:
            metrics_lst.append(
                convert_metrics_to_df(metrics=compute_recall(cossim_values, query_cossim, recall_thr_pairs), source=s)
            )
    if len(metrics_lst) == 0:
        return pd.DataFrame()
    return pd.concat(metrics_lst, ignore_index=True)


def aggregate(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    keys_to_remove: List[str] = [],
    method: str = "mean",
) -> Bunch:
    final_embeddings = []
    metadata = metadata.reset_index(drop=True)
    aggr_func = np.mean if method == "mean" else np.median if method == "median" else None
    if aggr_func is None:
        raise ValueError(f"Invalid aggregation method: {method}")
    grouping = metadata.groupby(pert_col)
    unique_perts = []
    for pert, group in grouping:
        if pert in keys_to_remove:
            continue
        final_emb = aggr_func(embeddings[group.index.values, :], axis=0)
        final_embeddings.append(final_emb)
        unique_perts.append(pert)
    return Bunch(
        features=pd.DataFrame(np.vstack(final_embeddings)), metadata=pd.DataFrame.from_dict({pert_col: unique_perts})
    )


# =========================
# Perturbation consistency benchmark
# =========================

def cv_stats(
    X: np.ndarray,
    n_examples_per_group: np.ndarray,
    stat: str,
    return_mean: bool = False,
    aggregation_fn: str = "mean",
) -> np.ndarray:
    EPSILON = 1e-7
    n_examples = X.shape[1]

    if aggregation_fn == "mean":
        group_sum = np.sum(X, axis=1)

    stats = []
    for i in range(n_examples):
        current_slice = X[:, i, :]

        if aggregation_fn == "mean":
            agg = (group_sum - current_slice) / (n_examples_per_group - 1)[:, np.newaxis]
        else:
            idx = np.ones(X.shape[1], dtype=bool)
            idx[i] = False
            other_slices = X[:, idx, :]
            agg = aggregation_fn(other_slices)
            if agg.shape == ():
                raise ValueError("aggregation_fn aggregated across all values rather than a single dimension")

        rowwise_dot = np.sum(current_slice * agg, axis=1)
        agg_norm = np.linalg.norm(agg, axis=1)
        current_slice_norm = np.linalg.norm(current_slice, axis=1)
        cossim = rowwise_dot / (current_slice_norm * agg_norm + EPSILON)
        cossim = np.clip(cossim, -1.0, 1.0)

        if stat == "cossim":
            stats.append(cossim)
        elif stat == "angle":
            angles = np.arccos(cossim)
            angles = angles * 180 / np.pi
            stats.append(angles)
        else:
            raise ValueError(f"{stat} is not a supported statistic")

    if return_mean:
        return np.mean(stats, axis=0)
    else:
        return np.array(stats).T


def compute_sim_between_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    stat: str,
    EPSILON: float = 1e-7,
) -> np.ndarray:
    dot_product_matrix = np.dot(arr1, arr2.T)
    norms_arr1 = np.linalg.norm(arr1, axis=1)
    norms_arr2 = np.linalg.norm(arr2, axis=1)
    norm_matrix = np.outer(norms_arr1, norms_arr2)
    cos_angles = dot_product_matrix / (norm_matrix + EPSILON)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    if stat == "angle":
        sim_matrix = np.arccos(cos_angles) * (180 / np.pi)
    elif stat == "cossim":
        sim_matrix = cos_angles
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    return sim_matrix


def compute_query(
    X: np.ndarray,
    obs: pd.DataFrame,
    stat: str,
    pert_col: str,
    gene_col: str,
    control_label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_groups = obs[pert_col].nunique()
    max_obs = obs[pert_col].value_counts().max()
    groups = obs[pert_col].unique()

    X_LOO = np.zeros((n_groups, max_obs, X.shape[1]), dtype=np.float32)
    group_n_examples = np.zeros(n_groups)
    group_names = []
    target_gene_names_list = []

    for ind, g in enumerate(groups):
        cur_mask = (obs[pert_col] == g).values
        cur_x = X[cur_mask]
        group_n_examples[ind] = len(cur_x)
        group_names.append(g)
        name = obs.loc[cur_mask, gene_col].iloc[0]
        target_gene_names_list.append(name)
        X_LOO[ind, : len(cur_x), :] = cur_x

    cv_stats_query = cv_stats(X_LOO, group_n_examples, stat, return_mean=False)

    query_dist = np.zeros(n_groups)
    for i in range(n_groups):
        query_dist[i] = cv_stats_query[i, : int(group_n_examples[i])].mean() if group_n_examples[i] > 0 else np.nan

    control_guides = list(obs[obs[gene_col] == control_label][pert_col].unique())
    control_inds = np.array([i in control_guides for i in group_names])
    query_guides = np.array(group_names)[~control_inds]
    query_dist = query_dist[~control_inds]
    group_n_examples = group_n_examples[~control_inds]
    target_gene_names = np.array(target_gene_names_list)[~control_inds]

    return query_dist, query_guides, group_n_examples, target_gene_names


def compute_null(
    X: np.ndarray,
    obs: pd.DataFrame,
    stat: str,
    pert_col: str,
    num_null_cells: int,
) -> np.ndarray:
    grouped_expression = (
        pd.DataFrame(X, index=obs[pert_col]).reset_index().groupby(pert_col, observed=True).mean()
    )

    rand_ind = np.random.choice(len(X), min(num_null_cells, len(X)), replace=False)
    X_null = X[rand_ind]
    null_LOO = compute_sim_between_arrays(X_null, grouped_expression.values, stat)

    guides_single_cell = np.array(obs[pert_col].iloc[rand_ind])
    guides_mean_array = np.array(grouped_expression.index)
    index_2_keep = guides_single_cell[:, np.newaxis] != guides_mean_array
    null_LOO = null_LOO[index_2_keep]
    return null_LOO


def compute_pval(query: np.ndarray, null: np.ndarray, stat: str) -> np.ndarray:
    sorted_null = np.sort(null)
    if stat == "angle":
        pval = np.searchsorted(sorted_null, query) / len(sorted_null)
    elif stat == "cossim":
        pval = 1 - np.searchsorted(sorted_null, query) / len(sorted_null)
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    return pval


def pert_signal_consistency_benchmark(
    features: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    neg_ctrl_perts: list,
    keys_to_drop: Optional[List[str]] = None,
    stat: str = "cossim",
    num_null_cells: int = 10000,
) -> pd.DataFrame:
    if keys_to_drop:
        keep = ~metadata[pert_col].isin(keys_to_drop)
        features = features[keep]
        metadata = metadata.loc[keep].reset_index(drop=True)

    query_dist, query_guides, _, _ = compute_query(
        features,
        metadata,
        stat=stat,
        pert_col=pert_col,
        gene_col=pert_col,
        control_label="__DOES_NOT_EXIST__",
    )

    null_dist = compute_null(features, metadata, stat=stat, pert_col=pert_col, num_null_cells=num_null_cells)
    pvals = compute_pval(query_dist, null_dist, stat)

    keep_ctrl = ~np.isin(query_guides, neg_ctrl_perts)
    perts = query_guides[keep_ctrl]
    sims = query_dist[keep_ctrl]
    pvals = pvals[keep_ctrl]

    col_sim = f"avg{stat}"
    return pd.DataFrame({
        "pert": perts,
        col_sim: sims,
        "pval": pvals
    }).reset_index(drop=True)


# Default recall pairs for the second metric
DEFAULT_RECALL_THR_PAIRS = [(0.05, 0.95)]


