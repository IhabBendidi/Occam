"""Prior knowledge graph loading for perturbation GNN.

Ported from TxPM's GSPGraph (gspp/data/graphmodule.py), adapted for standalone
use without omegaconf/hydra dependencies.
"""

import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import pickle as _pickle

import numpy as np
import pandas as pd
import torch


def _read_pickle_numpy2_compat(file_path: str) -> pd.DataFrame:
    """Read a pickle file that may have been saved with numpy>=2.0.

    NumPy 2.0 renamed ``numpy.core`` to ``numpy._core``. Pickles saved under
    numpy>=2.0 reference ``numpy._core.multiarray`` etc., which causes a
    segfault (not a Python exception) when loaded under numpy<2.0.  This helper
    uses a custom ``Unpickler`` that remaps the module path at the protocol
    level, avoiding the crash.
    """

    class _Numpy2Unpickler(_pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with open(file_path, "rb") as fh:
        return _Numpy2Unpickler(fh).load()


def load_multi_kg_graph(
    pert2id: Dict[str, int],
    graph_types: list,
    cache_dir: str,
    reduce2perts: bool = True,
    edge_filter_mode: Optional[str] = None,
    kg_norm_weights: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load multiple KGs, merge edges, return binary-weighted graph.

    Args:
        pert2id: Gene symbol -> int mapping.
        graph_types: List of graph types to load and merge.
        cache_dir: Root directory for graph data files.
        reduce2perts: Keep only edges between genes in pert2id.
        edge_filter_mode: Applied to ALL KGs (e.g., "top_20").
        kg_norm_weights: KG names that get norm_weights=True. Defaults to ["go"].

    Returns:
        (edge_index, edge_weight, num_nodes) — merged, deduplicated, binary weights.
    """
    REVOLUTION_SUBTYPES = ["revolution-hepg2", "revolution-rpe1", "revolution-jurkat"]
    expanded = []
    for gt in graph_types:
        if gt == "revolution":
            expanded.extend(REVOLUTION_SUBTYPES)
        else:
            expanded.append(gt)
    graph_types = expanded

    norm_set = set(kg_norm_weights or ["go"])
    all_edge_indices = []
    for gt in graph_types:
        nw = gt in norm_set
        g = PerturbationGraph(
            pert2id=pert2id,
            graph_type=gt,
            cache_dir=cache_dir,
            reduce2perts=reduce2perts,
            norm_weights=nw,
            edge_filter_mode=edge_filter_mode,
        )
        print(f"  {gt}: {g.edge_weight.shape[0]} edges (norm_weights={nw})")
        all_edge_indices.append(g.edge_index)

    merged = torch.cat(all_edge_indices, dim=1)
    merged = torch.unique(merged, dim=1)
    edge_weight = torch.ones(merged.size(1), dtype=torch.float)
    num_nodes = len(pert2id)
    print(f"  Merged: {merged.size(1)} unique edges, {num_nodes} nodes")
    return merged, edge_weight, num_nodes


def scan_data_pert_genes(
    data_paths: list,
    pert_gene_col: str = "gene_name",
) -> set:
    """Scan h5ad files for perturbation gene names (lightweight, backed mode).

    Returns set of non-control perturbation gene symbols found in data.
    """
    import anndata

    all_genes: set = set()
    for path in data_paths:
        adata = anndata.read_h5ad(path, backed="r")
        is_ctrl = adata.obs["is_control"].values.astype(bool)
        non_ctrl_genes = adata.obs.loc[~is_ctrl, pert_gene_col].unique().tolist()
        all_genes.update(g for g in non_ctrl_genes if g != "non-targeting")
        adata.file.close()
    return all_genes


def filter_pert2id(
    pert2id: Dict[str, int],
    keep_genes: set,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Filter and re-index pert2id to only genes in keep_genes."""
    present = sorted(g for g in keep_genes if g in pert2id)
    new_pert2id = {g: i for i, g in enumerate(present)}
    new_id2pert = {i: g for g, i in new_pert2id.items()}
    return new_pert2id, new_id2pert


def load_pert2id(
    master_gene_set_path: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load perturbation-to-ID mapping from master gene set CSV.

    The CSV has columns: gene_id, gene_name (gene symbols).

    Returns:
        (pert2id, id2pert) — gene_name→int and int→gene_name mappings.
    """
    df = pd.read_csv(master_gene_set_path)
    pert2id = dict(zip(df["gene_name"], df["gene_id"]))
    id2pert = dict(zip(df["gene_id"], df["gene_name"]))
    return pert2id, id2pert


class PerturbationGraph:
    """Loads a prior knowledge gene-gene graph and converts it to PyG tensors.

    Supported graph types: go, string, reactome, pxmap, txmap, depmap, dense,
    and their variants (pxmap-top1, txmap-top1, afm-conf, afm-full, depmap-top1).

    Args:
        pert2id: Gene symbol → integer ID mapping.
        graph_type: Which graph to load.
        cache_dir: Root directory containing graph data files.
        reduce2perts: If True, keep only edges between genes in pert2id.
        norm_weights: If True, normalize edge weights to [0, 1].
        edge_filter_mode: Optional filtering, e.g. "top_20", "percentile_90".
    """

    def __init__(
        self,
        pert2id: Dict[str, int],
        graph_type: str = "go",
        cache_dir: str = "data/graphs",
        reduce2perts: bool = True,
        norm_weights: bool = False,
        edge_filter_mode: Optional[str] = None,
    ):
        self.pert2id = pert2id
        self.cache_dir = cache_dir
        self.reduce2perts = reduce2perts
        self.norm_weights = norm_weights
        self.edge_filter_mode = edge_filter_mode

        self.edge_index, self.edge_weight, self.num_nodes = self._load(graph_type)

    def _load(self, graph_type: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load and process a graph by type."""
        graph_args: Dict[str, Any] = {}
        if self.edge_filter_mode is not None:
            graph_args["mode"] = self.edge_filter_mode

        if graph_type == "go":
            network = pd.read_csv(f"{self.cache_dir}/go/go_essential_all.csv")
            return self._process_graph(network, **graph_args)

        if graph_type == "reactome":
            network = pd.read_csv(f"{self.cache_dir}/reactome/gsp_reactome.csv")
            return self._process_graph(network, **graph_args)

        if graph_type == "string":
            network = pd.read_parquet(f"{self.cache_dir}/string/v11.5.parquet")
            return self._process_graph(network, **graph_args)

        if graph_type == "pxmap":
            return self._process_large_network(
                f"{self.cache_dir}/pxmap/full/ph_all_gene_pair_similarities_v2.pickle",
                **graph_args,
            )

        if graph_type == "txmap":
            return self._process_large_network(
                f"{self.cache_dir}/txmap/full/tx_all_gene_pair_similarities_v2.pickle",
                **graph_args,
            )

        if graph_type == "pxmap-top1":
            network = _read_pickle_numpy2_compat(
                f"{self.cache_dir}/pxmap/top1/top_phenomics_similarities.pickle"
            )
            return self._process_graph(network, **graph_args)

        if graph_type == "txmap-top1":
            network = _read_pickle_numpy2_compat(
                f"{self.cache_dir}/txmap/top1/top_transcriptomic_similarities.pickle"
            )
            return self._process_graph(network, **graph_args)

        if graph_type == "afm-conf":
            network = _read_pickle_numpy2_compat(f"{self.cache_dir}/afm/AFM_screen_WS3.pkl")
            return self._process_graph(network, **graph_args)

        if graph_type == "afm-full":
            network = _read_pickle_numpy2_compat(f"{self.cache_dir}/afm/AFM_screen_WS3_full.pkl")
            return self._process_graph(network, **graph_args)

        if graph_type == "depmap":
            return self._process_large_network(
                f"{self.cache_dir}/depmap/DepMap_WS3_full.pkl", **graph_args
            )

        if graph_type == "depmap-top1":
            network = _read_pickle_numpy2_compat(f"{self.cache_dir}/depmap/DepMap_WS3_top1.pkl")
            return self._process_graph(network, **graph_args)

        if graph_type == "revolution-hepg2":
            network = _read_pickle_numpy2_compat(
                f"{self.cache_dir}/revolution/revolution_knn20_hepg2.pickle"
            )
            return self._process_graph(network, **graph_args)

        if graph_type == "revolution-rpe1":
            network = _read_pickle_numpy2_compat(
                f"{self.cache_dir}/revolution/revolution_knn20_rpe1.pickle"
            )
            return self._process_graph(network, **graph_args)

        if graph_type == "revolution-jurkat":
            network = _read_pickle_numpy2_compat(
                f"{self.cache_dir}/revolution/revolution_knn20_jurkat.pickle"
            )
            return self._process_graph(network, **graph_args)

        if graph_type == "dense":
            num_nodes = len(self.pert2id)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
            return edge_index, edge_weight, num_nodes

        # Cell-line-specific KGs built from that line's CONTROL cells alone
        # (scripts/h1_kg/build_h1_kg.py). Every other graph here is general biology — the
        # same edges whatever the cell line — while the task is scored on one held-out
        # line, so this is the axis those graphs cannot express.
        #
        # Prefix-matched rather than enumerated: the builder emits one file per
        # (map source x percentile x sign) and that grid changes often. Anything named
        # `h1ctrl-*` resolves to `{cache_dir}/h1ctrl/{graph_type}.csv`, so new variants
        # need no code change.
        if graph_type.startswith("h1ctrl-"):
            path = f"{self.cache_dir}/h1ctrl/{graph_type}.csv"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"KG '{graph_type}' not found at {path}. Build it first with "
                    f"scripts/h1_kg/build_h1_kg.py --name <src> --pcts <p>."
                )
            network = pd.read_csv(path)
            return self._process_graph(network, **graph_args)

        raise ValueError(f"Unknown graph type: {graph_type}")

    # ------------------------------------------------------------------
    # Graph processing
    # ------------------------------------------------------------------

    def _process_graph(
        self,
        network: pd.DataFrame,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert a network DataFrame to (edge_index, edge_weight, num_nodes).

        Handles various column naming conventions and weight columns.
        """
        # Detect column naming
        if "regulator" in network.columns and "target" in network.columns:
            src_col, tgt_col = "regulator", "target"
        elif "source" in network.columns and "target" in network.columns:
            src_col, tgt_col = "source", "target"
        elif "gene1" in network.columns and "gene2" in network.columns:
            src_col, tgt_col = "gene1", "gene2"
        else:
            raise ValueError(
                "Network must have regulator/target, source/target, or gene1/gene2 columns"
            )

        # Normalise weight column name
        for alt in ("similarity", "importance", "normalized_iptm_conf", "iptm"):
            if alt in network.columns:
                network = network.rename(columns={alt: "weight"})
                break

        # Rename to canonical (regulator, target)
        if src_col != "regulator":
            network = network.rename(columns={src_col: "regulator", tgt_col: "target"})

        # Map gene symbols to integer IDs
        if self.reduce2perts:
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)
        else:
            all_genes = set(network["regulator"]).union(set(network["target"]))
            delta_genes = all_genes - set(self.pert2id.keys()) - {""}
            pert2id = self.pert2id.copy()
            next_id = max(self.pert2id.values()) + 1
            for g in sorted(delta_genes):
                pert2id[g] = next_id
                next_id += 1
            network["regulator"] = network["regulator"].map(pert2id)
            network["target"] = network["target"].map(pert2id)
            network = network.dropna()
            num_nodes = len(pert2id)

        # Normalise weights
        if self.norm_weights:
            network["weight"] = abs(
                network["weight"].transform(lambda x: x / x.max())
            )

        # Edge filtering
        if mode is not None:
            network = self._filter_edges(network, mode)

        # Convert to tensors
        edge_index = torch.from_numpy(
            np.stack([network["regulator"].to_numpy(), network["target"].to_numpy()])
        ).long()
        edge_weight = torch.tensor(
            network["weight"].to_numpy(), dtype=torch.float
        )

        return edge_index, edge_weight, num_nodes

    @staticmethod
    def _filter_edges(network: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Apply edge filtering: top_N, threshold_X, or percentile_P."""
        parts = mode.split("_")
        filter_type = parts[0]
        arg = int(parts[1])

        if filter_type == "top":
            network = (
                network.groupby("target")
                .apply(lambda x: x.nlargest(arg, ["weight"]))
                .reset_index(drop=True)
            )
        elif filter_type == "threshold":
            network = (
                network.groupby("target")
                .apply(lambda x: x[x["weight"] >= arg])
                .reset_index(drop=True)
            )
        elif filter_type == "percentile":
            threshold = np.percentile(network["weight"], arg)
            network = network[network["weight"] >= threshold].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown edge filter mode: {mode}")

        return network

    def _process_large_network(
        self, file_path: str, chunk_size: int = 1_000_000, **graph_args
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Process large network files in chunks (pxmap, txmap, depmap full)."""
        mode = graph_args.pop("mode", None)
        mode_type, mode_arg = None, None

        if mode is not None:
            if "/" in mode:
                mode, mode_post = mode.split("/")
                graph_args["mode"] = mode_post
            parts = mode.split("_")
            mode_type = parts[0]
            mode_arg = int(parts[1])

        full_data = _read_pickle_numpy2_compat(file_path)

        if isinstance(full_data, pd.DataFrame):
            chunks = [
                full_data.iloc[i : i + chunk_size]
                for i in range(0, len(full_data), chunk_size)
            ]
        else:
            chunks = [full_data]

        processed_chunks = []
        for chunk in chunks:
            if mode_type == "percentile":
                threshold = np.percentile(chunk["similarity"].abs(), mode_arg)
                chunk = chunk[chunk["similarity"].abs() >= threshold]
            elif mode_type == "top":
                chunk = (
                    chunk.groupby("gene2")
                    .apply(lambda x: x.nlargest(mode_arg, "similarity"))
                    .reset_index(drop=True)
                )
            if len(chunk) > 0:
                processed_chunks.append(chunk)

        if not processed_chunks:
            raise ValueError("No data remained after filtering.")

        network = pd.concat(processed_chunks, ignore_index=True)
        return self._process_graph(network, **graph_args)

    def downsample(
        self, p: float = 1.0, seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Randomly keep a fraction p of edges."""
        if p >= 1.0:
            return self.edge_index, self.edge_weight, self.num_nodes

        n = self.edge_weight.shape[0]
        n_keep = int(p * n)
        perm = list(range(n))
        random.seed(seed)
        random.shuffle(perm)

        self.edge_index = self.edge_index[:, perm[:n_keep]]
        self.edge_weight = self.edge_weight[perm[:n_keep]]
        return self.edge_index, self.edge_weight, self.num_nodes
