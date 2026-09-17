from typing import Any, Union, List, Dict, Tuple
from omegaconf import ListConfig, DictConfig, OmegaConf

import torch
import random
import pandas as pd
import sys
import warnings
import numpy as np
import importlib
import types

import joblib

from collections import defaultdict

from gspp.utils import read_pickle_in_chunks


def _ensure_numpy_pickle_compat() -> None:
    """Expose NumPy 1.x core modules under the NumPy ≥2 namespace expected by serialized pickles."""
    try:
        import numpy._core.numeric  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    core_module = sys.modules.get("numpy._core")
    if core_module is None:
        np_core = getattr(np, "core", None)
        if np_core is None:
            return

        core_module = types.ModuleType("numpy._core")
        core_module.__dict__.update(np_core.__dict__)
        core_module.__package__ = "numpy"
        core_module.__path__ = getattr(np_core, "__path__", [])
        sys.modules["numpy._core"] = core_module
        if not hasattr(np, "_core"):
            setattr(np, "_core", core_module)

    submodules = (
        "numeric",
        "multiarray",
        "umath",
        "fromnumeric",
        "function_base",
        "records",
        "shape_base",
    )

    for name in submodules:
        alias = f"numpy._core.{name}"
        if alias in sys.modules:
            continue
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules[alias] = module
        setattr(core_module, name, module)


_ensure_numpy_pickle_compat()


class GSPGraph:
    """
    Class to create graphs from raw data and downsample them. Support for multiple graph types in case of a hybrid GNN.

    Args:
        pert2id (dict): Dictionary mapping perturbations to IDs.
        gene2id (dict): Master dictionary mapping genes to IDs regardless if they are in the perturbation set.
        graph_cfg (Union[str, List, Dict]): The type of graph including optional arguments.
            - str: Provide graph type (e.g. "go", "string", "pxmap", "txmap", "dense").
            - List: Provide a list of graph types, e.g., ["go", "string"].
            - Dict: Provide a dictionary with graph identifiers as keys and optional arguments as values, e.g., {"graph1": {"graph_type": "string",...}, "graph2": {"graph_type": "pxmap",...}}.
        cache_dir (str): The directory to store the graphs.
    """

    def __init__(
        self,
        pert2id: Dict[str, int],
        gene2id: Dict[str, int],
        graph_cfg: Union[str, List, Dict] = "go",
        cache_dir: str = "/rxrx/data/valence/pathfinder/modules/gspp/graphs",
    ):
        self.pert2id = pert2id
        self.gene2id = gene2id
        self.cache_dir = cache_dir

        if isinstance(graph_cfg, str):
            graph_cfg = {graph_cfg: {"graph_type": graph_cfg}}
        elif isinstance(graph_cfg, ListConfig):
            graph_cfg = {
                graph_type: {"graph_type": graph_type} for graph_type in graph_cfg
            }

        self.graph_dict = {}
        for graph_name, graph_args in graph_cfg.items():
            if isinstance(graph_args, DictConfig):
                graph_args = OmegaConf.to_container(graph_args, resolve=True)

            graph_type = graph_args.pop("graph_type", "string")
            p_downsample = graph_args.pop("p_downsample", 1.0)
            p_rewire_src = graph_args.pop("p_rewire_src", 0.0)
            p_rewire_tgt = graph_args.pop("p_rewire_tgt", 0.0)
            random_seed = graph_args.pop("random_seed", 42)
            seed_downsample = graph_args.pop("seed_downsample", None)

            graph = self.create_graph(graph_type, graph_args)

            # Donwsample graph if desired
            if p_downsample < 1.0:
                graph = self.downsample_graph(graph, p_downsample, random_seed)

            if p_rewire_src > 0.0 or p_rewire_tgt > 0.0:
                print('Rewiring graph with p_rewires:', (p_rewire_src, p_rewire_tgt))
                graph = self.random_rewire(graph, p_rewire_src=p_rewire_src, p_rewire_tgt=p_rewire_tgt, random_seed=random_seed)

            self.graph_dict[graph_name] = graph

    def create_graph(self, graph_type: str, graph_args: Dict[str, Any] = None):
        """
        Load the graph based on the graph type.
        """
        if graph_type == "go-gears":
            edge_index = joblib.load(f"{self.cache_dir}/go-grears/gears-go-edges.pkl")
            edge_weight = joblib.load(
                f"{self.cache_dir}/go-grears/gears-go-weights.pkl"
            )
            graph = edge_index, edge_weight, len(self.pert2id)

        elif graph_type == "go":
            network = pd.read_csv(f"{self.cache_dir}/go/go_essential_all.csv")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "reactome":
            network = pd.read_csv(f"{self.cache_dir}/reactome/gsp_reactome.csv")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "string":
            network = pd.read_parquet(f"{self.cache_dir}/string/v11.5.parquet")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "pxmap":
            graph = self.process_large_network(
                f"{self.cache_dir}/pxmap/full/ph_all_gene_pair_similarities_v2.pickle",
                **graph_args,
            )

        elif graph_type == "txmap":
            graph = self.process_large_network(
                f"{self.cache_dir}/txmap/full/tx_all_gene_pair_similarities_v2.pickle",
                **graph_args,
            )

        elif graph_type == "pxmap-top1":
            network = pd.read_pickle(
                f"{self.cache_dir}/pxmap/top1/top_phenomics_similarities.pickle"
            )
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "txmap-top1":
            network = pd.read_pickle(
                f"{self.cache_dir}/txmap/top1/top_transcriptomic_similarities.pickle"
            )
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "afm-conf":
            network = pd.read_pickle(
                f"{self.cache_dir}/afm/AFM_screen_WS3.pkl"
            )
            graph = self.process_graph(network, **graph_args)
            
        elif graph_type == "afm-full":
            network = pd.read_pickle(
                f"{self.cache_dir}/afm/AFM_screen_WS3_full.pkl"
            )
            graph = self.process_graph(network, **graph_args)
            
        elif graph_type == "depmap":
            graph = self.process_large_network(
                f"{self.cache_dir}/depmap/DepMap_WS3_full.pkl",
                **graph_args,
            )
            
        elif graph_type == "depmap-top1":
            network = pd.read_pickle(
                f"{self.cache_dir}/depmap/DepMap_WS3_top1.pkl"
            )
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "dense":
            num_nodes = len(self.pert2id)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
            graph = edge_index, edge_weight, len(self.pert2id)

        else:
            raise ValueError("Invalid graph mode")

        return graph

    def process_graph(
        self,
        network: pd.DataFrame,
        reduce2perts: bool = True,
        reduce2positive: bool = False,
        norm_weights: bool = False,
        mode: str = None,
    ):
        """
        Allows processing the network and converting it to PyG format.

        Args:
            network (pd.DataFrame): The network to process.
            reduce2perts (bool): Whether to reduce the network to only perturbation genes.
            reduce2positive (bool): Whether to reduce the network to only positive weights.
            norm_weights (bool): Whether to normalize the weights to [0,1]. TODO (Cass?): Integrate other normalization options.
            mode (str): The mode of edge selection. Currently supports "top_n" edges per target, "quantile_q" edges per target, edges with weight about "threshold", "cossim_thresh" for signal cut-off (Ph and Tx)), and "top_k" for the absolute top signals (must be <5).
        """
        # Check which column naming convention is used
        if "regulator" in network.columns and "target" in network.columns:
            source_col = "regulator"
            target_col = "target"
        elif "source" in network.columns and "target" in network.columns:
            source_col = "source"
            target_col = "target"
        elif "gene1" in network.columns and "gene2" in network.columns:
            source_col = "gene1"
            target_col = "gene2"
        else:
            raise ValueError(
                "Network must have either regulator/target or gene1/gene2 columns"
            )

        # Check and rename similarity column if it exists
        if "similarity" in network.columns:
            network = network.rename(columns={"similarity": "weight"})

        # Check and rename importance column if it exists
        if "importance" in network.columns:
            network = network.rename(columns={"importance": "weight"})

        # Check and rename iptm column if it exists
        if "normalized_iptm_conf" in network.columns:
            network = network.rename(columns={"normalized_iptm_conf": "weight"})

        # Check and rename iptm column if it exists
        if "iptm" in network.columns:
            network = network.rename(columns={"iptm": "weight"})

        # Rename columns to standard format
        if source_col != "regulator":
            network = network.rename(
                columns={source_col: "regulator", target_col: "target"}
            )

        # Reduce to edges between perturbation genes
        if reduce2perts:
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)
        else:
            network["regulator"] = network["regulator"].map(self.gene2id)
            network["target"] = network["target"].map(self.gene2id)
            network = network.dropna()
            num_nodes = len(self.gene2id)

            # all_genes = set(network["regulator"]).union(set(network["target"]))
            # delta_genes = all_genes.difference(set(self.pert2id.keys()))
            # delta_genes = delta_genes.difference({""})
            # pert2id = self.pert2id.copy()
            # for idx, g in enumerate(delta_genes):
            #     pert2id[g] = max(self.pert2id.values()) + idx + 1

            # network["regulator"] = network["regulator"].map(pert2id)
            # network["target"] = network["target"].map(pert2id)
            # network = network.dropna()
            # num_nodes = len(pert2id)

        # Reduce to positive weights only
        if reduce2positive:
            network = network[network["weight"] > 0]
            network = network.reset_index(drop=True)

        # Normalize the weights to [0,1] - added abs() to fix negative weights from Ph/Tx in this case
        if norm_weights:
            network["weight"] = abs(network["weight"].transform(lambda x: x / x.max()))

        # Determine the mode of edge selection
        if mode is not None:
            mode, arg = mode.split("_")
            arg = int(arg)

        # Per target gene, only keep n edges with the highest weights
        if mode is None:
            pass

        elif mode == "top":
            # Equivalent to the original `groupby("target").apply(nlargest(arg, ["weight"]))`,
            # rewritten because pandas >=3 no longer passes the grouping column into the applied
            # function, which silently dropped "target" from the result (KeyError downstream).
            # A *stable* descending sort + head(arg) per group reproduces nlargest(keep="first"):
            # ties keep the rows that appear first in the original frame.
            network = (
                network.sort_values("weight", ascending=False, kind="mergesort")
                .groupby("target", sort=False, group_keys=False)
                .head(arg)
                .reset_index(drop=True)
            )

        # Per target gene, only keep edges with weights above a certain threshold; be carful when also using `norm_weights`
        elif mode == "threshold":
            # The original grouped by "target" before filtering, but the predicate is per-row,
            # so the groupby was a no-op (and broke the same way under pandas >=3).
            network = network[network["weight"] >= arg].reset_index(drop=True)

        # Per target gene, only use edges with cosine similarity above a certain threshold
        elif "cossim-threshold" in mode:
            args = args / 100

            if not 0 <= arg <= 1:
                raise ValueError("Cosine similarity threshold must be between 0 and 1")
            if arg < 0.1:
                warnings.warn("The threshold is too low for cosine similarity.")
                sys.exit(1)

            # Filter based on absolute threshold
            if "abs" in mode:
                network = network[network["weight"].abs() >= arg]
            else:
                network = network[network["weight"] >= arg]

            network = network.reset_index(drop=True)

        # Per target gene, only use edges with a cosine similarity in the specified percentile of absolute values
        elif "percentile" in mode:
            # Calculate threshold value based on absolute similarities if desired
            if "abs" in mode:
                threshold = np.percentile(network["weight"].abs(), arg)
                network = network[network["weight"].abs() >= threshold]
            else:
                threshold = np.percentile(network["weight"], arg)
                network = network[network["weight"] >= threshold]

            network = network.reset_index(drop=True)

        # Package the graph into PyG format
        edge_index = torch.tensor(
            [network["regulator"].to_numpy(), network["target"].to_numpy()],
            dtype=torch.long,
        )
        edge_weight = torch.tensor(network["weight"].to_numpy(), dtype=torch.float)

        return edge_index, edge_weight, num_nodes

    def process_large_network(self, file_path, chunk_size=1_000_000, **graph_args):
        """
        Process large network file in chunks to compute absolute similarities

        Args:
            file_path (str): Path to the pickle file containing the network data
            chunk_size (int): Size of chunks to process at a time
            **graph_args: Additional arguments to pass to process_graph

        Returns:
            tuple: (edge_index, edge_weight, num_nodes) processed graph tuple
        """
        processed_chunks = []
        total_rows = 0

        mode = graph_args.pop("mode", None)

        if mode is not None:
            if "/" in mode:
                mode, mode_post = mode.split("/")
                graph_args["mode"] = mode_post
            mode, arg = mode.split("_")
            arg = int(arg)

        # For percentile mode, we need to first calculate the global threshold
        if "percentile" in mode:
            # First pass to calculate the absolute similarity threshold
            similarities = []
            for chunk in read_pickle_in_chunks(file_path, chunk_size=chunk_size):
                if "abs" in mode:
                    similarities.extend(chunk["similarity"].abs().tolist())
                else:
                    similarities.extend(chunk["similarity"].tolist())

            # Calculate the percentile threshold
            threshold = np.percentile(similarities, arg)
            del similarities

        for chunk_idx, chunk in enumerate(
            read_pickle_in_chunks(file_path, chunk_size=chunk_size)
        ):
            # Apply early filtering based on mode
            if "cossim_threshold" in mode:
                args = args / 100

                if not 0 <= arg <= 1:
                    raise ValueError(
                        "Cosine similarity threshold must be between 0 and 1"
                    )
                if arg < 0.1:
                    warnings.warn("The threshold is too low for cosine similarity.")
                    sys.exit(1)

                if "abs" in mode:
                    chunk = chunk[chunk["similarity"].abs() >= arg]
                else:
                    chunk = chunk[chunk["similarity"] >= arg]

            elif "percentile" in mode:
                # Use pre-calculated threshold for filtering
                if "abs" in mode:
                    chunk = chunk[chunk["similarity"].abs() >= threshold]
                else:
                    chunk = chunk[chunk["similarity"] >= threshold]

            elif mode == "top":
                # First pass to collect all weights for each target
                target_weights = defaultdict(list)
                for chunk in read_pickle_in_chunks(file_path, chunk_size=chunk_size):
                    for target, group in chunk.groupby("gene2"):
                        target_weights[target].extend(
                            list(zip(group.index, group["similarity"]))
                        )

                # Calculate thresholds for each target
                target_thresholds = {}
                for target, weights in target_weights.items():
                    if len(weights) > arg:
                        # Sort by weight and get the threshold for the Nth highest weight
                        sorted_weights = sorted(
                            weights, key=lambda x: abs(x[1]), reverse=True
                        )
                        threshold = abs(sorted_weights[arg - 1][1])
                        target_thresholds[target] = threshold
                    else:
                        # If we have fewer entries than arg, keep all of them
                        target_thresholds[target] = 0
                del target_weights

                # Second pass to filter chunks based on thresholds
                for chunk_idx, chunk in enumerate(
                    read_pickle_in_chunks(file_path, chunk_size=chunk_size)
                ):
                    # Filter rows where weight is >= threshold for that target
                    chunk = chunk[
                        chunk.apply(
                            lambda row: abs(row["similarity"])
                            >= target_thresholds[row["gene2"]],
                            axis=1,
                        )
                    ]

            if len(chunk) > 0:
                processed_chunks.append(chunk)
                total_rows += len(chunk)
                print(f"Processed chunk {chunk_idx + 1}, total rows: {total_rows:,}")

        if not processed_chunks:
            raise ValueError(
                "No data remained after filtering. Consider adjusting the threshold."
            )

        network = pd.concat(processed_chunks, ignore_index=True)

        # Process the final graph using the existing process_graph method
        return self.process_graph(network, **graph_args)

    def downsample_graph(
        self,
        graph: Tuple[torch.Tensor, torch.Tensor],
        p_downsample: float = 1.0,
        seed_downsample: int = None,
    ):
        """
        Downsample the graph to a random fraction of the original edges.

        Args:
            graph (Tuple[torch.Tensor, torch.Tensor, num_nodes]): The graph (edge_indes, edge_weight & num_nodes) to downsample.
            p_downsample (float): The fraction of edges to keep.
            seed_downsample (int): The seed for the downsample operation.
        """
        edge_index, edge_weight, num_nodes = graph

        n = len(edge_weight)
        n_downsample = int(p_downsample * n)

        perm = list(range(n))
        if seed_downsample is not None:
            random.seed(seed_downsample)
        random.shuffle(perm)

        edge_index = edge_index[:, perm]
        edge_weight = edge_weight[perm]

        edge_index = edge_index[:, :n_downsample]
        edge_weight = edge_weight[:n_downsample]

        return edge_index, edge_weight, num_nodes
    

    def random_rewire(
        self,
        graph: Tuple[torch.Tensor, torch.Tensor],
        p_rewire_src: float = 0.0,
        p_rewire_tgt: float = 0.0,
        random_seed: int = None,
    ):
        """
        Rewire each edge with a given probability.

        Args:
            graph (Tuple[torch.Tensor, torch.Tensor, num_nodes]): The graph (edge_indes, edge_weight & num_nodes) to rewire.
            p_rewire (float): The chance of each edge to randomly get rewired.
            random_seed (int): The seed for the rewiring operation.
        """
        edge_index, edge_weight, num_nodes = graph

        num_edges = edge_index.size(1)
        if random_seed is not None:
            random_gen = random.Random(random_seed)
        else:
            random_gen = random.Random()
        for i in range(num_edges):
            if random_gen.random() < p_rewire_src:
                new_source = random_gen.randint(0, num_nodes - 1)
                edge_index[0, i] = new_source
            if random_gen.random() < p_rewire_tgt:
                new_target = random_gen.randint(0, num_nodes - 1)
                edge_index[1, i] = new_target

        return edge_index, edge_weight, num_nodes