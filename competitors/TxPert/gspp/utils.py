import torch

import random
import numpy as np
import pandas as pd

from omegaconf import DictConfig


def set_seed(seed) -> None:
    """
    Seeding all the random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def modify_config(cfg: DictConfig) -> DictConfig:
    """
    Modify the configuration based on the global variables. This allows easier definiton of wandb sweeps.
    """
    # Subsample the GNNs/graphs based on the indices defined in 'global.used_graphs'
    global_vars = cfg.get("global", None)
    if global_vars is not None:
        used_graphs = global_vars.get("used_graphs", None)

        if used_graphs is not None:
            cfg.graph.graph_cfg = {
                key: val
                for idx, (key, val) in enumerate(cfg.graph.graph_cfg.copy().items())
                if idx in used_graphs
            }
            
            if "gnns_config" in cfg.model.pert_model.keys():
                cfg.model.pert_model.gnns_config = {
                    key: val
                    for idx, (key, val) in enumerate(cfg.model.pert_model.gnns_config.copy().items())
                    if idx in used_graphs
                }

        if not global_vars.get("use_added_data", False):
            cfg.datamodule.added_data_args = {}

    # Make `no_basal_model` configurable under `model.cntr_model.model_type`
    if "cntr_model" in cfg.model.keys():
        if "model_type" in cfg.model.cntr_model.keys():
            basal_model_type = cfg.model.cntr_model.get("model_type", "vae")
            if basal_model_type == "none":
                cfg.model.no_basal_model = True

    # Tmp fix because wandb sweeps cannot pass `None` as a value
    for key, val in cfg.datamodule.items():
        if val == "None":
            cfg.datamodule.base_datamodule[key] = None
    if "no_basal_model" in cfg.model.keys() and "no_pert_model" in cfg.model.keys():
        if cfg.model.no_basal_model and cfg.model.no_pert_model:
            cfg.training.epochs = 0

    # Lower `min_cells` to 10 when dealing with TrekSeq to avoid filtering out 2/3 of compounds
    if "trekseq" in cfg.datamodule["test_cell_type"]:
        cfg.datamodule["min_cells"] = 10

    return cfg


def tmp_cfg_sanity_checks(cfg: DictConfig) -> None:
    """
    Temoporary function to perform sanity checks on the configuration to avoid missing features under development.
    """
    # Control matching currently only supported when no data is loaded from disk
    if cfg.datamodule.match_cntr:
        assert not cfg["global"][
            "use_added_data"
        ], "Currently, control matching is only supported when no data is loaded from disk while the datamodule is being refactored."


def read_pickle_in_chunks(file_path: str, chunk_size: int):
    """
    Read a pickle file in chunks.
    """
    # Read the entire DataFrame
    df = pd.read_pickle(file_path)

    # Generate chunks
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]

