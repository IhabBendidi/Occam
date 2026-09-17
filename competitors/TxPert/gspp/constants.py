from pathlib import Path

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR

from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GATConv


ACTIVATION_DICT = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "none": None,
    "abs": torch.abs,
    "id": lambda x: x,
}

GNN_DICT = {
    "gcn": GCNConv,
    "gin": GINConv,
    "gat": GATConv,
    "gat_v2": GATv2Conv,
}

SCHEDULER_DICT = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "linear": LinearLR,
    "sequential": SequentialLR,
    "none": None,
}

# Molecular perturbations are out of scope for this arm (the benchmark is CRISPRi-only),
# so the molfeat fingerprint calculators upstream registers here are dropped.
MOL_FEATURIZER_DICT = {}

# The only local data this arm reads: the perturbation / gene vocabularies in ./data.
# Upstream additionally hardcoded per-cell-type h5ad paths under other users' `outgoing/`
# directories; those are gone -- expression comes from the canonical zarr datasets, whose
# locations live in a manifest TOML rather than in code.
DATA_DIR = Path(__file__).parent.parent / "data"

PUBLIC_CELL_TYPES = [
    "K562",
    "RPE1",
    "hepg2",
    "jurkat",
]

PUBLIC_CELL_TYPE_DATASETS = {
    "K562": "replogle_k562_essential",
    "RPE1": "replogle_rpe1_essential",
    "hepg2": "nadig_hepg2",
    "jurkat": "nadig_jurkat",
}

GENE_NAME = "gene_name"  # <- this is the column that will match pert2id (I think)
MOL_NAME = "SMILES"
# the other 'must use' column (beyond this demo), is 'is_control', which is a boolean mask to get only the controls
CONTROL = "control"
CELL_TYPE = "cell_line"
CONDITION_NAME = "condition_name"
CONDITION = "condition"
DOSE = "dose"
DOSE_RAW = "dose (nM)"
IS_CONTROL = "is_control"
BATCH = "batch"
BATCH_RAW = "batch_num"
CONTROL_LABEL = "ctrl"


TRAIN = "train"
TEST = "test"
VAL = "val"

FAST = "fast"
EXTENDED = "extended"
SLOW = "slow"

# dictionary keys used when passing results around, especially from the predictor to the evaluation functions
RES_PRED = "pred"
RES_TRUTH = "truth"
RES_PERT_CAT = "pert_cat"
RES_DOSE_CAT = "dose_cat"
RES_CELL_TYPE = "cell_type"
RES_EXP_BATCH = "experimental_batch"


class ObsmKey:
    RAW = "raw"
    SCGPT = 'scgpt'
