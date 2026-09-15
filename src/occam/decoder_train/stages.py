"""Stage entry point: :func:`run_decoder`.

Extracted from ``decoder_training_pipeline.py:3179-3219``, where it was
the research tree's decoder entry point (see docs/reproducing_chain_a.md).
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import sys
import json
import csv
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import anndata
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from occam.model import AutoEncoder, MODEL_CONFIGS, resolve_decoder_hidden
from occam.decoder.external import HybridNormGlobalSkipDecoder
from occam.transforms import NormalizePerCell, Log1p
from occam.data.datasets import OccamChunkedDataset, OccamAdataDataset
from occam.data.datasets import optim_train_val_chunk_split, compute_global_chunk_info
from occam.ddp import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    is_rank0_env,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Numpy 2.x -> 1.x compatibility shim
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

# ---- Constants ----

VALID_INIT_SCHEMES = (
    "default", "kaiming_normal", "kaiming_uniform",
    "xavier_normal", "xavier_uniform", "orthogonal", "trunc_normal",
)

_DEFAULT_AUG_HPARAMS = {
    'initial_gene_jitter_strength': 0.2,
    'final_gene_jitter_strength': 0.05,
    'gene_jitter_prob': 0.6,
    'initial_gene_mask_rate': 0.05,
    'final_gene_mask_rate': 0.0,
    'gene_mask_prob': 0.5,
    'max_effective_mask_rate': 0.25,
    'initial_mask_activation_threshold': 0.5,
    'final_mask_activation_threshold': 0.0,
    'initial_sensitivity_p': 2.0,
    'final_sensitivity_p': 0.25,
    'sensitivity_p_phase1_epochs_ratio': 0.25,
    'initial_sparsity_modulator_strength': 0.75,
    'final_sparsity_modulator_strength': 0.1,
    'initial_cov_influence_weight': 1.5,
    'final_cov_influence_weight': 0.5,
    'gvf_lower_p': 0.1,
    'gvf_upper_p': 0.95,
}

# BMDB benchmark paths
_BMDB_DIR = None  # resolved lazily
from occam.decoder_train.config import DecoderTrainConfig
from occam.decoder_train.trainer import DecoderTrainer

__all__ = ["run_decoder"]


def run_decoder(
    encoder_checkpoint: str,
    train_data: str,
    output_dir: str,
    config: Optional[DecoderTrainConfig] = None,
    val_data: Optional[str] = None,
    eval_data: Optional[List[str]] = None,
    gene_col: Optional[str] = None,
    eval_gene_col: Optional[str] = None,
    device: str = "cuda",
) -> DecoderTrainer:
    """Convenience function to train an Occam Decoder.

    Args:
        encoder_checkpoint: Path to trained Occam encoder checkpoint.
        train_data: Path to training h5ad file or folder.
        output_dir: Directory to save results.
        config: Optional DecoderTrainConfig (created with defaults if None).
        val_data: Optional validation h5ad file.
        eval_data: Optional list of h5ad files for decoder evaluation.
        gene_col: Gene name column in data.
        eval_gene_col: Gene name column in eval data.
        device: Device for training.

    Returns:
        Trained DecoderTrainer instance.
    """
    if config is None:
        config = DecoderTrainConfig()
    config.encoder_checkpoint = encoder_checkpoint

    trainer = DecoderTrainer(config, device=device)
    trainer.train(
        train_data=train_data,
        output_dir=output_dir,
        val_data=val_data,
        eval_data=eval_data,
        gene_col=gene_col,
        eval_gene_col=eval_gene_col,
    )
    return trainer
