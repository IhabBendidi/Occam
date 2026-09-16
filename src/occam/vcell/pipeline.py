"""Virtual Cell training/evaluation pipeline.

Orchestrates: encoder loading, GNN + graph setup, dataset creation,
training loop, evaluation, checkpointing, baselines, and wandb logging.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import gc
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from occam.vcell.config import VirtualCellConfig
from occam.vcell.stream_v2 import SparseBatch, densify_csr, rss_gb
from occam.vcell.dataset import (
    CellTypeControlMatcher,
    PerturbationBatch,
    PseudobulkDataset,
    VirtualCellDataset,
    VirtualCellStreamingDataset,
    build_eval_dataset,
    append_control_identity,
    precompute_paired_control_embeddings,
    build_mixed_dataset,
    build_pseudobulk_dataset,
    prepare_virtual_cell_data,
    set_transform_flags as _set_dataset_transform_flags,
    set_combo_decomposition as _set_dataset_combo_decomposition,
    resolve_pert_indices as _resolve_pert_indices,
)
from occam.vcell.gnn import PerturbationGNN
from occam.vcell.graph import (
    PerturbationGraph,
    load_pert2id,
    load_multi_kg_graph,
    scan_data_pert_genes,
    filter_pert2id,
)
from occam.vcell.metrics import compute_perturbation_metrics, _safe_pearsonr
from occam.vcell.model import VirtualCellModel
from occam.vcell.splits import PerturbationSplitter


# Import Occam components (from parent codebase — not modified)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from occam.model import AutoEncoder, MODEL_CONFIGS, resolve_decoder_hidden
from occam.decoder.external import HybridNormGlobalSkipDecoder
from occam.gpl_loss import GeometricPlackettLuceLoss


def _append_h1_control_identity(train_dataset, config, encoder_gene_names,
                                skip_normalize: bool, skip_log1p: bool):
    """Append H1 control-in/control-out samples to the training set.

    These carry `pert_indices=[-1]` (the model's null token) and `cell_types="H1"`. The
    dataset stores pre-paired tensors, so no control pool is consulted at training time and
    the H1 rows cannot leak into any other cell line's sampling.
    """
    from occam.vcell.panel import build_control_identity_samples

    try:
        ctrl, tgt, names, indices, cts, mask = build_control_identity_samples(
            config, encoder_gene_names, skip_normalize, skip_log1p,
            mode=getattr(config, "panel_control_identity_mode", None)
            or getattr(config, "control_identity_mode", "exact"),
            n=int(getattr(config, "panel_train_control_n", 0)),
            seed=config.seed,
        )
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [panel] could NOT add H1 control identity samples: {exc}")
        raise

    n_before = len(train_dataset)
    train_dataset.control_profiles = torch.cat([train_dataset.control_profiles, ctrl])
    train_dataset.perturbed_profiles = torch.cat([train_dataset.perturbed_profiles, tgt])
    train_dataset.pert_gene_names = train_dataset.pert_gene_names + names
    train_dataset.pert_indices = train_dataset.pert_indices + indices
    train_dataset.cell_types = train_dataset.cell_types + cts
    train_dataset.gene_masks = torch.cat([train_dataset.gene_masks, mask])
    if train_dataset.batch_ids is not None:
        train_dataset.batch_ids = train_dataset.batch_ids + [None] * len(names)
    if train_dataset.is_sc_flags is not None:
        train_dataset.is_sc_flags = train_dataset.is_sc_flags + [True] * len(names)
    print(f"  [panel] training set {n_before} -> {len(train_dataset)} "
          f"(+{len(names)} H1 control identity, "
          f"{100 * len(names) / max(len(train_dataset), 1):.1f}% of samples)")
    return train_dataset


def _force_h1_perts_into_train(split_info: Dict, config) -> Dict:
    """Move every H1-panel perturbation out of val/test and into train.

    Returns the split unchanged if the H1 panel cannot be read. Reports how many moved so
    the effect is visible in the log rather than implicit.
    """
    import anndata as anndata_

    panel_dir = Path(getattr(config, "panel_dir", "VCCv2/scratch/h1_panel"))
    real_path = (Path(config.panel_real_path) if getattr(config, "panel_real_path", "")
                 else panel_dir / "panel_real.h5ad")
    if not real_path.exists():
        print(f"  [panel] panel not found at {real_path}; leaving the split untouched")
        return split_info
    try:
        a = anndata_.read_h5ad(real_path, backed="r")
        col = getattr(config, "panel_pert_col", "target_gene")
        h1 = set(map(str, a.obs[col].unique())) - {
            getattr(config, "panel_control_label", "non-targeting"), "non-targeting"}
        a.file.close()
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [panel] could not read the panel ({exc}); leaving the split untouched")
        return split_info

    # Only ever move perturbations the splitter already emitted, so nothing is invented
    # that the training corpus does not actually contain.
    moved = set()
    for key in ("val", "test_unseen_pert"):
        present = split_info.get(key, [])
        moved |= h1 & set(present)
        split_info[key] = [p for p in present if p not in h1]
    if moved:
        split_info["train"] = sorted(set(split_info["train"]) | moved)
    print(f"  [panel] moved {len(moved)} H1-panel perturbation(s) into train "
          f"(train={len(split_info['train'])}, val={len(split_info['val'])}, "
          f"test_unseen_pert={len(split_info['test_unseen_pert'])})")
    return split_info


def _run_cell_eval(
    pred_path: str,
    gt_path: str,
    output_dir: str,
    num_threads: int = 16,
    timeout: int = 600,
) -> Dict[str, float]:
    """Run cell-eval via subprocess (jupyter pyenv) and parse agg_results.csv.

    Returns dict of metric_name -> mean_value, or empty dict on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "cell-eval", "run",
        "-ap", pred_path,
        "-ar", gt_path,
        "-o", output_dir,
        "--num-threads", str(num_threads),
        "--profile", "full",
    ]
    env = os.environ.copy()
    env["PYENV_VERSION"] = "jupyter"

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  cell-eval failed (exit {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return {}
    except subprocess.TimeoutExpired:
        print("  cell-eval timed out (600s)")
        return {}
    except FileNotFoundError:
        print("  cell-eval not found (is pyenv jupyter available?)")
        return {}

    agg_path = os.path.join(output_dir, "agg_results.csv")
    if not os.path.exists(agg_path):
        print(f"  agg_results.csv not found at {agg_path}")
        return {}

    try:
        with open(agg_path) as f:
            reader = csv.DictReader(f)
            rows = {row["statistic"]: row for row in reader}
        mean_row = rows.get("mean", {})
        metrics = {}
        for k, v in mean_row.items():
            if k == "statistic":
                continue
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                pass
        print(f"  cell-eval done: {len(metrics)} metrics parsed")
        return metrics
    except Exception as e:
        print(f"  Failed to parse agg_results.csv: {e}")
        return {}


class LogCoshLoss(nn.Module):
    """Log-cosh loss: behaves like L2 for small errors, L1 for large errors.
    Twice differentiable everywhere (smoother than SmoothL1/Huber).
    Operates in log1p space — drop-in replacement for MSE/SmoothL1.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        # log(cosh(x)) = x + softplus(-2x) - log(2), numerically stable
        loss = diff + F.softplus(-2.0 * diff) - 0.6931471805599453
        if self.reduction == "none":
            return loss
        return loss.mean()


class PoissonCountLoss(nn.Module):
    """Poisson NLL computed in count space from log1p-space inputs.
    Both input and target are expected in log1p space (log(1 + counts)).
    Converts to count space via expm1, then computes Poisson NLL.
    No learnable parameters.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert from log1p space to count space
        mu = torch.expm1(input).clamp(min=self.eps)      # predicted counts
        y = torch.expm1(target).clamp(min=0.0)            # observed counts
        # Poisson NLL: mu - y * log(mu)  (dropping constant log(y!) term)
        loss = mu - y * torch.log(mu)
        if self.reduction == "none":
            return loss
        return loss.mean()


# Largest value allowed into `expm1` in the count-space losses below.
#
# WHY THIS EXISTS (F148). These losses take log1p-space tensors and undo the log with
# `expm1`. In float32 `expm1` overflows to `inf` above ~88.7. An `inf` per-element loss is
# survivable on its own, but `_recon_loss` masks absent genes by MULTIPLYING by a 0/1 mask,
# and `inf * 0 == NaN` — so a single unsupervised gene whose decoder output has drifted large
# turns the whole batch's loss into NaN. That is how a corpus with many absent genes (CD4 is a
# 12,342-gene panel inside a 44,728-gene vocabulary, i.e. 72% masked out) poisons training.
#
# 12 is chosen tight ON PURPOSE. `expm1(12) ~ 1.6e5` is still ~80x above the largest
# physically possible value (a gene holding an ENTIRE library is log1p ~7.6 at target_sum
# 2000, ~9.2 at 1e4), so it cannot touch real data — verified: at log1p 7.6 the capped and
# uncapped losses agree to 10 significant figures.
# ⚠️ It was 30 first, and that was too loose: `expm1(30) ~ 1e13` keeps the loss FINITE but
# lets it reach ~1e14, whose gradients overflow to inf — and an inf gradient is fatal (see
# the gradient guard in train_epoch). Bounding the forward is not enough; the cap has to be
# tight enough to bound the BACKWARD too.
_EXPM1_MAX = 12.0


class NBDecoderLoss(nn.Module):
    """Negative Binomial NLL with learnable per-gene dispersion (theta), log1p-space I/O.
    Ported from decoder_training_pipeline so a zinb/nb-pretrained decoder can be finetuned in
    VC with a matching loss. Dispersion params start FRESH (the decoder ckpt does not store
    log_theta) and adapt during VC finetuning.
    """
    def __init__(self, num_genes: int, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ⚠️ clamp log_theta BEFORE exp, not after. `log_theta.exp().clamp(max=1e6)` has a
        # FINITE forward (1e6) but a **NaN backward** once log_theta > ~88.7: exp overflows to
        # inf, clamp's derivative there is 0, and `0 * inf = NaN`. Measured directly.
        theta = self.log_theta.clamp(max=13.8).exp().clamp(min=self.eps)
        # clamp BEFORE expm1 — see _EXPM1_MAX. Clamping the loss afterwards would not help:
        # torch.clamp(nan) is nan, so the overflow has to be prevented, not truncated.
        mu = torch.expm1(input.clamp(min=0, max=_EXPM1_MAX)).clamp(min=self.eps)
        y = torch.expm1(target.clamp(min=0, max=_EXPM1_MAX))
        loss = (
            torch.lgamma(theta)
            - torch.lgamma(y + theta)
            + theta * torch.log1p(mu / theta)
            + y * torch.log1p(theta / mu)
        )
        if self.reduction == "none":
            return loss
        return loss.mean()


class ZINBDecoderLoss(nn.Module):
    """Zero-Inflated NB NLL with learnable per-gene theta + zero-inflation pi, log1p-space I/O.
    Ported from decoder_training_pipeline. Params start FRESH on VC finetune (not stored in ckpt).
    """
    def __init__(self, num_genes: int, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.logit_pi = nn.Parameter(torch.full((num_genes,), -2.0))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # clamp BEFORE exp — see NBDecoderLoss (0 * inf = NaN in backward).
        theta = self.log_theta.clamp(max=13.8).exp().clamp(min=self.eps)
        pi = torch.sigmoid(self.logit_pi)
        mu = torch.expm1(input.clamp(min=0, max=_EXPM1_MAX)).clamp(min=self.eps)
        y = torch.expm1(target.clamp(min=0, max=_EXPM1_MAX))
        nb_log_prob = (
            torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
            + theta * torch.log(theta / (mu + theta))
            + y * torch.log(mu / (mu + theta) + self.eps)
        )
        nb_log_zero = theta * torch.log(theta / (mu + theta))
        zero_mask = (y < 0.5).float()
        zero_case = torch.log(pi + (1 - pi) * nb_log_zero.exp() + self.eps)
        nonzero_case = torch.log(1 - pi + self.eps) + nb_log_prob
        nll = -(zero_mask * zero_case + (1 - zero_mask) * nonzero_case)
        if self.reduction == "none":
            return nll
        return nll.mean()


def _seed_worker(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    import random as _random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def _set_all_seeds(seed: int):
    """Set seed for all randomness sources: Python, NumPy, PyTorch, CUDA."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for deterministic cuBLAS operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Force ALL PyTorch ops to use deterministic algorithms
    # (covers scatter_add, index_add, etc. used by PyG GATv2Conv)
    torch.use_deterministic_algorithms(True)


class VirtualCellTrainer:
    """Trains the VirtualCellModel (GNN + decoder, encoder optionally frozen).

    Args:
        config: VirtualCellConfig with all hyperparameters.
        device: Device string ("cuda" or "cpu").
    """

    def __init__(self, config: VirtualCellConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.best_metric = -float("inf")
        self.current_epoch = 0

        # Set all seeds for reproducibility (comprehensive seeding)
        _set_all_seeds(config.seed)

        # Stage C: dispersion (log_theta/logit_pi) warm-start from the unified ckpt, if present.
        self._ckpt_dispersion = None

        # Load encoder + decoder
        if config.encoder_checkpoint:
            # Unified checkpoint: load both encoder and pretrained decoder
            self._load_from_unified_checkpoint()
        elif config.pretrained_encoder:
            # Encoder-only checkpoint: train decoder from scratch
            self._load_encoder()
            self.decoder = None  # built below
        else:
            raise ValueError(
                "Must provide either --encoder_checkpoint (unified encoder+decoder) "
                "or --pretrained_encoder (encoder-only)"
            )

        # Load pert2id and graph
        print(f"Loading pert2id from {config.master_gene_set_path}")
        self.pert2id, self.id2pert = load_pert2id(config.master_gene_set_path)
        print(f"  {len(self.pert2id)} genes in perturbation vocabulary")

        # Optional: filter to data-present genes
        if config.data_perts_only:
            original_count = len(self.pert2id)
            data_genes = scan_data_pert_genes(config.data_paths, config.pert_gene_col)
            self.pert2id, self.id2pert = filter_pert2id(self.pert2id, data_genes)
            print(f"  Filtered pert2id: {len(self.pert2id)} genes (from {original_count})")

        # Load graph (single or multi-KG)
        if config.no_gnn:
            num_nodes = len(self.pert2id)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_weight = torch.zeros(0)
            print(f"  No graph prior (no_gnn=True): {num_nodes} nodes, learnable embeddings")
        elif config.graph_types is not None:
            print(f"Loading multi-KG graph: {config.graph_types}")
            edge_index, edge_weight, num_nodes = load_multi_kg_graph(
                pert2id=self.pert2id,
                graph_types=config.graph_types,
                cache_dir=config.graph_cache_dir,
                reduce2perts=config.graph_reduce2perts,
                edge_filter_mode=config.graph_edge_filter_mode,
                kg_norm_weights=config.kg_norm_weights,
            )
        else:
            print(f"Loading graph: {config.graph_type}")
            graph = PerturbationGraph(
                pert2id=self.pert2id,
                graph_type=config.graph_type,
                cache_dir=config.graph_cache_dir,
                reduce2perts=config.graph_reduce2perts,
                edge_filter_mode=config.graph_edge_filter_mode,
            )
            edge_index = graph.edge_index
            edge_weight = graph.edge_weight
            num_nodes = graph.num_nodes
            print(
                f"  Graph: {num_nodes} nodes, "
                f"{edge_weight.shape[0]} edges"
            )

        # Create perturbation model
        if config.no_gnn and config.pert_model_type == "adjacency_map":
            from occam.vcell.adjacency_map import LearnableEmbeddingModel
            pert_model = LearnableEmbeddingModel(
                num_genes=num_nodes,
                out_dim=config.pert_latent_dim,
            )
            print(f"  Learnable embeddings: {num_nodes} genes, dim={config.pert_latent_dim}")
        elif config.pert_model_type == "adjacency_map":
            from occam.vcell.adjacency_map import SimpleAdjacencyMapModel
            pert_model = SimpleAdjacencyMapModel(
                num_genes=num_nodes,
                edge_index=edge_index,
                hidden_dim=config.adj_hidden_dim,
                out_dim=config.pert_latent_dim,
                dropout=config.adj_dropout,
                use_ffn=config.adj_use_ffn,
                normalize=config.adj_normalize,
                adj_deg=config.adj_deg,
                self_loops=config.adj_self_loops,
                only_reverse=config.adj_only_reverse,
                make_undirected=config.adj_make_undirected,
                activation=config.adj_activation,
            )
            print(
                f"  Adjacency map: {num_nodes} nodes, "
                f"adj_matrix {pert_model.adj_matrix.shape}"
            )
        else:
            pert_model = PerturbationGNN(
                num_genes=num_nodes,
                hidden_dim=config.gnn_hidden_dim,
                out_dim=config.pert_latent_dim,
                layer_type=config.gnn_layer_type,
                num_layers=config.gnn_num_layers,
                skip=config.gnn_skip,
                dropout=config.gnn_dropout,
                num_heads=config.gnn_num_heads,
                use_edge_weight=config.use_edge_weight,
                no_gnn=config.no_gnn,
                activation=config.gnn_activation,
            )

        # Create decoder (from scratch if not loaded from unified checkpoint)
        hurdle_mode = config.hurdle_loss_weight > 0
        if self.decoder is None:
            decoder = HybridNormGlobalSkipDecoder(
                input_dim=self.hidden_size,
                output_dim=self.num_genes,
                hidden_dim=config.decoder_hidden_dim,
                num_blocks=config.decoder_num_blocks,
                dropout_rate=config.decoder_dropout,
                se_reduction=config.decoder_se_reduction,
                output_activation=getattr(config, "decoder_output_activation",
                                          "rectified_sigmoid"),
                hurdle_mode=hurdle_mode,
            )
            print(
                f"Decoder (from scratch): input_dim={self.hidden_size}, "
                f"output_dim={self.num_genes}"
                + (", hurdle_mode=True" if hurdle_mode else "")
            )
        else:
            decoder = self.decoder
            print(
                f"Decoder (pretrained): input_dim={decoder.input_dim}, "
                f"output_dim={decoder.output_dim}, "
                f"hidden_dim={decoder.hidden_dim}"
            )

        # Assemble model
        interaction_kwargs = {
            "film_scale": config.film_scale,
            "film_hidden_dim": config.film_hidden_dim,
            "film_activation": config.film_activation,
            "film_mode": config.film_mode,
            "gated_mlp_hidden_dim": config.gated_mlp_hidden_dim,
            "gated_mlp_num_layers": config.gated_mlp_num_layers,
        }
        self.model = VirtualCellModel(
            encoder=self.encoder,
            gnn=pert_model,
            decoder=decoder,
            edge_index=edge_index,
            edge_weight=edge_weight,
            encoder_hidden_size=self.hidden_size,
            pert_latent_dim=config.pert_latent_dim,
            freeze_encoder=config.freeze_encoder,
            freeze_decoder=config.freeze_decoder,
            interaction_type=config.interaction_type,
            interaction_kwargs=interaction_kwargs,
            # F181/E3. Slots are allocated generously (not from the stream, which is built
            # later); unused rows stay at their zero init and never receive gradient.
            source_cond=str(getattr(config, "source_cond", "none")),
            n_sources=(int(getattr(config, "source_cond_slots", 64))
                       if str(getattr(config, "source_cond", "none")) != "none" else 0),
            source_dropout=float(getattr(config, "source_dropout", 0.25)),
            film_mean_free=bool(getattr(config, "film_mean_free", False)),
        ).to(self.device)
        if str(getattr(config, "source_cond", "none")) != "none":
            print(f"  [source_cond={config.source_cond}] slots="
                  f"{int(getattr(config, 'source_cond_slots', 64))} "
                  f"dropout={float(getattr(config, 'source_dropout', 0.25))} — "
                  f"H1 inference will use the NEUTRAL source + KNOCKDOWN modality")

        if config.interaction_type != "additive":
            print(f"  Interaction: {config.interaction_type}")

        # v2: Validate expression_loss_mode + frozen encoder incompatibility
        if config.expression_loss_mode == "importance_weighted" and config.freeze_encoder:
            if not config.pseudobulk:
                raise ValueError(
                    "--expression_loss_mode importance_weighted with --freeze_encoder "
                    "is only supported with --pseudobulk. "
                    "In streaming/in_memory mode, controls are pre-computed as embeddings "
                    "and raw expression is not available for importance weighting."
                )
            else:
                print("  Note: importance_weighted + freeze_encoder OK in pseudobulk mode "
                      "(controls are raw expression)")

        # Optimizer — use model's get_trainable_params for correct freeze handling
        # Separate LRs: lr for GNN/interaction, decoder_lr for pretrained decoder
        self.decoder_lr = config.decoder_lr if config.decoder_lr is not None else config.lr
        param_groups = []
        for pg in self.model.get_trainable_params():
            if pg["name"] == "encoder":
                lr = config.encoder_lr
            elif pg["name"] == "decoder":
                lr = self.decoder_lr
            else:
                lr = config.lr  # GNN + interaction
            param_groups.append({"params": pg["params"], "lr": lr, "name": pg["name"]})

        # v2: Projection head for "projected" latent loss
        self.latent_proj_head = None
        if config.latent_loss_type == "projected":
            self.latent_proj_head = nn.Sequential(
                nn.Linear(self.hidden_size, config.latent_proj_dim),
                nn.GELU(),
                nn.Linear(config.latent_proj_dim, config.latent_proj_dim),
            ).to(self.device)
            param_groups.append({
                "params": self.latent_proj_head.parameters(),
                "lr": config.lr,
                "name": "latent_proj_head",
            })

        self.optimizer = AdamW(
            param_groups, lr=config.lr, weight_decay=config.weight_decay
        )

        # v2: Decoder LR warmup — find decoder param group index
        self.decoder_lr_warmup_epochs = config.decoder_lr_warmup_epochs
        self._decoder_pg_idx = None
        if self.decoder_lr_warmup_epochs > 0:
            for i, pg in enumerate(self.optimizer.param_groups):
                if pg.get("name") == "decoder":
                    self._decoder_pg_idx = i
                    break
            if self._decoder_pg_idx is not None:
                print(f"  Decoder LR warmup: {self.decoder_lr_warmup_epochs} epochs")

        # Encoder LR delay: hold the encoder frozen (lr=0) for the first
        # encoder_lr_delay_frac of training, then release it. Lets the pert head
        # and decoder settle against a FIXED representation before the encoder is
        # allowed to move, which otherwise chases a moving target early on.
        self.encoder_lr_delay_frac = float(getattr(config, "encoder_lr_delay_frac", 0.0))
        self._encoder_pg_idx = None
        if self.encoder_lr_delay_frac > 0 and not config.freeze_encoder:
            for i, pg in enumerate(self.optimizer.param_groups):
                if pg.get("name") == "encoder":
                    self._encoder_pg_idx = i
                    break
            if self._encoder_pg_idx is not None:
                self._encoder_release_epoch = max(
                    1, int(round(self.encoder_lr_delay_frac * config.epochs)))
                print(f"  Encoder LR delay: frozen until epoch "
                      f"{self._encoder_release_epoch}/{config.epochs} "
                      f"({self.encoder_lr_delay_frac:.0%}), then lr={config.encoder_lr}")

        # Scheduler (optional linear warmup + cosine decay)
        if config.warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer, start_factor=1e-2, total_iters=config.warmup_epochs
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs - config.warmup_epochs,
                eta_min=config.eta_min,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[config.warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=config.eta_min
            )

        # Loss
        if config.loss_type == "smooth_l1":
            self.recon_loss_fn = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
        elif config.loss_type == "mse":
            self.recon_loss_fn = nn.MSELoss()
        elif config.loss_type == "log_cosh":
            self.recon_loss_fn = LogCoshLoss()
        elif config.loss_type == "poisson":
            self.recon_loss_fn = PoissonCountLoss()
        elif config.loss_type == "poisson_smooth_l1":
            self.recon_loss_fn = None  # handled inline (combo of two losses)
            self._poisson_fn = PoissonCountLoss()
            self._smooth_l1_fn = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
            self._poisson_weight = config.poisson_weight
        elif config.loss_type == "nb":
            # reduction='none' — per-gene dispersion needs the gene dim intact; masking/mean
            # is handled in _recon_loss (NOT via the 1D gene_mask flatten the others use).
            self.recon_loss_fn = NBDecoderLoss(num_genes=self.num_genes, reduction="none").to(self.device)
        elif config.loss_type == "zinb":
            self.recon_loss_fn = ZINBDecoderLoss(num_genes=self.num_genes, reduction="none").to(self.device)
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")

        # ---- Stage C dispersion handling (nb/zinb only; smooth_l1 etc. have no dispersion = no-op) ----
        _has_disp = (self.recon_loss_fn is not None
                     and any(True for _ in self.recon_loss_fn.parameters()))
        if _has_disp:
            # (1) WARM-START log_theta/logit_pi from the unified ckpt's saved dispersion (gene-aligned).
            disp = self._ckpt_dispersion
            if getattr(config, "vc_load_dispersion", True) and disp is not None \
                    and disp.get("log_theta") is not None \
                    and disp["log_theta"].shape[0] == self.num_genes:
                with torch.no_grad():
                    self.recon_loss_fn.log_theta.copy_(disp["log_theta"].to(self.device))
                    if hasattr(self.recon_loss_fn, "logit_pi") and disp.get("logit_pi") is not None:
                        self.recon_loss_fn.logit_pi.copy_(disp["logit_pi"].to(self.device))
                print(f"  [Stage C] warm-started {config.loss_type} dispersion from checkpoint "
                      f"(ckpt loss_type={disp.get('loss_type')})")
            elif getattr(config, "vc_load_dispersion", True) and disp is not None:
                print(f"  [Stage C] WARNING: ckpt dispersion shape mismatch "
                      f"({disp.get('log_theta').shape[0] if disp.get('log_theta') is not None else None} "
                      f"vs num_genes {self.num_genes}) -> using default init")

            # (2) FREEZE vs FINETUNE. Rule: freezing the decoder forces dispersion frozen too.
            dispersion_frozen = config.freeze_decoder or getattr(config, "vc_freeze_dispersion", False)
            if dispersion_frozen:
                for p in self.recon_loss_fn.parameters():
                    p.requires_grad_(False)
                why = "freeze_decoder" if config.freeze_decoder else "vc_freeze_dispersion"
                print(f"  [Stage C] dispersion FROZEN ({why}=True)")
            else:
                disp_lr = getattr(config, "vc_dispersion_lr", None)
                grp = {"params": list(self.recon_loss_fn.parameters()),
                       "weight_decay": 0.0, "name": "dispersion"}
                if disp_lr is not None and disp_lr > 0:
                    grp["lr"] = disp_lr           # separate dispersion LR (the Stage C lever)
                self.optimizer.add_param_group(grp)
                print(f"  [Stage C] dispersion TRAINABLE (lr={disp_lr if disp_lr else config.lr})")
        self.cosine_loss_weight = config.cosine_loss_weight
        self.recon_loss_scale = config.recon_loss_scale

        # GPL ranking loss (auxiliary, expression space)
        self.gpl_weight = config.gpl_weight
        if self.gpl_weight > 0:
            self.gpl_fn = GeometricPlackettLuceLoss(reduction="mean")
        else:
            self.gpl_fn = None

        # v2: Latent loss and regularization config
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.latent_loss_alpha = config.latent_loss_alpha
        self.delta_reg_type = config.delta_reg_type
        self.delta_reg_lambda = config.delta_reg_lambda
        self.delta_reg_tau = config.delta_reg_tau
        self.expression_loss_mode = config.expression_loss_mode

        # AMP scaler
        self._amp_dtype = (torch.bfloat16 if str(getattr(self.config, "amp_dtype", "fp16")) == "bf16"
                           else torch.float16)
        # bf16 has fp32's exponent range: no loss scaling, and the GradScaler's zero-scale
        # failure mode (PIPELINE_AUDIT_2026-09-09 §3) cannot occur. fp16 keeps the old path.
        self.scaler = (torch.amp.GradScaler("cuda")
                       if (device == "cuda" and self._amp_dtype == torch.float16) else None)
        print(f"  [amp] autocast dtype={'bf16' if self._amp_dtype == torch.bfloat16 else 'fp16'} "
              f"grad_scaler={'on' if self.scaler is not None else 'off'}")

        # DDP
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.is_distributed = self.world_size > 1
            if self.is_distributed:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))]
                )

    @classmethod
    def from_vcell_checkpoint(cls, ckpt_path: str, device: str = "cuda",
                              output_activation: Optional[str] = None):
        """Build a VirtualCellTrainer for eval-only from a saved vcell checkpoint.

        Loads encoder, GNN, decoder, and interaction weights directly from the
        checkpoint — no need for the original encoder_checkpoint or graph files.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not ckpt.get("is_virtual_cell_checkpoint"):
            raise ValueError("Not a virtual cell checkpoint")

        tc = ckpt["training_config"]
        gc_ = ckpt["gnn_config"]
        ic = ckpt["interaction_config"]
        dc = dict(ckpt["decoder_config"])
        # Checkpoints written before output_activation was persisted default to
        # rectified_sigmoid, which for a softplus-trained decoder is a dead model rather
        # than a small discrepancy. Refuse to guess silently.
        if output_activation is not None:
            dc["output_activation"] = output_activation
        elif "output_activation" not in dc:
            print("  WARNING: this checkpoint predates output_activation being saved; "
                  "assuming 'rectified_sigmoid'. If the decoder was trained with softplus "
                  "the output will be ~99% zeros — pass output_activation explicitly.")

        trainer = object.__new__(cls)
        trainer.device = device
        trainer.best_metric = ckpt.get("best_metric", -float("inf"))
        trainer.current_epoch = ckpt.get("epoch", 0)

        # Encoder config
        enc_config = ckpt["encoder_config"]
        trainer.model_size = enc_config.get("model_size", "small")
        trainer.hidden_size = enc_config.get("hidden_size", 512)
        trainer.encoder_gene_names = ckpt["gene_names"]
        trainer.num_genes = len(trainer.encoder_gene_names)
        trainer.gene_to_index = ckpt.get(
            "gene_to_index",
            {g: i for i, g in enumerate(trainer.encoder_gene_names)},
        )
        trainer.target_sum = ckpt.get("target_sum", 1e4)
        trainer.pert2id = ckpt["pert2id"]
        trainer.id2pert = {v: k for k, v in trainer.pert2id.items()}

        # Build encoder
        full_model = AutoEncoder(
            num_genes=trainer.num_genes,
            model_size=trainer.model_size,
            hidden_size=trainer.hidden_size,
        )
        enc_state = ckpt["encoder_state_dict"]
        cleaned = {
            k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v
            for k, v in enc_state.items()
        }
        full_model.load_state_dict(cleaned, strict=False)
        trainer.encoder = full_model.encoder.to(device)
        trainer._encoder_state_dict = enc_state
        del full_model

        # Build decoder
        decoder = HybridNormGlobalSkipDecoder(
            input_dim=dc["input_dim"],
            output_dim=dc["output_dim"],
            hidden_dim=dc["hidden_dim"],
            num_blocks=dc["num_blocks"],
            dropout_rate=dc.get("dropout_rate", 0.3),
            se_reduction=dc.get("se_reduction", 16),
            # see the note at the unified-checkpoint decoder build: parameterless activations
            # make a wrong output head load cleanly and fail silently
            output_activation=dc.get("output_activation", "rectified_sigmoid"),
            hurdle_mode=dc.get("hurdle_mode", False),
        )

        # Build perturbation model (adjacency map or GNN)
        # Reconstruct edge_index/edge_weight from the saved state dict
        num_pert_nodes = gc_["num_genes"]
        pert_model_type = gc_.get("pert_model_type", tc.get("pert_model_type", "gnn"))

        no_gnn = gc_.get("no_gnn", False)
        if pert_model_type == "adjacency_map" and no_gnn:
            from occam.vcell.adjacency_map import LearnableEmbeddingModel
            pert_model = LearnableEmbeddingModel(
                num_genes=num_pert_nodes,
                out_dim=gc_.get("out_dim", 64),
            )
        elif pert_model_type == "adjacency_map":
            from occam.vcell.adjacency_map import SimpleAdjacencyMapModel
            adj_state = ckpt["gnn_state_dict"]
            adj_matrix_shape = adj_state["adj_matrix"].shape
            dummy_edge_index = torch.zeros(2, 0, dtype=torch.long)
            pert_model = SimpleAdjacencyMapModel(
                num_genes=num_pert_nodes,
                edge_index=dummy_edge_index,
                hidden_dim=gc_.get("adj_hidden_dim", 128),
                out_dim=gc_.get("out_dim", 64),
                dropout=gc_.get("adj_dropout", 0.2),
                normalize=gc_.get("adj_normalize", "none"),
                adj_deg=gc_.get("adj_deg", 1),
                only_reverse=gc_.get("adj_only_reverse", True),
                make_undirected=gc_.get("adj_make_undirected", True),
                activation=gc_.get("adj_activation", "relu"),
            )
        else:
            pert_model = PerturbationGNN(
                num_genes=num_pert_nodes,
                hidden_dim=gc_.get("hidden_dim", 64),
                out_dim=gc_.get("out_dim", 64),
                layer_type=gc_.get("layer_type", "gat_v2"),
                num_layers=gc_.get("num_layers", 4),
                skip=gc_.get("skip", "none"),
                num_heads=gc_.get("num_heads", 2),
                use_edge_weight=gc_.get("use_edge_weight", False),
                no_gnn=gc_.get("no_gnn", False),
            )

        # Dummy edge_index/weight for VirtualCellModel (not used at eval time
        # for adjacency_map, and GNN weights are already trained)
        dummy_ei = torch.zeros(2, 0, dtype=torch.long)
        dummy_ew = torch.zeros(0)

        interaction_kwargs = {
            "film_scale": ic.get("film_scale", 2.0),
            "film_hidden_dim": ic.get("film_hidden_dim", 0),
            "film_activation": ic.get("film_activation", "none"),
            "film_mode": ic.get("film_mode", "full"),
            "gated_mlp_hidden_dim": ic.get("gated_mlp_hidden_dim"),
            "gated_mlp_num_layers": ic.get("gated_mlp_num_layers", 2),
        }
        trainer.model = VirtualCellModel(
            encoder=trainer.encoder,
            gnn=pert_model,
            decoder=decoder,
            edge_index=dummy_ei,
            edge_weight=dummy_ew,
            encoder_hidden_size=trainer.hidden_size,
            pert_latent_dim=gc_.get("out_dim", 64),
            freeze_encoder=True,
            freeze_decoder=True,
            interaction_type=ic.get("interaction_type", "additive"),
            interaction_kwargs=interaction_kwargs,
            # Read back from the checkpoint, never from a config: scoring a mean-free model
            # without the correction changes every predicted delta.
            film_mean_free=bool(ic.get("film_mean_free", False)),
        ).to(device)

        # Load trained weights
        fwd = trainer.model
        fwd.gnn.load_state_dict(ckpt["gnn_state_dict"])
        fwd.decoder.load_state_dict(ckpt["decoder_state_dict"])
        fwd.interaction.load_state_dict(ckpt["interaction_state_dict"])

        # Minimal config for evaluate()
        config = VirtualCellConfig.__new__(VirtualCellConfig)
        config.loss_type = tc.get("loss_type", "smooth_l1")
        config.cosine_loss_weight = tc.get("cosine_loss_weight", 0.0)
        config.target_sum = trainer.target_sum
        trainer.config = config

        # Loss function (needed by evaluate)
        smooth_l1_beta = tc.get("smooth_l1_beta", 2.0)
        if config.loss_type == "smooth_l1":
            trainer.recon_loss_fn = nn.SmoothL1Loss(beta=smooth_l1_beta)
        elif config.loss_type == "mse":
            trainer.recon_loss_fn = nn.MSELoss()
        elif config.loss_type == "log_cosh":
            trainer.recon_loss_fn = LogCoshLoss()
        elif config.loss_type == "poisson":
            trainer.recon_loss_fn = PoissonCountLoss()
        elif config.loss_type == "poisson_smooth_l1":
            trainer.recon_loss_fn = None
            trainer._poisson_fn = PoissonCountLoss()
            trainer._smooth_l1_fn = nn.SmoothL1Loss(beta=smooth_l1_beta)
            trainer._poisson_weight = tc.get("poisson_weight", 0.5)
        elif config.loss_type == "nb":
            trainer.recon_loss_fn = NBDecoderLoss(num_genes=trainer.num_genes, reduction="none").to(trainer.device)
        elif config.loss_type == "zinb":
            trainer.recon_loss_fn = ZINBDecoderLoss(num_genes=trainer.num_genes, reduction="none").to(trainer.device)
        else:
            trainer.recon_loss_fn = nn.SmoothL1Loss(beta=smooth_l1_beta)

        trainer.cosine_loss_weight = config.cosine_loss_weight
        trainer.recon_loss_scale = tc.get("recon_loss_scale", 1.0)
        trainer.gpl_fn = None
        trainer.gpl_weight = 0.0
        trainer.is_distributed = False
        trainer.rank = 0
        trainer.world_size = 1
        trainer.latent_proj_head = None

        print(f"Loaded vcell checkpoint: epoch={trainer.current_epoch}, "
              f"best_metric={trainer.best_metric:.4f}")
        print(f"  Encoder: {trainer.model_size}, {trainer.num_genes} genes")
        print(f"  Pert model: {pert_model_type}, {num_pert_nodes} nodes")

        return trainer, ckpt

    def _load_encoder(self):
        """Load Occam encoder from encoder-only checkpoint."""
        config = self.config
        ckpt_path = config.pretrained_encoder
        print(f"Loading encoder-only checkpoint: {ckpt_path}")
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        is_unified = checkpoint.get("is_unified_checkpoint", False)

        if is_unified:
            ckpt_config = checkpoint.get("encoder_config", {})
        else:
            ckpt_config = checkpoint.get("config", {})

        # Gene names
        self.encoder_gene_names = checkpoint.get("gene_names", None)
        if self.encoder_gene_names is None:
            dataset_info = checkpoint.get("dataset_info", {})
            self.encoder_gene_names = dataset_info.get("gene_names", None)
        if self.encoder_gene_names is None:
            raise ValueError("Checkpoint does not contain gene_names")

        self.num_genes = len(self.encoder_gene_names)
        self.gene_to_index = {g: i for i, g in enumerate(self.encoder_gene_names)}

        # Target sum
        if is_unified:
            self.target_sum = checkpoint.get("target_sum", config.target_sum)
        else:
            self.target_sum = ckpt_config.get("target_sum", config.target_sum)
            dataset_info = checkpoint.get("dataset_info", {})
            if dataset_info.get("target_sum") is not None:
                self.target_sum = dataset_info["target_sum"]
        config.target_sum = self.target_sum

        # Transform-skip flags (inherit from ckpt config/training_config, OR with CLI).
        tcfg = checkpoint.get("training_config", {}) or {}
        ckpt_skip_norm = bool(ckpt_config.get("skip_normalize") or tcfg.get("skip_normalize") or False)
        ckpt_skip_log1p = bool(ckpt_config.get("skip_log1p") or tcfg.get("skip_log1p") or False)
        self.skip_normalize = ckpt_skip_norm or bool(getattr(config, "skip_normalize", False))
        self.skip_log1p = ckpt_skip_log1p or bool(getattr(config, "skip_log1p", False))
        config.skip_normalize, config.skip_log1p = self.skip_normalize, self.skip_log1p
        _set_dataset_transform_flags(self.skip_normalize, self.skip_log1p)
        _set_dataset_combo_decomposition(getattr(config, "combo_decomposition", False))
        if self.skip_normalize or self.skip_log1p:
            print(f"  Transform-skip: skip_normalize={self.skip_normalize}, "
                  f"skip_log1p={self.skip_log1p}")

        # Model architecture
        model_size = ckpt_config.get("model_size", "small")
        hidden_size = ckpt_config.get("hidden_size", None)
        dropout = ckpt_config.get("dropout", 0.0)
        if hidden_size is None:
            hidden_size = MODEL_CONFIGS.get(model_size, {}).get("hidden_size", 512)
        self.hidden_size = hidden_size
        self.model_size = model_size

        if is_unified:
            state_dict = checkpoint["encoder_state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]

        # Only `.encoder` is kept below -- but a non-unified checkpoint is loaded with
        # strict=True, so the throwaway MAE decoder still has to match the saved shapes
        # or load_state_dict raises. (This is the MAE's internal decoder, not the
        # HybridNormGlobalSkipDecoder built further down.)
        full_model = AutoEncoder(
            num_genes=self.num_genes,
            model_size=model_size,
            hidden_size=hidden_size,
            decoder_hidden=resolve_decoder_hidden(ckpt_config, state_dict, model_size),
            dropout=dropout,
            reconstruction_loss=ckpt_config.get("reconstruction_loss", "mse"),
            clamp_max_log1p=ckpt_config.get("clamp_max_log1p", 5.5),
        )
        cleaned = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        full_model.load_state_dict(cleaned, strict=not is_unified)

        self.encoder = full_model.encoder.to(self.device)
        self._encoder_state_dict = {
            k: v for k, v in cleaned.items() if k.startswith("encoder.")
        }
        self.decoder = None  # no decoder from this checkpoint

        # Strip unused components
        del full_model
        gc.collect()

        print(
            f"  Encoder loaded: model_size={model_size}, hidden_size={hidden_size}, "
            f"num_genes={self.num_genes}, target_sum={self.target_sum}"
        )

    def _load_from_unified_checkpoint(self):
        """Load both encoder and pretrained decoder from a phase 3 unified checkpoint."""
        config = self.config
        ckpt_path = config.encoder_checkpoint
        print(f"Loading unified encoder+decoder checkpoint: {ckpt_path}")
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        if not checkpoint.get("is_unified_checkpoint", False):
            raise ValueError(
                f"--encoder_checkpoint expects a unified checkpoint "
                f"(is_unified_checkpoint=True). Got a non-unified checkpoint. "
                f"Use --pretrained_encoder instead for encoder-only checkpoints."
            )

        # Encoder config
        enc_config = checkpoint["encoder_config"]
        model_size = enc_config.get("model_size", "small")
        hidden_size = enc_config.get("hidden_size", 512)
        num_genes = enc_config.get("num_genes")
        self.model_size = model_size
        self.hidden_size = hidden_size

        # Gene names
        self.encoder_gene_names = checkpoint["gene_names"]
        self.num_genes = len(self.encoder_gene_names)
        self.gene_to_index = checkpoint.get(
            "gene_to_index",
            {g: i for i, g in enumerate(self.encoder_gene_names)},
        )

        # Target sum
        self.target_sum = checkpoint.get("target_sum", config.target_sum)
        config.target_sum = self.target_sum

        # Transform-skip flags: inherit from the checkpoint (encoder_config or
        # training_config), OR with CLI flags. The model's input space must match the
        # encoder's training space (replogle: data already library-normalized + log1p).
        tcfg = checkpoint.get("training_config", {}) or {}
        ckpt_skip_norm = bool(enc_config.get("skip_normalize") or tcfg.get("skip_normalize") or False)
        ckpt_skip_log1p = bool(enc_config.get("skip_log1p") or tcfg.get("skip_log1p") or False)
        self.skip_normalize = ckpt_skip_norm or bool(getattr(config, "skip_normalize", False))
        self.skip_log1p = ckpt_skip_log1p or bool(getattr(config, "skip_log1p", False))
        config.skip_normalize, config.skip_log1p = self.skip_normalize, self.skip_log1p
        _set_dataset_transform_flags(self.skip_normalize, self.skip_log1p)
        _set_dataset_combo_decomposition(getattr(config, "combo_decomposition", False))
        if self.skip_normalize or self.skip_log1p:
            print(f"  Transform-skip: skip_normalize={self.skip_normalize}, "
                  f"skip_log1p={self.skip_log1p}")

        # Build and load encoder
        full_model = AutoEncoder(
            num_genes=self.num_genes,
            model_size=model_size,
            hidden_size=hidden_size,
        )
        enc_state = checkpoint["encoder_state_dict"]
        cleaned = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v for k, v in enc_state.items()}
        full_model.load_state_dict(cleaned, strict=False)
        self.encoder = full_model.encoder.to(self.device)
        self._encoder_state_dict = {
            k: v for k, v in cleaned.items() if k.startswith("encoder.")
        }
        del full_model
        gc.collect()

        if config.random_encoder:
            print("  Random encoder: re-initializing encoder weights")
            for m in self.encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # Build and load decoder
        dec_config = checkpoint["decoder_config"]
        hurdle_mode = self.config.hurdle_loss_weight > 0
        ckpt_hurdle = dec_config.get("hurdle_mode", False)

        self.decoder = HybridNormGlobalSkipDecoder(
            input_dim=dec_config["input_dim"],
            output_dim=dec_config["output_dim"],
            hidden_dim=dec_config["hidden_dim"],
            num_blocks=dec_config["num_blocks"],
            dropout_rate=dec_config.get("dropout_rate", 0.3),
            se_reduction=dec_config.get("se_reduction", 16),
            # MUST come from the checkpoint. Activation modules hold no parameters, so a
            # softplus-trained decoder loads strict=True under the rectified_sigmoid default
            # and silently runs the wrong nonlinearity — which is the dead-unit trap of
            # NEURIPS_SCALING_AND_SEED_RESULTS.md §12.7, not a small discrepancy.
            output_activation=dec_config.get("output_activation", "rectified_sigmoid"),
            hurdle_mode=hurdle_mode,
        )

        if hurdle_mode and not ckpt_hurdle:
            self._load_pretrained_into_hurdle_decoder(
                self.decoder, checkpoint["decoder_state_dict"]
            )
        else:
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.decoder = self.decoder.to(self.device)

        # Stage C: capture the decoder's learned zinb/nb dispersion (per-gene log_theta/logit_pi) for
        # warm-starting the VC loss (only used when loss_type in {nb,zinb}; ignored otherwise).
        self._ckpt_dispersion = checkpoint.get("dispersion")

        if config.random_decoder:
            print("  Random decoder: re-initializing decoder weights")
            for m in self.decoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        assert dec_config["input_dim"] == hidden_size, (
            f"Decoder input_dim ({dec_config['input_dim']}) != encoder hidden_size "
            f"({hidden_size}). Checkpoint is inconsistent."
        )

        print(
            f"  Encoder loaded: model_size={model_size}, hidden_size={hidden_size}, "
            f"num_genes={self.num_genes}, target_sum={self.target_sum}"
        )
        print(
            f"  Decoder loaded: input_dim={dec_config['input_dim']}, "
            f"hidden_dim={dec_config['hidden_dim']}, "
            f"num_blocks={dec_config['num_blocks']}"
        )

    # ------------------------------------------------------------------
    # v2: Latent loss, delta regularization, decoder warmup
    # ------------------------------------------------------------------

    def _apply_decoder_warmup(self, epoch: int):
        """Override decoder param group LR during warmup period.

        Linear ramp from 0 to decoder_lr over decoder_lr_warmup_epochs.
        Called at the start of train_epoch, before the batch loop.
        """
        if self.decoder_lr_warmup_epochs <= 0 or self._decoder_pg_idx is None:
            return
        if epoch > self.decoder_lr_warmup_epochs:
            return  # warmup complete, scheduler controls decoder LR
        warmup_lr = (epoch / self.decoder_lr_warmup_epochs) * self.decoder_lr
        self.optimizer.param_groups[self._decoder_pg_idx]["lr"] = warmup_lr

    def _apply_encoder_delay(self, epoch: int):
        """Zero the encoder LR until the delay period has elapsed."""
        if self._encoder_pg_idx is None:
            return
        if epoch < self._encoder_release_epoch:
            self.optimizer.param_groups[self._encoder_pg_idx]["lr"] = 0.0
        elif epoch == self._encoder_release_epoch:
            self.optimizer.param_groups[self._encoder_pg_idx]["lr"] = self.config.encoder_lr
            print(f"  [epoch {epoch}] encoder released, lr={self.config.encoder_lr}")

    @staticmethod
    def _load_pretrained_into_hurdle_decoder(
        hurdle_decoder: nn.Module,
        pretrained_state: Dict[str, torch.Tensor],
    ):
        """Load non-hurdle checkpoint weights into a hurdle-mode decoder.

        Copies shared layers (input_layer, res_blocks, output_head BN/GELU)
        and the mean-channel weights from the pretrained output Linear.
        logit_pi channel weights are left at default (near-zero).
        """
        own_state = hurdle_decoder.state_dict()
        for k, v in pretrained_state.items():
            if k == "output_head.2.weight":
                # Pretrained: (output_dim, hidden_dim) → hurdle: (2*output_dim, hidden_dim)
                own_state[k][:v.shape[0], :] = v
            elif k == "output_head.2.bias":
                own_state[k][:v.shape[0]] = v
            elif k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
        hurdle_decoder.load_state_dict(own_state)

    def _init_hurdle_params(self, train_dataset):
        """Initialize hurdle parameters from empirical per-gene statistics.

        Scans SC samples in the training dataset to compute:
        - pi_hat_j: zero fraction per gene → logit_pi bias
        - k_j: Gamma shape from method of moments → log_shape_param
        """
        fwd = self.model.module if hasattr(self.model, "module") else self.model
        decoder = fwd.decoder
        if not getattr(decoder, "hurdle_mode", False):
            return

        profiles = train_dataset.perturbed_profiles  # (N, G) tensor
        sc_flags = getattr(train_dataset, "is_sc_flags", None)

        if sc_flags is not None:
            sc_mask = torch.tensor(sc_flags, dtype=torch.bool)
            if sc_mask.any():
                data = profiles[sc_mask].numpy()
            else:
                data = profiles.numpy()
        else:
            data = profiles.numpy()

        n_genes = data.shape[1]
        pi_hat = np.zeros(n_genes, dtype=np.float32)
        m_plus = np.zeros(n_genes, dtype=np.float32)
        var_plus = np.zeros(n_genes, dtype=np.float32)

        for j in range(n_genes):
            col = data[:, j]
            is_zero = col == 0
            pi_hat[j] = is_zero.mean()
            pos = col[~is_zero]
            if len(pos) > 1:
                m_plus[j] = pos.mean()
                var_plus[j] = pos.var(ddof=1)
            elif len(pos) == 1:
                m_plus[j] = pos[0]
                var_plus[j] = 0.1  # fallback

        # Clamp pi_hat to (0.001, 0.999) for logit stability
        pi_hat = np.clip(pi_hat, 0.001, 0.999)
        logit_pi_bias = np.log(pi_hat / (1 - pi_hat)).astype(np.float32)

        # Gamma shape from method of moments: k_j = m_plus^2 / var_plus
        var_plus = np.clip(var_plus, 0.05**2, 1.5**2)
        m_plus_safe = np.maximum(m_plus, 1e-6)
        k_init = (m_plus_safe ** 2) / var_plus
        k_init = np.clip(k_init, 0.05, 100.0)

        # Set logit_pi bias in the output Linear
        with torch.no_grad():
            out_linear = decoder.output_head[2]  # nn.Linear(hidden, 2*G)
            out_dim = decoder.output_dim
            out_linear.bias[out_dim:] = torch.from_numpy(logit_pi_bias)
            out_linear.weight[out_dim:].zero_()

            # Set log_shape_param via inverse softplus: x = log(exp(k - 0.05) - 1)
            k_shifted = np.maximum(k_init - 0.05, 1e-4)
            log_shape_init = np.log(np.expm1(k_shifted))
            decoder.log_shape_param.copy_(
                torch.from_numpy(log_shape_init.astype(np.float32))
            )

        print(f"  Hurdle init from {data.shape[0]} SC samples: "
              f"pi_hat=[{pi_hat.min():.3f}, {np.median(pi_hat):.3f}, {pi_hat.max():.3f}], "
              f"k_init=[{k_init.min():.2f}, {np.median(k_init):.2f}, {k_init.max():.2f}]")

    def _compute_hurdle_nll(
        self,
        model_output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        gene_mask: torch.Tensor,
        stop_grad_pi: bool = False,
    ) -> torch.Tensor:
        """Mean-preserving hurdle-Gamma NLL for SC samples."""
        m = model_output["m"]
        k = model_output["shape"]  # (G,) per-gene
        logit_pi = model_output["logit_pi"]

        is_zero = (target == 0).float()
        is_pos = 1.0 - is_zero
        gene_mask_f = gene_mask.float()

        log_pi = F.logsigmoid(logit_pi)
        log_one_minus_pi = F.logsigmoid(-logit_pi)

        pi = torch.sigmoid(logit_pi)
        one_minus_pi = torch.clamp(1 - pi, min=self.config.hurdle_q_min, max=1.0)
        if stop_grad_pi:
            one_minus_pi = one_minus_pi.detach()
        m_plus = m / one_minus_pi

        k_expanded = k.unsqueeze(0)
        rate = k_expanded / m_plus.clamp(min=1e-6)
        y_safe = target.clamp(min=1e-6)

        log_gamma = (
            k_expanded * torch.log(rate.clamp(min=1e-8))
            - torch.lgamma(k_expanded)
            + (k_expanded - 1) * torch.log(y_safe)
            - rate * target
        )

        nll = -(is_zero * log_pi + is_pos * (log_one_minus_pi + log_gamma))
        nll_masked = (nll * gene_mask_f).sum() / gene_mask_f.sum().clamp(min=1)
        return nll_masked

    def _compute_latent_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target_expression: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute latent-space loss and diagnostic metrics.

        Encodes target_expression via the frozen encoder to get z_target,
        then computes the latent loss based on self.latent_loss_type.

        Returns:
            (loss_tensor, metrics_dict). loss is 0 if latent_loss_type == "none".
        """
        if self.latent_loss_type == "none":
            return torch.tensor(0.0, device=self.device), {}

        delta_z = model_output["delta_z"]
        z_hat = model_output["z_hat"]
        z_ctrl = model_output["z_ctrl"]

        # Compute z_target from target expression via encoder (always no_grad)
        fwd_model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            z_target = fwd_model.encoder(target_expression).detach()

        delta_z_target = z_target - z_ctrl.detach()

        # Compute loss based on type
        if self.latent_loss_type == "delta_z":
            loss = F.mse_loss(delta_z, delta_z_target)
        elif self.latent_loss_type == "z_hat":
            loss = F.mse_loss(z_hat, z_target)
        elif self.latent_loss_type == "both":
            loss_dz = F.mse_loss(delta_z, delta_z_target)
            loss_zh = F.mse_loss(z_hat, z_target)
            loss = self.latent_loss_alpha * loss_dz + (1 - self.latent_loss_alpha) * loss_zh
        elif self.latent_loss_type == "projected":
            z_hat_proj = self.latent_proj_head(z_hat)
            z_target_proj = self.latent_proj_head(z_target).detach()
            loss = F.mse_loss(z_hat_proj, z_target_proj)
        else:
            raise ValueError(f"Unknown latent_loss_type: {self.latent_loss_type}")

        # Diagnostic metrics (always computed when latent loss is active)
        with torch.no_grad():
            dz_norm = delta_z.norm(dim=-1)
            zc_norm = z_ctrl.norm(dim=-1).clamp(min=1e-8)
            metrics = {
                "delta_z_norm": dz_norm.mean().item(),
                "delta_z_ratio": (dz_norm / zc_norm).mean().item(),
                "delta_z_cosine_w_target": F.cosine_similarity(
                    delta_z, delta_z_target, dim=-1
                ).mean().item(),
                "z_hat_mse_w_target": F.mse_loss(z_hat, z_target).item(),
            }

        return loss, metrics

    def _compute_latent_diagnostics(
        self, model_output: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Latent-space health checks that run REGARDLESS of the latent loss.

        Design note in `VCCv2/OCCAM_STATE_HYBRID.md` §8. These catch two failure modes that
        a loss curve cannot see and that both produce a plausible-looking score rather than
        an obvious crash — which is why they are unconditional rather than opt-in.

        **Residual degeneracy.** `z_hat = z_ctrl + delta_z`. If `||delta_z|| << ||z_ctrl||`
        the perturbation is a rounding error on the control and the model has quietly
        collapsed to the identity. That scores at the identity floor (H1 `seen` 0.0982), not
        at zero, so it reads as a weak model rather than a broken one. `delta_z_ratio` is the
        tell. The same numbers are computed inside `_compute_latent_loss`, but ONLY when a
        latent loss is active — and the arms most at risk are exactly the ones that run
        without one.

        **Off-manifold latents.** Any interaction module not trained through the decoder (the
        STATE candidate, or any latent-only objective) can place `z_hat` where the decoder was
        never fit; the decoder then emits confident nonsense. `z_hat_offmanifold` re-encodes
        the decoded output — if `z_hat` sits on the manifold, `enc(dec(z_hat))` comes back
        close to it.

        ⚠️ The encoder/decoder pair is NOT a perfect autoencoder even on real cells, so the
        raw residual is uninterpretable on its own. `z_ctrl_offmanifold` is measured on the
        SAME batch as the reference and `offmanifold_ratio` is the actual diagnostic: ~1 means
        `z_hat` is no worse than a real cell's latent, >> 1 means the head is drifting off the
        manifold and any score it produces is decoder extrapolation.
        """
        every = int(getattr(self.config, "latent_diag_every", 50) or 0)
        if every <= 0:
            return {}
        self._diag_step = getattr(self, "_diag_step", -1) + 1
        if self._diag_step % every != 0:
            return {}

        z_hat = model_output.get("z_hat")
        z_ctrl = model_output.get("z_ctrl")
        delta_z = model_output.get("delta_z")
        if z_hat is None or z_ctrl is None or delta_z is None:
            return {}

        n = int(getattr(self.config, "latent_diag_rows", 256) or 256)
        fwd = self.model.module if hasattr(self.model, "module") else self.model
        out: Dict[str, float] = {}
        try:
            with torch.no_grad():
                dz = delta_z.norm(dim=-1)
                zc = z_ctrl.norm(dim=-1).clamp(min=1e-8)
                out["delta_z_norm"] = dz.mean().item()
                out["delta_z_ratio"] = (dz / zc).mean().item()

                if not bool(getattr(self.config, "latent_diag_manifold", True)):
                    return out

                # One extra decoder+encoder pass on a subsample — the decoder is 128-341M
                # params, so this is throttled by `latent_diag_every` and capped at
                # `latent_diag_rows` rather than run on the full batch.
                zh, zc_s = z_hat[:n].detach(), z_ctrl[:n].detach()
                was_training = fwd.decoder.training
                fwd.decoder.eval()
                try:
                    def _resid(z):
                        d = fwd.decoder(z)
                        d = d["prediction"] if isinstance(d, dict) else d
                        return ((fwd.encoder(d) - z).norm(dim=-1)
                                / z.norm(dim=-1).clamp(min=1e-8)).mean().item()
                    r_hat = _resid(zh)
                    r_ctrl = _resid(zc_s)
                finally:
                    if was_training:
                        fwd.decoder.train()
                out["z_hat_offmanifold"] = r_hat
                out["z_ctrl_offmanifold"] = r_ctrl
                out["offmanifold_ratio"] = r_hat / max(r_ctrl, 1e-8)
        except Exception as exc:      # diagnostics must never take training down
            if self._diag_step == 0:
                print(f"  [diag] latent diagnostics unavailable ({exc}); continuing")
            return out
        return out

    def _compute_delta_reg(
        self, model_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute delta regularization loss."""
        if self.delta_reg_type == "none":
            return torch.tensor(0.0, device=self.device)

        delta_z = model_output["delta_z"]

        if self.delta_reg_type == "l2":
            return self.delta_reg_lambda * (delta_z ** 2).sum(dim=-1).mean()
        elif self.delta_reg_type == "norm_ratio":
            z_ctrl = model_output["z_ctrl"]
            ratio = delta_z.norm(dim=-1) / z_ctrl.norm(dim=-1).clamp(min=1e-8)
            excess = F.relu(ratio - self.delta_reg_tau)
            return self.delta_reg_lambda * (excess ** 2).mean()
        else:
            raise ValueError(f"Unknown delta_reg_type: {self.delta_reg_type}")

    def _compute_ranking_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        ctrl: torch.Tensor,
        gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss on gene DE magnitudes."""
        if self.config.ranking_loss_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        pred_delta = (prediction - ctrl).abs() * gene_mask.float()
        true_delta = (target - ctrl).abs() * gene_mask.float()

        B, G = pred_delta.shape
        M = self.config.ranking_n_pairs

        idx_i = torch.randint(0, G, (B, M), device=self.device)
        idx_j = torch.randint(0, G, (B, M), device=self.device)

        true_i = true_delta.gather(1, idx_i)
        true_j = true_delta.gather(1, idx_j)
        pred_i = pred_delta.gather(1, idx_i)
        pred_j = pred_delta.gather(1, idx_j)

        target_sign = torch.sign(true_i - true_j)
        valid = target_sign != 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        return F.margin_ranking_loss(
            pred_i[valid], pred_j[valid], target_sign[valid],
            margin=self.config.ranking_margin,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------ streaming v2
    def _materialise_sparse(self, batch):
        """SparseBatch -> (ctrl_embeddings, dense target, pert_indices, gene_mask) on the GPU.

        Targets are densified on the device (no 44,728-wide CPU tensor). Controls are looked up
        in the GPU-resident embedding pool; identity rows (ctrl_idx < 0) use the frozen encoder
        on their own target row, which is exactly what `control_identity_mode=exact` means.
        """
        st = self._stream_v2_state
        dev = self.device
        vals = batch.vals.to(dev, non_blocking=True)
        cols = batch.cols.to(dev, non_blocking=True)
        rowptr = batch.rowptr.to(dev, non_blocking=True)
        rowsum = batch.rowsum.to(dev, non_blocking=True)
        target = densify_csr(vals, cols, rowptr, rowsum, st["num_genes"], st["target_sum"],
                             st["skip_normalize"], st["skip_log1p"])
        gid = batch.group_id.to(dev, non_blocking=True)
        # ---- denoised target (F170 lever 1): replace the single-cell target by a k-cell mean.
        # `gm + (cell - gm)/sqrt(k)` has exactly the mean and variance of a mean of k cells of
        # that (line, pert) group, so for an MSE-family loss it is equivalent to first and second
        # order while needing no random access. k <= 0 means the exact group mean (full pseudobulk).
        # Control-identity rows (gid < 0) are never denoised — their task IS to reproduce one cell.
        k = int(st.get("target_pb_k", 0) or 0)
        if st.get("gmeans") is not None and k != 1:
            has = gid >= 0
            if bool(has.any()):
                gm = st["gmeans"][gid.clamp(min=0)]
                if k <= 0:
                    new = gm
                else:
                    new = gm + (target - gm) / float(np.sqrt(k))
                target = torch.where(has.unsqueeze(1), new, target)
                del gm, new
        # ---- F181/E4: per-source TARGET-DELTA rescaling -----------------------------------
        # Sources span 5.9x in effect norm (median ||Δ||₂ 0.99 HEK293T .. 5.84 joung) because
        # knockdown penetrance differs: relative self-delta is -0.40/-0.56 for Heracles vs
        # -0.82/-0.90 for the essential CRISPRi screens. An unconditioned regression on that
        # mixture fits the source-weighted MEAN of Δ, which is shrunk — the measurement-layer
        # origin of F170's magnitude deficit. This maps every source's delta onto the essential
        # scale:  target' = base + (target - base) / s_source,  base = that group's CONTROL mean.
        # NOTE this rescales the LABEL, which SRCW (a sample weight) does not touch. It cannot
        # fix a SIGN conflict — that is what E3's modality conditioning is for.
        _sc = getattr(self, "_source_delta_scale", None)
        if _sc is not None and st.get("cmeans") is not None and st.get("gline") is not None:
            sid = batch.mask_ids.to(dev, non_blocking=True)
            has = gid >= 0
            if bool(has.any()):
                base = st["cmeans"][st["gline"][gid.clamp(min=0)]]
                sfac = _sc[sid.clamp(0, _sc.numel() - 1)].clamp(min=1e-3).unsqueeze(1)
                new = base + (target - base) / sfac
                target = torch.where(has.unsqueeze(1), new, target)
                del base, sfac, new
        ctrl_idx = batch.ctrl_idx.to(dev, non_blocking=True)
        if ctrl_idx.dim() == 2:                     # k control cells, averaged in EMBEDDING space
            ident = ctrl_idx[:, 0] < 0
            ctrl = st["pool"][ctrl_idx.clamp(min=0)].mean(dim=1)
        else:
            ident = ctrl_idx < 0
            ctrl = st["pool"][ctrl_idx.clamp(min=0)]
        if bool(ident.any()):
            fwd = self.model.module if hasattr(self.model, "module") else self.model
            with torch.no_grad():
                ctrl = ctrl.clone()
                ctrl[ident] = fwd.encoder(target[ident]).to(ctrl.dtype)
        gene_mask = st["mask_table"][batch.mask_ids.to(dev, non_blocking=True)]
        self._last_group_id = gid
        self._last_source_id = batch.mask_ids.to(dev, non_blocking=True)
        return ctrl, target, batch.pert_indices, gene_mask

    def _build_stream_v2(self, config, file_metas, control_matcher, train_perts,
                         train_cell_lines, extra):
        """Control-embedding pools + StreamingDatasetV2 + multi-worker DataLoader."""
        from occam.vcell.stream_v2 import (StreamingDatasetV2, build_control_embedding_pool,
                                            dense_to_csr, encoder_fingerprint)
        fwd = self.model.module if hasattr(self.model, "module") else self.model
        if not config.freeze_encoder:
            raise ValueError("stream_v2 needs a frozen encoder: control embeddings are precomputed")
        if getattr(self, "expression_loss_mode", "full") == "importance_weighted":
            raise ValueError("stream_v2 does not carry raw control expression; "
                             "expression_loss_mode=importance_weighted is unavailable")
        if not hasattr(control_matcher, "pending_pools"):
            raise RuntimeError("stream_v2 expects prepare_virtual_cell_data(build_pools=False)")
        t0 = time.time()
        enc = fwd.encoder
        enc.eval()

        @torch.no_grad()
        def encode_fn(x):
            return enc(x)

        key = encoder_fingerprint(enc, extra=str(getattr(config, "encoder_checkpoint", "")))
        n_enc = len(self.encoder_gene_names)
        extra_dense = None
        if extra is not None:
            e_ctrl, e_tgt, _e_names, _e_pidx, _e_cts, e_mask = extra
            extra_dense = {"H1": e_ctrl}
        pools, means = build_control_embedding_pool(
            control_matcher.pending_pools, encode_fn, n_enc, self.target_sum,
            self.skip_normalize, self.skip_log1p, self.device,
            cache_dir=getattr(config, "control_cache_dir", None), enc_key=key,
            chunk_size=config.optim_chunk_size,
            threads=int(getattr(config, "control_scan_threads", 6)),
            extra_dense=extra_dense,
            # The gene axis is part of the cache identity: embeddings computed under a
            # different axis are a different input, not the same input cached.
            axis_sha256=str(getattr(config, "bench_axis_sha256", "") or ""),
            axis_scope=str(getattr(config, "bench_axis_scope", "") or ""))
        control_matcher.control_pool = pools
        control_matcher.set_control_embeddings(pools)
        self._v2_control_means = means
        order = sorted(pools)
        pool_index, off = {}, 0
        for k in order:
            pool_index[k] = (off, int(pools[k].shape[0]))
            off += int(pools[k].shape[0])
        pool_gpu = torch.cat([pools[k] for k in order], 0).to(self.device)
        print(f"  [stream_v2] control-embedding pool on GPU: {tuple(pool_gpu.shape)} "
              f"({pool_gpu.numel() * 4 / 1e9:.2f} GB), {len(order)} lines")
        extras_csr = None
        if extra is not None:
            ev, ec, erp = dense_to_csr(e_tgt)
            ers = np.full(e_tgt.shape[0], -1.0, dtype=np.float32)   # -1 = already transformed
            mode = (getattr(config, "panel_control_identity_mode", None)
                    or getattr(config, "control_identity_mode", "exact"))
            h_off, h_n = pool_index["H1"]
            ectrl = (np.full(e_tgt.shape[0], -1, dtype=np.int64) if mode == "exact"
                     else h_off + np.arange(e_tgt.shape[0], dtype=np.int64))
            extras_csr = (ev, ec, erp, ers, ectrl, -1)
            del e_ctrl, e_tgt
        _gm_idx = _gmeans = _cmeans = _gline = None
        if (getattr(config, "target_pb_k", 1) != 1) or float(getattr(config, "delta_mag_weight", 0.0)) > 0:
            from occam.vcell.stream_v2 import load_group_means
            from occam.vcell.stream_v2 import source_name_of as _srcname
            _gm_idx, _gmeans, _cmeans, _gline = load_group_means(
                str(getattr(config, "pert_mean_dir", "cache/pert_means")), self.device,
                sources={_srcname(fm.path) for fm in file_metas})
        _snr_min = getattr(config, "min_pert_snr", None)
        _snr_max = getattr(config, "max_pert_snr", None)
        _snr_tab, _keep = None, None
        if _snr_min is not None or _snr_max is not None:
            from occam.vcell.stream_v2 import load_pert_snr_table
            _snr_tab = load_pert_snr_table(str(getattr(config, "pert_effect_dir", "cache/pert_effects")))
            _kp = getattr(config, "effect_keep_perts", None)
            if _kp:
                import pandas as _pd
                _df = _pd.read_csv(_kp)
                _col = "target_gene" if "target_gene" in _df.columns else _df.columns[0]
                _keep = set(_df[_col].astype(str))
                print(f"  [snr-filter] {len(_keep)} perturbations exempt from the filter ({Path(_kp).name})")
        ds = StreamingDatasetV2(
            file_metas=file_metas, pert2id=self.pert2id, split_pert_names=train_perts,
            control_matcher=control_matcher, encoder_gene_names=self.encoder_gene_names,
            target_sum=self.target_sum, n_controls=1, batch_size=config.batch_size,
            chunk_size=config.optim_chunk_size, buffer_chunks=config.optim_buffer_chunks,
            shuffle=True, seed=config.seed,
            include_controls_as_training=config.include_controls_as_training,
            train_cell_lines=train_cell_lines,
            control_identity_mode=getattr(config, "control_identity_mode", "paired"),
            max_cells_per_pert=getattr(config, "max_cells_per_pert_stream", 0),
            min_cells_per_pert=getattr(config, "min_cells_per_pert_stream", 0),
            balance_sources=getattr(config, "balance_sources", True),
            fixed_pairs=getattr(config, "stream_fixed_pairs", False),
            batch_match=getattr(config, "stream_batch_match", False),
            pool_index=pool_index, mask_table=None, mask_id_of_meta={}, extras_csr=None,
            interleave=getattr(config, "stream_interleave", "roundrobin"),
            pert_snr=_snr_tab, min_pert_snr=_snr_min, keep_perts=_keep, max_pert_snr=_snr_max,
            group_index=_gm_idx, ctrl_avg_k=int(getattr(config, "ctrl_avg_k", 1)))
        masks = [fm.gene_mask for fm in ds.active_metas]
        if extras_csr is not None:
            masks.append(e_mask[0] if e_mask.ndim == 2 else e_mask)
            extras_csr = extras_csr[:5] + (len(masks) - 1,)
        mask_table = torch.stack([m.bool() for m in masks], 0)
        ds.mask_table = mask_table
        ds.mask_id_of_meta = {mi: mi for mi in range(len(ds.active_metas))}
        ds.extras_csr = extras_csr
        ds._n_extra = 0 if extras_csr is None else int(extras_csr[2].shape[0] - 1)
        self._stream_v2_state = {
            "pool": pool_gpu, "mask_table": mask_table.to(self.device), "num_genes": n_enc,
            "target_sum": self.target_sum, "skip_normalize": self.skip_normalize,
            "skip_log1p": self.skip_log1p,
            "gmeans": _gmeans, "cmeans": _cmeans, "gline": _gline,
            "target_pb_k": int(getattr(config, "target_pb_k", 1)),
        }
        if _gmeans is not None:
            print(f"  [stream_v2] target_pb_k={getattr(config, 'target_pb_k', 1)} "
                  f"ctrl_avg_k={getattr(config, 'ctrl_avg_k', 1)} "
                  f"delta_mag_weight={getattr(config, 'delta_mag_weight', 0.0)}")
        # ---- F181/E3: per-source ASSAY MODALITY table (0 = knockdown, 1 = activation) ----
        # Measured from cache/pert_means: the self-gene delta (perturbing g must move g) is
        # negative for every source except joung_2023, whose median relative self-delta is
        # +0.847 with only 14.3% of perturbations going down — a TF OVEREXPRESSION atlas.
        _nslots = int(getattr(config, "source_cond_slots", 64))
        _mod = torch.zeros(_nslots + 1, dtype=torch.long)
        _mod[_nslots] = 2                                   # NEUTRAL slot -> neutral modality
        if str(getattr(config, "source_cond", "none")) != "none":
            _act = []
            for mi, fm in enumerate(ds.active_metas):
                k = int(ds.mask_id_of_meta[mi])
                if k <= _nslots and "joung" in Path(fm.path).name.lower():
                    _mod[k] = 1
                    _act.append(Path(fm.path).name)
            print(f"  [source_cond] modality table: {len(_act)} ACTIVATION source(s) "
                  f"{_act or '(none)'}; all others = knockdown")
        self._source_modality = _mod.to(self.device)

        # ---- F181/E4: per-source delta scale (label calibration) ----
        self._source_delta_scale = None
        if str(getattr(config, "source_target_scale", "none")) != "none":
            import json as _json
            _sp = Path(getattr(config, "source_scale_json", "")
                       or (Path(__file__).resolve().parents[1] / "cache/source_delta_scale.json"))
            if not _sp.exists():
                print(f"  ⚠ [source_target_scale] {_sp} missing — DISABLED "
                      f"(regenerate with the F181 probe)")
            else:
                _tab = _json.loads(_sp.read_text())["scale"]
                _v = torch.ones(_nslots + 1, dtype=torch.float32)
                _hit = []
                for mi, fm in enumerate(ds.active_metas):
                    k = int(ds.mask_id_of_meta[mi])
                    stem = Path(fm.path).name
                    match = next((kk for kk in _tab
                                  if kk.lower() in stem.lower()
                                  or kk.lower().replace("_", "") in stem.lower().replace("_", "")),
                                 None)
                    if match is not None and k <= _nslots:
                        _v[k] = float(_tab[match]); _hit.append((stem, match, float(_tab[match])))
                    else:
                        print(f"  ⚠ [source_target_scale] no scale entry for {stem} — left at 1.0")
                self._source_delta_scale = _v.to(self.device)
                print(f"  [source_target_scale] delta divided by, per source "
                      f"(reference = mean ||Δ|| of the essential 4):")
                for stem, match, v in _hit:
                    print(f"      {stem:<40} {match:<10} s={v:6.3f}")

        # ---- per-source weights (F174) : equalise each source's gradient share ----
        self._source_w = None
        if str(getattr(config, "source_weight", "none")) != "none":
            counts = torch.ones(mask_table.shape[0], dtype=torch.float64)
            is_file = torch.zeros(mask_table.shape[0], dtype=torch.bool)
            for mi, fm in enumerate(ds.active_metas):
                counts[ds.mask_id_of_meta[mi]] = float(ds._static_ok[mi].sum() + ds._ctrl_ok[mi].sum())
                is_file[ds.mask_id_of_meta[mi]] = True
            n_extra = 0.0
            if extras_csr is not None:
                n_extra = float(ds._n_extra or 0)
                counts[extras_csr[5]] = max(n_extra, 1.0)
            counts = counts.clamp(min=1.0)
            mode = str(getattr(config, "source_weight", "none"))
            # ⚠️ Balance the TRAINING FILES only. The H1 control-identity rows are a deliberately
            # sized auxiliary task (18,400 = all of pool A); treating them as a 13th "source" would
            # up-weight them 10.7x on essential and ~230x on all9, which would confound this lever
            # with "train much harder on H1 reconstruction". They keep weight exactly 1.0.
            w = torch.ones_like(counts)
            cf = counts[is_file]
            raw = (1.0 / cf) if mode == "balanced" else (1.0 / cf.sqrt())
            w[is_file] = raw * (cf.sum() / (raw * cf).sum())      # sample-mean 1 over FILE samples
            self._source_w = w.float().to(self.device)
            print(f"  [source-weight={mode}] per-sample weights (sample-mean 1):")
            for mi, fm in enumerate(ds.active_metas):
                k = ds.mask_id_of_meta[mi]
                print(f"      {Path(fm.path).name:<40} n={int(counts[k]):>10,}  w={float(w[k]):8.3f}")
            if extras_csr is not None:
                print(f"      {'h1_control_identity':<40} n={int(counts[extras_csr[5]]):>10,}  "
                      f"w={float(w[extras_csr[5]]):8.3f}")
        nw = int(getattr(config, "num_workers", 0) or 0)
        loader = DataLoader(ds, batch_size=None, num_workers=nw, pin_memory=(nw > 0),
                            prefetch_factor=(4 if nw > 0 else None), persistent_workers=False,
                            worker_init_fn=_seed_worker if nw > 0 else None)
        print(f"  [stream_v2] dataset ready: interleave={ds.interleave} workers={nw} "
              f"extras={ds._n_extra} masks={mask_table.shape[0]}; setup {time.time() - t0:.0f}s; "
              f"RSS {rss_gb():.1f} GB", flush=True)
        return ds, loader

    def _params_finite(self) -> bool:
        for n, t in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if t is not None and t.is_floating_point() and not torch.isfinite(t).all():
                print(f"  [save-guard] non-finite tensor {n}; this epoch cannot become best")
                return False
        return True

    @property
    def _pending_scores(self):
        if not hasattr(self, "_pending_scores_list"):
            self._pending_scores_list = []
        return self._pending_scores_list


    def _ok_to_save_best(self) -> bool:
        """Refuse to checkpoint a model whose last eval substituted non-finite predictions or
        whose parameters/buffers are non-finite (F164: NaN models were saved as best)."""
        try:
            from occam.vcell.panel import predict_panel as _p
            bad = int(getattr(_p, "last_nonfinite_rows", 0))
        except Exception:  # noqa: BLE001
            bad = 0
        bad += int(getattr(self, "_eval_nonfinite_rows", 0) or 0)
        if bad:
            print(f"  [save-guard] NOT saving best: {bad:,} non-finite prediction rows in the "
                  f"last evaluation (F158/F161). The score is a substitution artefact.")
            return False
        for n, t in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if t is not None and t.is_floating_point() and not torch.isfinite(t).all():
                print(f"  [save-guard] NOT saving best: non-finite tensor {n}")
                return False
        return True

    def train_epoch(
        self, train_loader, wandb_run=None, epoch: int = 0, total_epochs: int = 0
    ) -> Dict[str, float]:
        """Run one training epoch with progress bar and per-batch wandb logging."""
        model = self.model
        model.train()
        fwd_model = model.module if hasattr(model, "module") else model

        # v2: Apply decoder LR warmup at start of epoch
        self._apply_decoder_warmup(epoch)
        self._apply_encoder_delay(epoch)

        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_grad_norm = 0          # how many batches actually recorded one (F151)
        grad_norm_max = 0.0      # max over FINITE norms (the mean hides a single bad batch)
        n_bad_gradnorm = 0       # non-finite grad-norms, counted separately (F152)
        # v2: accumulators for diagnostic metrics
        epoch_recon_loss = 0.0
        self._per_source = {}
        epoch_latent_loss = 0.0
        epoch_delta_reg = 0.0
        epoch_gpl_loss = 0.0
        epoch_ranking_loss = 0.0
        epoch_hurdle_loss = 0.0
        epoch_mag_loss = 0.0
        epoch_hurdle_pi_sum = 0.0
        epoch_hurdle_n = 0
        epoch_latent_metrics: Dict[str, float] = {}
        n_batches = 0
        n_samples = 0
        _bench = int(getattr(self.config, "bench_batches", 0) or 0)
        _t_bench = time.time()
        # NaN guard bookkeeping (F147) — see the guard below for why this is always on.
        n_nonfinite = 0
        n_nonfinite_samples = 0

        # Streaming dataset __len__ returns sample count, not batch count
        ds = train_loader.dataset
        if hasattr(ds, 'batch_size'):
            n_total_batches = (len(ds) + ds.batch_size - 1) // ds.batch_size
        elif train_loader.batch_size is not None:
            n_total_batches = (len(ds) + train_loader.batch_size - 1) // train_loader.batch_size
        else:
            n_total_batches = None

        pbar = tqdm(
            train_loader,
            total=n_total_batches,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=False,
            disable=(self.rank != 0),
        )

        for batch in pbar:
            if isinstance(batch, PerturbationBatch):
                ctrl = batch.control_expression.to(self.device, non_blocking=True)
                target = batch.perturbed_expression.to(self.device, non_blocking=True)
                pert_idx = batch.pert_indices
                gene_mask = batch.gene_masks.to(self.device, non_blocking=True)
            elif isinstance(batch, SparseBatch):
                ctrl, target, pert_idx, gene_mask = self._materialise_sparse(batch)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            bs = ctrl.shape[0]
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=getattr(self, "_amp_dtype", torch.float16), enabled=(self.device == "cuda")):
                # v2: forward returns dict with intermediates
                _combo_seq = getattr(self.config, "combo_decomposition", False)
                if _combo_seq:
                    # Sequential composition, chained in latent space, decoded once.
                    # Doubles average the A-then-B and B-then-A cell states, so the
                    # prediction is order-invariant by construction (no penalty needed).
                    _base = fwd_model.module if hasattr(fwd_model, "module") else fwd_model
                    model_output = _base.forward_sequential(ctrl, pert_idx)
                else:
                    # F181/E3: hand the estimand's identity to the model. No-op when
                    # source_cond="none" (both stay None -> neutral slot, zero embedding).
                    _sid = _mid = None
                    _base0 = fwd_model.module if hasattr(fwd_model, "module") else fwd_model
                    if getattr(_base0, "source_emb", None) is not None:
                        _sid = getattr(self, "_last_source_id", None)
                        _mt = getattr(self, "_source_modality", None)
                        if _sid is not None and _mt is not None:
                            _mid = _mt[_sid.clamp(0, _mt.numel() - 1)]
                    model_output = fwd_model(ctrl, pert_idx, source_ids=_sid, modality_ids=_mid)
                prediction = model_output["prediction"]

                # GPL ranking loss (needs 2D structure, before gene_mask flattening)
                if self.gpl_fn is not None:
                    gpl_loss = self.gpl_fn(prediction, target, mask=gene_mask)

                # Per-sample gene mask loss
                masked_pred = prediction[gene_mask]
                masked_target = target[gene_mask]

                # v2: Expression loss mode
                if self.expression_loss_mode == "importance_weighted":
                    with torch.no_grad():
                        importance = (target - ctrl).abs()
                        importance = importance / (importance.mean() + 1e-8)
                        importance = importance ** self.config.importance_scale
                        importance = importance.clamp(min=0.1, max=10.0)
                        importance_masked = importance[gene_mask]
                    if self.config.loss_type == "smooth_l1":
                        per_elem = F.smooth_l1_loss(
                            masked_pred, masked_target,
                            beta=self.config.smooth_l1_beta, reduction="none",
                        )
                    elif self.config.loss_type == "mse":
                        per_elem = F.mse_loss(masked_pred, masked_target, reduction="none")
                    elif self.config.loss_type == "log_cosh":
                        per_elem = LogCoshLoss(reduction="none")(masked_pred, masked_target)
                    elif self.config.loss_type == "poisson":
                        per_elem = PoissonCountLoss(reduction="none")(masked_pred, masked_target)
                    elif self.config.loss_type == "poisson_smooth_l1":
                        p_elem = PoissonCountLoss(reduction="none")(masked_pred, masked_target)
                        s_elem = F.smooth_l1_loss(
                            masked_pred, masked_target,
                            beta=self.config.smooth_l1_beta, reduction="none",
                        )
                        per_elem = self._poisson_weight * p_elem + (1 - self._poisson_weight) * s_elem
                    elif self.config.loss_type in ("nb", "zinb"):
                        raise ValueError(
                            "nb/zinb not supported with expression_loss_mode=importance_weighted "
                            "(per-gene dispersion needs the 2D gene dim; use expression_loss_mode=full)"
                        )
                    recon_loss = (importance_masked * per_elem).mean()
                elif self.config.loss_type == "poisson_smooth_l1":
                    recon_loss = (
                        self._poisson_weight * self._poisson_fn(masked_pred, masked_target)
                        + (1 - self._poisson_weight) * self._smooth_l1_fn(masked_pred, masked_target)
                    )
                elif (getattr(self.config, "loss_norm", "pooled") != "pooled"
                      or getattr(self, "_source_w", None) is not None):
                    # F174 levers. The default `pooled` reduction is a mean over the batch's
                    # FLATTENED masked elements, so a cell's weight is proportional to how many
                    # genes ITS FILE measures: a Heracles cell (29,422 aligned genes) counts 3.4x
                    # an essential cell (8,950), and inside all9 the essential corpus ends up with
                    # 2.5 % of the gradient. Two independent corrections:
                    #   loss_norm=per_cell -> mean over each cell's own masked genes, then over cells
                    #   source_weight      -> per-sample weight so each SOURCE contributes equally
                    if self.config.loss_type in ("nb", "zinb"):
                        raise ValueError("loss_norm/source_weight are implemented for the "
                                         "elementwise losses only, not nb/zinb")
                    per_elem = self._recon_elem(prediction, target)      # (B, G), no reduction
                    gmf = gene_mask.float()
                    w = None
                    if getattr(self, "_source_w", None) is not None:
                        sid = getattr(self, "_last_source_id", None)
                        if sid is not None:
                            w = self._source_w[sid]                      # (B,), sample-mean 1
                    if str(getattr(self.config, "loss_norm", "pooled")) == "per_cell":
                        per_cell = (per_elem * gmf).sum(dim=1) / gmf.sum(dim=1).clamp(min=1.0)
                        recon_loss = (per_cell.mean() if w is None
                                      else (per_cell * w).sum() / w.sum().clamp(min=1e-8))
                    else:
                        ww = gmf if w is None else gmf * w.unsqueeze(1)
                        recon_loss = (per_elem * ww).sum() / ww.sum().clamp(min=1e-8)
                    del per_elem, gmf
                else:
                    recon_loss = self._recon_loss(prediction, target, gene_mask)

                # Apply loss scaling (e.g., to normalize Poisson magnitude)
                if self.recon_loss_scale != 1.0:
                    recon_loss = recon_loss / self.recon_loss_scale

                if self.cosine_loss_weight > 0:
                    gene_mask_f = gene_mask.float()
                    cos_loss = 1 - F.cosine_similarity(
                        prediction * gene_mask_f, target * gene_mask_f, dim=-1
                    ).mean()
                    expr_loss = (1 - self.cosine_loss_weight) * recon_loss + self.cosine_loss_weight * cos_loss
                else:
                    expr_loss = recon_loss

                # Hurdle-Gamma NLL for SC samples (PB keeps expr_loss above)
                # ---- delta-magnitude term (F170 lever 2). Trains the SIZE of the predicted
                # effect against the group-level truth, which is what shrink currently patches
                # post hoc. Both references are precomputed means, so no extra forward pass, and
                # the single-cell target's noise (3-70x the effect) never enters the magnitude.
                mag_loss = torch.tensor(0.0, device=self.device)
                _mw = float(getattr(self.config, "delta_mag_weight", 0.0))
                if _mw > 0 and getattr(self, "_last_group_id", None) is not None:
                    st = self._stream_v2_state
                    gid = self._last_group_id
                    has = gid >= 0
                    if bool(has.any()) and st.get("gmeans") is not None:
                        g = gid.clamp(min=0)
                        gm = st["gmeans"][g]
                        cm = st["cmeans"][st["gline"][g]]
                        gmf = gene_mask.float()
                        true_n = (((gm - cm) * gmf) ** 2).sum(dim=1).sqrt()
                        pred_n = (((prediction - cm) * gmf) ** 2).sum(dim=1).sqrt()
                        ratio = pred_n / true_n.clamp(min=1e-6)
                        mag_loss = ((ratio - 1.0) ** 2)[has].mean()
                        del gm, cm, gmf, true_n, pred_n, ratio
                hurdle_loss = torch.tensor(0.0, device=self.device)
                if self.config.hurdle_loss_weight > 0 and "logit_pi" in model_output:
                    is_sc = batch.is_sc
                    if is_sc is not None and is_sc.any():
                        sc_mask = is_sc.to(self.device)
                        sc_output = {}
                        for k, v in model_output.items():
                            if isinstance(v, torch.Tensor):
                                if v.dim() > 0 and v.shape[0] == sc_mask.shape[0]:
                                    sc_output[k] = v[sc_mask]
                                else:
                                    sc_output[k] = v
                        sc_target = target[sc_mask]
                        sc_gene_mask = gene_mask[sc_mask]

                        stop_grad = (epoch <= self.config.hurdle_warmup_epochs)
                        hurdle_loss = self._compute_hurdle_nll(
                            sc_output, sc_target, sc_gene_mask,
                            stop_grad_pi=stop_grad,
                        )

                        # Recompute expr_loss for PB only, then combine
                        pb_mask = ~sc_mask
                        if pb_mask.any():
                            pb_pred = prediction[pb_mask]
                            pb_target_t = target[pb_mask]
                            pb_gm = gene_mask[pb_mask]
                            pb_recon = self._recon_loss(pb_pred, pb_target_t, pb_gm)
                            if self.recon_loss_scale != 1.0:
                                pb_recon = pb_recon / self.recon_loss_scale
                            if self.cosine_loss_weight > 0:
                                pb_gm_f = pb_gm.float()
                                pb_cos = 1 - F.cosine_similarity(
                                    pb_pred * pb_gm_f, pb_target_t * pb_gm_f, dim=-1
                                ).mean()
                                pb_expr = (1 - self.cosine_loss_weight) * pb_recon + self.cosine_loss_weight * pb_cos
                            else:
                                pb_expr = pb_recon
                        else:
                            pb_expr = torch.tensor(0.0, device=self.device)

                        expr_loss = pb_expr + self.config.hurdle_loss_weight * hurdle_loss

                        # Monitoring accumulators
                        with torch.no_grad():
                            sc_pi = torch.sigmoid(sc_output["logit_pi"])
                            epoch_hurdle_pi_sum += sc_pi.mean().item() * sc_mask.sum().item()
                            epoch_hurdle_n += sc_mask.sum().item()

                # v2: Latent loss (supervise delta_z or z_hat)
                latent_loss, latent_metrics = self._compute_latent_loss(
                    model_output, target,
                )
                # Health checks that must run even when there is no latent loss — the arms
                # without one are precisely the ones that can collapse to the identity or
                # drift off the decoder's manifold unnoticed. §8 of OCCAM_STATE_HYBRID.md.
                for _k, _v in self._compute_latent_diagnostics(model_output).items():
                    latent_metrics.setdefault(_k, _v)

                # v2: Delta regularization
                delta_reg = self._compute_delta_reg(model_output)

                # v2: Ranking loss on gene DE magnitudes
                ranking_loss = self._compute_ranking_loss(prediction, target, ctrl, gene_mask)

                # v2: Total loss
                loss = expr_loss + self.latent_loss_weight * latent_loss + delta_reg
                if _mw > 0:
                    loss = loss + _mw * mag_loss
                if self.gpl_fn is not None:
                    loss = loss + self.gpl_weight * gpl_loss
                if self.config.ranking_loss_weight > 0:
                    loss = loss + self.config.ranking_loss_weight * ranking_loss

            # NaN provenance. The streaming single-cell path produced `loss=nan` on epoch 1
            # even on the essential-only corpus that trains fine cached, so the cause is in
            # the path, not the data. An epoch-level "loss=nan" says nothing about WHICH
            # batch or WHY, so isolate the first offending batch and report enough to act
            # on: which loss term went non-finite, and what the inputs looked like.
            # Diagnostic only — enabled by --debug_nan, off by default.
            if getattr(self.config, "debug_nan", False) and not torch.isfinite(loss):
                def _d(name, t):
                    if not isinstance(t, torch.Tensor):
                        return f"{name}={t}"
                    f = torch.isfinite(t)
                    return (f"{name}: finite={int(f.sum())}/{t.numel()} "
                            f"min={t[f].min().item() if f.any() else float('nan'):.4g} "
                            f"max={t[f].max().item() if f.any() else float('nan'):.4g}")
                print("\n" + "=" * 60)
                print(f"[nan] FIRST NON-FINITE LOSS — batch {n_batches}, epoch {epoch}")
                for nm, t in [("expr_loss", expr_loss), ("latent_loss", latent_loss),
                              ("delta_reg", delta_reg), ("total", loss)]:
                    print(f"  {nm} = {t.item() if isinstance(t, torch.Tensor) and t.numel()==1 else t}")
                print("  " + _d("control_in ", ctrl))
                print("  " + _d("target     ", target))
                print("  " + _d("prediction ", prediction))
                print("  " + _d("gene_mask  ", gene_mask.float() if gene_mask is not None else None))
                if gene_mask is not None:
                    per_row = gene_mask.sum(dim=1)
                    print(f"  genes present per row: min={int(per_row.min())} "
                          f"max={int(per_row.max())}  ROWS WITH ZERO PRESENT GENES="
                          f"{int((per_row == 0).sum())}")
                nulls = sum(1 for p in batch.pert_names if p == "non-targeting")
                print(f"  batch: n={len(batch.pert_names)} null-token(control-identity)={nulls}")
                print(f"  cell_types: {sorted(set(batch.cell_types))[:8]}")
                print("=" * 60 + "\n", flush=True)
                raise RuntimeError(
                    "non-finite loss with --debug_nan; see the dump above")

            # PER-SOURCE DIAGNOSTICS. Which dataset actually drives learning? The corpus
            # mixes sources with wildly different depth — median cells per
            # (cell_line, perturbation) is 45-121 for the essential files and
            # feng_targeted, but 1-2 for Feng genome-wide — so an aggregate loss cannot
            # say whether a source contributes signal or noise. Three quantities, all
            # cheap:
            #   n           samples seen from that source
            #   err         mean |prediction - target| over PRESENT genes
            #   dmag        mean |prediction - control|, i.e. is the model even emitting
            #               a delta for this source, or collapsing to the control?
            #   gnorm       ||d loss / d prediction|| for that source's rows — the direct
            #               measure of how hard the source pushes the model. Obtained from
            #               the existing backward via retain_grad(), so it costs one extra
            #               tensor, not an extra pass.
            # ================= NaN GUARD — ALWAYS ON (F147) =================
            # A single non-finite loss, taken through backward+step ONCE, turns every
            # weight into NaN permanently. The model then decodes one constant profile
            # regardless of the perturbation, the run keeps "training" at full speed, and
            # nothing complains. That is exactly how E8 burned ~190 GPU-hours across 14
            # arms: 5,955 corrupt CD4 cells (0.084% of the corpus) NaN'd the entire grid,
            # every arm checkpointed a NaN model as "New best model", and every arm scored
            # 0.0591 — below the identity floor.
            #
            # The pre-existing `--debug_nan` dump above is a DIAGNOSTIC and is off by
            # default, so it never fired. This is a GUARD: it is on unless explicitly
            # disabled, it costs one `torch.isfinite` on a scalar per batch, and it drops
            # the offending batch BEFORE backward so the weights are never poisoned.
            #
            # Skipping is not silently tolerating: a corpus that is systematically broken
            # must fail loudly rather than quietly train on a subset, so the epoch-end
            # check aborts once the non-finite fraction exceeds `nan_guard_max_frac`.
            if getattr(self.config, "nan_guard", True) and not torch.isfinite(loss):
                n_nonfinite += 1
                n_nonfinite_samples += bs
                if n_nonfinite == 1 and self.rank == 0:
                  # A guard that crashes the run it is guarding is worse than no guard,
                  # so the report itself is best-effort.
                  try:
                    with torch.no_grad():
                        def _bad(t):
                            if not isinstance(t, torch.Tensor):
                                return "n/a"
                            nf = (~torch.isfinite(t))
                            return f"{int(nf.sum())}/{t.numel()} non-finite"
                        libs = ctrl.sum(dim=1)
                        n_zero_lib = int((libs == 0).sum())
                        # WHICH TERM went bad. Without this the guard says "loss is NaN" and
                        # leaves you guessing between the NB reconstruction, the latent loss and
                        # the regulariser — which is exactly the position the first E8 failure
                        # left us in. `expm1` in NBDecoderLoss overflows float32 above ~88.7 in
                        # log1p space, so the input maxima are printed alongside.
                        _terms = []
                        for _nm in ("expr_loss", "recon_loss", "latent_loss", "delta_reg",
                                    "gpl_loss", "ranking_loss", "hurdle_loss"):
                            _t = locals().get(_nm, None)
                            if _t is None:
                                continue
                            try:
                                _terms.append(f"{_nm}={_t.item():.6g}")
                            except Exception:
                                pass
                        _rng = (f"target[min={target.min().item():.4g} max={target.max().item():.4g}] "
                                f"pred[min={prediction.min().item():.4g} max={prediction.max().item():.4g}] "
                                f"ctrl[max={ctrl.max().item():.4g}]")
                        print(
                            f"\n[nan-guard] epoch {epoch} batch {n_batches}: NON-FINITE LOSS "
                            f"— batch dropped, weights untouched.\n"
                            f"            loss={loss.item()}  control={_bad(ctrl)}  "
                            f"target={_bad(target)}  prediction={_bad(prediction)}\n"
                            f"            TERMS: {'  '.join(_terms) if _terms else '(unavailable)'}\n"
                            f"            RANGES: {_rng}   (expm1 overflows fp32 above ~88.7)\n"
                            f"            ZERO-LIBRARY control rows in batch: {n_zero_lib}/{ctrl.shape[0]}"
                            f"   <- a zero-library cell is the classic cause\n"
                            f"            cell_types: {sorted(set(batch.cell_types))[:8]}\n"
                            f"            (re-run with --debug_nan to raise on the first one instead)",
                            flush=True,
                        )
                  except Exception as _e:  # noqa: BLE001
                    print(f"[nan-guard] non-finite loss at epoch {epoch} batch {n_batches} "
                          f"(detail unavailable: {type(_e).__name__})", flush=True)
                self.optimizer.zero_grad(set_to_none=True)
                # EARLY trip. Waiting for the epoch-end verdict means burning a full epoch
                # (~4h15m on the 18.7M corpus) before a systematically broken corpus is
                # reported. Once enough batches have been seen for the fraction to mean
                # something, fail here instead.
                _seen = n_batches + n_nonfinite
                _maxf = float(getattr(self.config, "nan_guard_max_frac", 0.01))
                if _seen >= 500 and (n_nonfinite / _seen) > _maxf:
                    raise RuntimeError(
                        f"[nan-guard] ABORTING EARLY at epoch {epoch}: {n_nonfinite:,} of the "
                        f"first {_seen:,} batches had a non-finite loss "
                        f"({100 * n_nonfinite / _seen:.2f}%, limit {100 * _maxf:.2f}%). "
                        f"The corpus is broken — fix it rather than training on the remainder. "
                        f"For the CD4 int32-overflow case see scripts/repair_cd4_indptr.py."
                    )
                continue
            # ================= end NaN guard =================

            _ps = getattr(self.config, "log_per_source", False)
            if _ps and prediction.requires_grad:
                prediction.retain_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    trainable_params = [
                        p for pg in self.optimizer.param_groups for p in pg["params"]
                    ]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.config.gradient_clip
                    ).item()
                else:
                    grad_norm = None
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                trainable_params = [
                    p for pg in self.optimizer.param_groups for p in pg["params"]
                ]
                # ============ GRADIENT GUARD — must run BEFORE clipping (F150) ============
                # The loss guard above is not sufficient: gradients can go non-finite while the
                # loss stays perfectly healthy. Measured on this corpus — arms reported
                # `Epoch 1/12 | loss=0.1821 | grad_norm=inf`, i.e. a fine loss and an inf gradient.
                #
                # 🔴 And an inf gradient is FATAL, because `clip_grad_norm_` CONVERTS IT TO NaN:
                # with total_norm = inf, clip_coef = max_norm / (inf + 1e-6) = 0, and then
                # `grad * 0` is `inf * 0` = **NaN**. Verified directly:
                #     grads=inf -> total_norm=inf -> grads_after=[nan, nan, nan]
                #     grads=nan -> total_norm=nan -> grads_after=[nan, nan, nan]
                # So clipping does not rescue a bad gradient, it launders it into NaN and the
                # next optimizer step writes NaN into every weight. This is the same
                # `inf * 0 = NaN` shape as F149, this time inside PyTorch's own utility.
                #
                # Therefore: check finiteness FIRST, and if anything is non-finite drop the step
                # entirely rather than clipping it.
                if getattr(self.config, "nan_guard", True):
                    _bad_grad = False
                    for _p in trainable_params:
                        if _p.grad is not None and not torch.isfinite(_p.grad).all():
                            _bad_grad = True
                            break
                    if _bad_grad:
                        n_nonfinite += 1
                        n_nonfinite_samples += bs
                        if n_nonfinite == 1 and self.rank == 0:
                            print(
                                f"\n[nan-guard] epoch {epoch} batch {n_batches}: NON-FINITE "
                                f"GRADIENT with a FINITE loss ({loss.item():.6g}) — step skipped, "
                                f"weights untouched.\n"
                                f"            (clip_grad_norm_ would have turned this into NaN "
                                f"weights: inf * 0 = NaN)\n"
                                f"            cell_types: {sorted(set(batch.cell_types))[:8]}",
                                flush=True,
                            )
                        self.optimizer.zero_grad(set_to_none=True)
                        _seen = n_batches + n_nonfinite
                        _maxf = float(getattr(self.config, "nan_guard_max_frac", 0.01))
                        if _seen >= 500 and (n_nonfinite / _seen) > _maxf:
                            raise RuntimeError(
                                f"[nan-guard] ABORTING EARLY at epoch {epoch}: {n_nonfinite:,} of "
                                f"the first {_seen:,} batches had a non-finite loss or gradient "
                                f"({100 * n_nonfinite / _seen:.2f}%, limit {100 * _maxf:.2f}%)."
                            )
                        continue
                # ============ end gradient guard ============
                if self.config.gradient_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.config.gradient_clip
                    ).item()
                    # ===== SECOND HOLE, closed (F152) =====
                    # Checking each gradient for finiteness is NOT sufficient. FINITE gradients
                    # can overflow when SQUARED AND SUMMED, so `clip_grad_norm_` can return
                    # inf/NaN while every individual gradient passed the check above. Measured:
                    #     grad=1e18 (finite) -> total_norm=inf
                    # The code smoke `17880956` showed exactly this — `grad_norm=nan` with the
                    # guard firing ZERO times — and it only survived because that run had lr=0,
                    # which makes the optimizer step a no-op. At lr>0 it destroys the model.
                    # So the RETURNED norm must be checked too, and the step dropped if it is bad
                    # (the gradients have already been scaled by a garbage coefficient by then).
                    if getattr(self.config, "nan_guard", True) and not math.isfinite(grad_norm):
                        n_nonfinite += 1
                        n_nonfinite_samples += bs
                        n_bad_gradnorm += 1
                        if n_bad_gradnorm == 1 and self.rank == 0:
                            try:
                                worst = sorted(
                                    ((float(_p.grad.detach().abs().max()), _n)
                                     for _n, _p in self.model.named_parameters()
                                     if _p.grad is not None),
                                    reverse=True,
                                )[:4]
                                wtxt = "  ".join(f"{_n}:{_v:.4g}" for _v, _n in worst)
                            except Exception:
                                wtxt = "(unavailable)"
                            print(
                                f"\n[nan-guard] epoch {epoch} batch {n_batches}: grad_norm="
                                f"{grad_norm} from FINITE gradients (sum-of-squares overflowed "
                                f"fp32) — step skipped.\n"
                                f"            loss was {loss.item():.6g}; largest |grad| per "
                                f"tensor: {wtxt}\n"
                                f"            cell_types: {sorted(set(batch.cell_types))[:8]}",
                                flush=True,
                            )
                        self.optimizer.zero_grad(set_to_none=True)
                        _seen = n_batches + n_nonfinite
                        _maxf = float(getattr(self.config, "nan_guard_max_frac", 0.01))
                        if _seen >= 500 and (n_nonfinite / _seen) > _maxf:
                            raise RuntimeError(
                                f"[nan-guard] ABORTING EARLY at epoch {epoch}: {n_nonfinite:,} of "
                                f"the first {_seen:,} batches had a non-finite loss, gradient or "
                                f"grad-norm ({100 * n_nonfinite / _seen:.2f}%, limit "
                                f"{100 * _maxf:.2f}%).\n"
                                f"[nan-guard] TWO DIFFERENT CAUSES look like this — do not assume:\n"
                                f"  (a) LR FRAGILITY (weights finite, isolated batches overflow). "
                                f"Signature: a few % of batches, the rest train normally, and the "
                                f"periodic weight check has NOT fired. Fix: lower decoder_lr "
                                f"(F155 — 1e-2 gave 2.6-6.3%, 5e-3 gave 0%), or raise "
                                f"--nan_guard_max_frac if the loss of steps is acceptable.\n"
                                f"  (b) CORRUPT INPUT (zero-library or unreadable cells). "
                                f"Signature: 'ZERO-LIBRARY control rows' > 0 in the dump above, or "
                                f"a very high rate from batch 1. Fix: check the input files "
                                f"(scripts/repair_cd4_indptr.py covers the CD4 int32-indptr case)."
                            )
                        continue
                else:
                    grad_norm = None
                self.optimizer.step()

            # Invalidate GNN cache after weight update
            fwd_model.invalidate_gnn_cache()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_mag_loss += float(mag_loss)
            n_batches += 1
            if _bench and n_batches == max(1, _bench // 2):
                _t_half, _n_half = time.time(), n_samples + bs
            if _bench and n_batches >= _bench:
                _dt = time.time() - _t_bench
                _dt2 = time.time() - _t_half
                _n2 = (n_samples + bs) - _n_half
                print(f"[bench] {n_batches} batches, {n_samples + bs:,} samples in {_dt:.1f}s -> "
                      f"{(n_samples + bs) / max(_dt, 1e-9):,.0f} samples/s overall; "
                      f"second half {_n2:,} samples in {_dt2:.1f}s -> {_n2 / max(_dt2, 1e-9):,.0f} samples/s (steady state) | "
                      f"RSS main {rss_gb():.1f} GB | PSS main+workers {rss_gb(True):.1f} GB (true footprint)", flush=True)
                break
            # WEIGHT CHECK, PERIODIC (F156). Once per epoch was not enough, and the gap cost a
            # wrong diagnosis: an intermittent all-NaN *batch* (weights fine, next batch fine) and
            # genuinely NaN *weights* (unrecoverable, every batch dead) look identical in a single
            # guard dump. F155 was written asserting the second when the truth was the first.
            # This makes the distinction mechanical instead of inferential, and it is cheap —
            # one finiteness reduction over the parameters every `_wchk` batches.
            _wchk = int(getattr(self.config, "weight_check_every", 500))
            if _wchk > 0 and n_batches % _wchk == 0:
                _dead = [n for n, p_ in list(self.model.named_parameters()) + list(self.model.named_buffers())
                         if p_ is not None and not torch.isfinite(p_).all()]
                if _dead:
                    raise RuntimeError(
                        f"[nan-guard] ABORTING at epoch {epoch} batch {n_batches}: "
                        f"{len(_dead)} PARAMETER tensors are non-finite (first: {_dead[:3]}). "
                        f"The MODEL is destroyed — this is not a per-batch data issue and no "
                        f"amount of batch-dropping recovers it. Lower the learning rate "
                        f"(decoder_lr is the usual culprit; see F155) rather than raising "
                        f"--nan_guard_max_frac."
                    )
            n_samples += bs
            if grad_norm is not None:
                # Only FINITE norms enter the mean/max; non-finite ones are counted in
                # n_bad_gradnorm instead. My first attempt used `not (g <= max)` and was
                # NaN-LOSING: once max was NaN, the next finite g satisfied it and
                # OVERWROTE the NaN, hiding the very event it was meant to surface.
                if math.isfinite(grad_norm):
                    epoch_grad_norm += grad_norm
                    n_grad_norm += 1
                    if grad_norm > grad_norm_max:
                        grad_norm_max = grad_norm
                else:
                    n_bad_gradnorm += 1

            if _ps:
                with torch.no_grad():
                    src_of = getattr(self, "_cell_line_source", {})
                    srcs = np.array([src_of.get(c, "other") for c in batch.cell_types])
                    gm = gene_mask.bool() if gene_mask is not None else None
                    ae = (prediction - target).abs()
                    # under stream_v2 the control is a 768-d embedding: no raw-control delta
                    dm = ((prediction - ctrl).abs() if ctrl.shape[1] == prediction.shape[1]
                          else torch.full_like(ae, float("nan")))
                    if gm is not None:
                        n_on = gm.sum(1).clamp(min=1)
                        per_err = (ae * gm).sum(1) / n_on
                        per_dm = (dm * gm).sum(1) / n_on
                    else:
                        per_err, per_dm = ae.mean(1), dm.mean(1)
                    pg = prediction.grad
                    for s in np.unique(srcs):
                        m = torch.from_numpy(srcs == s).to(prediction.device)
                        d = self._per_source.setdefault(
                            s, {"n": 0, "err": 0.0, "dmag": 0.0, "gnorm": 0.0})
                        k = int(m.sum())
                        d["n"] += k
                        d["err"] += float(per_err[m].sum())
                        d["dmag"] += float(per_dm[m].sum())
                        if pg is not None:
                            d["gnorm"] += float(pg[m].norm())

            # v2: Accumulate diagnostic metrics
            epoch_recon_loss += expr_loss.item()
            if isinstance(latent_loss, torch.Tensor) and latent_loss.requires_grad:
                epoch_latent_loss += latent_loss.item()
            elif isinstance(latent_loss, torch.Tensor):
                epoch_latent_loss += latent_loss.item()
            if isinstance(delta_reg, torch.Tensor) and delta_reg.requires_grad:
                epoch_delta_reg += delta_reg.item()
            elif isinstance(delta_reg, torch.Tensor):
                epoch_delta_reg += delta_reg.item()
            if self.gpl_fn is not None:
                epoch_gpl_loss += gpl_loss.item()
            if self.config.ranking_loss_weight > 0:
                epoch_ranking_loss += ranking_loss.item()
            if self.config.hurdle_loss_weight > 0:
                epoch_hurdle_loss += hurdle_loss.item()
            for k, v in latent_metrics.items():
                epoch_latent_metrics[k] = epoch_latent_metrics.get(k, 0.0) + v

            # v2: Gate diagnostics for gated MLP
            if (
                self.config.interaction_type == "gated_mlp"
                and hasattr(fwd_model.interaction, "_last_gate")
                and fwd_model.interaction._last_gate is not None
            ):
                gate = fwd_model.interaction._last_gate
                epoch_latent_metrics["gate_activation_mean"] = (
                    epoch_latent_metrics.get("gate_activation_mean", 0.0)
                    + gate.mean().item()
                )
                epoch_latent_metrics["gate_active_dims"] = (
                    epoch_latent_metrics.get("gate_active_dims", 0.0)
                    + (gate > 0.1).float().sum(dim=-1).mean().item()
                )

            # Update progress bar
            avg_loss = epoch_loss / n_batches
            pbar.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{avg_loss:.4f}")

        pbar.close()
        self.scheduler.step()

        # ⚠️ This used to be `... if epoch_grad_norm > 0 else None`, which made the field
        # DISAPPEAR from the log whenever it was 0 — and `grad_norm` is the single field that
        # exposed F150 (`loss=0.1821 | grad_norm=inf`). Never let telemetry vanish silently.
        avg_grad_norm = (epoch_grad_norm / n_grad_norm) if n_grad_norm > 0 else None
        result = {
            "loss": epoch_loss / max(n_batches, 1),
            "lr": self.scheduler.get_last_lr()[0],
            "n_batches": n_batches,
            "n_samples": n_samples,
            "grad_norm": avg_grad_norm,
            "grad_norm_max": (grad_norm_max if n_grad_norm > 0 else None),
            "n_grad_norm_batches": n_grad_norm,
            "n_bad_gradnorm": n_bad_gradnorm,
            "mag_loss": (epoch_mag_loss / max(n_batches, 1)),
        }

        # v2: Add latent metrics to result
        if n_batches > 0:
            result["recon_loss"] = epoch_recon_loss / n_batches
            if self.latent_loss_type != "none":
                result["latent_loss"] = epoch_latent_loss / n_batches
                recon_avg = epoch_recon_loss / n_batches
                result["latent_recon_ratio"] = (
                    (epoch_latent_loss / n_batches) / max(recon_avg, 1e-8)
                )
            if self.delta_reg_type != "none":
                result["delta_reg"] = epoch_delta_reg / n_batches
            if self.gpl_fn is not None:
                result["gpl_loss"] = epoch_gpl_loss / n_batches
            if self.config.ranking_loss_weight > 0:
                result["ranking_loss"] = epoch_ranking_loss / n_batches
            if self.config.hurdle_loss_weight > 0:
                result["hurdle_loss"] = epoch_hurdle_loss / n_batches
                if epoch_hurdle_n > 0:
                    result["hurdle_pi_mean"] = epoch_hurdle_pi_sum / epoch_hurdle_n
                    fwd = self.model.module if hasattr(self.model, "module") else self.model
                    if hasattr(fwd.decoder, "log_shape_param"):
                        k_vals = (nn.functional.softplus(fwd.decoder.log_shape_param) + 0.05).clamp(max=100.0)
                        result["hurdle_k_min"] = k_vals.min().item()
                        result["hurdle_k_median"] = k_vals.median().item()
                        result["hurdle_k_max"] = k_vals.max().item()
            for k, v in epoch_latent_metrics.items():
                result[k] = v / n_batches

        # ---- NaN guard verdict for the epoch (F147) ----
        # Report even when zero, so "0 dropped" is on the record rather than inferred from
        # silence — the whole failure mode this guards against was invisible by default.
        result["n_nonfinite_batches"] = n_nonfinite
        if n_nonfinite:
            seen = n_batches + n_nonfinite
            frac = n_nonfinite / max(seen, 1)
            result["nonfinite_frac"] = frac
            max_frac = float(getattr(self.config, "nan_guard_max_frac", 0.01))
            msg = (f"[nan-guard] epoch {epoch}: dropped {n_nonfinite:,}/{seen:,} batches "
                   f"({100 * frac:.3f}%, {n_nonfinite_samples:,} samples) for non-finite loss")
            if frac > max_frac:
                raise RuntimeError(
                    f"{msg}\n"
                    f"[nan-guard] ABORTING: that exceeds nan_guard_max_frac={max_frac:.3%}. "
                    f"This is a corpus problem, not noise — training on the remainder would "
                    f"silently drop data. Check the input files for zero-library or corrupt "
                    f"cells (see scripts/repair_cd4_indptr.py for the CD4 int32-overflow case), "
                    f"then re-run. Raise --nan_guard_max_frac only if you have established the "
                    f"dropped cells are genuinely negligible."
                )
            if self.rank == 0:
                print(msg + " — under the abort threshold, continuing.", flush=True)

        # Weight-level backstop: the guard above prevents the known route to NaN weights,
        # but anything that slipped through (a NaN already inside the optimizer state, a
        # non-finite scheduler value) still ends the run here rather than 12 epochs later.
        if n_batches > 0:
            bad = [n for n, p in self.model.named_parameters()
                   if p is not None and not torch.isfinite(p).all()]
            if bad:
                raise RuntimeError(
                    f"[nan-guard] epoch {epoch}: {len(bad)} parameter tensors are non-finite "
                    f"(first: {bad[:3]}). The model is destroyed; every downstream checkpoint "
                    f"and score would be meaningless. Aborting instead of continuing."
                )

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _recon_elem(self, full_pred, full_target):
        """Per-element reconstruction loss, (B, G), no reduction — for the F174 reductions."""
        lt = self.config.loss_type
        if lt == "smooth_l1":
            return F.smooth_l1_loss(full_pred, full_target, beta=self.config.smooth_l1_beta,
                                    reduction="none")
        if lt == "mse":
            return F.mse_loss(full_pred, full_target, reduction="none")
        raise ValueError(f"_recon_elem does not support loss_type={lt!r}")

    def _recon_loss(self, full_pred, full_target, gmask):
        """Masked reconstruction loss. For per-gene losses (nb/zinb) the gene dim must stay
        intact (their log_theta/logit_pi are per-gene), so compute on the full (B,G) tensors
        and apply gmask as a multiplicative mask; for the others, keep the original 1D
        gene_mask flatten behaviour."""
        if self.config.loss_type in ("nb", "zinb"):
            per_elem = self.recon_loss_fn(full_pred, full_target)  # (B, G), reduction='none'
            gm = gmask.bool()
            # Mask with `where`, NOT with multiplication (F148). `per_elem * gm` evaluates
            # `inf * 0` for an absent gene whose loss overflowed, and that is NaN — which then
            # propagates to the batch loss, the gradient and every weight. `where` drops the
            # masked entries outright, so a non-finite value on a gene we are not even training
            # on can never contaminate the genes we are. The _EXPM1_MAX clamp above should stop
            # per_elem going infinite in the first place; this is the second line of defence,
            # and the two are deliberately independent.
            n_on = gm.sum().clamp(min=1)
            return torch.where(gm, per_elem, torch.zeros_like(per_elem)).sum() / n_on
        return self.recon_loss_fn(full_pred[gmask], full_target[gmask])

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        control_means: Dict[str, np.ndarray],
        split_name: str = "val",
        return_predictions: bool = False,
    ):
        """Evaluate on a split, computing loss + pearson_delta + gene-level metrics."""
        model = self.model
        model.eval()

        # Reset the eval control draw so the metric is a function of the WEIGHTS ONLY.
        # `CellTypeControlMatcher.sample_controls` draws from a shared, mutable generator, so
        # without this each val cell gets a different control every epoch and the selector
        # moves for reasons that have nothing to do with the model.
        _ds = getattr(eval_loader, "dataset", None)
        _cm = getattr(_ds, "control_matcher", None)
        if _cm is not None and hasattr(_cm, "rng"):
            _cm.rng = np.random.default_rng(int(getattr(self.config, "seed", 42)))
        fwd_model = model.module if hasattr(model, "module") else model

        all_preds = []
        self._eval_nonfinite = 0
        self._eval_nonfinite_rows = 0
        all_truths = []
        all_pert_names = []
        all_cell_types = []
        all_gene_masks = []
        all_z_hat = []
        all_z_target = []
        all_z_ctrl = []

        # Compute val/test loss
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        n_loss_samples = 0
        all_ctrls = []

        for batch in eval_loader:
            if isinstance(batch, PerturbationBatch):
                ctrl = batch.control_expression.to(self.device, non_blocking=True)
                target = batch.perturbed_expression.to(self.device, non_blocking=True)
                pert_idx = batch.pert_indices
                gene_mask = batch.gene_masks.to(self.device, non_blocking=True)
            elif isinstance(batch, SparseBatch):
                ctrl, target, pert_idx, gene_mask = self._materialise_sparse(batch)
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            with torch.amp.autocast("cuda", dtype=getattr(self, "_amp_dtype", torch.float16), enabled=(self.device == "cuda")):
                model_output = fwd_model(ctrl, pert_idx)
                prediction = model_output["prediction"]
                # Count non-finite eval predictions so `[val] loss=nan` EXPLAINS itself instead
                # of just printing `nan` (F158). Diagnostic only — deliberately NOT repaired: the
                # val metrics are a health signal and silently fixing them would hide exactly
                # what we want to see. The H1 prediction path DOES repair and report, because
                # there one bad cell destroys the whole panel score instead of one number.
                if not torch.isfinite(prediction).all():
                    _nfm = ~torch.isfinite(prediction)
                    self._eval_nonfinite = getattr(self, "_eval_nonfinite", 0) + int(_nfm.sum())
                    self._eval_nonfinite_rows = (getattr(self, "_eval_nonfinite_rows", 0)
                                                 + int(_nfm.any(dim=1).sum()))

                # GPL ranking loss (needs 2D structure, before flattening)
                if self.gpl_fn is not None:
                    gpl_loss_val = self.gpl_fn(prediction, target, mask=gene_mask)

                # Loss (same computation as training)
                masked_pred = prediction[gene_mask]
                masked_target = target[gene_mask]
                if self.config.loss_type == "poisson_smooth_l1":
                    recon_loss = (
                        self._poisson_weight * self._poisson_fn(masked_pred, masked_target)
                        + (1 - self._poisson_weight) * self._smooth_l1_fn(masked_pred, masked_target)
                    )
                else:
                    recon_loss = self._recon_loss(prediction, target, gene_mask)

                if self.recon_loss_scale != 1.0:
                    recon_loss = recon_loss / self.recon_loss_scale

                if self.cosine_loss_weight > 0:
                    gene_mask_f = gene_mask.float()
                    cos_loss = 1 - F.cosine_similarity(
                        prediction * gene_mask_f, target * gene_mask_f, dim=-1
                    ).mean()
                    loss = (1 - self.cosine_loss_weight) * recon_loss + self.cosine_loss_weight * cos_loss
                else:
                    loss = recon_loss

                if self.gpl_fn is not None:
                    loss = loss + self.gpl_weight * gpl_loss_val

                # Gene-level metrics on masked genes
                mse = (masked_pred - masked_target).pow(2).mean()
                mae = (masked_pred - masked_target).abs().mean()

            bs = ctrl.shape[0]
            total_loss += loss.item() * bs
            total_mse += mse.item() * bs
            total_mae += mae.item() * bs
            n_loss_samples += bs

            all_preds.append(prediction.cpu().numpy())
            all_truths.append(target.cpu().numpy())
            # per-cell paired control: the reference for the single-cell delta metrics.
            # Only meaningful in gene space, which is why eval datasets are NOT run through
            # precompute_paired_control_embeddings (train only).
            if ctrl.shape[1] == target.shape[1]:
                all_ctrls.append(ctrl.cpu().numpy())
            all_pert_names.extend(batch.pert_names)
            all_cell_types.extend(batch.cell_types)
            all_gene_masks.append(batch.gene_masks.numpy())

            if self.latent_loss_type != "none" and "z_hat" in model_output:
                z_target = fwd_model.encoder(target).detach()
                all_z_hat.append(model_output["z_hat"].detach().cpu().numpy())
                all_z_target.append(z_target.cpu().numpy())
                all_z_ctrl.append(model_output["z_ctrl"].detach().cpu().numpy())

        predictions = np.concatenate(all_preds)
        ground_truth = np.concatenate(all_truths)
        pert_names = np.array(all_pert_names)
        cell_types = np.array(all_cell_types)

        # Build per-cell-type gene masks (each dataset has different genes)
        gene_masks_np = np.concatenate(all_gene_masks)
        gene_masks_per_ct = {}
        for ct in np.unique(cell_types):
            ct_idx = np.where(cell_types == ct)[0][0]
            gene_masks_per_ct[ct] = gene_masks_np[ct_idx].astype(bool)

        # Compute perturbation-level metrics (pearson_delta)
        # Single-cell metrics: each cell scored against ITS OWN paired control, no cell
        # averaging anywhere. Falls back to the pseudobulk path only if the eval dataset
        # carried embeddings instead of gene-space controls.
        controls_np = np.concatenate(all_ctrls) if all_ctrls else None
        if controls_np is None:
            print("  WARNING: no gene-space controls in eval batches — "
                  "falling back to PSEUDOBULK metrics (cells will be averaged)")
        aggregate, per_pert = compute_perturbation_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            pert_names=pert_names,
            cell_types=cell_types,
            control_means=control_means,
            gene_masks_per_ct=gene_masks_per_ct,
            controls=controls_np,
        )

        # Add loss and per-sample gene-level metrics (matches training loss computation)
        aggregate["loss"] = total_loss / max(n_loss_samples, 1)
        aggregate["gene_mse"] = total_mse / max(n_loss_samples, 1)
        aggregate["gene_mae"] = total_mae / max(n_loss_samples, 1)

        # Latent-space Pearson delta (per-pert, per-ct)
        if all_z_hat:
            z_hat_all = np.concatenate(all_z_hat)
            z_target_all = np.concatenate(all_z_target)
            z_ctrl_all = np.concatenate(all_z_ctrl)

            latent_pds = []
            unique_perts_lat = np.unique(pert_names)
            unique_cts_lat = np.unique(cell_types)
            for pert in unique_perts_lat:
                if pert == "non-targeting":
                    continue
                for ct in unique_cts_lat:
                    mask = (pert_names == pert) & (cell_types == ct)
                    if mask.sum() == 0:
                        continue
                    dz_pred = (z_hat_all[mask] - z_ctrl_all[mask]).mean(axis=0)
                    dz_true = (z_target_all[mask] - z_ctrl_all[mask]).mean(axis=0)
                    latent_pds.append(_safe_pearsonr(dz_pred, dz_true))

            if latent_pds:
                aggregate["latent_pearson_delta_mean"] = float(np.mean(latent_pds))
                aggregate["latent_pearson_delta_median"] = float(np.median(latent_pds))

        if getattr(self, "_eval_nonfinite", 0):
            print(f"  [{split_name}] ⚠️ NON-FINITE PREDICTIONS during eval: "
                  f"{self._eval_nonfinite:,} values over {self._eval_nonfinite_rows:,} cells "
                  f"— this is WHY the metrics below are nan. The model is unstable on some "
                  f"inputs even though training may look healthy (F158).", flush=True)
        print(
            f"  [{split_name}] loss={aggregate['loss']:.4f} | "
            f"gene_mse={aggregate['gene_mse']:.4f} | "
            f"gene_mae={aggregate['gene_mae']:.4f}\n"
            f"    pearson_delta: mean={aggregate['pearson_delta_mean']:.4f}, "
            f"median={aggregate['pearson_delta_median']:.4f} | "
            f"pearson: mean={aggregate['pearson_mean']:.4f}, "
            f"median={aggregate['pearson_median']:.4f}\n"
            f"    mse_delta: {aggregate['mse_delta_mean']:.4f} | "
            f"mae_delta: {aggregate['mae_delta_mean']:.4f} | "
            f"mse: {aggregate['mse_mean']:.4f} | mae: {aggregate['mae_mean']:.4f} | "
            f"({aggregate['n_pert_celltype_pairs']} pairs)\n"
            f"    overlap@20: mean={aggregate.get('overlap_20_mean', 0):.4f} | "
            f"overlap@50: mean={aggregate.get('overlap_50_mean', 0):.4f}"
        )
        if "latent_pearson_delta_mean" in aggregate:
            print(
                f"    latent_pearson_delta: mean={aggregate['latent_pearson_delta_mean']:.4f}, "
                f"median={aggregate.get('latent_pearson_delta_median', 0):.4f}"
            )

        # Per cell type breakdown
        unique_cts = np.unique(cell_types)
        ct_means = []
        per_ct_metrics = {}
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
            per_ct_metrics[ct] = ct_agg
            ct_means.append(ct_agg["pearson_delta_mean"])
            print(
                f"    {ct}: pearson_delta_mean="
                f"{ct_agg['pearson_delta_mean']:.4f} "
                f"({ct_agg['n_pert_celltype_pairs']} pairs)"
            )

        if len(ct_means) > 1:
            ct_arr = np.array(ct_means)
            aggregate["pearson_delta_arithmetic_mean_across_ct"] = float(np.mean(ct_arr))
            ct_clipped = np.clip(ct_arr, 1e-10, None)
            aggregate["pearson_delta_geometric_mean_across_ct"] = float(
                np.exp(np.mean(np.log(ct_clipped)))
            )
            print(
                f"    Avg across cell types: "
                f"arithmetic={aggregate['pearson_delta_arithmetic_mean_across_ct']:.4f}  "
                f"geometric={aggregate['pearson_delta_geometric_mean_across_ct']:.4f}"
            )

        aggregate["per_cell_type"] = per_ct_metrics
        if return_predictions:
            return aggregate, predictions, pert_names
        return aggregate

    @staticmethod
    def _log_eval_metrics(wandb_run, split_name: str, metrics: Dict, epoch: int):
        """Log evaluation metrics to wandb, including per-cell-type breakdown.

        Uses a single wandb.log call to avoid step conflicts.
        """
        per_ct = metrics.pop("per_cell_type", {})
        log_dict = {"epoch": epoch}
        # Overall metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log_dict[f"{split_name}/{k}"] = v
        # Per cell type
        for ct_name, ct_metrics in per_ct.items():
            for k, v in ct_metrics.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{split_name}/{ct_name}/{k}"] = v
        wandb_run.log(log_dict)
        # Restore for caller
        metrics["per_cell_type"] = per_ct

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _weights_from_checkpoint(self, ckpt_path: str, why: str = ""):
        """Temporarily swap the live model's weights for those in `ckpt_path`.

        The end-of-training H1 evaluation used to run on whatever weights the loop
        happened to finish on — the LAST epoch — even when `--h1_select_on_celleval`
        had already selected a much earlier checkpoint as best. Measured across the
        30-config LR sweep (`17666463`/`17666464`): the selected checkpoint beat the
        final one in **29 of 30** runs, mean vcc2026 `overall` 0.1389 vs 0.1244, worst
        case 0.1568 (epoch 20) vs 0.1146 (epoch 100). Every seen/unseen and cell-eval
        v1 number in that sweep therefore describes a model the run had already
        rejected, at ~2 h of eval per run.

        Only the modules that `save_checkpoint` actually persists AND that live on the
        model are swapped: gnn, decoder, interaction, latent_proj_head. The encoder is
        deliberately NOT swapped — `self._encoder_state_dict` is a snapshot taken at
        init, so it is byte-identical in every checkpoint a run writes. That holds only
        while the encoder is frozen (the default); if it is trainable the snapshot is
        stale in the checkpoint too, which is a separate pre-existing bug, so this
        warns rather than silently half-restoring.

        Restoring on the way out is load-bearing, not tidiness: `final_model.pth` is
        written AFTER this evaluation, so leaking the best weights would make the
        "final" checkpoint a mislabelled copy of the best one.
        """
        fwd_model = self.model.module if hasattr(self.model, "module") else self.model

        if not os.path.exists(ckpt_path):
            print(f"  [ckpt] {ckpt_path} not found — evaluating the final-epoch weights")
            yield False
            return

        if not self.config.freeze_encoder:
            print("  [ckpt] WARNING: encoder is trainable, but checkpoints store an "
                  "init-time encoder snapshot. Swapping gnn/decoder/interaction only; "
                  "the encoder stays at its final-epoch weights.")

        modules = {
            "gnn_state_dict": getattr(fwd_model, "gnn", None),
            "decoder_state_dict": getattr(fwd_model, "decoder", None),
            "interaction_state_dict": getattr(fwd_model, "interaction", None),
            "latent_proj_head_state_dict": self.latent_proj_head,
        }
        ckpt = torch.load(ckpt_path, map_location=self.device)
        saved: Dict[str, Dict] = {}
        try:
            for key, mod in modules.items():
                if mod is None or key not in ckpt:
                    continue
                saved[key] = copy.deepcopy(mod.state_dict())
                mod.load_state_dict(ckpt[key])
            # MANDATORY after swapping gnn weights. `_gnn_cache` memoises the GNN's
            # perturbation embeddings `z_graph` and forward() returns it verbatim when
            # populated. Training invalidates it after every optimizer step, but by the
            # time the end-of-training evaluation runs it holds embeddings computed from
            # the FINAL weights. Loading a state_dict does not touch it, so without this
            # the model becomes a chimera — best-epoch decoder/interaction driven by
            # final-epoch perturbation embeddings.
            #
            # Measured, before this line existed (round-2 LR sweep, 32/32 runs): the
            # final `all` score undershot the selection score in 32 of 32 runs, by a
            # margin that shrank monotonically with the selected epoch — 0.032 at epoch
            # 10, 0.026 at 20, and exactly 0.0000 at epoch 100, where the swap is a no-op
            # and no chimera can form. That gradient is the fingerprint of this cache.
            fwd_model.invalidate_gnn_cache()
            print(f"  [ckpt] evaluating {os.path.basename(ckpt_path)} "
                  f"(epoch {ckpt.get('epoch', '?')}) {why} — swapped "
                  f"{', '.join(k.replace('_state_dict', '') for k in saved) or 'nothing'}"
                  f"; gnn cache invalidated")
            del ckpt
            yield True
        finally:
            for key, sd in saved.items():
                modules[key].load_state_dict(sd)
            if saved:
                fwd_model.invalidate_gnn_cache()
                print("  [ckpt] restored final-epoch weights; gnn cache invalidated")

    def save_checkpoint(self, path: str, split_info: Dict = None):
        """Save unified checkpoint."""
        fwd_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        bench = getattr(self, "benchmark", None)
        checkpoint = {
            # The benchmark this model was trained under. The scoring side must refuse a
            # checkpoint whose gene_axis_sha256 disagrees with the axis it is scored on: a
            # model trained on 6,983 input columns and scored on 9,624 is silent and severe.
            "benchmark": ({
                "id": bench.benchmark_id,
                "benchmark_sha256": bench.sha256,
                "gene_axis_sha256": bench.axis_sha256,
                "n_axis_genes": len(bench.axis_ensembl),
                "axis_scope": str(getattr(self.config, "bench_axis_scope", "")),
                "eval_cell_lines": list(bench.eval_cell_lines),
                "excluded_perts": {k: sorted(v)
                                   for k, v in bench.exclude_file_perts.items()},
            } if bench is not None else None),
            "val_cell_lines": list(getattr(self, "_val_cell_lines", []) or []),
            "encoder_state_dict": self._encoder_state_dict,
            "gnn_state_dict": fwd_model.gnn.state_dict(),
            "decoder_state_dict": fwd_model.decoder.state_dict(),
            "interaction_state_dict": fwd_model.interaction.state_dict(),
            "interaction_type": fwd_model.interaction_type,
            # v1 backward compat: also save as pert_projection_state_dict for additive
            **({"pert_projection_state_dict": fwd_model.interaction.state_dict()}
               if fwd_model.interaction_type == "additive" else {}),
            # v2: projection head (if used)
            **({"latent_proj_head_state_dict": self.latent_proj_head.state_dict()}
               if self.latent_proj_head is not None else {}),
            "encoder_config": {
                "model_size": self.model_size,
                "hidden_size": self.hidden_size,
                "num_genes": self.num_genes,
            },
            "gnn_config": {
                "pert_model_type": self.config.pert_model_type,
                "num_genes": fwd_model.gnn.num_genes,
                "hidden_dim": self.config.gnn_hidden_dim,
                "out_dim": self.config.pert_latent_dim,
                "layer_type": self.config.gnn_layer_type,
                "num_layers": self.config.gnn_num_layers,
                "skip": self.config.gnn_skip,
                "num_heads": self.config.gnn_num_heads,
                "use_edge_weight": self.config.use_edge_weight,
                "no_gnn": self.config.no_gnn,
                **({"adj_hidden_dim": self.config.adj_hidden_dim,
                    "adj_dropout": self.config.adj_dropout,
                    "adj_deg": self.config.adj_deg,
                    "adj_normalize": self.config.adj_normalize,
                    "adj_only_reverse": self.config.adj_only_reverse,
                    "adj_make_undirected": self.config.adj_make_undirected,
                    "adj_activation": self.config.adj_activation,
                   } if self.config.pert_model_type == "adjacency_map" else {}),
            },
            "interaction_config": {
                "interaction_type": self.config.interaction_type,
                "pert_latent_dim": self.config.pert_latent_dim,
                "hidden_size": self.hidden_size,
                "film_scale": self.config.film_scale,
                "film_hidden_dim": self.config.film_hidden_dim,
                "film_activation": self.config.film_activation,
                "film_mode": self.config.film_mode,
                "film_mean_free": bool(getattr(self.config, "film_mean_free", False)),
                "gated_mlp_hidden_dim": self.config.gated_mlp_hidden_dim,
                "gated_mlp_num_layers": self.config.gated_mlp_num_layers,
            },
            "decoder_config": {
                # MUST be saved. Activation modules hold no parameters, so a softplus
                # decoder reloads strict=True under the rectified_sigmoid default and
                # silently runs the wrong nonlinearity (measured: 99.0% exact zeros).
                "output_activation": getattr(
                    self.decoder, "output_activation", "rectified_sigmoid"),
                "skip_normalize": getattr(self, "skip_normalize", False),
                "skip_log1p": getattr(self, "skip_log1p", False),
                "input_dim": self.hidden_size,
                "output_dim": self.num_genes,
                "hidden_dim": fwd_model.decoder.hidden_dim,
                "num_blocks": len(fwd_model.decoder.res_blocks),
                "dropout_rate": self.config.decoder_dropout,
                "se_reduction": self.config.decoder_se_reduction,
                "hurdle_mode": getattr(fwd_model.decoder, "hurdle_mode", False),
            },
            "gene_names": list(self.encoder_gene_names),
            "gene_to_index": dict(self.gene_to_index),
            "pert2id": dict(self.pert2id),
            "target_sum": self.target_sum,
            "split_info": split_info,
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "training_config": {
                "pert_model_type": self.config.pert_model_type,
                "graph_type": self.config.graph_type,
                "graph_types": self.config.graph_types,
                "data_perts_only": self.config.data_perts_only,
                "loss_type": self.config.loss_type,
                "cosine_loss_weight": self.config.cosine_loss_weight,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "freeze_encoder": self.config.freeze_encoder,
                "freeze_decoder": self.config.freeze_decoder,
                "encoder_checkpoint": self.config.encoder_checkpoint,
                "pretrained_encoder": self.config.pretrained_encoder,
                # v2
                "interaction_type": self.config.interaction_type,
                "film_mode": self.config.film_mode,
                "film_scale": self.config.film_scale,
                "film_hidden_dim": self.config.film_hidden_dim,
                "film_mean_free": bool(getattr(self.config, "film_mean_free", False)),
                "latent_loss_type": self.config.latent_loss_type,
                "latent_loss_weight": self.config.latent_loss_weight,
                "delta_reg_type": self.config.delta_reg_type,
                "expression_loss_mode": self.config.expression_loss_mode,
                "decoder_lr_warmup_epochs": self.config.decoder_lr_warmup_epochs,
                "hurdle_loss_weight": self.config.hurdle_loss_weight,
                "hurdle_warmup_epochs": self.config.hurdle_warmup_epochs,
                "hurdle_q_min": self.config.hurdle_q_min,
            },
            "is_virtual_cell_checkpoint": True,
        }
        # Stage C: persist the (possibly finetuned) nb/zinb dispersion so this VC ckpt can chain.
        lf = getattr(self, "recon_loss_fn", None)
        if lf is not None and hasattr(lf, "parameters") and any(True for _ in lf.parameters()):
            checkpoint["loss_state_dict"] = lf.state_dict()
            disp = {"loss_type": self.config.loss_type}
            if hasattr(lf, "log_theta"):
                disp["log_theta"] = lf.log_theta.detach().cpu()
            if hasattr(lf, "logit_pi"):
                disp["logit_pi"] = lf.logit_pi.detach().cpu()
            checkpoint["dispersion"] = disp
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    # ------------------------------------------------------------------
    # Full training
    # ------------------------------------------------------------------

    def train(self, output_dir: str = None, run_dir: str = None):
        """Full training pipeline.

        `run_dir`, when given, is used VERBATIM. The orchestrator always passes it, because
        the timestamp+urandom fallback below makes a 24-arm array job unattributable -- the
        exact failure that left the research tree with checkpoints nobody could match to a
        recipe. The fallback is kept only for library callers with no run directory of their
        own.
        """
        config = self.config
        output_dir = output_dir or config.output_dir
        if run_dir:
            run_dir = Path(run_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_suffix = os.urandom(3).hex()
            run_dir = Path(output_dir) / f"occam_vcell_{timestamp}_{unique_suffix}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        wandb_run = None
        if config.use_wandb and self.rank == 0:
            try:
                import wandb

                # Build config dict with names aligned to pretraining where shared
                wandb_config = {
                    # --- Shared with pretraining (same key names) ---
                    "model_size": self.model_size,
                    "hidden_size": self.hidden_size,
                    "num_genes": self.num_genes,
                    "target_sum": self.target_sum,
                    "learning_rate": config.lr,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "weight_decay": config.weight_decay,
                    "dropout": config.decoder_dropout,
                    "device": str(self.device),
                    # --- Virtual cell specific ---
                    "phase": "virtual_cell",
                    "encoder_checkpoint": config.encoder_checkpoint,
                    "pretrained_encoder": config.pretrained_encoder,
                    "freeze_encoder": config.freeze_encoder,
                    "freeze_decoder": config.freeze_decoder,
                    "encoder_lr": config.encoder_lr,
                    "decoder_lr": self.decoder_lr,
                    "graph_type": config.graph_type,
                    "gnn_hidden_dim": config.gnn_hidden_dim,
                    "gnn_num_layers": config.gnn_num_layers,
                    "gnn_num_heads": config.gnn_num_heads,
                    "gnn_layer_type": config.gnn_layer_type,
                    "gnn_skip": config.gnn_skip,
                    "gnn_dropout": config.gnn_dropout,
                    "gnn_activation": config.gnn_activation,
                    "pert_latent_dim": config.pert_latent_dim,
                    "use_edge_weight": config.use_edge_weight,
                    "no_gnn": config.no_gnn,
                    "graph_reduce2perts": config.graph_reduce2perts,
                    "graph_edge_filter_mode": config.graph_edge_filter_mode,
                    "pert_model_type": config.pert_model_type,
                    "graph_types": config.graph_types,
                    "data_perts_only": config.data_perts_only,
                    "kg_norm_weights": config.kg_norm_weights,
                    **({"adj_hidden_dim": config.adj_hidden_dim,
                        "adj_dropout": config.adj_dropout,
                        "adj_deg": config.adj_deg,
                        "adj_normalize": config.adj_normalize,
                        "adj_only_reverse": config.adj_only_reverse,
                        "adj_make_undirected": config.adj_make_undirected,
                        "adj_activation": config.adj_activation,
                       } if config.pert_model_type == "adjacency_map" else {}),
                    # v2
                    "interaction_type": config.interaction_type,
                    "film_scale": config.film_scale,
                    "film_hidden_dim": config.film_hidden_dim,
                    "film_activation": config.film_activation,
                    "film_mode": config.film_mode,
                    "gated_mlp_hidden_dim": config.gated_mlp_hidden_dim,
                    "gated_mlp_num_layers": config.gated_mlp_num_layers,
                    "latent_loss_type": config.latent_loss_type,
                    "latent_loss_weight": config.latent_loss_weight,
                    "latent_loss_alpha": config.latent_loss_alpha,
                    "latent_proj_dim": config.latent_proj_dim,
                    "delta_reg_type": config.delta_reg_type,
                    "delta_reg_lambda": config.delta_reg_lambda,
                    "delta_reg_tau": config.delta_reg_tau,
                    "expression_loss_mode": config.expression_loss_mode,
                    "decoder_lr_warmup_epochs": config.decoder_lr_warmup_epochs,
                    "decoder_hidden_dim": config.decoder_hidden_dim,
                    "decoder_num_blocks": config.decoder_num_blocks,
                    "decoder_dropout": config.decoder_dropout,
                    "decoder_se_reduction": config.decoder_se_reduction,
                    "decoder_input_dim": self.hidden_size,
                    "loss_type": config.loss_type,
                    "smooth_l1_beta": config.smooth_l1_beta,
                    "cosine_loss_weight": config.cosine_loss_weight,
                    "recon_loss_scale": config.recon_loss_scale,
                    "poisson_weight": config.poisson_weight,
                    "gpl_weight": config.gpl_weight,
                    "hurdle_loss_weight": config.hurdle_loss_weight,
                    "hurdle_warmup_epochs": config.hurdle_warmup_epochs,
                    "hurdle_q_min": config.hurdle_q_min,
                    "gradient_clip": config.gradient_clip,
                    "eta_min": config.eta_min,
                    "warmup_epochs": config.warmup_epochs,
                    "n_controls_per_sample": config.n_controls_per_sample,
                    "include_controls_as_training": config.include_controls_as_training,
                    "data_paths": config.data_paths,
                    "gene_name_col": config.gene_name_col,
                    "pert_gene_col": config.pert_gene_col,
                    "train_fraction": config.train_fraction,
                    "val_fraction": config.val_fraction,
                    "test_fraction": config.test_fraction,
                    "holdout_cell_line": config.holdout_cell_line,
                    "split_seed": config.split_seed,
                    "eval_freq": config.eval_freq,
                    "seed": config.seed,
                    "pseudobulk": config.pseudobulk,
                    "pseudobulk_by_cellline": config.pseudobulk_by_cellline,
                    "batch_col": config.batch_col if config.pseudobulk and not config.pseudobulk_by_cellline else None,
                    "run_dir": str(run_dir),
                }

                wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name or f"vcell_{timestamp}",
                    tags=config.wandb_tags or ["virtual_cell"],
                    config=wandb_config,
                )

                # All metrics use epoch as x-axis
                wandb.define_metric("epoch")
                wandb.define_metric("*", step_metric="epoch")

            except Exception as e:
                print(f"wandb init failed: {e}")

        # ---- Data preparation ----
        print("=" * 60)
        print("Preparing data...")
        # Batch labels are needed by pseudobulk aggregation AND by the batch-based
        # cell-line filters. Omitting them for the latter makes the filters unsatisfiable,
        # which `prepare_virtual_cell_data` raises on rather than silently passing every
        # line — but the fix is to load them, not to weaken the check.
        need_batch_col = (
            (config.pseudobulk and not config.pseudobulk_by_cellline)
            or getattr(config, "min_control_batches_per_cell_line", 0) > 0
            or getattr(config, "require_control_batch_overlap", False)
            # stream_batch_match needs a batch id on every row to build per-batch pools
            or getattr(config, "stream_batch_match", False)
        )
        # ---- Frozen benchmark definition -------------------------------------------
        # Loaded before any data is read, so a moved source file or an edited artifact stops
        # the run here rather than producing a plausible number three hours later.
        self.benchmark = None
        _restrict_ids = None
        _exclude_file_perts = None
        if getattr(config, "bench_dir", None):
            from occam.vcell.benchmark import load_benchmark

            self.benchmark = load_benchmark(
                config.bench_dir,
                expect_id=str(getattr(config, "bench_id", "") or ""),
                expect_sha256=str(getattr(config, "bench_sha256", "") or ""),
                data_paths=list(config.data_paths),
                encoder_gene_names=self.encoder_gene_names,
            )
            print(f"  [bench] {self.benchmark.summary()}")
            # Record the axis hash on the config so it reaches the control-embedding cache
            # key and the checkpoint without being threaded through six call sites.
            config.bench_axis_sha256 = self.benchmark.axis_sha256
            _exclude_file_perts = self.benchmark.exclude_file_perts
            scope = str(getattr(config, "bench_axis_scope", "input_and_loss"))
            if scope == "input_and_loss":
                _restrict_ids = self.benchmark.axis_set
            elif scope != "loss_only":
                raise ValueError(
                    f"bench_axis_scope must be 'input_and_loss' or 'loss_only', "
                    f"got {scope!r}")
            print(f"  [bench] axis scope {scope}: "
                  f"{'input and loss' if _restrict_ids else 'loss mask only'}")

        file_metas, control_matcher = prepare_virtual_cell_data(
            data_paths=config.data_paths,
            encoder_gene_names=self.encoder_gene_names,
            pert2id=self.pert2id,
            target_sum=self.target_sum,
            gene_name_col=config.gene_name_col,
            pert_gene_col=config.pert_gene_col,
            batch_col=config.batch_col if need_batch_col else None,
            seed=config.seed,
            min_controls_per_cell_line=config.min_controls_per_cell_line,
            cell_line_cols=getattr(config, "cell_line_cols", None),
            max_controls_per_cell_line=getattr(config, "max_controls_per_cell_line", 0),
            control_fallback_col=getattr(config, "control_fallback_col", None),
            control_only_paths=getattr(config, "control_only_data_paths", None),
            min_control_batches_per_cell_line=getattr(
                config, "min_control_batches_per_cell_line", 0),
            require_control_batch_overlap=getattr(
                config, "require_control_batch_overlap", False),
            control_force_group=getattr(config, "control_force_group", False),
            build_pools=not getattr(config, "stream_v2", False),
            restrict_gene_ids=_restrict_ids,
            exclude_file_perts=_exclude_file_perts,
        )

        if self.benchmark is not None:
            # loss_only: keep the full input scatter, narrow the loss mask instead.
            if str(getattr(config, "bench_axis_scope", "")) == "loss_only":
                axis_cols = torch.zeros(len(self.encoder_gene_names), dtype=torch.bool)
                idx = [self.gene_to_index[g] for g in self.benchmark.axis_ensembl
                       if g in self.gene_to_index]
                axis_cols[idx] = True
                for _fm in file_metas:
                    _fm.gene_mask = _fm.gene_mask & axis_cols
            # Every context must present the SAME columns; that is the whole point.
            _sums = {int(_fm.gene_mask.sum()) for _fm in file_metas}
            if len(_sums) != 1:
                raise ValueError(
                    f"gene masks differ across files ({sorted(_sums)}). The benchmark axis "
                    f"exists so that cell line is the only difference between contexts; "
                    f"unequal masks also change the pooled loss normalisation per context.")
            print(f"  [bench] all {len(file_metas)} contexts share a "
                  f"{_sums.pop():,}-gene mask")

        # cell_line -> source-file map, for the per-source diagnostics. Cell lines are
        # unique to a file in this corpus (essential lines, Feng donor ids, T-cell
        # donor x condition), so the mapping is well defined.
        self._cell_line_source = {}
        for _fm in file_metas:
            _src = Path(_fm.path).stem.replace(".assigned_guide", "")
            for _cl in set(_fm.obs_cell_lines):
                self._cell_line_source[_cl] = _src
        self._cell_line_source["H1"] = "h1_control_identity"

        if config.pseudobulk_by_cellline:
            for fm in file_metas:
                fm.batch_ids = fm.obs_cell_lines.copy()



        # ---- Splits ----
        print("=" * 60)
        print("Splitting perturbations...")
        if config.split_json and Path(config.split_json).exists():
            split_info = PerturbationSplitter.load(config.split_json)
        else:
            # Collect perturbation names per cell line (using per-row cell lines)
            # Only include cell lines that have a control pool
            # A cell line is usable if it has controls of its own OR can borrow its
            # group's (see CellTypeControlMatcher.has_pool / --control_fallback_col).
            valid_cls = {
                cl for fm in file_metas for cl in set(map(str, fm.obs_cell_lines))
                if control_matcher.has_pool(cl)
            }
            pert_names_per_cl: Dict[str, set] = {}
            for fm in file_metas:
                if getattr(fm, "controls_only", False):
                    continue
                _drop = getattr(fm, "excluded_pert_lines", None) or set()
                for i in range(fm.n_obs):
                    if fm.is_control[i]:
                        continue
                    if fm.obs_cell_lines[i] in _drop:
                        continue
                    pname = fm.pert_gene_names[i]
                    # resolve_pert_indices, not a bare pert2id lookup: with combo
                    # decomposition on, 'A+B' is resolvable via its members even
                    # though the joint label is not itself a graph node. A bare
                    # lookup here silently removed every double from the candidate
                    # pool BEFORE the splitter ever saw it.
                    if pname == "non-targeting" or _resolve_pert_indices(pname, self.pert2id) is None:
                        continue
                    cl = fm.obs_cell_lines[i]
                    if cl not in valid_cls:
                        continue
                    if cl not in pert_names_per_cl:
                        pert_names_per_cl[cl] = set()
                    pert_names_per_cl[cl].add(pname)

            # One splitter. The research tree had five, selected by mutually
            # exclusive dataset-mode flags; four of them were dataset-specific
            # and are dropped (see docs/reproducing_chain_a.md). `PerturbationSplitter`
            # is the general one, and the only one that supports the axes that
            # matter here: `holdout_cell_line` (which produces the unseen-context
            # test set) and `all_perts_train`.
            splitter = PerturbationSplitter(
                train_fraction=config.train_fraction,
                val_fraction=config.val_fraction,
                test_fraction=config.test_fraction,
                seed=config.split_seed,
                holdout_cell_line=config.holdout_cell_line,
                all_perts_train=getattr(config, "all_perts_train", False),
            )
            split_info = splitter.split(pert_names_per_cl)

            # Optional: pull every H1-panel perturbation into train, maximising the
            # seen-perturbation panel. NOT the default, because it also empties the
            # unseen-perturbation panel — with K562 whole-genome all 238 available H1
            # perturbations would become seen, leaving only the 17 that appear in no
            # training file, which is too few for a meaningful pds scale. The natural
            # split leaves both sub-panels populated, which is what the seen/unseen
            # decomposition needs.
            #
            # It is not leakage either way: H1 CELLS are never trained on, only the same
            # gene knocked down in other cell lines.
            if getattr(config, "panel_eval_enabled", False) and \
                    getattr(config, "panel_force_perts_in_train", False):
                split_info = _force_h1_perts_into_train(split_info, config)

            # Save splits
            split_path = run_dir / "splits.json"
            PerturbationSplitter.save(split_info, str(split_path))

        train_perts = set(split_info["train"])
        # Kept for the H1 evaluation's seen/unseen stratification: which H1 perturbations
        # the model actually saw (in OTHER cell lines) separates unseen-perturbation from
        # unseen-cell-line generalisation, which is otherwise conflated.
        self._train_pert_names = set(train_perts)
        val_perts = set(split_info["val"])
        test_unseen_pert = set(split_info["test_unseen_pert"])
        test_unseen_cell = set(split_info["test_unseen_cell"])
        train_cell_lines = split_info.get("cell_lines_train")
        test_cell_lines = split_info.get("cell_lines_test")

        # Eval cell lines follow the split. The research tree also allowed a
        # dataset-specific override that pinned val/test to one benchmark cell
        # line; with the general splitter that is expressed as
        # `holdout_cell_line` instead.

        # ---- Datasets ----
        print("=" * 60)
        print("Building datasets...")
        eval_batch_size = config.batch_size * 4  # no gradients → larger batches

        exclude_cl_perts = None

        if config.pseudobulk:
            if config.mixed_pb_sc:
                # ---- Mixed mode: pseudobulk + single-cell ----
                print("Mode: MIXED PSEUDOBULK + SINGLE-CELL")
                train_dataset = build_mixed_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=train_perts,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    cell_lines=train_cell_lines,
                    exclude_cl_perts=exclude_cl_perts,
                    seed=config.seed,
                    sc_only=config.sc_only,
                    max_sc_pairs_per_pert=getattr(config, "max_sc_pairs_per_pert", 0),
                    cache_dir=getattr(config, "sc_cache_dir", None),
                    include_pb=not config.sc_only,
                )
                # Control identity for the TRAINING cell lines. Appended after the cache so
                # the exact/paired/none ablation shares one perturbation-pair cache instead
                # of building a separate multi-hundred-GB copy per mode.
                train_dataset = append_control_identity(
                    train_dataset, file_metas, self.encoder_gene_names,
                    target_sum=self.target_sum, cell_lines=train_cell_lines,
                    ratio=getattr(config, "control_identity_ratio", 0.0),
                    mode=getattr(config, "control_identity_mode", "exact"),
                    seed=config.seed,
                )
                # H1 enters training HERE and nowhere else: its model-visible controls as
                # identity samples under the null token. No H1 perturbation is ever a
                # target. This mirrors the real task, where the unseen cell line's
                # non-targeting controls are exactly what you are given.
                if getattr(config, "panel_eval_enabled", False) and \
                        getattr(config, "panel_train_controls", False):
                    train_dataset = _append_h1_control_identity(
                        train_dataset, config, self.encoder_gene_names,
                        self.skip_normalize, self.skip_log1p)
                # Frozen encoder + fixed pairs => the control embedding is recomputed
                # identically every epoch. Cache it once. Exact, not an approximation:
                # the encoder is dropout=0 Linear/LayerNorm/Mish, hence deterministic.
                if config.freeze_encoder and \
                        getattr(config, "precompute_paired_embeddings", True):
                    train_dataset = precompute_paired_control_embeddings(
                        train_dataset, self.encoder, self.hidden_size, self.device)
            else:
                # ---- Pseudobulk mode: deterministic aggregated profiles ----
                print("Mode: PSEUDOBULK")
                print("Building pseudobulk training dataset...")
                train_dataset = build_pseudobulk_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=train_perts,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    cell_lines=train_cell_lines,
                    exclude_cl_perts=exclude_cl_perts,
                )
            loader_g = torch.Generator()
            loader_g.manual_seed(config.seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=PseudobulkDataset.collate_fn,
                pin_memory=True,
                drop_last=True,
                generator=loader_g,
                worker_init_fn=_seed_worker,
            )

            if val_perts:
                print("Building pseudobulk validation dataset...")
                val_dataset = build_pseudobulk_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=val_perts,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    cell_lines=train_cell_lines,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=PseudobulkDataset.collate_fn,
                )
            else:
                print("No validation perturbations — skipping val_loader")
                val_loader = None

            # Test: unseen perturbations
            print("Building pseudobulk test_unseen_pert dataset...")
            test_pert_dataset = None
            test_pert_loader = None
            # Off by default. build_eval_dataset loads EVERY cell of the split, dense, at
            # full encoder width -- gigabytes -- and nothing evaluates it during training.
            # It is also the test set, which should not be touched here at all.
            if test_unseen_pert and getattr(config, "bench_build_test_datasets", True):
                test_pert_dataset = build_pseudobulk_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=test_unseen_pert,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    cell_lines=train_cell_lines,
                )
                test_pert_loader = DataLoader(
                    test_pert_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=PseudobulkDataset.collate_fn,
                )

            # Test: unseen cell line
            print("Building pseudobulk test_unseen_cell dataset...")
            test_cell_dataset = None
            test_cell_loader = None
            if test_unseen_cell and test_cell_lines:
                test_cell_dataset = build_pseudobulk_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=test_unseen_cell,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    cell_lines=test_cell_lines,
                )
                test_cell_loader = DataLoader(
                    test_cell_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=PseudobulkDataset.collate_fn,
                )

        else:
            # ---- Default mode: streaming with random control matching ----
            # H1 controls are the one part of H1 that trains, and H1 is structurally
            # absent from data_paths, so they cannot be streamed — they must be injected.
            # The cached path does this via _append_h1_control_identity; without the
            # equivalent here, --stream_single_cell would silently train with NO H1
            # controls and quietly break the premise of the whole H1 setup.
            _extra = None
            if (getattr(config, "panel_eval_enabled", False)
                    and not getattr(config, "no_h1_train_controls", False)):
                from occam.vcell.panel import build_control_identity_samples
                _extra = build_control_identity_samples(
                    config, self.encoder_gene_names,
                    self.skip_normalize, self.skip_log1p,
                    mode=getattr(config, "panel_control_identity_mode", None)
                    or getattr(config, "control_identity_mode", "exact"),
                    n=int(getattr(config, "panel_train_control_n", 0)),
                    seed=config.seed,
                )
                print(f"  [panel] streaming mode: injecting {len(_extra[0])} H1 control "
                      f"identity samples spread across the epoch")

            if getattr(config, "stream_v2", False):
                train_dataset, train_loader = self._build_stream_v2(
                    config, file_metas, control_matcher, train_perts, train_cell_lines, _extra)
            else:
                # Training: optim streaming mode
                train_dataset = VirtualCellStreamingDataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=train_perts,
                    control_matcher=control_matcher,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    n_controls=config.n_controls_per_sample,
                    batch_size=config.batch_size,
                    chunk_size=config.optim_chunk_size,
                    buffer_chunks=config.optim_buffer_chunks,
                    shuffle=True,
                    seed=config.seed,
                    include_controls_as_training=config.include_controls_as_training,
                    train_cell_lines=train_cell_lines,
                    control_identity_mode=getattr(config, "control_identity_mode", "paired"),
                    max_cells_per_pert=getattr(config, "max_cells_per_pert_stream", 0),
                    min_cells_per_pert=getattr(config, "min_cells_per_pert_stream", 0),
                    balance_sources=getattr(config, "balance_sources", True),
                    extra_samples=_extra,
                    fixed_pairs=getattr(config, "stream_fixed_pairs", False),
                    batch_match=getattr(config, "stream_batch_match", False),
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=None, num_workers=0
                )

            # Validation: in-memory
            if val_perts:
                print("Building validation dataset...")
                val_dataset = build_eval_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=val_perts,
                    control_matcher=control_matcher,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    n_controls=config.n_controls_per_sample,
                    # The benchmark's eval cell lines, NOT train_cell_lines. The held-out
                    # perturbations are TRAINING data in the other contexts, so drawing val
                    # from them would make the selector two-thirds a training metric.
                    cell_lines=(getattr(config, "bench_eval_cell_lines", None)
                                or train_cell_lines),
                )
                self._val_cell_lines = list(
                    getattr(config, "bench_eval_cell_lines", None) or train_cell_lines)
                print(f"  val cell lines: {self._val_cell_lines}  "
                      f"({len(val_dataset)} samples, {len(val_perts)} perturbations)")
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=VirtualCellDataset.collate_fn,
                )
            else:
                print("No validation perturbations — skipping val_loader")
                val_loader = None

            # Test: unseen perturbations (from training cell lines)
            print("Building test_unseen_pert dataset...")
            test_pert_dataset = None
            test_pert_loader = None
            if test_unseen_pert:
                test_pert_dataset = build_eval_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=test_unseen_pert,
                    control_matcher=control_matcher,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    n_controls=config.n_controls_per_sample,
                    cell_lines=train_cell_lines,
                )
                test_pert_loader = DataLoader(
                    test_pert_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=VirtualCellDataset.collate_fn,
                )

            # Test: unseen cell line
            print("Building test_unseen_cell dataset...")
            test_cell_dataset = None
            test_cell_loader = None
            if test_unseen_cell and test_cell_lines:
                test_cell_dataset = build_eval_dataset(
                    file_metas=file_metas,
                    pert2id=self.pert2id,
                    split_pert_names=test_unseen_cell,
                    control_matcher=control_matcher,
                    encoder_gene_names=self.encoder_gene_names,
                    target_sum=self.target_sum,
                    n_controls=config.n_controls_per_sample,
                    cell_lines=test_cell_lines,
                )
                test_cell_loader = DataLoader(
                    test_cell_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=VirtualCellDataset.collate_fn,
                )

        # ---- H1 validation (external held-out cell line) ----
        # With all_perts_train there is no in-domain validation split left, by design.
        # Validation instead comes from H1 itself, so checkpoint selection targets the
        # actual objective (cross-cell-line transfer) rather than in-domain loss.
        # ⚠ This is selection on the evaluation distribution; the perturbations used are
        # excluded from the scored panels so the reported score stays uncontaminated.
        self._h1_val_perts = []
        if val_loader is None and getattr(config, "panel_eval_enabled", False) \
                and float(getattr(config, "panel_val_frac", 0.0)) > 0:
            from occam.vcell.panel import build_panel_val_dataset
            val_dataset, self._h1_val_perts = build_panel_val_dataset(
                config, self.encoder_gene_names, self.pert2id,
                self.skip_normalize, self.skip_log1p, seed=config.split_seed)
            val_loader = DataLoader(
                val_dataset, batch_size=eval_batch_size, shuffle=False,
                num_workers=0, collate_fn=PseudobulkDataset.collate_fn,
            )

        # ---- Update wandb with data/model info ----
        if wandb_run:
            import wandb
            fwd = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            n_trainable = sum(
                p.numel() for p in fwd.parameters() if p.requires_grad
            )
            n_total = sum(p.numel() for p in fwd.parameters())
            wandb.config.update(
                {
                    "model_parameters": n_total,
                    "trainable_parameters": n_trainable,
                    "n_train_perts": len(train_perts),
                    "n_val_perts": len(val_perts),
                    "n_test_unseen_pert": len(test_unseen_pert),
                    "n_test_unseen_cell": len(test_unseen_cell),
                    "n_train_samples": len(train_dataset),
                    "n_val_samples": len(val_dataset) if val_loader is not None else 0,
                    "n_test_pert_samples": len(test_pert_dataset) if test_pert_dataset else 0,
                    "n_test_cell_samples": len(test_cell_dataset) if test_cell_dataset else 0,
                    "train_cell_lines": train_cell_lines,
                    "test_cell_lines": test_cell_lines,
                    "n_control_cells": {
                        ct: pool.shape[0]
                        for ct, pool in control_matcher.control_pool.items()
                    },
                },
                allow_val_change=True,
            )

        # ---- Control means for evaluation ----
        control_means = {}
        if getattr(config, "stream_v2", False):
            control_means = dict(getattr(self, "_v2_control_means", {}))   # raw log1p means from the scan
        else:
            for ct, pool in control_matcher.control_pool.items():
                control_means[ct] = pool.mean(dim=0).numpy()

        # ---- Pre-compute encoder embeddings (frozen encoder only) ----
        fwd_model = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        # Streaming + "exact" control identity cannot use precomputed embeddings: the
        # identity input is the cell's own RAW expression (num_genes) while the pool holds
        # embeddings (hidden_size), so the two cannot be stacked into one batch.
        # VirtualCellStreamingDataset raises on that combination; skip the precompute here
        # so the run proceeds on the encoder path instead of dying at the first batch.
        # Costs an encoder forward per control input, which is why it is not the default.
        _stream_exact = (getattr(config, "stream_single_cell", False)
                         and getattr(config, "control_identity_mode", "paired") == "exact"
                         and config.include_controls_as_training)
        if config.freeze_encoder and not _stream_exact and not getattr(config, "stream_v2", False):
            print("Pre-computing encoder embeddings for control pools...")
            fwd_model.precompute_control_embeddings(
                control_matcher.control_pool, device=self.device
            )
            # Set on control_matcher so datasets return embeddings directly
            control_matcher.set_control_embeddings(fwd_model._control_embeddings)
        elif _stream_exact:
            print("Skipping control-embedding precompute: streaming + exact control "
                  "identity needs raw control expression (encoder runs per batch).")

        # ---- Hurdle-Gamma initialization ----
        if config.hurdle_loss_weight > 0 and config.pseudobulk and config.mixed_pb_sc:
            print("Initializing hurdle-Gamma parameters from SC training data...")
            self._init_hurdle_params(train_dataset)

        # ---- Baselines ----
        if config.run_baselines and self.rank == 0:
            print("=" * 60)
            print("Running baselines...")
            self._run_baselines(
                train_dataset=None,
                val_dataset=val_dataset if val_loader is not None else None,
                test_pert_dataset=test_pert_dataset,
                test_cell_dataset=test_cell_dataset,
                control_means=control_means,
                wandb_run=wandb_run,
                file_metas=file_metas,
                train_perts=train_perts,
                train_cell_lines=train_cell_lines,
                control_matcher=control_matcher,
                exclude_cl_perts=exclude_cl_perts,
            )

        # ---- Training loop ----
        print("=" * 60)
        print(f"Starting training for {config.epochs} epochs...")
        for epoch in range(1, config.epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()

            # Set epoch for streaming dataset (PseudobulkDataset uses DataLoader shuffle)
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)

            metrics = self.train_epoch(
                train_loader,
                wandb_run=wandb_run,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            elapsed = time.time() - t0
            if int(getattr(config, "bench_batches", 0) or 0) > 0:
                print("[bench] benchmark complete — stopping before evaluation/checkpointing", flush=True)
                return str(run_dir)

            if self.rank == 0:
                samples_per_sec = metrics["n_samples"] / max(elapsed, 1e-6)
                if metrics.get("grad_norm") is not None:
                    _gm = metrics.get("grad_norm_max")
                    grad_str = (f"grad_norm={metrics['grad_norm']:.4f} "
                                f"(max {_gm:.4g}) | " if _gm is not None
                                else f"grad_norm={metrics['grad_norm']:.4f} | ")
                else:
                    # Say so explicitly rather than omitting the field.
                    grad_str = f"grad_norm=NOT-RECORDED({metrics.get('n_grad_norm_batches', 0)} batches) | "
                print(
                    f"Epoch {epoch}/{config.epochs} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"{grad_str}"
                    f"{metrics['n_batches']} batches, "
                    f"{metrics['n_samples']} samples | "
                    f"{samples_per_sec:.0f} samples/s | "
                    f"time={elapsed:.1f}s"
                )
                # Per-source table: which dataset actually drives learning?
                ps = getattr(self, "_per_source", None)
                if ps:
                    print(f"  [per-source epoch {epoch}]  "
                          f"{'source':<18}{'n':>10}{'err':>10}{'dmag':>10}{'gnorm/samp':>12}")
                    for s_, d in sorted(ps.items(), key=lambda kv: -kv[1]["n"]):
                        n = max(d["n"], 1)
                        print(f"  [per-source epoch {epoch}]  {s_:<18}{d['n']:>10,}"
                              f"{d['err']/n:>10.4f}{d['dmag']/n:>10.4f}{d['gnorm']/n:>12.3e}")
                    if wandb_run:
                        for s_, d in ps.items():
                            n = max(d["n"], 1)
                            wandb_run.log({f"per_source/{s_}/err": d["err"]/n,
                                           f"per_source/{s_}/dmag": d["dmag"]/n,
                                           f"per_source/{s_}/gnorm": d["gnorm"]/n,
                                           f"per_source/{s_}/n": d["n"], "epoch": epoch})

                if wandb_run:
                    train_log = {
                        "epoch": epoch,
                        "train/epoch_loss": metrics["loss"],
                        "train/lr": metrics["lr"],
                        "train/samples_per_sec": samples_per_sec,
                        "train/epoch_time_s": elapsed,
                    }
                    if metrics.get("grad_norm") is not None:
                        train_log["train/avg_grad_norm"] = metrics["grad_norm"]
                    # v2 metrics
                    if "recon_loss" in metrics:
                        train_log["train/recon_loss"] = metrics["recon_loss"]
                    if "latent_loss" in metrics:
                        train_log["train/latent_loss"] = metrics["latent_loss"]
                    if "latent_recon_ratio" in metrics:
                        train_log["train/latent_recon_ratio"] = metrics["latent_recon_ratio"]
                    if "delta_reg" in metrics:
                        train_log["train/delta_reg"] = metrics["delta_reg"]
                    if "gpl_loss" in metrics:
                        train_log["train/gpl_loss"] = metrics["gpl_loss"]
                    if "ranking_loss" in metrics:
                        train_log["train/ranking_loss"] = metrics["ranking_loss"]
                    for hk in ["hurdle_loss", "hurdle_pi_mean",
                               "hurdle_k_min", "hurdle_k_median", "hurdle_k_max"]:
                        if hk in metrics:
                            train_log[f"hurdle/{hk.replace('hurdle_', '')}"] = metrics[hk]
                    for key in [
                        "delta_z_norm", "delta_z_ratio",
                        "delta_z_cosine_w_target", "z_hat_mse_w_target",
                        # off-manifold detector: read `offmanifold_ratio`, not the raw
                        # residual — the AE is imperfect even on real cells, so `z_hat` is
                        # only interpretable against `z_ctrl` on the same batch. ~1 healthy.
                        "z_hat_offmanifold", "z_ctrl_offmanifold", "offmanifold_ratio",
                        "gate_activation_mean", "gate_active_dims",
                    ]:
                        if key in metrics:
                            train_log[f"train/{key}"] = metrics[key]
                    # Decoder warmup LR
                    if (
                        self.decoder_lr_warmup_epochs > 0
                        and self._decoder_pg_idx is not None
                    ):
                        train_log["train/decoder_lr"] = (
                            self.optimizer.param_groups[self._decoder_pg_idx]["lr"]
                        )
                    wandb_run.log(train_log)

            # Evaluation
            if epoch % config.eval_freq == 0 or epoch == config.epochs:
                if self.rank == 0:
                    eval_t0 = time.time()
                    print(f"  Evaluating at epoch {epoch}...")

                    # Validation (skipped when there is no val_loader)
                    if val_loader is not None:
                        val_metrics = self.evaluate(val_loader, control_means, "val")
                        if wandb_run:
                            self._log_eval_metrics(wandb_run, "val", val_metrics, epoch)

                        # Check for best model, on per-cell validation pearson_delta.
                        #
                        # ⚠️ This criterion was MEASURED to rank checkpoints wrongly on the
                        # real panel in the research tree: it picked one arm and put the
                        # eventual bake-off winner last. The alternative there was to
                        # select on the panel score itself, but that needed the other
                        # scorer -- cell-eval v1 takes ~60 min per evaluation, far too slow
                        # to run every few epochs. So selection is val pearson_delta here,
                        # and the panel score is an END-OF-TRAINING measurement.
                        #
                        # Practical consequence: do not read the selected checkpoint as
                        # "the best on the panel". Score the final checkpoint too, and
                        # compare. See docs/evaluation.md.
                        val_score = val_metrics["pearson_delta_mean"]
                        sel_name = "val_pearson_delta"

                        if val_score > self.best_metric and self._ok_to_save_best():
                            self.best_metric = val_score
                            self.save_checkpoint(
                                str(ckpt_dir / "best_model.pth"), split_info
                            )
                            print(f"  New best model: {sel_name}={val_score:.4f}")

                        if wandb_run:
                            wandb_run.log(
                                {
                                    "epoch": epoch,
                                    "val/best_pearson_delta_mean": self.best_metric,
                                }
                            )

                    # ---- Panel evaluation ------------------------------------
                    # Single cell in, single cell out, scored with cell-eval v1.
                    # The panel lives in its own corpus and is never in
                    # data_paths, so the held-out context is structural rather
                    # than split-enforced.
                    #
                    # END OF TRAINING ONLY. cell-eval v1 takes ~60 min for a
                    # 255-perturbation panel, which is far too slow to drive
                    # checkpoint selection; selection uses val pearson_delta,
                    # with the caveat recorded above.
                    if getattr(config, "panel_eval_enabled", False) and epoch == config.epochs:
                        print("Running panel evaluation...")
                        try:
                            from occam.vcell.panel import (
                                load_panel, predict_panel, run_panel_evaluation,
                            )
                            from occam.vcell.panel.calibrate import (
                                apply_quantile_map, fit_control_quantile_map,
                            )

                            # Score the SELECTED checkpoint, not the final epoch.
                            # These runs peak early and then decay, so the last
                            # epoch is systematically the wrong model to spend an
                            # hour of evaluation on. Set panel_eval_checkpoint
                            # to "final" for the other behaviour.
                            _which = str(getattr(config, "panel_eval_checkpoint", "best"))
                            _ctx = (
                                self._weights_from_checkpoint(
                                    str(ckpt_dir / "best_model.pth"),
                                    why=f"(selected on {self.best_metric:.4f})")
                                if _which == "best" else contextlib.nullcontext(False)
                            )
                            with _ctx:
                                panel = load_panel(config, self.encoder_gene_names)
                                pred, labels, ctrl_used = predict_panel(
                                    fwd_model, panel, self.pert2id, self.device, config,
                                )
                                calibrated = None
                                if str(getattr(config, "panel_calibrate", "both")).lower() \
                                        in ("quantile", "both"):
                                    src_q, ref_q = fit_control_quantile_map(
                                        fwd_model, panel, self.pert2id, self.device,
                                        predict_panel,
                                        max_cells=getattr(config, "panel_calib_max_cells", 20000),
                                        seed=config.seed,
                                    )
                                    calibrated = apply_quantile_map(
                                        pred, src_q, ref_q, self.device)
                                panel_metrics = run_panel_evaluation(
                                    panel=panel,
                                    pred=pred,
                                    labels=labels,
                                    ctrl_used=ctrl_used,
                                    out_dir=str(run_dir / "panel"),
                                    train_pert_names=getattr(self, "_train_pert_names", None),
                                    panels=str(getattr(config, "panel_panels", "all")).split(","),
                                    calibrate=str(getattr(config, "panel_calibrate", "both")),
                                    calibrated=calibrated,
                                    shrinks=[
                                        float(x) for x in
                                        str(getattr(config, "panel_shrink", "1.0")).split(",")
                                        if x
                                    ],
                                )
                            if wandb_run and panel_metrics:
                                flat = {
                                    f"panel/{p}/{a}/{m}": v
                                    for p, arms in panel_metrics.items()
                                    for a, ms in arms.items()
                                    for m, v in ms.items()
                                }
                                wandb_run.log(flat)
                        except Exception as e:
                            print(f"  Panel evaluation failed: {e}")
                            import traceback
                            traceback.print_exc()

                    # Test unseen pert

                    # Test unseen cell
                    if test_cell_loader is not None:
                        test_cell_metrics = self.evaluate(
                            test_cell_loader, control_means, "test_unseen_cell"
                        )
                        if wandb_run:
                            self._log_eval_metrics(
                                wandb_run, "test_unseen_cell", test_cell_metrics, epoch
                            )

                    eval_elapsed = time.time() - eval_t0
                    print(f"  Evaluation took {eval_elapsed:.1f}s")
                    if wandb_run:
                        wandb_run.log(
                            {"epoch": epoch, "eval/time_s": eval_elapsed}
                        )

        # Final checkpoint
        if self.rank == 0:
            self.save_checkpoint(str(ckpt_dir / "final_model.pth"), split_info)

        if wandb_run:
            wandb_run.finish()

        print("=" * 60)
        if self.best_metric > -float("inf"):
            print(f"Training complete. Best pearson_delta: {self.best_metric:.4f}")
        else:
            print("Training complete. No validation — using final model only.")
        print(f"Checkpoints: {ckpt_dir}")
        return str(run_dir)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _run_baselines(
        self,
        train_dataset,
        val_dataset,
        test_pert_dataset,
        test_cell_dataset,
        control_means,
        wandb_run,
        file_metas,
        train_perts,
        train_cell_lines,
        control_matcher,
        exclude_cl_perts=None,
    ):
        """Run MeanBaseline and ExperimentalAccuracy."""
        from occam.vcell.baselines import run_baseline_evaluation

        # Build a simple in-memory train dataset for baselines
        print("  Building training dataset for baselines...")
        if self.config.pseudobulk:
            train_eval_dataset = build_pseudobulk_dataset(
                file_metas=file_metas,
                pert2id=self.pert2id,
                split_pert_names=train_perts,
                encoder_gene_names=self.encoder_gene_names,
                target_sum=self.target_sum,
                cell_lines=train_cell_lines,
                exclude_cl_perts=exclude_cl_perts,
            )
        else:
            train_eval_dataset = build_eval_dataset(
                file_metas=file_metas,
                pert2id=self.pert2id,
                split_pert_names=train_perts,
                control_matcher=control_matcher,
                encoder_gene_names=self.encoder_gene_names,
                target_sum=self.target_sum,
                n_controls=self.config.n_controls_per_sample,
                cell_lines=train_cell_lines,
            )

        eval_datasets = []
        if val_dataset is not None:
            eval_datasets.append(("val", val_dataset))
        if test_pert_dataset is not None:
            eval_datasets.append(("test_unseen_pert", test_pert_dataset))
        if test_cell_dataset is not None:
            eval_datasets.append(("test_unseen_cell", test_cell_dataset))

        results = run_baseline_evaluation(
            train_dataset=train_eval_dataset,
            eval_datasets=eval_datasets,
            control_means=control_means,
            seed=self.config.seed,
        )

        # Log to wandb — use same paths as model eval so they share charts.
        # Baselines log at epoch=0 as flat reference lines alongside model curves.
        # e.g. val/pearson_delta_mean (model) vs val/mean_baseline_pearson_delta_mean
        if wandb_run:
            for baseline_name, split_metrics in results.items():
                tag = baseline_name if baseline_name != "experimental_accuracy" else "exp_accuracy"
                log_dict = {"epoch": 0}
                for split_name, metrics in split_metrics.items():
                    overall = metrics.get("overall", {})
                    for k, v in overall.items():
                        if isinstance(v, (int, float)):
                            log_dict[f"{split_name}/{tag}_{k}"] = v
                    # Per cell type
                    for ct_name, ct_metrics in metrics.items():
                        if ct_name == "overall":
                            continue
                        for k, v in ct_metrics.items():
                            if isinstance(v, (int, float)):
                                log_dict[f"{split_name}/{ct_name}/{tag}_{k}"] = v
                wandb_run.log(log_dict)

        del train_eval_dataset
        gc.collect()
