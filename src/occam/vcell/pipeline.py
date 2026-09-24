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

        # Opt-in frozen per-gene table (ablation A4 GenePT / A5 ESM2). Empty = off.
        pert_emb_path = str(getattr(config, "pert_embedding_path", "") or "")
        if pert_emb_path and config.pert_model_type != "adjacency_map":
            raise ValueError("pert_embedding_path requires pert_model_type=adjacency_map "
                             f"(got {config.pert_model_type!r})")
        self._pert_embedding_info = None

        # Load graph (single or multi-KG)
        if pert_emb_path:
            num_nodes = max(self.pert2id.values()) + 1
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_weight = torch.zeros(0)
            print(f"  No graph (pert_embedding_path set): {num_nodes} nodes, frozen table")
        elif config.no_gnn:
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
        if pert_emb_path:
            from occam.vcell.adjacency_map import (FrozenEmbeddingModel,
                                                   build_pert_embedding_matrix)
            emb_mat, emb_present, emb_info = build_pert_embedding_matrix(
                pert_emb_path, self.pert2id,
                norm=str(getattr(config, "pert_embedding_norm", "zscore")))
            self._pert_embedding_info = emb_info
            pert_model = FrozenEmbeddingModel(
                embedding_matrix=emb_mat,
                present_mask=emb_present,
                hidden_dim=config.adj_hidden_dim,
                out_dim=config.pert_latent_dim,
                dropout=config.adj_dropout,
                use_ffn=config.adj_use_ffn,
                activation=config.adj_activation,
            )
            print(f"  Frozen pert embeddings: {emb_info['path']} (sha256 {emb_info['sha256'][:12]}), "
                  f"dim={emb_info['dim']}, norm={emb_info['norm']}")
            print(f"  [pert_embedding coverage] {emb_info['n_present']}/{emb_info['n_vocab']} "
                  f"vocab genes present, {emb_info['n_missing']} missing -> shared learned "
                  f"missing vector (table has {emb_info['n_table_keys']} keys)")
        elif config.no_gnn and config.pert_model_type == "adjacency_map":
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
        # ⚠️ ENCODER GROUP FLOOR (encoder ablation, 2026-09-23). CosineAnnealingLR's `eta_min`
        # is ABSOLUTE and shared by every param group, so an encoder group whose base lr is
        # below it (encoder_lr 1e-5 vs the recipe's eta_min 3e-4) is annealed UPWARD to 3e-4
        # -- 30x its intended lr for most of training. Only when an encoder group exists
        # (freeze_encoder=false; impossible on stream_v2, which refuses it) the cosine phase
        # is rebuilt with a per-group floor: every non-encoder group keeps exactly
        # eta_min (the same curve as before), and the encoder floors at
        # eta_min * encoder_lr / lr, i.e. the same RELATIVE floor as the rest of the model.
        _enc_groups = [i for i, pg in enumerate(self.optimizer.param_groups)
                       if pg.get("name") == "encoder"]
        if _enc_groups:
            import math
            from torch.optim.lr_scheduler import LambdaLR
            _T = max(1, config.epochs - max(0, config.warmup_epochs))
            _lams = []
            for i, pg in enumerate(self.optimizer.param_groups):
                base = float(pg.get("initial_lr", pg["lr"]))
                if i in _enc_groups:
                    r = float(config.eta_min) / float(config.lr) if config.lr else 0.0
                else:
                    r = float(config.eta_min) / base if base > 0 else 0.0
                _lams.append(lambda t, r=r: r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * min(t, _T) / _T)))
            # Rebuild from a clean state: the schedulers above already stepped once at
            # construction, so restore each group's base lr before re-creating them.
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(pg.get("initial_lr", pg["lr"]))
                pg.pop("initial_lr", None)
            if config.warmup_epochs > 0:
                warmup = LinearLR(self.optimizer, start_factor=1e-2,
                                  total_iters=config.warmup_epochs)
                self.scheduler = SequentialLR(
                    self.optimizer, schedulers=[warmup, LambdaLR(self.optimizer, _lams)],
                    milestones=[config.warmup_epochs])
            else:
                self.scheduler = LambdaLR(self.optimizer, _lams)
            print(f"  [encoder] trainable, encoder_lr={config.encoder_lr:g}, floor "
                  f"{float(config.eta_min) * float(config.encoder_lr) / float(config.lr):.3g} "
                  f"(relative eta_min; non-encoder groups unchanged)")

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
        elif config.loss_type == "nb_smooth_l1":
            # ⚠️ THE TWO TERMS ARE ON DIFFERENT SCALES -- see config.nb_weight. Kept as two
            # functions and combined in `_recon_loss`, because NB needs the gene dim intact
            # (per-gene log_theta) while SmoothL1 uses the 1D masked flatten.
            self.recon_loss_fn = None
            self._nb_fn = NBDecoderLoss(num_genes=self.num_genes, reduction="none").to(self.device)
            self._sl1_fn = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
            self._nb_weight = float(config.nb_weight)
            self._nb_scale = float(config.nb_scale)
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
        if bool(getattr(config, "recon_split", False)):
            # ⚠️ The delta term feeds NEGATIVE values to the reconstruction loss. That is fine
            # for smooth_l1/mse/log_cosh, which are functions of a residual, and meaningless
            # for nb/zinb/poisson, which model COUNTS and assume a non-negative argument.
            # Refuse at construction rather than producing NaNs or a silently wrong number.
            _ok = ("smooth_l1", "mse", "log_cosh")
            if str(config.loss_type) not in _ok:
                raise ValueError(
                    f"recon_split=True requires loss_type in {_ok}, got "
                    f"{config.loss_type!r}. The delta term recon(pred - D(z_ctrl), "
                    f"target - ctrl) is negative by construction, which count likelihoods "
                    f"cannot represent.")
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.latent_loss_alpha = config.latent_loss_alpha
        self.delta_reg_type = config.delta_reg_type
        self.delta_reg_lambda = config.delta_reg_lambda
        self.delta_reg_tau = config.delta_reg_tau
        self.expression_loss_mode = config.expression_loss_mode
        # F-disp1 dispersion terms: the EMA centres are allocated lazily on first use, so an
        # arm with every dispersion weight at 0 pays nothing (the gene-space EMA is 1.28 GB).
        self._disp_ema_rad = None      # (G,)          model's own mean ||pred - ctrl_mean||
        self._disp_ema_mu = None       # (G, n_genes)  model's own per-group mean profile
        self._disp_seen_rad = None     # (G,) bool     has this group's radial centre been set?
        self._disp_seen_mu = None      # (G,) bool     has this group's profile centre been set?
        self._disp_acc = {"cos_pred": 0.0, "cos_real": 0.0, "n_cos": 0,
                          "rad_ratio": 0.0, "n_rad": 0, "var_ratio": 0.0, "n_var": 0}

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
        if pert_model_type == "adjacency_map" and gc_.get("pert_embedding"):
            # Frozen-table arm: the (normalised) table is a buffer in gnn_state_dict, so the
            # shell only needs its shape; the table file itself is NOT re-read.
            from occam.vcell.adjacency_map import FrozenEmbeddingModel
            pert_model = FrozenEmbeddingModel.empty_like_state(
                ckpt["gnn_state_dict"],
                hidden_dim=gc_.get("adj_hidden_dim", 128),
                out_dim=gc_.get("out_dim", 64),
                dropout=gc_.get("adj_dropout", 0.2),
                use_ffn=gc_.get("adj_use_ffn", True),
                activation=gc_.get("adj_activation", "relu"),
            )
        elif pert_model_type == "adjacency_map" and no_gnn:
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
        elif config.loss_type == "nb_smooth_l1":
            # the reload path must know this loss too, or a nb_smooth_l1 checkpoint falls
            # through to the `else: raise` below and cannot be resumed or evaluated.
            trainer.recon_loss_fn = None
            trainer._nb_fn = NBDecoderLoss(num_genes=trainer.num_genes, reduction="none").to(trainer.device)
            trainer._sl1_fn = nn.SmoothL1Loss(beta=smooth_l1_beta)
            trainer._nb_weight = float(tc.get("nb_weight", 0.5))
            trainer._nb_scale = float(tc.get("nb_scale", 1.0))
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
        _wlc = float(getattr(self.config, "latent_cos_weight", 0.0))
        _z = torch.tensor(0.0, device=self.device)
        if (self.latent_loss_type == "none" and _wlc <= 0
                and float(getattr(self.config, "vicreg_var_weight", 0.0)) <= 0
                and float(getattr(self.config, "vicreg_cov_weight", 0.0)) <= 0):
            return _z, _z, _z, _z, {}

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
        elif self.latent_loss_type == "none":
            # reachable only when latent_cos_weight > 0: the cosine is wanted without the MSE
            loss = torch.tensor(0.0, device=self.device)
        else:
            raise ValueError(f"Unknown latent_loss_type: {self.latent_loss_type}")

        # ─── LATENT COSINE: the DIRECTION of the perturbation in latent space ──────────────
        # The existing latent losses above are all MSE, i.e. purely magnitude-and-position.
        # This is their direction-only counterpart, and the latent analogue of the gene-space
        # `delta_cos_weight` that produced M03 (F-loss7): 1 - cos(delta_z, delta_z_target).
        #
        # ⚠️ Scale is NOT comparable to the MSE terms and must be measured, never assumed --
        # `1 - cos` is bounded [0,2] and dimensionless, while the latent MSE scales with
        # ||z||^2. F-loss5 found the gene-space pair differed by ~100x, and reading the
        # configured weight as a blend was the single most costly mistake in this campaign.
        #
        # ⚠️ UNLIKE THE MSE, THE TARGET CHOICE MATTERS HERE. MSE(delta_z, delta_z_target) and
        # MSE(z_hat, z_target) are the SAME function (F-latent2): the residuals are the same
        # tensor. Cosine is NOT translation-invariant, so the two are genuinely different:
        #   `delta_z` -> the direction of the perturbation EFFECT
        #   `z_hat`   -> the direction of the absolute latent STATE, which is dominated by the
        #                shared control baseline. Expect it near 0 (i.e. cos near 1) for every
        #                perturbation, the same way the gene-space `cosine_loss_weight` on the
        #                absolute profile scores ~0.99 regardless of what was predicted.
        # `delta_z` is the default for that reason; `z_hat` exists to MEASURE the difference.
        # ─── VICReg VARIANCE + COVARIANCE (latent only) ───────────────────────────────────
        # Our latent MSE already plays VICReg's INVARIANCE role (it pulls delta_z onto
        # delta_z_target). These are the two regularisers that keep an invariance term from
        # collapsing the representation:
        #   variance   : mean_j max(0, gamma - std_j(Z))    -- a HINGE, exactly zero above gamma
        #   covariance : sum_{i!=j} C(Z)_ij^2 / d           -- decorrelates the dimensions
        #
        # ⚠️ gamma is NOT VICReg's 1.0. Measured (F-latent7): per-dim std is ~0.55-0.60 here, so
        # gamma=1 would make the hinge permanently active and turn the variance term into a
        # plain "make everything bigger" penalty. Default is 0 = OFF; set it from the measured
        # std distribution.
        #
        # ⚠️ WHAT THE MEASUREMENT SAYS TO EXPECT (pre-registered, F-latent7): only 0.6% of dims
        # are dead (std<0.01) so the VARIANCE term has little to fix, while the effective rank
        # is 33/512 on Q02 and 25/512 on M03, so the COVARIANCE term has a real target.
        #
        # Statistics are over the BATCH (one row per cell), which is where a collapsed
        # dimension shows up. Targets `delta_z` or `z_hat` per `vicreg_target`; unlike the MSE
        # these are NOT the same, because a per-row-varying control shifts z_hat but not
        # delta_z.
        var_loss = torch.tensor(0.0, device=self.device)
        cov_loss = torch.tensor(0.0, device=self.device)
        _wv = float(getattr(self.config, "vicreg_var_weight", 0.0))
        _wcv = float(getattr(self.config, "vicreg_cov_weight", 0.0))
        if _wv > 0 or _wcv > 0:
            _vt = str(getattr(self.config, "vicreg_target", "delta_z"))
            if _vt == "delta_z":
                Z = delta_z
            elif _vt == "z_hat":
                Z = z_hat
            else:
                raise ValueError(
                    f"Unknown vicreg_target: {_vt!r}. Use 'delta_z' or 'z_hat'.")
            if Z.shape[0] < 2:
                pass                      # std/cov undefined on a single row
            else:
                _gamma = float(getattr(self.config, "vicreg_gamma", 0.0))
                Zc = Z - Z.mean(dim=0, keepdim=True)
                if _wv > 0:
                    if _gamma <= 0:
                        raise ValueError(
                            "vicreg_var_weight > 0 needs vicreg_gamma > 0. The variance term "
                            "is a hinge max(0, gamma - std); with gamma=0 it is identically "
                            "zero and the weight would be silently inert.")
                    std = torch.sqrt(Z.var(dim=0, unbiased=True) + 1e-8)
                    var_loss = torch.clamp(_gamma - std, min=0.0).mean()
                if _wcv > 0:
                    n, d = Z.shape
                    C = (Zc.T @ Zc) / (n - 1)
                    cov_loss = (C.pow(2).sum() - C.pow(2).diagonal().sum()) / d

        cos_loss = torch.tensor(0.0, device=self.device)
        _ct = str(getattr(self.config, "latent_cos_target", "delta_z"))
        if _wlc > 0:
            if _ct == "delta_z":
                cos_loss = (1 - F.cosine_similarity(delta_z, delta_z_target, dim=-1)).mean()
            elif _ct == "z_hat":
                cos_loss = (1 - F.cosine_similarity(z_hat, z_target, dim=-1)).mean()
            else:
                raise ValueError(
                    f"Unknown latent_cos_target: {_ct!r}. Use 'delta_z' (effect direction) "
                    f"or 'z_hat' (absolute state direction).")

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
                "latent_cos_loss": float(cos_loss),
                "vicreg_var_loss": float(var_loss),
                "vicreg_cov_loss": float(cov_loss),
                "latent_cos_delta_z": (1 - F.cosine_similarity(
                    delta_z, delta_z_target, dim=-1)).mean().item(),
                "latent_cos_z_hat": (1 - F.cosine_similarity(
                    z_hat, z_target, dim=-1)).mean().item(),
                "latent_mse_loss": float(loss),
            }

        return loss, cos_loss, var_loss, cov_loss, metrics

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
        self._last_ctrl_idx = ctrl_idx   # recon_split_mode='raw' indexes the raw pool with it
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
        # keep_raw only when recon_split_mode='raw' asks for it: it costs ~5 GB of CPU RAM,
        # and every other mode uses the gene-space control MEANS already in _stream_v2_state.
        _keep_raw = (bool(getattr(config, "recon_split", False))
                     and str(getattr(config, "recon_split_mode", "cmean")) == "raw")
        if _keep_raw and getattr(config, "control_cache_dir", None):
            # The embedding cache short-circuits the dense scan, so the raw rows would never
            # be materialised and the pool would misalign. Bypass it for this run only.
            print("  [stream_v2] recon_split_mode='raw': bypassing the control-embedding disk "
                  "cache so the dense rows are materialised.", flush=True)
        pools, means, raw_pools = build_control_embedding_pool(
            control_matcher.pending_pools, encode_fn, n_enc, self.target_sum,
            self.skip_normalize, self.skip_log1p, self.device,
            cache_dir=(None if _keep_raw else getattr(config, "control_cache_dir", None)),
            enc_key=key,
            chunk_size=config.optim_chunk_size,
            threads=int(getattr(config, "control_scan_threads", 6)),
            extra_dense=extra_dense,
            # The gene axis is part of the cache identity: embeddings computed under a
            # different axis are a different input, not the same input cached.
            axis_sha256=str(getattr(config, "bench_axis_sha256", "") or ""),
            axis_scope=str(getattr(config, "bench_axis_scope", "") or ""),
            keep_raw=_keep_raw)
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
        # ⚠️ SAME `order`, or every ctrl_idx points at the wrong cell. Concatenating these two
        # independently is the single way this feature can produce a plausible wrong number,
        # so the row counts are asserted rather than trusted.
        raw_pool_cpu = None
        if raw_pools is not None:
            raw_pool_cpu = torch.cat([raw_pools[k] for k in order], 0)
            if raw_pool_cpu.shape[0] != pool_gpu.shape[0]:
                raise RuntimeError(
                    f"raw control pool has {raw_pool_cpu.shape[0]} rows but the embedding pool "
                    f"has {pool_gpu.shape[0]}; ctrl_idx would index the wrong cells.")
            print(f"  [stream_v2] raw control pool: {tuple(raw_pool_cpu.shape)} "
                  f"({raw_pool_cpu.numel() * 4 / 2**30:.2f} GB, CPU), aligned to the "
                  f"embedding pool", flush=True)
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
        # ⚠️ EVERY CONSUMER OF THE GROUP MEANS MUST BE IN THIS GATE. The table is only loaded
        # when something asks for it, and a term whose weight is set but whose inputs were
        # never loaded is SILENTLY ZERO -- its guard (`st.get("gmeans") is not None`) simply
        # fails. That is exactly what happened when `delta_cos_weight` was added as a consumer
        # and not added here: three DCOS arms at weights 0.25/0.5/1.0 produced bit-identical
        # validation curves, because the term contributed nothing in all three.
        _needs_group_means = (
            (getattr(config, "target_pb_k", 1) != 1)
            or float(getattr(config, "delta_mag_weight", 0.0)) > 0
            or float(getattr(config, "delta_cos_weight", 0.0)) > 0
            # ⚠️ recon_split in cmean/delta_only mode reads `cmeans`/`gmeans` from
            # _stream_v2_state. Omitting it here is precisely how delta_cos_weight ran
            # silently zero for three arms (F-loss1) -- the guard below then REFUSES rather
            # than degrading, so a missing table stops the run instead of faking a result.
            or (bool(getattr(config, "recon_split", False))
                and (str(getattr(config, "recon_split_mode", "cmean")) in ("cmean", "delta_only")
                     or str(getattr(config, "recon_delta_target", "cell")) == "group_mean"))
            # F-disp1: all three dispersion terms read `gmeans`/`cmeans`/`gline` for the
            # reference the spread is measured around, so they belong in THIS gate too --
            # and they need the dispersion table on top of it (loaded just below).
            or float(getattr(config, "dcos_conc_weight", 0.0)) > 0
            or float(getattr(config, "radial_spread_weight", 0.0)) > 0
            or float(getattr(config, "gene_var_weight", 0.0)) > 0
        )
        _needs_disp = (float(getattr(config, "dcos_conc_weight", 0.0)) > 0
                       or float(getattr(config, "radial_spread_weight", 0.0)) > 0
                       or float(getattr(config, "gene_var_weight", 0.0)) > 0)
        if _needs_group_means:
            # ⚠️ REFUSE, DO NOT DEGRADE. A weight that is set and has no effect is worse than
            # a crash: three DCOS arms burned a GPU each producing bit-identical curves
            # because the term was silently zero (F-loss1). If a term's inputs cannot be
            # loaded, the run must not start.
            _mean_dir = str(getattr(config, "pert_mean_dir", "cache/pert_means"))
            if not (Path(_mean_dir).is_dir() and any(Path(_mean_dir).glob("*.npz"))):
                _asked = [n for n, v in (
                    ("target_pb_k", getattr(config, "target_pb_k", 1) != 1),
                    ("delta_mag_weight", float(getattr(config, "delta_mag_weight", 0.0)) > 0),
                    ("delta_cos_weight", float(getattr(config, "delta_cos_weight", 0.0)) > 0),
                    ("recon_split(%s)" % getattr(config, "recon_split_mode", "cmean"),
                     bool(getattr(config, "recon_split", False))),
                    ("dcos_conc_weight", float(getattr(config, "dcos_conc_weight", 0.0)) > 0),
                    ("radial_spread_weight",
                     float(getattr(config, "radial_spread_weight", 0.0)) > 0),
                    ("gene_var_weight", float(getattr(config, "gene_var_weight", 0.0)) > 0),
                ) if v]
                raise FileNotFoundError(
                    f"{_asked} require the per-(cell line, perturbation) mean table, but "
                    f"{_mean_dir} has no *.npz. Build it first "
                    f"(scripts/pert_group_means.py). Refusing to train: these terms would "
                    f"be SILENTLY ZERO and the run would look successful.")
            from occam.vcell.stream_v2 import load_group_means
            from occam.vcell.stream_v2 import source_name_of as _srcname
            _gm_idx, _gmeans, _cmeans, _gline = load_group_means(
                str(getattr(config, "pert_mean_dir", "cache/pert_means")), self.device,
                sources={_srcname(fm.path) for fm in file_metas})
        _disp = None
        if _needs_disp:
            # ⚠️ REFUSE, DO NOT DEGRADE -- same rule as the means table above. A dispersion
            # weight whose target is missing would train against zeros, which is not "off":
            # matching a variance target of 0 is an instruction to COLLAPSE, the exact
            # pathology F-disp1 measured.
            _disp_dir = str(getattr(config, "pert_disp_dir", "cache/pert_disp"))
            if not (Path(_disp_dir).is_dir() and any(Path(_disp_dir).glob("*.npz"))):
                raise FileNotFoundError(
                    f"dcos_conc_weight/radial_spread_weight/gene_var_weight require the "
                    f"per-group DISPERSION table, but {_disp_dir} has no *.npz. Build it "
                    f"with scripts/pert_group_dispersion.py. Refusing to train: the terms "
                    f"would match a variance target of ZERO, i.e. ask for the collapse they "
                    f"exist to fix.")
            from occam.vcell.stream_v2 import load_group_dispersion
            _dv, _drm, _drs, _dcm, _dcs, _dnc = load_group_dispersion(
                _disp_dir, _gm_idx, int(_gmeans.shape[0]), self.device,
                sources={_srcname(fm.path) for fm in file_metas})
            _disp = {"gvar": _dv, "rad_mean": _drm, "rad_std": _drs,
                     "cos_mean": _dcm, "cos_std": _dcs, "n_cells": _dnc}
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
            "raw_pool": raw_pool_cpu,
            "target_pb_k": int(getattr(config, "target_pb_k", 1)),
            "disp": _disp,
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
        # F-disp1: per-epoch dispersion diagnostics. Reset here so the printed ratio is the
        # epoch's, not the run's -- a cumulative average would hide the term taking effect.
        self._disp_acc = {"cos_pred": 0.0, "cos_real": 0.0, "n_cos": 0,
                          "rad_ratio": 0.0, "n_rad": 0, "var_ratio": 0.0, "n_var": 0}
        epoch_recon_ctrl = 0.0
        epoch_recon_delta = 0.0
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
            if isinstance(batch, PerturbationBatch) and getattr(batch, "compact_cols", None) is not None:
                # Fast non-stream loader: compact rows, densified here on the GPU (zeros
                # outside the compact axis -- exactly what the CPU alignment produced).
                _cols = batch.compact_cols.to(self.device, non_blocking=True)
                _G = int(batch.mask_table.shape[1])
                _t = batch.perturbed_expression.to(self.device, non_blocking=True)
                target = torch.zeros(_t.shape[0], _G, device=self.device, dtype=_t.dtype)
                target[:, _cols] = _t
                _c = batch.control_expression.to(self.device, non_blocking=True)
                if batch.ctrl_compact:
                    ctrl = torch.zeros(_c.shape[0], _G, device=self.device, dtype=_c.dtype)
                    ctrl[:, _cols] = _c
                else:
                    ctrl = _c                      # precomputed control embeddings
                pert_idx = batch.pert_indices
                gene_mask = batch.mask_table.to(self.device, non_blocking=True)[
                    batch.mask_ids.to(self.device, non_blocking=True)]
                _gids = getattr(batch, "group_ids", None)
                self._last_group_id = (None if _gids is None
                                       else _gids.to(self.device, non_blocking=True))
                del _t, _c
            elif isinstance(batch, PerturbationBatch):
                ctrl = batch.control_expression.to(self.device, non_blocking=True)
                target = batch.perturbed_expression.to(self.device, non_blocking=True)
                pert_idx = batch.pert_indices
                gene_mask = batch.gene_masks.to(self.device, non_blocking=True)
                # Non-stream parity: group ids for delta_cos / delta_mag (None = terms off).
                _gids = getattr(batch, "group_ids", None)
                self._last_group_id = (None if _gids is None
                                       else _gids.to(self.device, non_blocking=True))
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
                    elif self.config.loss_type in ("nb", "zinb", "nb_smooth_l1"):
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
                    if self.config.loss_type in ("nb", "zinb", "nb_smooth_l1"):
                        raise ValueError("loss_norm/source_weight are implemented for the "
                                         "elementwise losses only, not nb/zinb/nb_smooth_l1")
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
                elif bool(getattr(self.config, "recon_split", False)):
                    # ─── SPLIT RECONSTRUCTION (F-recon1) ──────────────────────────────────
                    # `smooth_l1(pred, target)` is ~99.8% CONTROL RECONSTRUCTION on M03:
                    # ||D(z_ctrl) - c|| = 15.34 against a true effect size of 2.88, and the
                    # full-profile error ||pred - t|| = 15.58 is almost all inherited
                    # autoencoder error rather than perturbation error. The perturbation is
                    # therefore supervised by ~0.2% of this term's gradient.
                    #
                    # Splitting it lets the two be weighted independently:
                    #     L_ctrl  = recon(D(z_ctrl), ctrl)         pass the control through
                    #     L_delta = recon(pred - D(z_ctrl), target - ctrl)   predict the CHANGE
                    #
                    # ⚠️ NOT a re-parameterisation. recon(a-c, b-c) == recon(a,b) only when the
                    # SAME c is subtracted from both; here the prediction's control is the
                    # model's own D(z_ctrl) and the target's is the real c, so the residual
                    # carries the extra term (D(z_ctrl) - c). That is the whole point.
                    #
                    # ⚠️ Costs one EXTRA DECODER FORWARD per step, which the unsplit loss does
                    # not need.
                    _fwd = self.model.module if hasattr(self.model, "module") else self.model
                    _dec = _fwd.decoder(model_output["z_ctrl"])
                    recon_ctrl = _dec["prediction"] if isinstance(_dec, dict) else _dec
                    _mode = str(getattr(self.config, "recon_split_mode", "cmean"))
                    _st = self._stream_v2_state
                    _gid = getattr(self, "_last_group_id", None)

                    if _mode == "raw":
                        # ⚠️ Per-cell gene-space control. The streaming path serves 512-d pool
                        # embeddings, so this needs the RAW pool (~5 GB) built alongside it.
                        _raw = _st.get("raw_pool") if _st else None
                        if _raw is None:
                            raise RuntimeError(
                                "recon_split_mode='raw' needs the raw gene-space control pool, "
                                "which was not built. The streaming path keeps only the 512-d "
                                "embedding pool. Set stages.vcell.data.keep_raw_control_pool=true, "
                                "or use recon_split_mode='cmean'/'delta_only'.")
                        # The raw pool is CPU-resident (~5 GB); gather the batch's rows
                        # there and move only those to GPU.
                        _ci = self._last_ctrl_idx.clamp(min=0).cpu()
                        _cref = (_raw[_ci].mean(dim=1) if _ci.dim() == 2 else _raw[_ci])
                        _cref = _cref.to(self.device, non_blocking=True)
                    else:
                        # cmean / delta_only: the LINE CONTROL MEAN, the same gene-space
                        # reference `dcos` and `delta_mag` are defined against.
                        if _gid is None or _st is None or _st.get("cmeans") is None:
                            raise RuntimeError(
                                f"recon_split_mode={_mode!r} needs the control-mean table, but "
                                f"_stream_v2_state has no 'cmeans' (or no group ids). This is a "
                                f"GATE failure: _needs_group_means must include recon_split.")
                        _g = _gid.clamp(min=0)
                        _cref = _st["cmeans"][_st["gline"][_g]]

                    if _cref.shape[-1] != recon_ctrl.shape[-1]:
                        raise RuntimeError(
                            f"recon_split needs gene-space controls, but the reference is "
                            f"{tuple(_cref.shape)} while the decoder emits "
                            f"{tuple(recon_ctrl.shape)}.")

                    # ⚠️ The DELTA TARGET is orthogonal to the control reference, and it is
                    # the PHASE-2 axis. `cell` uses this row's single-cell target, whose noise
                    # is 3-70x the effect size; `group_mean` uses the denoised group mean, the
                    # same target `dcos` and `delta_mag` already use.
                    #   `delta_only` is NOT phase 2: it is phase 2's DELTA term with L_ctrl
                    #   dropped. Phase 2 proper is (cmean|raw) + delta_target=group_mean, which
                    #   keeps BOTH terms and denoises only the delta.
                    _dtgt = str(getattr(self.config, "recon_delta_target", "cell"))
                    if _dtgt == "group_mean" or _mode == "delta_only":
                        if _st.get("gmeans") is None or _gid is None:
                            raise RuntimeError(
                                "recon_delta_target='group_mean' needs the group-mean table; "
                                "_stream_v2_state has no 'gmeans'. GATE failure.")
                        _dref = _st["gmeans"][_gid.clamp(min=0)] - _cref
                    elif _dtgt == "cell":
                        _dref = target - _cref
                    else:
                        raise ValueError(
                            f"Unknown recon_delta_target: {_dtgt!r}. Use 'cell' or 'group_mean'.")

                    L_delta = self._recon_loss(prediction - recon_ctrl, _dref, gene_mask)
                    if _mode == "delta_only":
                        L_ctrl = torch.zeros((), device=self.device)
                    else:
                        L_ctrl = self._recon_loss(recon_ctrl, _cref, gene_mask)

                    _wc = float(getattr(self.config, "recon_ctrl_weight", 1.0))
                    _wd = float(getattr(self.config, "recon_delta_weight", 1.0))
                    recon_loss = _wc * L_ctrl + _wd * L_delta
                    self._last_recon_split = (float(L_ctrl), float(L_delta))
                    epoch_recon_ctrl += float(L_ctrl)
                    epoch_recon_delta += float(L_delta)
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

                # ─── DCOS: the DIRECTION of the effect ───────────────────────────────────
                # ⚠️ NOT `cosine_loss_weight`. That one is cos(prediction, target) on the
                # PROFILE (see above), and a profile is dominated by baseline expression that
                # is identical across perturbations -- two predictions that disagree entirely
                # about the effect can still score ~0.99 there. This scores
                #     1 - cos(pred - ctrl_mean, group_mean - ctrl_mean)
                # i.e. the direction of the perturbation effect itself.
                #
                # Both references are precomputed means already in `_stream_v2_state`, so no
                # extra forward pass, and the single cell's noise -- 3-70x the effect size --
                # never enters the target. Ported from the research tree's `delta_cos_weight`
                # (virtual_cell/pipeline.py:3397).
                dcos_loss = torch.tensor(0.0, device=self.device)
                _wc = float(getattr(self.config, "delta_cos_weight", 0.0))
                if _wc > 0 and getattr(self, "_last_group_id", None) is not None:
                    st = self._stream_v2_state
                    gid = self._last_group_id
                    has = gid >= 0
                    if bool(has.any()) and st.get("gmeans") is not None:
                        g = gid.clamp(min=0)
                        cm2 = st["cmeans"][st["gline"][g]]
                        gmf2 = gene_mask.float()
                        dp = (prediction - cm2) * gmf2          # predicted delta
                        dt = (st["gmeans"][g] - cm2) * gmf2     # true group delta
                        dcos_loss = (1 - F.cosine_similarity(dp[has], dt[has], dim=-1)).mean()
                        del cm2, gmf2, dp, dt
                # ─── F-disp1 DISPERSION TERMS ────────────────────────────────────────────
                # MEASURED on W23: predicted cells carry 2.6% of the real within-perturbation
                # dispersion (0.0026 vs 0.1009 per-gene std, 568/568 perturbations), and real
                # cell deltas align with their group direction at cos 0.246 +- 0.100 while
                # `dcos` above drives that cosine toward 1. So the collapse is what the loss
                # ASKS FOR: parallel directions (dcos) plus equal magnitudes (delta_mag) is a
                # constant delta, and `smooth_l1` -- the only per-cell term -- is ~2% of the
                # total in this regime.
                #
                # ⚠️ A BATCH HOLDS A MEDIAN OF ONE CELL PER PERTURBATION (measured: 1,169
                # distinct perts in 2,048 rows). Therefore:
                #   * a per-row penalty `(x_i - target)^2` on a SPREAD would drive every cell
                #     to the same value, i.e. reinforce the collapse. `radial_spread` and
                #     `gene_var` instead match a BATCH-LEVEL normalised second moment to 1,
                #     which is what "the dispersion is right on average" actually means;
                #   * the centre must be the MODEL'S OWN per-group mean, tracked as a detached
                #     EMA. Measuring spread around the REAL mean would let a constant bias
                #     satisfy the term without any dispersion at all.
                # `dcos_conc` is a FIRST moment and needs neither: it calibrates each cell's
                # alignment to the group's real mean alignment.
                disp_cos_loss = torch.zeros((), device=self.device)
                disp_rad_loss = torch.zeros((), device=self.device)
                disp_var_loss = torch.zeros((), device=self.device)
                _wdc = float(getattr(self.config, "dcos_conc_weight", 0.0))
                _wrs = float(getattr(self.config, "radial_spread_weight", 0.0))
                _wgv = float(getattr(self.config, "gene_var_weight", 0.0))
                if (_wdc > 0 or _wrs > 0 or _wgv > 0):
                    _st = self._stream_v2_state
                    _dsp = (_st or {}).get("disp")
                    _gid = getattr(self, "_last_group_id", None)
                    if _dsp is None or _gid is None:
                        raise RuntimeError(
                            "dispersion terms are ON but the dispersion table / group ids are "
                            "absent from _stream_v2_state. That is a GATE failure, not a data "
                            "problem: _needs_group_means and _needs_disp must both include "
                            "dcos_conc_weight / radial_spread_weight / gene_var_weight. "
                            "Refusing to train with a silently-zero term (F-loss1).")
                    _g = _gid.clamp(min=0)
                    # a group with n_cells < 2 has NO defined dispersion (NaN in the table)
                    _ok = (_gid >= 0) & (_dsp["n_cells"][_g] >= 2)
                    if bool(_ok.any()):
                        _cm3 = _st["cmeans"][_st["gline"][_g]]
                        _gmf3 = gene_mask.float()
                        _dp3 = (prediction - _cm3) * _gmf3

                        if _wdc > 0:
                            # DIRECTION CONCENTRATION. Calibrate each cell's alignment with its
                            # group's mean effect to the REAL mean alignment (measured 0.246,
                            # not 1) instead of maximising it the way `dcos` does.
                            #
                            # ⚠️ SLICE THE ROWS BEFORE THE OP, exactly as the `dcos` term above
                            # does. Computing on the full batch and indexing afterwards looks
                            # equivalent and is not: `cosine_similarity`'s backward is NaN for a
                            # zero vector, and autograd multiplies that NaN by the ZERO incoming
                            # gradient of a row this term excluded -- so an excluded row poisons
                            # the batch's gradient while the LOSS stays finite.
                            _ci = _ok.nonzero(as_tuple=True)[0]
                            _gc = _g[_ci]
                            _dt3 = (_st["gmeans"][_gc] - _cm3[_ci]) * _gmf3[_ci]
                            _c = F.cosine_similarity(_dp3[_ci], _dt3, dim=-1)
                            _ct = _dsp["cos_mean"][_gc]
                            disp_cos_loss = ((_c - _ct) ** 2).mean()
                            with torch.no_grad():
                                self._disp_acc["cos_pred"] += float(_c.mean())
                                self._disp_acc["cos_real"] += float(_ct.mean())
                                self._disp_acc["n_cos"] += 1
                            del _dt3, _c, _ct, _gc, _ci

                        if _wrs > 0:
                            # RADIAL SPREAD. Match the spread of ||pred - ctrl_mean|| to the real
                            # spread, measured around the MODEL's own mean radius per group (a
                            # detached EMA), so a bias in the radius -- which `delta_mag` already
                            # supervises -- cannot satisfy this term.
                            #
                            # ⚠️ NOT `.norm()`: its backward is `grad * x / ||x||`, i.e. 0/0 for a
                            # row whose masked delta is exactly zero. The clamped sum-of-squares
                            # form is differentiable everywhere.
                            _r = (_dp3 * _dp3).sum(dim=-1).clamp(min=1e-12).sqrt()
                            if self._disp_ema_rad is None:
                                # ⚠️ LAZY init on the model's OWN first observation per group, NOT
                                # on the real mean. Warm-starting on the real mean made the
                                # epoch-1 diagnostic read 1.0380, which was the model's BIAS
                                # against the real radius rather than its spread -- exactly the
                                # confound the EMA centre exists to remove.
                                self._disp_ema_rad = torch.zeros_like(_dsp["rad_mean"])
                                self._disp_seen_rad = torch.zeros(
                                    _dsp["rad_mean"].shape[0], dtype=torch.bool,
                                    device=self.device)
                            # a group whose centre has never been observed has EMA 0, and
                            # (r - 0)/std would read as an enormous spread. Skip it this batch;
                            # the update below centres it for the next one.
                            _ri = (_ok & self._disp_seen_rad[_g]).nonzero(as_tuple=True)[0]
                            if _ri.numel() > 0:
                                # rows sliced FIRST: rad_std is NaN for the one group with
                                # n_cells < 2, and a NaN the mask removes afterwards still
                                # poisons the backward pass (`0 * NaN = NaN`).
                                _gr = _g[_ri]
                                _rs_t = _dsp["rad_std"][_gr].clamp(min=1e-3)
                                _u = (_r[_ri] - self._disp_ema_rad[_gr]) / _rs_t
                                _m2 = (_u ** 2).mean()
                                disp_rad_loss = (_m2 - 1.0) ** 2
                                with torch.no_grad():
                                    # accumulate ONLY when the term actually ran: a fallback
                                    # value averaged in would read as "dispersion matched"
                                    self._disp_acc["rad_ratio"] += float(_m2.clamp(min=0).sqrt())
                                    self._disp_acc["n_rad"] += 1
                                del _gr, _rs_t, _u, _m2
                            del _ri
                            with torch.no_grad():
                                _mom = float(getattr(self.config, "disp_ema_momentum", 0.99))
                                _gu, _inv = torch.unique(_g[_ok], return_inverse=True)
                                _rd = _r[_ok].detach()
                                _sum = torch.zeros(_gu.numel(), device=self.device,
                                                   dtype=_rd.dtype).index_add_(0, _inv, _rd)
                                _cnt = torch.zeros(_gu.numel(), device=self.device,
                                                   dtype=_rd.dtype).index_add_(
                                                       0, _inv, torch.ones_like(_rd))
                                _new = _sum / _cnt.clamp(min=1)
                                # momentum 0 on a group's first touch = take the observation
                                _mm = torch.where(self._disp_seen_rad[_gu],
                                                  torch.full_like(_new, _mom),
                                                  torch.zeros_like(_new))
                                self._disp_ema_rad[_gu] = (_mm * self._disp_ema_rad[_gu]
                                                           + (1 - _mm) * _new)
                                self._disp_seen_rad[_gu] = True
                                del _gu, _inv, _rd, _sum, _cnt, _new, _mm
                            del _r

                        if _wgv > 0:
                            # PER-GENE VARIANCE. Same construction as the radial term, one degree
                            # of freedom per gene instead of one per cell.
                            if self._disp_ema_mu is None:
                                self._disp_ema_mu = torch.zeros_like(_st["gmeans"])
                                self._disp_seen_mu = torch.zeros(
                                    _st["gmeans"].shape[0], dtype=torch.bool,
                                    device=self.device)
                            _gi2 = (_ok & self._disp_seen_mu[_g]).nonzero(as_tuple=True)[0]
                            if _gi2.numel() > 0:
                                _gs = _g[_gi2]
                                _sd = _dsp["gvar"][_gs].clamp(min=1e-6).sqrt()
                                # only genes PRESENT in the row and with a real spread worth
                                # matching: a gene whose real std is ~0 carries no dispersion
                                # target, and dividing by it manufactures a huge gradient.
                                _keep = (_gmf3[_gi2] > 0) & (_sd > 1e-2)
                                _w3 = _keep.float()
                                # divide by 1 where the gene is dropped rather than by a 1e-6
                                # std the mask zeroes afterwards: a masked-out element still
                                # gets a backward pass, and `0 * huge` poisons a kept one.
                                _sds = torch.where(_keep, _sd, torch.ones_like(_sd))
                                _uu = ((prediction[_gi2] - self._disp_ema_mu[_gs]) / _sds) ** 2
                                _m2g = (_uu * _w3).sum() / _w3.sum().clamp(min=1.0)
                                disp_var_loss = (_m2g - 1.0) ** 2
                                with torch.no_grad():
                                    self._disp_acc["var_ratio"] += float(_m2g.clamp(min=0).sqrt())
                                    self._disp_acc["n_var"] += 1
                                del _gs, _sd, _sds, _keep, _w3, _uu, _m2g
                            del _gi2
                            with torch.no_grad():
                                _mom = float(getattr(self.config, "disp_ema_momentum", 0.99))
                                _gu, _inv = torch.unique(_g[_ok], return_inverse=True)
                                _P = prediction.detach()[_ok]
                                _sum = torch.zeros(_gu.numel(), _P.shape[1], device=self.device,
                                                   dtype=_P.dtype).index_add_(0, _inv, _P)
                                _cnt = torch.zeros(_gu.numel(), device=self.device,
                                                   dtype=_P.dtype).index_add_(
                                                       0, _inv, torch.ones(_P.shape[0],
                                                                           device=self.device,
                                                                           dtype=_P.dtype))
                                _new = _sum / _cnt.clamp(min=1).unsqueeze(1)
                                _mm = torch.where(self._disp_seen_mu[_gu],
                                                  torch.full_like(_cnt, _mom),
                                                  torch.zeros_like(_cnt)).unsqueeze(1)
                                self._disp_ema_mu[_gu] = (_mm * self._disp_ema_mu[_gu]
                                                          + (1 - _mm) * _new)
                                self._disp_seen_mu[_gu] = True
                                del _P, _sum, _cnt, _new, _mm, _gu, _inv

                        del _cm3, _gmf3, _dp3

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
                (latent_loss, latent_cos_loss, vicreg_var_loss, vicreg_cov_loss,
                 latent_metrics) = self._compute_latent_loss(
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
                _wlc = float(getattr(self.config, "latent_cos_weight", 0.0))
                if _wlc > 0:
                    loss = loss + _wlc * latent_cos_loss
                _wv = float(getattr(self.config, "vicreg_var_weight", 0.0))
                _wcv = float(getattr(self.config, "vicreg_cov_weight", 0.0))
                if _wv > 0:
                    loss = loss + _wv * vicreg_var_loss
                if _wcv > 0:
                    loss = loss + _wcv * vicreg_cov_loss
                if _mw > 0:
                    loss = loss + _mw * mag_loss
                if _wc > 0:
                    loss = loss + _wc * dcos_loss
                if _wdc > 0:
                    loss = loss + _wdc * disp_cos_loss
                if _wrs > 0:
                    loss = loss + _wrs * disp_rad_loss
                if _wgv > 0:
                    loss = loss + _wgv * disp_var_loss
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
            # F-disp1 diagnostics. `*_ratio` is pred dispersion / REAL dispersion: 1.0 = matched,
            # and F-disp1 measured 0.026 before any of these terms existed.
            "disp_cos_pred": (self._disp_acc["cos_pred"] / self._disp_acc["n_cos"]
                              if self._disp_acc["n_cos"] else None),
            "disp_cos_real": (self._disp_acc["cos_real"] / self._disp_acc["n_cos"]
                              if self._disp_acc["n_cos"] else None),
            "disp_rad_ratio": (self._disp_acc["rad_ratio"] / self._disp_acc["n_rad"]
                               if self._disp_acc["n_rad"] else None),
            "disp_var_ratio": (self._disp_acc["var_ratio"] / self._disp_acc["n_var"]
                               if self._disp_acc["n_var"] else None),
            "recon_ctrl_loss": (epoch_recon_ctrl / max(n_batches, 1)),
            "recon_delta_loss": (epoch_recon_delta / max(n_batches, 1)),
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
        if self.config.loss_type == "nb_smooth_l1":
            # NB half: gene dim intact, masked with `where` and NOT multiplication -- F148,
            # `per_elem * gm` evaluates inf*0 = NaN for an absent gene whose loss overflowed.
            per_elem = self._nb_fn(full_pred, full_target)
            gm = gmask.bool()
            n_on = gm.sum().clamp(min=1)
            nb_term = torch.where(gm, per_elem, torch.zeros_like(per_elem)).sum() / n_on
            # SmoothL1 half: the original 1D masked flatten.
            sl1_term = self._sl1_fn(full_pred[gmask], full_target[gmask])
            w = self._nb_weight
            return (1.0 - w) * sl1_term + w * (nb_term / max(self._nb_scale, 1e-8))
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
                # ⛔ THIS IS THE FORWARD THAT SELECTS best_model.pth. `val_pearson_delta`,
                # the selection criterion, is computed from these predictions. It called
                # `fwd_model(ctrl, pert_idx)` -- i.e. `forward()`, the `sum` mode -- so on a
                # combinatorial benchmark checkpoint selection optimised `sum` while every
                # reported number used `decode_seq_mean`.
                #
                # ⚠️ CORRECTION, 2026-09-23. `panel_compose` was first wired into the PANEL
                # evaluation path (predict_panel, ~line 4740), which is an end-of-training
                # measurement and selects NOTHING. All 30 arms of the norman sweep (jobs
                # 18363844 / 18363864) therefore ran with `panel_compose: decode_seq_mean`
                # in their resolved config and were STILL selected on `sum`: the
                # "[panel] validation composition" line never printed once. Found by
                # grepping the training logs for it. This is the site that matters.
                _vc = str(getattr(self.config, "panel_compose", "") or "")
                if _vc:
                    _vbase = fwd_model.module if hasattr(fwd_model, "module") else fwd_model
                    if not getattr(self, "_val_compose_announced", False):
                        print(f"  [val] selection metric composes pairs with: {_vc}",
                              flush=True)
                        self._val_compose_announced = True
                    model_output = _vbase.forward_composed(ctrl, pert_idx, _vc)
                else:
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
            # Under stream_v2 the control pool holds 512-d EMBEDDINGS, so `all_ctrls` stays
            # empty and every metric below is a GROUP-MEAN (pseudobulk) quantity, not the
            # per-cell one the name suggests. That is better for model selection -- a
            # group-mean delta has far lower variance -- but it must be a decision rather
            # than a side effect, so the aggregate carries a `__groupmean` alias and a
            # `metric_granularity` field and both are saved with the checkpoint.
            print("  WARNING: no gene-space controls in eval batches — "
                  "falling back to PSEUDOBULK metrics (cells will be averaged). Every "
                  "pearson_delta below is a GROUP-MEAN quantity; the selection metric is "
                  "logged as pearson_delta_mean__groupmean.")
        aggregate, per_pert = compute_perturbation_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            pert_names=pert_names,
            cell_types=cell_types,
            control_means=control_means,
            gene_masks_per_ct=gene_masks_per_ct,
            controls=controls_np,
        )

        # Name the granularity, always -- not only when it falls back. A number whose
        # granularity is only recoverable by re-reading the streaming config is a number
        # that will be compared against the wrong thing.
        aggregate["metric_granularity"] = (
            "single_cell" if controls_np is not None else "groupmean")
        # A numeric twin, because the wandb logger skips non-numeric values and the
        # granularity is exactly the thing you want beside the curve you are reading.
        aggregate["metric_is_single_cell"] = 1.0 if controls_np is not None else 0.0
        if controls_np is None:
            for _k in [k for k in list(aggregate) if k.startswith("pearson_delta")]:
                aggregate[f"{_k}__groupmean"] = aggregate[_k]

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

        # A trainable encoder is saved LIVE (save_checkpoint), so it is swapped with the
        # rest; the checkpoint keys carry the AutoEncoder's "encoder." prefix.
        _swap_encoder = not self.config.freeze_encoder

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
            if _swap_encoder and "encoder_state_dict" in ckpt:
                _enc_sd = {k[len("encoder."):]: v for k, v in ckpt["encoder_state_dict"].items()
                           if k.startswith("encoder.")}
                saved["encoder_state_dict"] = copy.deepcopy(fwd_model.encoder.state_dict())
                modules["encoder_state_dict"] = fwd_model.encoder
                fwd_model.encoder.load_state_dict(_enc_sd)
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

    def _encoder_is_live(self) -> bool:
        """True when the encoder's weights are not the init-time pretrained snapshot."""
        return (not self.config.freeze_encoder) or bool(getattr(self.config, "random_encoder", False))

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
            # ⚠️ `_encoder_state_dict` is an INIT-TIME snapshot of the PRETRAINED weights,
            # taken before `random_encoder` re-initialises them. Correct only for a frozen
            # pretrained encoder (every run to date). A trained or random encoder must save
            # its LIVE weights, or the checkpoint silently carries the pretrained encoder
            # (encoder ablation C1/C2/C3). Same "encoder."-prefixed keys as the snapshot.
            "encoder_state_dict": (
                {f"encoder.{k}": v.detach().cpu()
                 for k, v in fwd_model.encoder.state_dict().items()}
                if self._encoder_is_live() else self._encoder_state_dict),
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
                # Frozen-table arm only (absent for every other config): provenance of the
                # table (path + sha256 + coverage) and the MLP knobs the reload needs.
                **({"pert_embedding": dict(self._pert_embedding_info),
                    "adj_use_ffn": self.config.adj_use_ffn,
                   } if getattr(self, "_pert_embedding_info", None) else {}),
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
                **({"pert_embedding_path": self._pert_embedding_info["path"],
                    "pert_embedding_sha256": self._pert_embedding_info["sha256"],
                    "pert_embedding_norm": self._pert_embedding_info["norm"],
                   } if getattr(self, "_pert_embedding_info", None) else {}),
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
        # ⚠️ `nb_smooth_l1` KEEPS ITS NB TERM IN `_nb_fn`, NOT `recon_loss_fn` (which is None
        # for the combo losses), so looking only at `recon_loss_fn` silently dropped a TRAINED
        # per-gene dispersion: L02's checkpoint has no `log_theta` at all, while L01's does.
        # That loses the arm's own dispersion for chaining AND for NB emission, which is the
        # one thing a jointly-trained NB arm is uniquely able to provide.
        lf = getattr(self, "recon_loss_fn", None)
        if lf is None:
            lf = getattr(self, "_nb_fn", None)
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
                    **({"pert_embedding_path": self._pert_embedding_info["path"],
                        "pert_embedding_sha256": self._pert_embedding_info["sha256"],
                        "pert_embedding_norm": self._pert_embedding_info["norm"],
                        "pert_embedding_n_present": self._pert_embedding_info["n_present"],
                        "pert_embedding_n_vocab": self._pert_embedding_info["n_vocab"],
                       } if getattr(self, "_pert_embedding_info", None) else {}),
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

            # THE SPLIT COMES FROM THE ARTIFACT. Without this the auto-splitter runs and
            # invents its own 0.75/0.05/0.20 fractions -- measured once as
            # train=1789/val=119/test=476 instead of 2381/103/621, silently, with the axis
            # and the exclusions correctly applied around it. The exclusions still held, so
            # nothing failed; the benchmark was simply a different benchmark.
            _bench_split = os.path.join(self.benchmark.dir, "splits.json")
            if config.split_json and os.path.abspath(str(config.split_json)) != os.path.abspath(_bench_split):
                raise ValueError(
                    f"split_json={config.split_json!r} conflicts with benchmark "
                    f"{self.benchmark.benchmark_id!r}, whose split is {_bench_split}. "
                    f"A benchmark defines its own split; pick one.")
            config.split_json = _bench_split
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
            print(f"  loaded split from {config.split_json}")
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

        if self.benchmark is not None:
            # The split that is ACTUALLY in force must be the artifact's. Asserted rather
            # than assumed: the first grid launch ran with the auto-splitter's
            # train=1789/val=119/test=476 while every other benchmark guard passed, because
            # the axis and the exclusions were correct and only the split was not.
            want = self.benchmark.split
            for key in ("train", "val", "test_unseen_pert"):
                got_n, want_n = len(split_info[key]), len(want[key])
                if set(split_info[key]) != set(want[key]):
                    raise ValueError(
                        f"split '{key}' in force has {got_n} perturbations, the "
                        f"{self.benchmark.benchmark_id} artifact declares {want_n}. The "
                        f"benchmark defines the split; something else supplied it.")
            print(f"  [bench] split verified against the artifact: "
                  f"train {len(train_perts):,} · val {len(val_perts)} · "
                  f"test {len(test_unseen_pert)}")

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
                # ⚠️ recon_split needs the RAW control EXPRESSION. The precompute replaces
                # each control with its 512-d embedding, after which `ctrl` is a latent and
                # `L_ctrl = recon(D(z_ctrl), ctrl)` compares 44,728 genes to 512 latents.
                # That is what killed 10/12 arms of job 18216421 with an IndexError deep
                # inside _recon_loss. The smoke test missed it because the smoke config takes
                # the streaming path, which skips the precompute for its own reasons -- so the
                # ONE configuration I verified was the one that could not hit the bug.
                if config.freeze_encoder and bool(getattr(config, "recon_split", False)):
                    print("Disabling paired control-embedding precompute: recon_split needs "
                          "the RAW control expression (L_ctrl reconstructs it). Costs one "
                          "encoder forward per batch.", flush=True)
                if config.freeze_encoder and \
                        not bool(getattr(config, "recon_split", False)) and \
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
                # ⚠️ NON-stream_v2 parity with the stream_v2 recipe (encoder ablation,
                # 2026-09-23). The loss terms below read `self._stream_v2_state` and
                # `self._last_group_id`, which only the v2 loader used to set -- so on this
                # path `delta_cos_weight` and `delta_mag_weight` were SILENTLY ZERO (the
                # F-loss1 shape), i.e. a non-stream "Y06" trained a different loss.
                # The shared loss code keeps reading the same attribute name; only the
                # three tables it needs are provided here.
                _v1_group_index = None
                if (int(getattr(config, "target_pb_k", 1)) != 1
                        or int(getattr(config, "ctrl_avg_k", 1)) != 1):
                    raise ValueError(
                        "target_pb_k / ctrl_avg_k != 1 are implemented by stream_v2 only; "
                        "the non-stream loader would silently ignore them.")
                if (float(getattr(config, "delta_mag_weight", 0.0)) > 0
                        or float(getattr(config, "delta_cos_weight", 0.0)) > 0
                        or bool(getattr(config, "recon_split", False))):
                    _mean_dir = str(getattr(config, "pert_mean_dir", "cache/pert_means"))
                    if not (Path(_mean_dir).is_dir() and any(Path(_mean_dir).glob("*.npz"))):
                        raise FileNotFoundError(
                            f"delta_mag/delta_cos/recon_split need the group-mean table, but "
                            f"{_mean_dir} has no *.npz (scripts/pert_group_means.py). "
                            f"Refusing to train with a silently-zero term.")
                    from occam.vcell.stream_v2 import load_group_means, source_name_of
                    _v1_group_index, _gm, _cm, _gl = load_group_means(
                        _mean_dir, self.device,
                        sources={source_name_of(fm.path) for fm in file_metas})
                    self._stream_v2_state = {"gmeans": _gm, "cmeans": _cm, "gline": _gl,
                                             "raw_pool": None, "disp": None}
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
                    interleave=getattr(config, "stream_interleave", "roundrobin"),
                    row_group_index=_v1_group_index,
                    # Compact rows + GPU densify + background-thread prefetch. Same rows,
                    # predicates and chunk order as the slow loader; a different random
                    # stream for control draws / within-group shuffles. Only the H1-extras
                    # and batch_match configurations fall back to the slow loader.
                    fast=(_extra is None and not getattr(config, "stream_batch_match", False)),
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=None, num_workers=0
                )
                print(f"  [stream_v1] fast={train_dataset.fast} "
                      f"interleave={getattr(config, 'stream_interleave', 'roundrobin')} "
                      f"group_means={'on' if _v1_group_index is not None else 'off'} "
                      f"freeze_encoder={config.freeze_encoder} "
                      f"random_encoder={getattr(config, 'random_encoder', False)}")

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
        # ⚠️ Gate on the REAL router. `stream_single_cell` no longer selects anything --
        #    `config.pseudobulk` does -- so keying this on it left the v1 streaming path
        #    precomputing embeddings and then dying at the first batch on
        #    `control_identity_mode='exact'`, which is the default. v2 was unaffected only
        #    because it is excluded separately below.
        _streaming = not bool(getattr(config, "pseudobulk", False))
        _stream_exact = (_streaming
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

                # ⚠️ THESE PRINTS MUST NOT SIT INSIDE `if wandb_run:`. Both of them did, which made
                # every diagnostic here invisible in offline runs -- and competitor/sweep runs set
                # WANDB_MODE=offline by default. That is the wandb-only-diagnostics failure F-latent1
                # recorded, and a term that is ON and unmeasured is how DCOS burned three GPUs
                # (F-loss1). They print unconditionally.
                _rs = {k: metrics[k] for k in ("recon_ctrl_loss", "recon_delta_loss") if k in metrics}
                if _rs and (_rs.get("recon_ctrl_loss", 0) or _rs.get("recon_delta_loss", 0)):
                    _rc = _rs.get("recon_ctrl_loss", 0.0)
                    _rd = _rs.get("recon_delta_loss", 0.0)
                    _tot = _rc + _rd
                    print("  [recon-split] L_ctrl %.6f (%.1f%%)  L_delta %.6f (%.1f%%)  ratio %.2fx"
                          "   (UNWEIGHTED; configured weights apply on top)"
                          % (_rc, 100*_rc/max(_tot,1e-12), _rd, 100*_rd/max(_tot,1e-12),
                             _rc/max(_rd,1e-12)), flush=True)
                # F-disp1 dispersion diagnostics, printed UNCONDITIONALLY for the same
                # reason as the block above: these arms run with WANDB_MODE=offline, and a
                # term that is ON and unmeasured is how DCOS burned three GPUs (F-loss1).
                # `*_ratio` = pred dispersion / REAL dispersion. 1.0 = matched. The
                # pre-term baseline, measured on W23, is 0.026.
                _dd = {k: metrics.get(k) for k in
                       ("disp_cos_pred", "disp_cos_real", "disp_rad_ratio", "disp_var_ratio")}
                if any(v is not None for v in _dd.values()):
                    _parts = []
                    if _dd["disp_cos_pred"] is not None:
                        _parts.append("cos(pred delta, group delta)=%.4f vs REAL %.4f"
                                      % (_dd["disp_cos_pred"], _dd["disp_cos_real"]))
                    if _dd["disp_rad_ratio"] is not None:
                        _parts.append("radial std_pred/std_real=%.4f" % _dd["disp_rad_ratio"])
                    if _dd["disp_var_ratio"] is not None:
                        _parts.append("per-gene std_pred/std_real=%.4f" % _dd["disp_var_ratio"])
                    print("  [disp] " + "  ".join(_parts)
                          + "   (ratio 1.0 = matched; 0.026 = the W23 baseline collapse)",
                          flush=True)
                _diag = {k: metrics[k] for k in
                         ("delta_z_ratio", "offmanifold_ratio",
                          "delta_z_cosine_w_target", "gate_active_dims")
                         if k in metrics}
                if _diag:
                    print("  [latent] " + "  ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in _diag.items())
                        + "   (delta_z_ratio ~0 = COLLAPSED to identity; "
                          "offmanifold_ratio ~1 = healthy, >>1 = decoder extrapolation)",
                        flush=True)
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

                    # ⚠️ ALSO TO STDOUT. These went ONLY to wandb, so in practice they were
                    # never read: grepping every run log in this repo for `delta_z_ratio`
                    # returns zero hits, and the arms have no local wandb directories. The
                    # two failure modes they exist to catch -- collapse to the identity
                    # (`delta_z_ratio` -> 0) and off-manifold prediction
                    # (`offmanifold_ratio` >> 1) -- both read as "a weak model" in the score
                    # alone, so a silent diagnostic is the same as no diagnostic. See
                    # F-latent1. wandb is `offline` for competitor runs anyway.
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

                        # ---- PER-EPOCH CHECKPOINTS (opt-in) --------------------------
                        # `best_model.pth` is selected on validation per-cell
                        # pearson_delta, a criterion this file already records as having
                        # "MEASURED to rank checkpoints wrongly on the real panel" (see the
                        # comment above). Testing that claim needs every epoch's weights,
                        # which are otherwise discarded -- only best/final survive.
                        #
                        # ⚠️ 2.13 GB per checkpoint. At stride 1 over 30 epochs that is
                        # 64 GB per run, so this is OFF by default and the stride is
                        # configurable.
                        _se = int(getattr(self.config, "save_every_epochs", 0) or 0)
                        if _se > 0 and (epoch % _se == 0):
                            _ep_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
                            self.save_checkpoint(str(_ep_path), split_info)
                            print(f"  [ckpt-sweep] saved {_ep_path.name} "
                                  f"(val_pearson_delta={val_score:.4f})", flush=True)

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
                                # BUG 1 was: load_panel(config, genes) -- 2 of 4 required
                                # args. The skip flags are the TRAINER's resolved values
                                # (checkpoint-inherited, then OR'd with the config), not
                                # config.* which is only one of the two inputs.
                                panel = load_panel(
                                    config, self.encoder_gene_names,
                                    self.skip_normalize, self.skip_log1p,
                                )
                                # BUG 2 was: `config` passed positionally into the
                                # `predict_fn` slot, then called as config(ct, pidx); and a
                                # 4-tuple unpacked into 3. Never pass input_override from a
                                # scoring path -- it feeds the true perturbed cell.
                                _p_avg_k = int(getattr(config, "panel_ctrl_avg_k", 1) or 1)
                                if _p_avg_k != int(getattr(config, "ctrl_avg_k", 1) or 1):
                                    raise ValueError(
                                        f"panel_ctrl_avg_k={_p_avg_k} != "
                                        f"ctrl_avg_k={getattr(config, 'ctrl_avg_k', 1)}. "
                                        f"Eval-side control averaging must match training "
                                        f"or the read-out is not comparable.")
                                # ⛔ COMPOSE THE VALIDATION PREDICTION THE SAME WAY THE
                                # REPORT DOES. predict_panel's default predict_fn is
                                # `model(ctrl, pert_idx)` -- the `sum` mode -- so on a
                                # combinatorial benchmark checkpoint selection optimised
                                # `sum` while every reported number used
                                # `decode_seq_mean`. The two differ by 0.04-0.08
                                # pearson_delta on the 72-cell matrix, i.e. more than most
                                # effects being selected between. Empty keeps the old
                                # behaviour so every pre-existing run reproduces.
                                _pc = str(getattr(config, "panel_compose", "") or "")
                                _pfn = None
                                if _pc:
                                    from occam.vcell.model import VirtualCellModel as _VCM
                                    if _pc not in _VCM.COMPOSE_MODES:
                                        raise ValueError(
                                            f"panel_compose={_pc!r} is not a composition "
                                            f"mode; expected one of "
                                            f"{list(_VCM.COMPOSE_MODES)}")
                                    _cbase = (fwd_model.module
                                              if hasattr(fwd_model, "module")
                                              else fwd_model)

                                    def _pfn(ct, pidx, _m=_cbase, _mode=_pc):
                                        return _m.forward_composed(ct, pidx,
                                                                   _mode)["prediction"]
                                    print(f"  [panel] validation composition: {_pc}",
                                          flush=True)
                                pred, labels, ctrl_used, cin_idx = predict_panel(
                                    fwd_model, panel, self.pert2id, self.device,
                                    predict_fn=_pfn,
                                    eval_control=str(getattr(
                                        config, "panel_eval_control", "single")),
                                    batch_size=int(getattr(
                                        config, "panel_batch_size", 512)),
                                    seed=int(getattr(config, "seed", 42)),
                                    ctrl_avg_k=_p_avg_k,
                                )
                                calibrated = None
                                if str(getattr(config, "panel_calibrate", "both")).lower() \
                                        in ("quantile", "both"):
                                    # BUG 3 was: predict_panel passed as predict_fn, which
                                    # is invoked as predict_fn(ct, pidx) -- a signature it
                                    # can never satisfy. The None default builds the right
                                    # closure itself.
                                    src_q, ref_q = fit_control_quantile_map(
                                        fwd_model, panel, self.pert2id, self.device,
                                        max_cells=getattr(config, "panel_calib_max_cells", 20000),
                                        seed=config.seed,
                                        ctrl_avg_k=_p_avg_k,
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
                                    anchors=tuple(
                                        a.strip() for a in
                                        str(getattr(config, "panel_anchors",
                                                    "ctrlmean,ctrlcell")).split(",")
                                        if a.strip()),
                                    min_panel_perts=int(getattr(
                                        config, "panel_min_perts", 25)),
                                    small_panel_warn=int(getattr(
                                        config, "panel_small_warn", 30)),
                                    celleval_kwargs={
                                        "num_threads": int(getattr(
                                            config, "panel_celleval_threads", 16)),
                                        "timeout": int(getattr(
                                            config, "panel_celleval_timeout", 43200)),
                                        "de_batch_size": int(getattr(
                                            config, "panel_celleval_de_batch_size", 2000)),
                                    },
                                )
                            if wandb_run and panel_metrics:
                                flat = {
                                    f"panel/{p}/{a}/{m}": v
                                    for p, arms in panel_metrics.items()
                                    for a, ms in arms.items()
                                    for m, v in ms.items()
                                }
                                wandb_run.log(flat)
                        except (FileNotFoundError, ImportError) as e:
                            # Only the two EXPECTED absences are tolerated: no panel
                            # directory on this machine, or the optional panel deps
                            # missing. Everything else propagates -- a bare `except
                            # Exception` here turned four TypeErrors into printed lines and
                            # hid them for the whole life of the port.
                            print(f"  Panel evaluation skipped: {type(e).__name__}: {e}")

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
