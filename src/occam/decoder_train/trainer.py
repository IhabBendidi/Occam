"""The external-decoder trainer (stage 3).

Extracted from ``decoder_training_pipeline.py:1203-3178``.

Two inheritance rules in ``_load_encoder`` are load-bearing and must not be
"tidied":

* ``skip_normalize`` / ``skip_log1p`` are ``inherit-then-OR``:
  ``bool(ckpt_cfg.get(k) or False) or bool(config.k)``. The ``.get(...) or
  False`` matters -- the released stage-2 config carries neither key, and
  indexing it would raise. The CLI flags can only force a transform *off*,
  never back on.
* ``target_sum`` comes from ``dataset_info`` in preference to ``config``. That
  ordering is why the release resolves 2000.0 rather than the 1e4 default.
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
import shutil
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
from occam.losses import (
    BasicPoissonLoss, CorrectPoissonLoss, LogCoshLoss, NBDecoderLoss,
    PoissonCountLoss, ZINBDecoderLoss,
)
from occam.decoder_train.augment import (
    AugmentationController, compute_gvf_chunked, init_weights,
    _compute_data_fingerprint,
)
from occam.decoder_train.config import DecoderTrainConfig
from occam.decoder_train.metrics import (
    real_de_cache_path,
    compute_all_metrics, compute_pearson_per_sample, compute_spearman_per_sample,
    run_cell_eval, run_decoder_benchmark, _pearson_torch, _spearman_torch,
)

__all__ = ["DecoderTrainer"]


class DecoderTrainer:
    """Trains a HybridNormGlobalSkipDecoder on top of a pretrained Occam encoder."""

    def __init__(self, config: DecoderTrainConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.best_val_loss = float('inf')
        self.best_combined_score = float('-inf')  # best decoder-eval combined score (selection metric)
        self.current_epoch = 0

        # DDP state (injected by train_decoder.py when using torchrun)
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        # GPU transform deferral (set in train() for optim mode)
        self._defer_transforms = False
        self._transforms = None

        # Load encoder
        self._load_encoder()

        # Create decoder
        self.decoder = HybridNormGlobalSkipDecoder(
            input_dim=self.hidden_size,
            output_dim=self.num_genes,
            hidden_dim=config.decoder_hidden_dim,
            num_blocks=config.decoder_num_blocks,
            dropout_rate=config.decoder_dropout,
            se_reduction=config.decoder_se_reduction,
            weight_tie=getattr(config, "decoder_weight_tie", False),
            bottleneck_k=getattr(config, "decoder_bottleneck_k", 1),
            monarch_blocks=getattr(config, "decoder_monarch_blocks", 0),
            monarch_mup=getattr(config, "decoder_monarch_mup", False),
            kron_rank=getattr(config, "decoder_kron_rank", 0),
            glu_mode=getattr(config, "decoder_glu", False),
            glu_ratio=getattr(config, "decoder_glu_ratio", 0.667),
            sparsity_24=getattr(config, "decoder_sparsity_24", False),
            sparsity_24_decay=getattr(config, "decoder_sparsity_24_decay", 2e-4),
            output_activation=getattr(config, "decoder_output_activation", "rectified_sigmoid"),
        ).to(device)

        # Stage B3/B6 warm-start: analytic dense->structured projection from a trained dense decoder.
        _ws = getattr(config, "decoder_monarch_warmstart", "") or ""
        _kr = getattr(config, "decoder_kron_rank", 0)
        _mb = getattr(config, "decoder_monarch_blocks", 0)
        if _ws and (_mb > 1 or _kr > 0):
            _ck = torch.load(_ws, map_location="cpu", weights_only=False)
            _dsd = _ck.get("decoder_state_dict", _ck.get("model_state_dict", _ck))
            if _kr > 0:
                from occam.decoder.external import warmstart_kron_from_dense
                _n = warmstart_kron_from_dense(self.decoder, _dsd, _kr)
                _tag = f"Kronecker(rank={_kr})"
            else:
                from occam.decoder.external import warmstart_monarch_from_dense
                _n = warmstart_monarch_from_dense(self.decoder, _dsd, _mb)
                _tag = f"Monarch(b={_mb})"
            self.decoder.to(device)
            if is_main_process():
                print(f"[warm-start] projected {_n} dense Linear(H,H) -> {_tag} from {_ws}")
            del _ck, _dsd

        # Weight init
        if config.weight_init != "default":
            assert config.weight_init in VALID_INIT_SCHEMES, (
                f"Unknown weight_init '{config.weight_init}'. Valid: {VALID_INIT_SCHEMES}"
            )
            self.decoder.apply(lambda m: init_weights(m, config.weight_init))
            logging.info(f"Applied weight init: {config.weight_init}")

        n_dec_params = sum(p.numel() for p in self.decoder.parameters())
        n_enc_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"Decoder parameters: {n_dec_params:,}")
        print(f"Encoder parameters: {n_enc_params:,} ({'frozen' if config.freeze_encoder else 'trainable'})")

        # Loss (built BEFORE the optimizer so NB/ZINB dispersion params, if any,
        # can be picked up by _setup_optimizer and trained jointly).
        self.loss_fn = self._build_loss_fn(reduction="mean")
        self.loss_fn = self.loss_fn.to(self.device)

        # Optimizer
        self._setup_optimizer()

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.eta_min,
        )

        # AMP (mixed precision). Disable with config.no_amp to train/eval in fp32.
        self.amp_enabled = not getattr(config, 'no_amp', False)
        self.scaler = torch.amp.GradScaler('cuda') if (device == "cuda" and self.amp_enabled) else None

        # Adversarial / perceptual (set up later if needed)
        self.feature_extractor = None
        self.feature_extractor_perc = None
        self.disc_head = None
        self.disc_optimizer = None
        self.disc_scaler = None

        if config.adv_loss or config.perc_loss:
            self._setup_adversarial_perceptual()

    def _load_encoder(self):
        """Load Occam encoder from checkpoint, strip decoder and mask predictor.

        Handles both standard Occam checkpoints (model_state_dict has full
        AutoEncoder weights) and unified decoder checkpoints (encoder_state_dict
        has encoder-only weights, is_unified_checkpoint=True).
        """
        config = self.config
        print(f"Loading Occam checkpoint: {config.encoder_checkpoint}")
        checkpoint = torch.load(
            config.encoder_checkpoint, map_location=self.device, weights_only=False,
        )

        is_unified = checkpoint.get('is_unified_checkpoint', False)

        # Extract config — unified checkpoints store it under 'encoder_config'
        if is_unified:
            ckpt_config = checkpoint.get('encoder_config', {})
        else:
            ckpt_config = checkpoint.get('config', {})

        self.encoder_gene_names = checkpoint.get('gene_names', None)
        if self.encoder_gene_names is None:
            dataset_info = checkpoint.get('dataset_info', {})
            self.encoder_gene_names = dataset_info.get('gene_names', None)
        if self.encoder_gene_names is None:
            raise ValueError("Checkpoint does not contain gene_names")

        self.num_genes = len(self.encoder_gene_names)
        self.gene_to_index = {g: i for i, g in enumerate(self.encoder_gene_names)}

        # Resolve target_sum
        if is_unified:
            self.target_sum = checkpoint.get('target_sum', 1e4)
        else:
            self.target_sum = ckpt_config.get('target_sum', 1e4)
            dataset_info = checkpoint.get('dataset_info', {})
            if dataset_info.get('target_sum') is not None:
                self.target_sum = dataset_info['target_sum']

        # Resolve transform-skip flags. The decoder's input space MUST match the
        # encoder it is attached to: if the encoder was trained on data that was
        # already library-normalized / log1p-transformed (skip flags True), the
        # decoder must NOT re-apply NormalizePerCell / Log1p (that would double-
        # transform, e.g. log1p(log1p(x))). Inherit from the encoder checkpoint
        # config; CLI flags can additionally force-enable (OR). Older checkpoints
        # without these keys default to False → transforms applied as before.
        ckpt_skip_norm = bool(ckpt_config.get('skip_normalize') or False)
        ckpt_skip_log1p = bool(ckpt_config.get('skip_log1p') or False)
        self.skip_normalize = ckpt_skip_norm or bool(getattr(self.config, 'skip_normalize', False))
        self.skip_log1p = ckpt_skip_log1p or bool(getattr(self.config, 'skip_log1p', False))
        if self.skip_normalize or self.skip_log1p:
            print(f"Transform-skip (encoder is pre-transformed): "
                  f"skip_normalize={self.skip_normalize}, skip_log1p={self.skip_log1p}")

        # Resolve model architecture
        model_size = ckpt_config.get('model_size', 'small')
        hidden_size = ckpt_config.get('hidden_size', None)
        dropout = ckpt_config.get('dropout', 0.0)

        if hidden_size is None:
            hidden_size = MODEL_CONFIGS.get(model_size, {}).get('hidden_size', 512)
        self.hidden_size = hidden_size
        self.model_size = model_size

        # Instantiate full model to load weights
        full_model = AutoEncoder(
            num_genes=self.num_genes,
            model_size=model_size,
            hidden_size=hidden_size,
            decoder_hidden=resolve_decoder_hidden(
                ckpt_config, checkpoint.get('model_state_dict'), model_size
            ),
            dropout=dropout,
            reconstruction_loss=ckpt_config.get('reconstruction_loss', 'mse'),
            clamp_max_log1p=ckpt_config.get('clamp_max_log1p', 5.5),
        )

        # Load state dict — unified checkpoints use encoder_state_dict (encoder
        # keys only), standard checkpoints use model_state_dict (full model)
        if is_unified:
            state_dict = checkpoint['encoder_state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
        cleaned = {k.replace('module.', '', 1).replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        full_model.load_state_dict(cleaned, strict=not is_unified)

        # Extract encoder only
        self.encoder = full_model.encoder
        self.encoder.to(self.device)

        # Store feature method on encoder for convenience
        # AutoEncoder.feature() just calls self.encoder(x), so we wrap it
        self._encoder_model = full_model  # keep reference for .feature() calls
        self._encoder_model.to(self.device)

        # Strip decoder and mask predictor from memory
        del full_model.decoder
        del full_model.mask_predictor
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Freeze encoder if configured
        if self.config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        print(f"Encoder loaded: {model_size} ({self.hidden_size}d), {self.num_genes} genes")
        print(f"Target sum: {self.target_sum}")

    @staticmethod
    def _libnorm_log1p(x, target_sum: float):
        """Rescale each cell to ``target_sum`` in COUNT space, return to log1p space.

        x is log1p(normalised counts). Per-cell rescale is monotone in x for a fixed
        cell, so per-cell Spearman across genes is exactly invariant to this; Pearson
        in log1p space and MAE are not.
        """
        c = torch.expm1(x.clamp(min=0))
        tot = c.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.log1p(c * (target_sum / tot))

    def _make_transforms(self):
        """Build the CPU transform pipeline, honoring encoder skip flags.

        Returns ``[NormalizePerCell(target_sum), Log1p()]`` by default, but omits
        either transform when the encoder was trained on already-transformed data
        (``skip_normalize`` / ``skip_log1p`` inherited from the encoder checkpoint).
        """
        transforms = []
        if not getattr(self, 'skip_normalize', False):
            transforms.append(NormalizePerCell(self.target_sum))
        if not getattr(self, 'skip_log1p', False):
            transforms.append(Log1p())
        return transforms

    def load_decoder_weights(self, checkpoint_path: str):
        """Load trained decoder weights from a unified checkpoint."""
        print(f"Loading decoder weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if 'decoder_state_dict' in ckpt:
            self.decoder.load_state_dict(ckpt['decoder_state_dict'])
        elif 'model_state_dict' in ckpt:
            self.decoder.load_state_dict(ckpt['model_state_dict'])
        else:
            raise ValueError("Checkpoint has neither decoder_state_dict nor model_state_dict")

        self.current_epoch = ckpt.get('epoch', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.best_combined_score = ckpt.get('best_combined_score', float('-inf'))
        print(f"Decoder loaded (epoch {self.current_epoch}, best_val_loss={self.best_val_loss:.6f})")

        # Restore nb/zinb per-gene dispersion. These live on the LOSS module, not the
        # decoder, and were previously saved but never loaded — so an --eval_only run
        # would silently fall back to the init values (theta=1, pi=0.12) instead of the
        # trained ones. That is harmless for smooth_l1/mse (no params) but would make
        # predictive sampling wrong, so it is restored here rather than at the call site.
        lf = getattr(self, "loss_fn", None)
        lsd = ckpt.get("loss_state_dict")
        if lf is not None and lsd:
            try:
                lf.load_state_dict(lsd)
                bits = []
                if hasattr(lf, "log_theta"):
                    th = lf.log_theta.detach().exp()
                    bits.append(f"theta mean={th.mean():.4f} min={th.min():.4g} max={th.max():.4g}")
                if hasattr(lf, "logit_pi"):
                    pi = torch.sigmoid(lf.logit_pi.detach())
                    bits.append(f"pi mean={pi.mean():.4f}")
                print(f"  Loss dispersion restored ({self.config.loss_type})"
                      + (": " + "; ".join(bits) if bits else ""))
            except Exception as e:
                print(f"  [warn] could not load loss_state_dict into "
                      f"{type(lf).__name__} ({self.config.loss_type}): {e}")

    def _celleval_dispersion(self):
        """(theta, pi) for predictive sampling, or None if the loss has no distribution.

        theta/pi are per-gene and live on ``self.loss_fn`` (restored from the checkpoint
        by ``load_decoder_weights``). pi is None for plain NB.
        """
        lf = getattr(self, "loss_fn", None)
        if lf is None or not hasattr(lf, "log_theta"):
            return None
        theta = lf.log_theta.detach().exp().clamp(min=1e-8, max=1e6).to(self.device)
        pi = None
        if hasattr(lf, "logit_pi"):
            pi = torch.sigmoid(lf.logit_pi.detach()).to(self.device)
        return theta, pi

    @staticmethod
    def _sample_predictive(preds: torch.Tensor, theta: torch.Tensor, pi=None):
        """Sample counts from the fitted NB/ZINB predictive, return them in log1p space.

        The decoder emits a deterministic MEAN, which is why its per-cell variance is far
        below real single cells: on nadig the ground truth has ~262 FDR-significant genes
        per perturbation while mean predictions produce ~6,300, a 25x inflation that
        collapses overlap_at_N. Drawing from the fitted predictive restores realistic
        dispersion instead of merely reweighting the loss.

        Parameterisation matches NBDecoderLoss/ZINBDecoderLoss exactly:
        mu = expm1(clamp(pred, 0)) in count space, theta = exp(log_theta),
        pi = sigmoid(logit_pi). Output is log1p(y) so it lands in the same space as the
        targets produced by _make_transforms().
        """
        mu = torch.expm1(preds.clamp(min=0)).clamp(min=1e-8)
        # NB(total_count=theta, logits=log(mu/theta)) has mean exactly mu.
        logits = torch.log(mu) - torch.log(theta)
        y = torch.distributions.NegativeBinomial(
            total_count=theta.expand_as(mu), logits=logits,
        ).sample()
        if pi is not None:
            y = torch.where(torch.rand_like(y) < pi, torch.zeros_like(y), y)
        return torch.log1p(y.clamp(min=0))

    def _build_loss_fn(self, reduction: str = "mean"):
        """Construct the reconstruction loss module for ``config.loss_type``.

        NB/ZINB carry learnable per-gene dispersion params; they are sized from
        ``self.num_genes`` (set in ``_load_encoder``) and trained jointly.
        """
        lt = self.config.loss_type
        if lt == "smooth_l1":
            return nn.SmoothL1Loss(beta=self.config.smooth_l1_beta, reduction=reduction)
        elif lt == "mse":
            return nn.MSELoss(reduction=reduction)
        elif lt == "log_cosh":
            return LogCoshLoss(reduction=reduction)
        elif lt == "poisson":
            return PoissonCountLoss(reduction=reduction)
        elif lt == "basic_poisson":
            return BasicPoissonLoss(reduction=reduction)
        elif lt == "correct_poisson":
            return CorrectPoissonLoss(reduction=reduction)
        elif lt == "nb":
            return NBDecoderLoss(
                num_genes=self.num_genes, reduction=reduction,
                theta_init=getattr(self.config, "disp_theta_init", 0.0),
            )
        elif lt == "zinb":
            return ZINBDecoderLoss(
                num_genes=self.num_genes, reduction=reduction,
                theta_init=getattr(self.config, "disp_theta_init", 0.0),
                pi_init=getattr(self.config, "zinb_pi_init", -2.0),
            )
        else:
            raise ValueError(f"Unknown loss type: {lt}")

    def _setup_optimizer(self):
        """Setup optimizer with separate param groups if joint training.

        NB/ZINB dispersion params (``log_theta``/``logit_pi``, if present on
        ``self.loss_fn``) are added in a dedicated weight-decay-free group.
        """
        config = self.config
        loss_params = list(self.loss_fn.parameters()) if hasattr(self.loss_fn, "parameters") else []
        if config.freeze_encoder:
            groups = [{"params": self.decoder.parameters()}]
        else:
            groups = [
                {"params": self.decoder.parameters(), "lr": config.lr},
                {"params": self.encoder.parameters(), "lr": config.encoder_lr},
            ]
        if loss_params:
            disp_group = {"params": loss_params, "weight_decay": 0.0}
            disp_lr = getattr(config, "dispersion_lr", None)
            if disp_lr is not None and disp_lr > 0:
                disp_group["lr"] = disp_lr  # separate LR for nb/zinb dispersion params
            groups.append(disp_group)
        self.optimizer = torch.optim.AdamW(
            groups, lr=config.lr, weight_decay=config.weight_decay,
        )

    def _setup_adversarial_perceptual(self):
        """Setup adversarial and perceptual loss components."""
        from occam.decoder.discriminator import (
            OccamFeatureExtractor, DiscriminatorHead,
            hinge_d_loss, hinge_g_loss, perceptual_loss,
        )
        config = self.config
        assert config.feature_extractor_checkpoint is not None, (
            "--feature_extractor_checkpoint required for adversarial/perceptual loss"
        )

        # Load a separate encoder for feature extraction
        fe_ckpt = torch.load(
            config.feature_extractor_checkpoint, map_location=self.device, weights_only=False,
        )
        fe_config = fe_ckpt.get('config', {})
        fe_gene_names = fe_ckpt.get('gene_names', None)
        if fe_gene_names is None:
            fe_gene_names = fe_ckpt.get('dataset_info', {}).get('gene_names')

        fe_model_size = fe_config.get('model_size', 'small')
        fe_hidden = fe_config.get('hidden_size', None)
        if fe_hidden is None:
            fe_hidden = MODEL_CONFIGS.get(fe_model_size, {}).get('hidden_size', 512)
        fe_target_sum = fe_config.get('target_sum', 1e4)
        fe_num_genes = len(fe_gene_names)

        fe_decoder_hidden = resolve_decoder_hidden(
            fe_config, fe_ckpt.get('model_state_dict'), fe_model_size
        )
        fe_model = AutoEncoder(
            num_genes=fe_num_genes,
            model_size=fe_model_size,
            hidden_size=fe_hidden,
            decoder_hidden=fe_decoder_hidden,
            dropout=fe_config.get('dropout', 0.0),
        )
        fe_state = fe_ckpt['model_state_dict']
        cleaned = {k.replace('module.', '', 1).replace('_orig_mod.', '', 1): v for k, v in fe_state.items()}
        fe_model.load_state_dict(cleaned)
        fe_model.to(self.device)

        # Set mode
        mode = config.adv_mode if config.adv_loss else config.perc_mode
        if mode == "frozen":
            fe_model.eval()
            for p in fe_model.parameters():
                p.requires_grad = False

        decoder_genes = list(self.encoder_gene_names)

        if config.adv_loss:
            self.feature_extractor = OccamFeatureExtractor(
                fe_model, fe_target_sum, decoder_genes, fe_gene_names,
            ).to(self.device)

        if config.perc_loss and not config.adv_loss:
            self.feature_extractor = OccamFeatureExtractor(
                fe_model, fe_target_sum, decoder_genes, fe_gene_names,
            ).to(self.device)
        elif config.perc_loss and config.adv_loss:
            if config.adv_mode == config.perc_mode == "frozen":
                self.feature_extractor_perc = None  # reuse adv extractor
            else:
                # Load separate for perc
                fe_model_p = AutoEncoder(
                    num_genes=fe_num_genes, model_size=fe_model_size,
                    hidden_size=fe_hidden, decoder_hidden=fe_decoder_hidden,
                    dropout=fe_config.get('dropout', 0.0),
                )
                fe_model_p.load_state_dict(cleaned)
                fe_model_p.to(self.device)
                if config.perc_mode == "frozen":
                    fe_model_p.eval()
                    for p in fe_model_p.parameters():
                        p.requires_grad = False
                self.feature_extractor_perc = OccamFeatureExtractor(
                    fe_model_p, fe_target_sum, decoder_genes, fe_gene_names,
                ).to(self.device)

        if config.adv_loss:
            self.disc_head = DiscriminatorHead(feature_dim=fe_hidden).to(self.device)
            disc_params = list(self.disc_head.parameters())
            if config.adv_mode.startswith("trainable"):
                disc_params += [
                    p for p in self.feature_extractor.encoder_model.parameters()
                    if p.requires_grad
                ]
            self.disc_optimizer = torch.optim.AdamW(
                disc_params, lr=config.disc_lr, weight_decay=0.0,
            )
            self.disc_scaler = torch.amp.GradScaler('cuda') if (self.device == "cuda" and self.amp_enabled) else None

        logging.info("Adversarial/perceptual loss components initialized")

    def _create_dataloader(
        self,
        data_path: str,
        gene_col: Optional[str],
        is_train: bool = True,
        batch_size: Optional[int] = None,
        chunk_indices: Optional[list] = None,
        rank: int = 0,
        world_size: int = 1,
        defer_transforms: bool = False,
        track_presence: bool = False,
        min_gene_overlap_pct: Optional[float] = None,
    ) -> DataLoader:
        """Create a DataLoader for training or validation.

        Args:
            defer_transforms: When True (optim mode), pass transforms=[] to
                the dataset so transforms run on GPU via _apply_gpu_transforms()
                instead of on CPU inside the dataset. Must be False for GVF
                computation which needs transformed data.
        """
        config = self.config
        bs = batch_size or config.batch_size
        transforms = self._make_transforms()

        if config.data_mode == "optim":
            dataset_transforms = [] if defer_transforms else transforms
            dataset = OccamChunkedDataset(
                adata_path=data_path,
                batch_size=bs,
                transforms=dataset_transforms,
                gene_name_col=gene_col,
                gene_vocabulary=self.encoder_gene_names,
                chunk_size=config.optim_chunk_size,
                buffer_chunks=config.optim_buffer_chunks,
                seed=42,
                shuffle=is_train,
                drop_last=is_train,
                chunk_indices=chunk_indices,
                rank=rank,
                world_size=world_size,
                track_presence=track_presence,
                min_gene_overlap_pct=min_gene_overlap_pct,
            )
            return DataLoader(dataset, batch_size=None, num_workers=0)
        else:
            dataset = OccamAdataDataset(
                adata_path=data_path,
                mode=config.data_mode,
                transforms=transforms,
                gene_name_col=gene_col,
                gene_vocabulary=self.encoder_gene_names,
            )
            return DataLoader(
                dataset, batch_size=bs, shuffle=is_train,
                drop_last=is_train, num_workers=4, pin_memory=True,
            )

    @staticmethod
    def _unpack_batch(batch):
        """Extract data tensor and optional gene presence mask from a batch.

        Returns ``(data, gene_present_or_None)``.

        OccamChunkedDataset with track_presence=True yields (data, mask_2D).
        OccamChunkedDataset with track_presence=False yields a plain tensor.
        OccamAdataDataset yields (data, label) — label is a 0-D scalar.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            second = batch[1]
            # Presence mask is 2-D (batch_size, n_genes); labels are 0-D scalars
            if second is not None and second.dim() == 2:
                return batch[0], second
            return batch[0], None
        if isinstance(batch, (list, tuple)):
            return batch[0], None
        return batch, None

    def _apply_gpu_transforms(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply NormalizePerCell + Log1p on GPU tensors (optim mode with deferred transforms)."""
        if self._transforms is None:
            return batch
        for t in self._transforms:
            if isinstance(t, NormalizePerCell):
                row_sums = batch.sum(dim=1, keepdim=True)
                row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
                batch = batch / row_sums * t.target_sum
            elif isinstance(t, Log1p):
                batch = torch.log1p(batch)
        return batch

    @property
    def _decoder_module(self) -> nn.Module:
        """Return the raw decoder module, unwrapping DDP if needed."""
        return self.decoder.module if isinstance(self.decoder, DDP) else self.decoder

    def _encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder forward pass, respecting freeze setting."""
        if self.config.freeze_encoder:
            with torch.no_grad():
                return self._encoder_model.feature(x)
        else:
            # Route through self.encoder (DDP-wrapped if distributed)
            # so gradient sync works correctly across ranks.
            return self.encoder(x)

    def train_epoch(
        self,
        train_loader: DataLoader,
        aug_controller: Optional[AugmentationController],
    ) -> Dict[str, float]:
        """Run one training epoch. Returns dict of metrics."""
        from occam.decoder.discriminator import hinge_d_loss, hinge_g_loss, perceptual_loss

        self.decoder.train()
        if not self.config.freeze_encoder:
            self.encoder.train()

        epoch_recon = 0.0
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_perc_loss = 0.0
        n_batches = 0

        _is_rank0 = not self.is_distributed or is_main_process()
        n_batches_approx = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else None
        pbar = tqdm(
            train_loader, total=n_batches_approx,
            desc=f"Train Epoch {self.current_epoch + 1}",
            unit="batch", dynamic_ncols=True, disable=not _is_rank0,
        )

        _gpm = getattr(self.config, 'gene_presence_mask', False)
        _pshift = getattr(self.config, 'presence_shift', 1.0)
        for batch in pbar:
            batch, gene_present = self._unpack_batch(batch)
            batch = batch.to(self.device, non_blocking=True)
            if gene_present is not None:
                gene_present = gene_present.to(self.device, non_blocking=True)

            # GPU transforms (optim mode: raw data from dataset, transform on GPU)
            if self._defer_transforms:
                batch = self._apply_gpu_transforms(batch)

            target_real = batch  # un-augmented for adv/perc

            # Shift encoder input if the encoder was trained with presence-as-input
            encoder_input = batch
            if gene_present is not None and _gpm:
                encoder_input = batch + gene_present * _pshift

            # Augment
            if aug_controller is not None and self.decoder.training:
                batch = aug_controller.apply(batch)
                # Re-shift augmented batch for encoder
                if gene_present is not None and _gpm:
                    encoder_input = batch + gene_present * _pshift
                else:
                    encoder_input = batch

            # Forward
            embeddings = self._encoder_forward(encoder_input)
            predictions = self.decoder(embeddings)

            # D step (discriminator)
            d_loss_val = 0.0
            if self.config.adv_loss and self.disc_head is not None:
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda" and self.amp_enabled)):
                    with torch.no_grad():
                        pred_d = self.decoder(self._encoder_forward(encoder_input))
                    fake_feat_d = self.feature_extractor(pred_d.detach())
                    real_feat_d = self.feature_extractor(target_real)
                    d_fake = self.disc_head(fake_feat_d)
                    d_real = self.disc_head(real_feat_d)
                    d_loss = hinge_d_loss(d_real, d_fake)

                self.disc_optimizer.zero_grad(set_to_none=True)
                if self.disc_scaler:
                    self.disc_scaler.scale(d_loss).backward()
                    self.disc_scaler.step(self.disc_optimizer)
                    self.disc_scaler.update()
                else:
                    d_loss.backward()
                    self.disc_optimizer.step()
                d_loss_val = d_loss.item()

            # G step (decoder + optional generator losses)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(self.device == "cuda" and self.amp_enabled)):
                # Re-forward for G step (needed because D step may have changed features)
                embeddings = self._encoder_forward(encoder_input)
                predictions = self.decoder(embeddings)
                if gene_present is not None:
                    if self.config.loss_type == "smooth_l1":
                        per_elem = F.smooth_l1_loss(
                            predictions, batch,
                            beta=self.config.smooth_l1_beta, reduction='none',
                        )
                    elif self.config.loss_type == "mse":
                        per_elem = F.mse_loss(predictions, batch, reduction='none')
                    elif self.config.loss_type == "log_cosh":
                        per_elem = LogCoshLoss(reduction='none')(predictions, batch)
                    elif self.config.loss_type == "poisson":
                        per_elem = PoissonCountLoss(reduction='none')(predictions, batch)
                    else:
                        raise ValueError(
                            f"loss_type={self.config.loss_type} is not supported with "
                            "gene_presence_mask (use smooth_l1/mse/log_cosh/poisson)"
                        )
                    per_elem = per_elem * gene_present
                    n_measured = gene_present.sum(dim=1, keepdim=True).clamp(min=1)
                    recon_loss = (per_elem.sum(dim=1) / n_measured.squeeze(1)).mean()
                else:
                    recon_loss = self.loss_fn(predictions, batch)
                total_loss = recon_loss

                g_loss_val = 0.0
                if self.config.adv_loss and self.feature_extractor is not None:
                    fake_feat_g = self.feature_extractor(predictions)
                    d_fake_g = self.disc_head(fake_feat_g)
                    g_adv = hinge_g_loss(d_fake_g)
                    total_loss = total_loss + self.config.lambda_adv * g_adv
                    g_loss_val = g_adv.item()

                perc_val = 0.0
                if self.config.perc_loss:
                    ext = (self.feature_extractor_perc
                           if self.feature_extractor_perc is not None
                           else self.feature_extractor)
                    fake_feat_p = ext(predictions)
                    real_feat_p = ext(target_real)
                    p_loss = perceptual_loss(real_feat_p, fake_feat_p)
                    total_loss = total_loss + self.config.lambda_perc * p_loss
                    perc_val = p_loss.item()

            if self.scaler:
                self.scaler.scale(total_loss).backward()
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    params = list(self.decoder.parameters())
                    if not self.config.freeze_encoder:
                        params += list(self.encoder.parameters())
                    torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.config.gradient_clip is not None:
                    params = list(self.decoder.parameters())
                    if not self.config.freeze_encoder:
                        params += list(self.encoder.parameters())
                    torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip)
                self.optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_d_loss += d_loss_val
            epoch_g_loss += g_loss_val
            epoch_perc_loss += perc_val
            n_batches += 1

            if _is_rank0 and n_batches % 10 == 0:
                pbar.set_postfix(loss=f"{epoch_recon / n_batches:.4f}")

        pbar.close()

        # DDP: sync metrics across ranks
        if self.is_distributed:
            stats = torch.tensor(
                [epoch_recon, epoch_d_loss, epoch_g_loss, epoch_perc_loss, float(n_batches)],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_recon, epoch_d_loss, epoch_g_loss, epoch_perc_loss = stats[:4].tolist()
            n_batches = int(stats[4].item())

        return {
            "train_loss": epoch_recon / max(n_batches, 1),
            "d_loss": epoch_d_loss / max(n_batches, 1),
            "g_adv": epoch_g_loss / max(n_batches, 1),
            "perc_loss": epoch_perc_loss / max(n_batches, 1),
            "n_batches": n_batches,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        decoder = self._decoder_module
        decoder.eval()
        total_loss = 0.0
        n_batches = 0

        _is_rank0 = not self.is_distributed or is_main_process()
        n_batches_approx = len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else None
        pbar = tqdm(
            val_loader, total=n_batches_approx, desc="Validate",
            unit="batch", dynamic_ncols=True, disable=not _is_rank0,
        )

        _gpm = getattr(self.config, 'gene_presence_mask', False)
        _pshift = getattr(self.config, 'presence_shift', 1.0)
        for batch in pbar:
            batch, gene_present = self._unpack_batch(batch)
            batch = batch.to(self.device, non_blocking=True)
            if gene_present is not None:
                gene_present = gene_present.to(self.device, non_blocking=True)

            # GPU transforms (optim mode)
            if self._defer_transforms:
                batch = self._apply_gpu_transforms(batch)

            # Shift encoder input if the encoder was trained with presence-as-input
            encoder_input = batch
            if gene_present is not None and _gpm:
                encoder_input = batch + gene_present * _pshift

            with torch.amp.autocast('cuda', enabled=False):  # val loss in fp32: AMP can overflow->NaN
                embeddings = self._encoder_model.feature(encoder_input)
                predictions = decoder(embeddings)
                if gene_present is not None:
                    if self.config.loss_type == "smooth_l1":
                        per_elem = F.smooth_l1_loss(
                            predictions, batch,
                            beta=self.config.smooth_l1_beta, reduction='none',
                        )
                    elif self.config.loss_type == "mse":
                        per_elem = F.mse_loss(predictions, batch, reduction='none')
                    elif self.config.loss_type == "log_cosh":
                        per_elem = LogCoshLoss(reduction='none')(predictions, batch)
                    elif self.config.loss_type == "poisson":
                        per_elem = PoissonCountLoss(reduction='none')(predictions, batch)
                    else:
                        raise ValueError(
                            f"loss_type={self.config.loss_type} is not supported with "
                            "gene_presence_mask (use smooth_l1/mse/log_cosh/poisson)"
                        )
                    per_elem = per_elem * gene_present
                    n_measured = gene_present.sum(dim=1, keepdim=True).clamp(min=1)
                    loss = (per_elem.sum(dim=1) / n_measured.squeeze(1)).mean()
                else:
                    loss = self.loss_fn(predictions, batch)
            total_loss += loss.item()
            n_batches += 1

        pbar.close()

        # DDP: sync val loss across ranks
        if self.is_distributed:
            stats = torch.tensor(
                [total_loss, float(n_batches)],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = stats[0].item()
            n_batches = int(stats[1].item())

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate_decoder_on_file(
        self,
        eval_path: str,
        gene_col: Optional[str] = None,
        batch_size: int = 1024,
    ) -> Dict[str, float]:
        """Evaluate decoder on a single h5ad file. Returns metrics dict.

        Computes metrics per-batch (streaming) so that neither the full
        predictions nor the full targets accumulate in memory. Explicitly
        frees the dataset and GPU cache after each file.
        """
        eval_name = Path(eval_path).stem
        decoder = self._decoder_module
        decoder.eval()

        logging.info(f"[{eval_name}] Loading dataset (in_memory)...")
        eval_dataset = OccamAdataDataset(
            adata_path=eval_path,
            mode="in_memory",
            transforms=self._make_transforms(),
            gene_name_col=gene_col,
            gene_vocabulary=self.encoder_gene_names,
        )
        n_samples = len(eval_dataset)
        n_batches = math.ceil(n_samples / batch_size)
        logging.info(
            f"[{eval_name}] {n_samples:,} cells x {eval_dataset.num_genes:,} genes, "
            f"{n_batches} batches"
        )

        n_genes = eval_dataset.num_genes

        # --- optional held-out per-gene quantile calibration -----------------------
        # Fit the pred->target map on one cell split, score on the complement. Fitting
        # and scoring on the same cells would be circular: the map is derived from the
        # very targets the metrics compare against.
        _calib = str(getattr(self.config, "eval_calibrate", "none")).lower()
        _ln = str(getattr(self.config, "eval_libnorm", "none")).lower()
        _ln_pre = _ln in ("pre", "both")
        _ln_post = _ln in ("post", "both")
        _ts = float(getattr(self, "target_sum", 1e4))
        if _ln != "none":
            logging.info(f"[{eval_name}] prediction libnorm: {_ln} (target_sum={_ts:g})")
        _src_q = _ref_q = None
        _score_idx = None
        if _calib == "quantile":
            _rng = np.random.default_rng(0)
            _on = str(getattr(self.config, "eval_calib_on", "controls")).lower()
            if _on == "controls":
                # Fit on non-targeting controls, score on perturbed cells. Disjoint, so
                # not circular, and it mirrors inference: controls are always available,
                # perturbed targets never are.
                _pc = getattr(self.config, "celleval_pert_col", "gene")
                _cl = getattr(self.config, "celleval_control_label", "non-targeting")
                _obs = anndata.read_h5ad(eval_path, backed='r').obs
                if _pc not in _obs.columns:
                    raise ValueError(f"[{eval_name}] eval_calib_on=controls but '{_pc}' "
                                     f"not in obs (have {list(_obs.columns)[:8]})")
                _lab = _obs[_pc].astype(str).to_numpy()
                _isc = (_lab == _cl) | (_lab == "non-targeting")
                _min_c = int(getattr(self.config, "eval_calib_min_controls", 200))
                if int(_isc.sum()) < _min_c:
                    # e.g. feng has 36 controls spread over 24 cell lines -- a per-gene
                    # 1001-point quantile map cannot be fit from that.
                    logging.warning(f"[{eval_name}] only {int(_isc.sum())} controls "
                                    f"(< {_min_c}) — skipping calibration for this file")
                    _calib = "none"
                    _fit_idx = _score_idx = None
                else:
                    _fit_idx = np.where(_isc)[0]
                    _score_idx = np.where(~_isc)[0]
                    logging.info(f"[{eval_name}] calibration fit on {len(_fit_idx):,} CONTROLS, "
                                 f"scored on {len(_score_idx):,} PERTURBED cells")
            else:
                _perm = _rng.permutation(n_samples)
                _n_fit = int(n_samples * float(getattr(self.config, "eval_calib_fit_frac", 0.5)))
                _fit_idx = _perm[:_n_fit]
                _score_idx = np.sort(_perm[_n_fit:])
        if _calib == "quantile":
            _cap = int(getattr(self.config, "eval_calib_max_fit_cells", 20000))
            if len(_fit_idx) > _cap:
                _fit_idx = _rng.choice(_fit_idx, _cap, replace=False)
            _fit_idx = np.sort(_fit_idx)
            logging.info(f"[{eval_name}] calibration: fitting on {len(_fit_idx):,} cells, "
                         f"scoring on {len(_score_idx):,} held-out cells")
            _fp, _ft = [], []
            _fl = DataLoader(torch.utils.data.Subset(eval_dataset, _fit_idx.tolist()),
                             batch_size=batch_size, shuffle=False, num_workers=0)
            for _b in _fl:
                _b, _ = self._unpack_batch(_b)
                _b = _b.to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=False):
                    _e = self._encoder_model.feature(
                        _b + getattr(self.config, 'presence_shift', 1.0)
                        if getattr(self.config, 'gene_presence_mask', False) else _b)
                    _p = decoder(_e)
                _p = _p.float()
                if _ln_pre:
                    _p = self._libnorm_log1p(_p, _ts)
                _fp.append(_p.cpu()); _ft.append(_b.float().cpu())
            _fp = torch.cat(_fp).numpy(); _ft = torch.cat(_ft).numpy()
            if bool(getattr(self.config, "eval_calib_ref_raw", False)):
                # log1p-only controls: reference lives in a DIFFERENT space from the
                # source, which breaks the consistency rule on purpose.
                _rawds = OccamAdataDataset(
                    adata_path=eval_path, mode="in_memory", transforms=[Log1p()],
                    gene_name_col=gene_col, gene_vocabulary=self.encoder_gene_names)
                _rl = DataLoader(torch.utils.data.Subset(_rawds, _fit_idx.tolist()),
                                 batch_size=batch_size, shuffle=False, num_workers=0)
                _ft = torch.cat([self._unpack_batch(_x)[0].float() for _x in _rl]).numpy()
                del _rawds
                logging.info(f"[{eval_name}] calibration reference = log1p-only controls "
                             f"(un-normalised) — negative control arm")
            _qs = np.linspace(0.0, 1.0, 1001)
            _src_q = np.quantile(_fp, _qs, axis=0).astype(np.float32)   # (1001, G)
            _ref_q = np.quantile(_ft, _qs, axis=0).astype(np.float32)
            _src_q = np.maximum.accumulate(_src_q, axis=0)
            del _fp, _ft
            gc.collect()
            # (G, 1001) for batched searchsorted
            _src_q = torch.from_numpy(_src_q.T).contiguous().to(self.device)
            _ref_q = torch.from_numpy(_ref_q.T).contiguous().to(self.device)
            eval_dataset = torch.utils.data.Subset(eval_dataset, _score_idx.tolist())
            n_samples = len(eval_dataset)
            n_batches = math.ceil(n_samples / batch_size)

        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        # Stream metrics: accumulate per-sample scalars, not full arrays
        all_pearson = []
        all_spearman = []
        all_mae = []
        all_mse = []
        n_total = 0

        pbar = tqdm(
            eval_loader, total=n_batches,
            desc=f"Eval {eval_name}",
            unit="batch",
            dynamic_ncols=True,
        )
        _gpm = getattr(self.config, 'gene_presence_mask', False)
        _pshift = getattr(self.config, 'presence_shift', 1.0)
        # Optionally score only the genes this file actually contains. The eval runs on
        # the full 44,728-gene encoder vocabulary, so most of a small-panel file is
        # structural zero padding (nadig 35,855/44,728 = 80.2%, feng 59.4%, TREK 1.4%).
        # Padding genes are identically 0 in the target, so a head that emits exact 0
        # (RectifiedSigmoid) has its prediction ties align with the target ties and
        # scores a high Spearman on them, while a strictly-positive head (softplus)
        # emits distinct small values whose ordering there is arbitrary. That alone
        # drove softplus Spearman to -19.39 on nadig vs +71.07, in perfect monotone
        # relation to the padding fraction, while Pearson and MAE were unchanged.
        # Restricting to present genes removes that confound from the comparison.
        _present_only = bool(getattr(self.config, "eval_present_genes_only", False))
        _zthr = float(getattr(self.config, "eval_zero_threshold", 0.0))
        _zpercell = bool(getattr(self.config, "eval_zero_percell", False))
        if _zthr > 0 or _zpercell:
            logging.info(f"[{eval_name}] zero-flooring predictions: "
                         + (f"count<{_zthr}" if _zthr > 0 else "per-cell min-nonzero target"))
        _pmask = None
        if _present_only:
            _ds0 = eval_dataset.dataset if isinstance(eval_dataset, torch.utils.data.Subset) else eval_dataset
            _pg = getattr(_ds0, "_gene_in_dataset", None)
            if _pg is not None:
                _pmask = _pg.to(self.device).bool()
                logging.info(f"[{eval_name}] scoring {int(_pmask.sum())}/{n_genes} present "
                             f"genes ({n_genes - int(_pmask.sum())} padding genes excluded)")
            else:
                logging.warning(f"[{eval_name}] eval_present_genes_only set but no gene mask")
        for batch in pbar:
            batch, _ = self._unpack_batch(batch)
            batch = batch.to(self.device, non_blocking=True)
            # Shift encoder input for single-file eval (all genes present)
            encoder_input = batch + _pshift if _gpm else batch
            with torch.amp.autocast('cuda', enabled=False):  # eval in fp32: AMP can overflow->NaN
                embeddings = self._encoder_model.feature(encoder_input)
                preds = decoder(embeddings)

            # Compute metrics on GPU (avoids slow CPU argsort for Spearman)
            preds_f = preds.float()
            batch_f = batch.float()
            if _ln_pre:
                preds_f = self._libnorm_log1p(preds_f, _ts)
            if _src_q is not None:
                # piecewise-linear per-gene quantile map, batched over genes
                v = preds_f.t().contiguous()                       # (G, B)
                i = torch.searchsorted(_src_q, v).clamp(1, _src_q.shape[1] - 1)
                x0 = torch.gather(_src_q, 1, i - 1); x1 = torch.gather(_src_q, 1, i)
                y0 = torch.gather(_ref_q, 1, i - 1); y1 = torch.gather(_ref_q, 1, i)
                w = torch.where(x1 > x0, (v - x0) / (x1 - x0).clamp(min=1e-12),
                                torch.zeros_like(v)).clamp(0, 1)
                preds_f = (y0 + w * (y1 - y0)).t().contiguous()
            if _ln_post:
                # NOTE this partially undoes the calibration: a per-cell rescale shifts
                # the per-gene marginals off the reference, cell-dependently. Measured
                # here rather than assumed.
                preds_f = self._libnorm_log1p(preds_f, _ts)
            if _pmask is not None:
                # subset AFTER the forward pass — the encoder still needs the full vector
                preds_f = preds_f[:, _pmask]
                batch_f = batch_f[:, _pmask]
            # Quantisation flooring: a prediction below the smallest count the assay can
            # report cannot correspond to a real observation. Applied in COUNT space.
            # NOTE after NormalizePerCell the unit is NOT a raw count -- the one-count
            # floor sits near 0.02 (nadig) / 0.0003 (TREK), and 90.7% / 98.0% of NONZERO
            # target values fall below 1.0, so a threshold of 1 would erase the data.
            if _zthr > 0.0:
                preds_f = torch.where(torch.expm1(preds_f.clamp(min=0)) < _zthr,
                                      torch.zeros_like(preds_f), preds_f)
            elif _zpercell:
                pos = torch.where(batch_f > 0, batch_f,
                                  torch.full_like(batch_f, float('inf')))
                floor = pos.min(dim=1, keepdim=True).values      # per-cell 1-count floor
                floor = torch.where(torch.isinf(floor), torch.zeros_like(floor), floor)
                preds_f = torch.where(preds_f < floor, torch.zeros_like(preds_f), preds_f)
            b_pearson = _pearson_torch(preds_f, batch_f).cpu().numpy()
            b_spearman = _spearman_torch(preds_f, batch_f).cpu().numpy()
            diff = (preds_f - batch_f).abs()
            b_mae = diff.mean(dim=1).cpu().numpy()
            b_mse = (diff.pow(2)).mean(dim=1).cpu().numpy()

            all_pearson.append(b_pearson)
            all_spearman.append(b_spearman)
            all_mae.append(b_mae)
            all_mse.append(b_mse)
            n_total += b_pearson.shape[0]

            # Running averages in the progress bar
            pbar.set_postfix(
                pearson=f"{np.mean(np.concatenate(all_pearson))*100:.1f}",
                spearman=f"{np.mean(np.concatenate(all_spearman))*100:.1f}",
                samples=f"{n_total:,}",
            )

        pearson = np.concatenate(all_pearson)
        spearman = np.concatenate(all_spearman)
        mae = np.concatenate(all_mae)
        mse = np.concatenate(all_mse)

        mean_pearson = float(np.mean(pearson)) * 100
        mean_spearman = float(np.mean(spearman)) * 100
        mean_mae = float(np.mean(mae))
        mean_mse = float(np.mean(mse))
        mae_inv = 1.0 / (1.0 + mean_mae) * 100
        mse_inv = 1.0 / (1.0 + mean_mse) * 100
        combined = (mean_pearson + mean_spearman + 0.1 * mse_inv) / 2.1

        logging.info(
            f"[{eval_name}] Done: {n_total:,} samples | "
            f"Pearson={mean_pearson:.2f} Spearman={mean_spearman:.2f} "
            f"MAE={mean_mae:.4f} MSE={mean_mse:.4f} Combined={combined:.2f}"
        )

        result = {
            "combined_score": combined,
            "pearson_mean": mean_pearson,
            "spearman_mean": mean_spearman,
            "mae_mean": mean_mae,
            "mse_mean": mean_mse,
            "mae_inverted": mae_inv,
            "mse_inverted": mse_inv,
            "n_samples": n_total,
            "n_genes": n_genes,
        }

        # Cleanup: free dataset memory before next eval file
        del eval_dataset, eval_loader, pbar
        del all_pearson, all_spearman, all_mae, all_mse
        del pearson, spearman, mae, mse
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def run_cell_eval_on_file(
        self,
        eval_path: str,
        output_dir: str,
        gene_col: Optional[str] = None,
        pert_col: str = "gene",
        control_label: str = "non-targeting",
        batch_size: int = 1024,
    ) -> Dict[str, Dict[str, float]]:
        """Run ARC cell-eval on the decoder's per-cell reconstruction of an eval file.

        Targets = the eval data preprocessed exactly as encoder input (``_make_transforms()``):
        for skip_normalize+skip_log1p encoders this is the data as-is; otherwise the same
        NormalizePerCell(target_sum)/Log1p the encoder was trained with. Predictions = the raw
        decoder output, passed exactly as the model emits them (no transform / inverse / renorm).
        Both are per-cell (matched single cells), avoiding the pseudobulk-variance pitfall.

        Runs TWO modes against the same (shared) real target, and returns
        ``{mode: {metric: value}}``:
          - ``injctrl`` (MAIN): control rows in the predictions are replaced by the real,
            already-preprocessed target controls (at the same indices — rows are aligned),
            so perturbation deltas are measured against a shared real-control baseline
            (mirrors the virtual-cell convention).
          - ``reconctrl``: predictions keep the decoder's own reconstructed controls.

        Writes ``{output_dir}/celleval/{name}_target.h5ad`` + ``{name}_pred_{mode}.h5ad`` and
        runs cell-eval into ``{output_dir}/cell_eval_{name}_{mode}/``. Returns {} per mode on failure.
        """
        eval_name = Path(eval_path).stem
        decoder = self._decoder_module
        decoder.eval()

        # --- Forward: collect full per-cell predictions + transformed targets ---
        eval_dataset = OccamAdataDataset(
            adata_path=eval_path,
            mode="in_memory",
            transforms=self._make_transforms(),
            gene_name_col=gene_col,
            gene_vocabulary=self.encoder_gene_names,
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        # Which encoder-vocabulary genes this file actually contains. Genes absent from
        # the file are zero-filled in the target but NOT in the prediction (the decoder
        # emits a value for every gene), so leaving them in makes cell-eval measure
        # hallucination on structurally-absent genes: they can never be FDR-significant
        # in the ground truth, but they can enter the prediction's top-N |log2FC| ranking
        # and dilute overlap@N. For nadig that is 35,855 of 44,728 genes (80%).
        _present = getattr(eval_dataset, "_gene_in_dataset", None)
        if _present is not None:
            present_mask = np.asarray(_present.cpu().numpy(), dtype=bool)
        else:
            present_mask = np.ones(len(self.encoder_gene_names), dtype=bool)
        _gpm = getattr(self.config, 'gene_presence_mask', False)
        _pshift = getattr(self.config, 'presence_shift', 1.0)
        # Optional: emit a draw from the fitted NB/ZINB predictive instead of its mean.
        _do_sample = bool(getattr(self.config, "celleval_sample_predictive", False))
        _disp = self._celleval_dispersion() if _do_sample else None
        if _do_sample:
            if _disp is None:
                print(f"  [warn] celleval_sample_predictive requested but loss_type="
                      f"{getattr(self.config, 'loss_type', '?')} has no dispersion "
                      f"params — falling back to mean predictions")
            else:
                torch.manual_seed(int(getattr(self.config, "celleval_sample_seed", 42)))
                _th, _pi = _disp
                print(f"  predictive sampling ON: NB{'/ZINB' if _pi is not None else ''}, "
                      f"theta mean={_th.mean():.4f}"
                      + (f", pi mean={_pi.mean():.4f}" if _pi is not None else "")
                      + f", seed={int(getattr(self.config, 'celleval_sample_seed', 42))}")
        preds_list, tgts_list = [], []
        for batch in tqdm(
            eval_loader, desc=f"cell-eval predict {eval_name}",
            unit="batch", dynamic_ncols=True,
        ):
            batch, _ = self._unpack_batch(batch)
            batch = batch.to(self.device, non_blocking=True)
            encoder_input = batch + _pshift if _gpm else batch
            with torch.amp.autocast('cuda', enabled=False):  # eval in fp32: AMP can overflow->NaN
                embeddings = self._encoder_model.feature(encoder_input)
                preds = decoder(embeddings)
            if _disp is not None:
                preds = self._sample_predictive(preds.float(), _disp[0], _disp[1])
            preds_list.append(preds.float().cpu().numpy())
            tgts_list.append(batch.float().cpu().numpy())
        preds_np = np.concatenate(preds_list, axis=0)
        tgts_np = np.concatenate(tgts_list, axis=0)
        del preds_list, tgts_list, eval_dataset, eval_loader
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # --- target_gene from obs (in_memory preserves cell order) ---
        adata_obs = anndata.read_h5ad(eval_path, backed='r').obs
        if pert_col not in adata_obs.columns:
            print(f"  cell-eval skipped: pert_col '{pert_col}' not in obs of {eval_name} "
                  f"(available: {list(adata_obs.columns)[:10]})")
            return {}
        target_gene = adata_obs[pert_col].astype(str).to_numpy()
        if len(target_gene) != preds_np.shape[0]:
            print(f"  cell-eval skipped: obs rows ({len(target_gene)}) != prediction rows "
                  f"({preds_np.shape[0]}) for {eval_name}")
            return {}
        if control_label != "non-targeting":
            target_gene = np.where(
                target_gene == control_label, "non-targeting", target_gene,
            )

        # --- Restrict to genes present in this eval file ---
        gene_names_ce = [str(g) for g in self.encoder_gene_names]
        if getattr(self.config, "celleval_present_genes_only", True):
            n_all = len(gene_names_ce)
            n_keep = int(present_mask.sum())
            if 0 < n_keep < n_all:
                preds_np = preds_np[:, present_mask]
                tgts_np = tgts_np[:, present_mask]
                gene_names_ce = [
                    g for g, keep in zip(gene_names_ce, present_mask) if keep
                ]
                print(f"  cell-eval genes: {n_keep}/{n_all} present in {eval_name} "
                      f"({n_all - n_keep} zero-filled genes dropped)")
            elif n_keep == 0:
                print(f"  cell-eval skipped: no encoder genes present in {eval_name}")
                return {}

        # --- Per-gene quantile calibration ----------------------------------------
        # Diagnosed cause of the DE inflation (scripts/de_inflation_diagnostic.py):
        # a DOMAIN SHIFT between decoder output and real data. On CONTROL cells, where
        # the true number of DE genes is zero by construction, Wilcoxon+BH returns
        #     real[A] vs real[B]  ->     0 genes   (test is calibrated)
        #     pred[A] vs pred[B]  ->     0 genes   (model-vs-model cancels)
        #     pred[A] vs real[B]  -> 8,147 / 8,873 on nadig, 19,692 / 44,113 on TREK
        # i.e. 92% / 45% of genes "significant" with NO perturbation at all. injctrl
        # compares model output (perturbed) against real data (controls), so that shift
        # contaminates every gene of every perturbation.
        #
        # Matching zeros, variance, mean, or mean+sd does NOT fix it (on TREK, mean
        # matching made it 42% WORSE). Only matching the FULL per-gene marginal drove
        # it to exactly 0 on both datasets. So the mismatch is in distribution SHAPE,
        # not any low-order moment — consistent with a 768-dim bottleneck reproducing
        # smooth systematic structure but not real single-cell marginals.
        #
        # The map is fit on CONTROL CELLS ONLY (no perturbation information) and applied
        # to all rows. It is monotone per gene, so within-gene cell rankings are
        # preserved: it cannot manufacture perturbation signal, only remove the shift.
        # It is nonetheless fit against real data and must be declared as a calibration
        # step, not treated as a free win.
        if str(getattr(self.config, "celleval_calibrate", "none")).lower() == "quantile":
            # target_gene is already normalised to 'non-targeting' above; ctrl_mask
            # itself is only built later, so derive it here.
            cm = (target_gene == "non-targeting")
            if cm.sum() < 50:
                print(f"  [warn] only {int(cm.sum())} control cells — skipping calibration")
            else:
                qs = np.linspace(0.0, 1.0, 1001)
                src_q = np.quantile(preds_np[cm], qs, axis=0).astype(np.float64)
                ref_q = np.quantile(tgts_np[cm], qs, axis=0).astype(np.float64)
                n_flat = 0
                for g in range(preds_np.shape[1]):
                    sq = src_q[:, g]
                    if sq[-1] - sq[0] < 1e-12:      # degenerate (e.g. all-zero) gene
                        n_flat += 1
                        continue
                    sq = np.maximum.accumulate(sq)  # np.interp needs non-decreasing xp
                    preds_np[:, g] = np.interp(preds_np[:, g], sq, ref_q[:, g])
                print(f"  quantile calibration: fit on {int(cm.sum())} control cells, "
                      f"applied to {preds_np.shape[1] - n_flat}/{preds_np.shape[1]} genes "
                      f"({n_flat} degenerate skipped) | pred 0-frac now "
                      f"{float((preds_np == 0).mean()):.4f} vs target "
                      f"{float((tgts_np == 0).mean()):.4f}")

        # --- Zero-handling intervention -------------------------------------------
        # Predictions carry small non-zero floats where the target is exactly 0. The
        # Wilcoxon test reads those as real signal, which is a candidate cause of the
        # ~25x inflation in predicted FDR-significant genes (nadig: ~6,600 predicted vs
        # 262 real). Measured zero fractions back this up on nadig — the ordering
        # zinb 0.530 > smooth_l1 0.401 > nb 0.228 matches the DE-count ordering
        # 5,319 < 6,569 < 6,929 exactly.
        _zmode = str(getattr(self.config, "celleval_zero_mode", "none")).lower()
        if _zmode != "none":
            tgt_zero = (tgts_np == 0)
            before = float((preds_np == 0).mean())
            if _zmode == "oracle":
                # Upper bound on what zero-fixing can buy. USES GROUND TRUTH, so it is a
                # diagnostic ceiling, not a deployable method.
                preds_np[tgt_zero] = 0.0
                thr_used = None
            elif _zmode == "threshold":
                thr = float(getattr(self.config, "celleval_zero_thresh", -1.0))
                if thr < 0:
                    # Auto: match the target's zero fraction. Estimated on a row sample
                    # so this stays cheap on 263k x 8.8k matrices.
                    n = preds_np.shape[0]
                    samp = preds_np[:: max(1, n // 4000)]
                    thr = float(np.quantile(np.abs(samp), tgt_zero.mean()))
                preds_np[np.abs(preds_np) <= thr] = 0.0
                thr_used = thr
            else:
                raise ValueError(f"unknown celleval_zero_mode: {_zmode}")
            print(f"  zero_mode={_zmode}"
                  + (f" thr={thr_used:.5f}" if thr_used is not None else "")
                  + f" | target 0-frac {tgt_zero.mean():.4f}"
                  + f" | pred 0-frac {before:.4f} -> {float((preds_np == 0).mean()):.4f}")

        # --- Build AnnData inputs (per-cell, matched) ---
        celleval_dir = os.path.join(output_dir, "celleval")
        os.makedirs(celleval_dir, exist_ok=True)
        var = pd.DataFrame(index=gene_names_ce)
        obs = pd.DataFrame(
            {"target_gene": target_gene},
            index=np.arange(preds_np.shape[0]).astype(str),
        )
        ctrl_mask = (target_gene == "non-targeting")
        n_ctrl = int(ctrl_mask.sum())

        # Shared real target (already _make_transforms()-preprocessed), written once.
        gt_path = os.path.join(celleval_dir, f"{eval_name}_target.h5ad")
        anndata.AnnData(X=tgts_np, obs=obs.copy(), var=var.copy()).write_h5ad(gt_path)

        # injctrl predictions: replace control rows with the real (preprocessed) target
        # controls at the same indices (rows are aligned). The injected controls are already
        # in encoder-input space (== tgts_np), so no extra preprocessing is needed.
        preds_inj = preds_np.copy()
        if n_ctrl > 0:
            preds_inj[ctrl_mask] = tgts_np[ctrl_mask]
        else:
            print(f"  [warn] no '{control_label}'->non-targeting controls found; "
                  f"injctrl == reconctrl for {eval_name}")
        print(f"  cell-eval inputs: {preds_np.shape[0]} cells x {preds_np.shape[1]} genes "
              f"({n_ctrl} non-targeting) -> {celleval_dir}")

        # --- Real-side DE reuse ------------------------------------------
        # cell-eval spends most of its time on the ground-truth differential
        # expression, and that depends ONLY on the target file -- not on the
        # decoder, the calibration or the mode. So it is identical for every arm
        # scored against the same target and for both modes within one run.
        # Keyed by the target's own sha256, so a target built from different
        # data, transforms or gene subset misses the cache instead of being
        # silently reused. Measured on nadig_2024_jurkat: ~1.5 h of a ~2.5 h run.
        _cache_dir = getattr(self.config, "celleval_de_real_cache", None)
        _de_real = None
        _cache_path = None
        if _cache_dir:
            os.makedirs(_cache_dir, exist_ok=True)
            _cache_path, _digest = real_de_cache_path(_cache_dir, gt_path)
            if os.path.isfile(_cache_path):
                _de_real = _cache_path
                print(f"  reusing real-side DE for target {_digest[:16]} "
                      f"({os.path.getsize(_cache_path) / 1e9:.1f} GB): {_cache_path}")
            else:
                print(f"  no cached real-side DE for target {_digest[:16]}; "
                      f"computing it and caching to {_cache_path}")

        # --- Run both modes against the shared target (injctrl = main, run first) ---
        results: Dict[str, Dict[str, float]] = {}
        _want = [m.strip() for m in
                 str(getattr(self.config, "celleval_modes", "injctrl,reconctrl")).split(",")
                 if m.strip()]
        for mode, pred_X in (("injctrl", preds_inj), ("reconctrl", preds_np)):
            if mode not in _want:
                continue
            pred_path = os.path.join(celleval_dir, f"{eval_name}_pred_{mode}.h5ad")
            anndata.AnnData(X=pred_X, obs=obs.copy(), var=var.copy()).write_h5ad(pred_path)
            ce_out = os.path.join(output_dir, f"cell_eval_{eval_name}_{mode}")
            _nthr = int(getattr(self.config, "celleval_num_threads", 16))
            _tmo = int(getattr(self.config, "celleval_timeout", 3600))
            _dbs = getattr(self.config, "celleval_de_batch_size", None)
            print(f"  [{mode}] running cell-eval "
                  f"(threads={_nthr}, timeout={_tmo/3600:.1f}h"
                  + (f", de_batch_size={_dbs}" if _dbs else "") + ")...")
            results[mode] = run_cell_eval(
                pred_path, gt_path, ce_out, num_threads=_nthr, timeout=_tmo,
                de_batch_size=_dbs, de_real=_de_real,
            )

            # Populate the cache from the first mode that produced one, and
            # reuse it for the second mode in this same run.
            _produced = os.path.join(ce_out, "real_de.csv")
            if _cache_path and _de_real is None and os.path.isfile(_produced):
                try:
                    shutil.copyfile(_produced, _cache_path)
                    _de_real = _cache_path
                    print(f"  cached real-side DE -> {_cache_path}")
                except OSError as exc:
                    print(f"  [warn] could not cache real-side DE: {exc}")

        del preds_np, tgts_np, preds_inj
        gc.collect()
        return results

    def _compute_gvf_distributed(
        self,
        train_data: str,
        gene_col: Optional[str],
        gvf_cache_dir: str,
        fingerprint_hash: str,
        fingerprint_meta: dict,
    ) -> torch.Tensor:
        """Compute GVF, with DDP-aware rank-0 compute + broadcast.

        In DDP mode:
        - All ranks check the cache first (instant if hit).
        - On cache miss: rank 0 computes GVF and broadcasts the result
          tensor to all other ranks via dist.broadcast().
        - GVF dataset always uses CPU transforms (defer_transforms=False)
          because GVF is computed in NormalizePerCell + Log1p space.
        """
        _is_rank0 = not self.is_distributed or is_main_process()
        cache_npy = os.path.join(gvf_cache_dir, f"gvf_{fingerprint_hash}.npy")

        if self.is_distributed:
            # DDP path: check cache, or rank-0 compute + broadcast
            if os.path.isfile(cache_npy):
                # All ranks load from cache independently
                try:
                    cached = np.load(cache_npy)
                    if cached.shape[0] == self.num_genes:
                        if _is_rank0:
                            logging.info(f"All ranks loaded GVF from cache: {cache_npy}")
                        return torch.from_numpy(cached).float()
                except Exception as e:
                    if _is_rank0:
                        logging.warning(f"Failed to load cached GVF: {e}. Rank 0 will recompute.")

            # Cache miss: rank 0 computes, broadcasts to all
            if _is_rank0:
                print("Computing GVF for augmentation (rank 0 only, other ranks waiting)...")
                gvf_dataset = self._create_dataloader(
                    train_data, gene_col, is_train=False,
                    defer_transforms=False,  # GVF needs CPU transforms
                    min_gene_overlap_pct=self.config.min_gene_overlap_pct,
                ).dataset
                gvf_tensor = compute_gvf_chunked(
                    gvf_dataset,
                    cache_dir=gvf_cache_dir,
                    fingerprint_hash=fingerprint_hash,
                    fingerprint_meta=fingerprint_meta,
                )
                del gvf_dataset
                gc.collect()
            else:
                gvf_tensor = torch.zeros(self.num_genes, dtype=torch.float32)

            # Broadcast GVF from rank 0 to all ranks
            gvf_tensor = gvf_tensor.to(self.device)
            dist.broadcast(gvf_tensor, src=0)
            gvf_tensor = gvf_tensor.cpu()

            if not _is_rank0:
                logging.info("Received GVF via broadcast from rank 0")

            return gvf_tensor
        else:
            # Single-GPU path
            print("Computing GVF for augmentation...")
            gvf_dataset = self._create_dataloader(
                train_data, gene_col, is_train=False,
                defer_transforms=False,  # GVF needs CPU transforms
                min_gene_overlap_pct=self.config.min_gene_overlap_pct,
            ).dataset
            gvf_tensor = compute_gvf_chunked(
                gvf_dataset,
                cache_dir=gvf_cache_dir,
                fingerprint_hash=fingerprint_hash,
                fingerprint_meta=fingerprint_meta,
            )
            del gvf_dataset
            gc.collect()
            return gvf_tensor

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save unified encoder+decoder checkpoint."""
        # Unwrap DDP if needed to get clean state_dict keys
        encoder_raw = self.encoder.module if isinstance(self.encoder, DDP) else self.encoder
        decoder_raw = self.decoder.module if isinstance(self.decoder, DDP) else self.decoder

        encoder_state = {
            k.replace('encoder.', '', 1) if k.startswith('encoder.') else k: v
            for k, v in encoder_raw.state_dict().items()
        }
        # Save full encoder model state dict for compatibility
        full_encoder_state = self._encoder_model.state_dict()
        # Remove decoder/mask_predictor keys if they still exist
        full_encoder_state = {
            k: v for k, v in full_encoder_state.items()
            if not k.startswith('decoder.') and not k.startswith('mask_predictor.')
        }

        decoder_state = decoder_raw.state_dict()

        checkpoint = {
            "encoder_state_dict": full_encoder_state,
            "decoder_state_dict": decoder_state,
            "model_state_dict": decoder_state,  # backward compat with load_decoder()
            "encoder_config": {
                "model_size": self.model_size,
                "hidden_size": self.hidden_size,
                "num_genes": self.num_genes,
            },
            "decoder_config": {
                "input_dim": self.hidden_size,
                "output_dim": self.num_genes,
                "hidden_dim": self.config.decoder_hidden_dim,
                "num_blocks": self.config.decoder_num_blocks,
                "dropout_rate": self.config.decoder_dropout,
                "se_reduction": self.config.decoder_se_reduction,
                "output_activation": getattr(self.config, "decoder_output_activation", "rectified_sigmoid"),
                "weight_tie": getattr(self.config, "decoder_weight_tie", False),
                "bottleneck_k": getattr(self.config, "decoder_bottleneck_k", 1),
                "monarch_blocks": getattr(self.config, "decoder_monarch_blocks", 0),
                "monarch_mup": getattr(self.config, "decoder_monarch_mup", False),
                "kron_rank": getattr(self.config, "decoder_kron_rank", 0),
                "glu_mode": getattr(self.config, "decoder_glu", False),
                "glu_ratio": getattr(self.config, "decoder_glu_ratio", 0.667),
                "sparsity_24": getattr(self.config, "decoder_sparsity_24", False),
                "sparsity_24_decay": getattr(self.config, "decoder_sparsity_24_decay", 2e-4),
            },
            "gene_names": list(self.encoder_gene_names),
            "gene_to_index": dict(self.gene_to_index),
            "target_sum": self.target_sum,
            "training_config": self.config.to_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "best_combined_score": self.best_combined_score,
            "parent_checkpoint_path": self.config.encoder_checkpoint,
            "is_unified_checkpoint": True,
        }

        # Stage C: persist learned nb/zinb dispersion params so downstream (e.g. virtual-cell
        # finetuning) can reuse the trained per-gene log_theta / logit_pi instead of re-initing.
        lf = getattr(self, "loss_fn", None)
        if lf is not None and hasattr(lf, "parameters") and len(list(lf.parameters())) > 0:
            checkpoint["loss_state_dict"] = lf.state_dict()
            disp = {"loss_type": getattr(self.config, "loss_type", None)}
            if hasattr(lf, "log_theta"):
                disp["log_theta"] = lf.log_theta.detach().cpu()
            if hasattr(lf, "logit_pi"):
                disp["logit_pi"] = lf.logit_pi.detach().cpu()
            checkpoint["dispersion"] = disp

        torch.save(checkpoint, path)
        if is_best:
            logging.info(f"Saved best checkpoint: {path}")

    def train(
        self,
        train_data: str,
        output_dir: str,
        val_data: Optional[str] = None,
        eval_data: Optional[List[str]] = None,
        gene_col: Optional[str] = None,
        eval_gene_col: Optional[str] = None,
    ):
        """Run the full decoder training loop."""
        config = self.config
        wandb_mod = None
        _is_rank0 = not self.is_distributed or is_main_process()

        # Wandb init (rank 0 only)
        if config.use_wandb and _is_rank0:
            try:
                import wandb
                run_name = config.wandb_run_name or f"decoder_{self.model_size}_{config.data_mode}"
                wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    tags=config.wandb_tags or ["decoder", "Occam"],
                    config={
                        **config.to_dict(),
                        "num_genes": self.num_genes,
                        "hidden_size": self.hidden_size,
                        "model_size": self.model_size,
                        "target_sum": self.target_sum,
                        "world_size": self.world_size,
                    },
                )
                wandb_mod = wandb
            except Exception as e:
                import traceback
                print(f"WARNING: wandb.init() failed: {e}")
                traceback.print_exc()
                print("Training will continue without wandb logging.")
                wandb_mod = None

        # DDP model wrapping (after optimizer creation in __init__)
        if self.is_distributed:
            if self.config.freeze_encoder:
                # Only decoder needs gradient sync; encoder is frozen
                self.decoder = DDP(
                    self.decoder, device_ids=[self.local_rank],
                    broadcast_buffers=False, gradient_as_bucket_view=True,
                )
            else:
                # Joint training: both encoder and decoder need gradient sync
                self.encoder = DDP(
                    self.encoder, device_ids=[self.local_rank],
                    broadcast_buffers=False, gradient_as_bucket_view=True,
                )
                self.decoder = DDP(
                    self.decoder, device_ids=[self.local_rank],
                    broadcast_buffers=False, gradient_as_bucket_view=True,
                )
            if _is_rank0:
                print(f"DDP: wrapped model for {self.world_size} GPUs")

        # GPU transform deferral (optim mode only)
        transforms = self._make_transforms()
        if config.data_mode == "optim":
            self._defer_transforms = True
            self._transforms = transforms
            if _is_rank0:
                print("Optim mode: transforms deferred to GPU")

        # Create output dirs
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create dataloaders
        train_chunk_ids = None
        val_chunk_ids = None
        val_loader = None

        if config.data_mode == "optim" and val_data is None:
            n_obs, n_chunks = compute_global_chunk_info(
                train_data, config.optim_chunk_size,
                gene_vocabulary=list(self.encoder_gene_names),
                gene_name_col=gene_col,
                min_gene_overlap_pct=self.config.min_gene_overlap_pct,
            )
            train_chunk_ids, val_chunk_ids = optim_train_val_chunk_split(
                total_chunks=n_chunks, val_fraction=config.val_split, seed=42,
            )
            if _is_rank0:
                print(
                    f"Chunk-based train/val split: {len(train_chunk_ids)} train, "
                    f"{len(val_chunk_ids)} val chunks"
                )

        if _is_rank0:
            print("Creating training dataloader...")
        train_loader = self._create_dataloader(
            train_data, gene_col, is_train=True, chunk_indices=train_chunk_ids,
            rank=self.rank, world_size=self.world_size,
            defer_transforms=self._defer_transforms,
            track_presence=config.gene_presence_mask,
            min_gene_overlap_pct=config.min_gene_overlap_pct,
        )

        if val_data is not None:
            if _is_rank0:
                print("Creating validation dataloader...")
            val_loader = self._create_dataloader(
                val_data, gene_col, is_train=False, batch_size=config.batch_size * 2,
                rank=self.rank, world_size=self.world_size,
                defer_transforms=self._defer_transforms,
                track_presence=config.gene_presence_mask,
                min_gene_overlap_pct=self.config.min_gene_overlap_pct,
            )
        elif val_chunk_ids is not None:
            if _is_rank0:
                print("Creating validation dataloader (chunk split)...")
            val_loader = self._create_dataloader(
                train_data, gene_col, is_train=False,
                batch_size=config.batch_size * 2, chunk_indices=val_chunk_ids,
                rank=self.rank, world_size=self.world_size,
                defer_transforms=self._defer_transforms,
                track_presence=config.gene_presence_mask,
                min_gene_overlap_pct=self.config.min_gene_overlap_pct,
            )

        # Compute GVF for augmentation
        aug_controller = None
        if not config.no_augmentation:
            # GVF cache lives in a shared directory (parent of timestamped run dir)
            # so it persists across runs. The fingerprint-based filename ensures
            # we only reuse when data + preprocessing are truly identical.
            gvf_cache_dir = config.gvf_cache_dir or os.path.join(
                os.path.dirname(output_dir), "gvf_cache"
            )
            fingerprint_hash, fingerprint_meta = _compute_data_fingerprint(
                train_data, self.target_sum, self.encoder_gene_names,
            )
            if _is_rank0:
                logging.info(
                    f"GVF data fingerprint: {fingerprint_hash} "
                    f"({fingerprint_meta['num_files']} file(s), "
                    f"{fingerprint_meta['num_genes']} genes, "
                    f"target_sum={fingerprint_meta['target_sum']})"
                )

            gvf_tensor = self._compute_gvf_distributed(
                train_data, gene_col, gvf_cache_dir,
                fingerprint_hash, fingerprint_meta,
            )

            aug_controller = AugmentationController(
                gvf_tensor=gvf_tensor,
                n_epochs=config.epochs,
                device=self.device,
                hparams_override=config.aug_hparams,
            )

        # Training loop
        if _is_rank0:
            print(f"\nStarting decoder training for {config.epochs} epochs...")
            print(f"  Freeze encoder: {config.freeze_encoder}")
            print(f"  Decoder LR: {config.lr}")
            if not config.freeze_encoder:
                print(f"  Encoder LR: {config.encoder_lr}")
            print(f"  Augmentation: {'disabled' if config.no_augmentation else 'enabled'}")
            print(f"  Decoder eval freq: {config.decoder_eval_freq}")
            print(f"  BMDB eval freq: {config.bmdb_eval_freq}")
            if config.gene_presence_mask:
                print(f"  Gene presence mask: enabled")
            if config.min_gene_overlap_pct is not None:
                print(f"  Min gene overlap: {config.min_gene_overlap_pct}%")
            if self.is_distributed:
                print(f"  DDP: {self.world_size} GPUs")
            print()

        for epoch in range(config.epochs):
            self.current_epoch = epoch

            if aug_controller is not None:
                aug_controller.update_epoch(epoch)

            # Reset dataset epoch for shuffling
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)

            # Train (all ranks participate)
            train_metrics = self.train_epoch(train_loader, aug_controller)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Validate (all ranks participate for correct all_reduce)
            val_loss = None
            if val_loader is not None and (
                (epoch + 1) % config.log_interval == 0
                or epoch == config.epochs - 1
            ):
                val_loss = self.validate(val_loader)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # best_decoder.pth is selected by combined score (see decoder-eval
                    # block below). Fall back to val-loss selection only when there are
                    # no eval files to compute a combined score from.
                    if _is_rank0 and not eval_data:
                        self.save_checkpoint(
                            os.path.join(ckpt_dir, "best_decoder.pth"), is_best=True,
                        )

            # Log (rank 0 only)
            if _is_rank0 and ((epoch + 1) % config.log_interval == 0 or epoch == config.epochs - 1):
                msg = (
                    f"Epoch {epoch + 1}/{config.epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.6f} | "
                    f"LR: {current_lr:.2e}"
                )
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)

            # Build wandb log dict (rank 0 only).
            # Deferred until after decoder eval so all metrics share the same step.
            if _is_rank0:
                wandb_log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["train_loss"],
                    "lr": current_lr,
                }
                if val_loss is not None:
                    wandb_log_dict["val_loss"] = val_loss
                if aug_controller is not None:
                    wandb_log_dict.update(aug_controller.get_wandb_dict())
                if config.adv_loss:
                    wandb_log_dict["loss/d_loss"] = train_metrics["d_loss"]
                    wandb_log_dict["loss/g_adv"] = train_metrics["g_adv"]
                    wandb_log_dict["loss/g_adv_weighted"] = config.lambda_adv * train_metrics["g_adv"]
                if config.perc_loss:
                    wandb_log_dict["loss/perc"] = train_metrics["perc_loss"]
                    wandb_log_dict["loss/perc_weighted"] = config.lambda_perc * train_metrics["perc_loss"]
                if config.adv_loss or config.perc_loss:
                    wandb_log_dict["loss/recon"] = train_metrics["train_loss"]

            # Decoder evaluation (rank 0 only; other ranks wait at next DDP forward)
            if (
                _is_rank0
                and eval_data
                and config.decoder_eval_freq > 0
                and ((epoch + 1) % config.decoder_eval_freq == 0 or epoch == config.epochs - 1)
            ):
                n_eval = len(eval_data)
                print(f"  Running decoder eval on {n_eval} file(s)...")
                eval_combined_scores = []
                for ei, eval_path in enumerate(eval_data, 1):
                    if not os.path.exists(eval_path):
                        print(f"  [{ei}/{n_eval}] File not found: {eval_path}, skipping")
                        continue
                    eval_name = Path(eval_path).stem
                    try:
                        metrics = self.evaluate_decoder_on_file(
                            eval_path, gene_col=eval_gene_col,
                        )
                        print(
                            f"  [{ei}/{n_eval}] {eval_name}: "
                            f"Pearson={metrics['pearson_mean']:.2f} "
                            f"Spearman={metrics['spearman_mean']:.2f} "
                            f"MAE={metrics['mae_mean']:.4f} "
                            f"Combined={metrics['combined_score']:.2f}"
                        )
                        eval_combined_scores.append(metrics['combined_score'])
                        for k, v in metrics.items():
                            wandb_log_dict[f"decoder_eval/{eval_name}/{k}"] = v
                    except Exception as e:
                        print(f"  [{ei}/{n_eval}] {eval_name}: Eval failed: {e}")

                # Select best_decoder.pth by mean combined score across eval files.
                if eval_combined_scores:
                    mean_combined = sum(eval_combined_scores) / len(eval_combined_scores)
                    wandb_log_dict["decoder_eval/mean_combined_score"] = mean_combined
                    if mean_combined > self.best_combined_score:
                        self.best_combined_score = mean_combined
                        self.save_checkpoint(
                            os.path.join(ckpt_dir, "best_decoder.pth"), is_best=True,
                        )
                        print(f"  ↑ New best combined score: {mean_combined:.2f} "
                              f"— saved best_decoder.pth")

            # Periodic BMDB benchmark (rank 0 only; lightweight, uses decoder weights)
            if (
                _is_rank0
                and config.bmdb_eval_freq > 0
                and ((epoch + 1) % config.bmdb_eval_freq == 0 or epoch == config.epochs - 1)
            ):
                try:
                    bmdb_results = run_decoder_benchmark(
                        self._decoder_module, self.encoder_gene_names,
                    )
                    if bmdb_results["mean_recalls"]:
                        for k, v in bmdb_results["mean_recalls"].items():
                            wandb_log_dict[f"bmdb/{k}"] = v
                        wandb_log_dict["bmdb/n_genes_benchmarked"] = bmdb_results["n_genes_benchmarked"]
                except Exception as e:
                    print(f"  BMDB benchmark failed: {e}")

            # Single wandb.log() per epoch so all metrics share the same step
            if wandb_mod is not None and _is_rank0:
                wandb_mod.log(wandb_log_dict)

        # Save final checkpoint (rank 0 only)
        if _is_rank0:
            print("\nSaving final checkpoint...")
            self.save_checkpoint(os.path.join(ckpt_dir, "final_decoder.pth"))

        # DDP cleanup: barrier, unwrap, destroy process group
        # Non-rank-0 processes exit here; rank 0 continues with post-training tasks.
        if self.is_distributed:
            dist.barrier()
            if isinstance(self.decoder, DDP):
                self.decoder = self.decoder.module
            if isinstance(self.encoder, DDP):
                self.encoder = self.encoder.module
            cleanup_distributed()

        # Post-training tasks run on rank 0 only.
        # After cleanup_distributed(), use is_rank0_env() (env-var based)
        # instead of is_main_process() (requires dist init).
        if not is_rank0_env():
            return

        # Collect all final metrics into one dict, log once at the end.
        final_log_dict = {}

        # All end-of-training evaluation (BMDB, reconstruction, cell-eval) runs on the BEST
        # checkpoint (best combined score), not the final epoch (which can be markedly worse
        # at high LR).
        best_ckpt = os.path.join(ckpt_dir, "best_decoder.pth")
        if os.path.exists(best_ckpt):
            try:
                _bd = torch.load(best_ckpt, map_location=self.device, weights_only=False)
                _sd = _bd.get("decoder_state_dict") or _bd.get("model_state_dict")
                self._decoder_module.load_state_dict(_sd)
                print(f"  Final eval: loaded BEST decoder checkpoint "
                      f"(best_combined_score={_bd.get('best_combined_score')})")
            except Exception as e:
                print(f"  Final eval: could not load best_decoder.pth ({e}); using final model")

        # Final BMDB benchmark (lightweight, no data loading)
        print("\nRunning BMDB benchmark...")
        try:
            benchmark_results = run_decoder_benchmark(
                self._decoder_module, self.encoder_gene_names,
            )
            final_log_dict["final/n_genes_benchmarked"] = benchmark_results["n_genes_benchmarked"]
            for k, v in benchmark_results["mean_recalls"].items():
                final_log_dict[f"final/{k}"] = v
        except Exception as e:
            print(f"BMDB benchmark failed: {e}")

        # Final decoder reconstruction eval
        if eval_data:
            n_eval = len(eval_data)
            print(f"\nFinal decoder reconstruction evaluation on {n_eval} file(s)...")
            print("-" * 60)
            for ei, eval_path in enumerate(eval_data, 1):
                if not os.path.exists(eval_path):
                    print(f"  [{ei}/{n_eval}] File not found: {eval_path}, skipping")
                    continue
                eval_name = Path(eval_path).stem
                print(f"  [{ei}/{n_eval}] {eval_name}")
                try:
                    metrics = self.evaluate_decoder_on_file(
                        eval_path, gene_col=eval_gene_col,
                    )
                    print(
                        f"    Pearson={metrics['pearson_mean']:.2f} "
                        f"Spearman={metrics['spearman_mean']:.2f} "
                        f"MAE={metrics['mae_mean']:.4f} MSE={metrics['mse_mean']:.4f} "
                        f"Combined={metrics['combined_score']:.2f}"
                    )
                    for k, v in metrics.items():
                        final_log_dict[f"final/decoder_eval/{eval_name}/{k}"] = v
                except Exception as e:
                    print(f"    Final eval failed: {e}")
            print("-" * 60)

        # Final cell-eval (ARC perturbation benchmark) on per-cell reconstructions
        # (runs on the BEST checkpoint loaded above).
        if eval_data and getattr(config, 'cell_eval_enabled', True):
            n_eval = len(eval_data)
            print(f"\nFinal cell-eval on {n_eval} file(s)...")
            for ei, eval_path in enumerate(eval_data, 1):
                if not os.path.exists(eval_path):
                    continue
                eval_name = Path(eval_path).stem
                print(f"  [{ei}/{n_eval}] cell-eval: {eval_name}")
                try:
                    ce_results = self.run_cell_eval_on_file(
                        eval_path, output_dir, gene_col=eval_gene_col,
                        pert_col=getattr(config, 'celleval_pert_col', 'gene'),
                        control_label=getattr(config, 'celleval_control_label', 'non-targeting'),
                    )
                    for mode, ce_metrics in ce_results.items():
                        for k, v in ce_metrics.items():
                            final_log_dict[f"final/cell_eval/{eval_name}/{mode}/{k}"] = v
                except Exception as e:
                    print(f"    cell-eval failed: {e}")

        # Single wandb.log() for all final metrics
        if wandb_mod is not None and final_log_dict:
            wandb_mod.log(final_log_dict)

        # Optional encoder benchmark
        if config.run_encoder_benchmark:
            print("\nRunning full encoder benchmark...")
            try:
                from occam.evaluation.benchmark_runner import run_end_of_training_benchmark

                # Create a lightweight wrapper matching the interface expected
                # by run_end_of_training_benchmark
                class _EncoderWrapper:
                    def __init__(self, encoder_model, device, gene_names, transforms):
                        self.model = encoder_model
                        self.device = device
                        self._gene_names = gene_names
                        self._transforms = transforms

                # Benchmark eval datasets are RAW counts → always Log1p; skip
                # NormalizePerCell only when the encoder itself skips normalization
                # (matches evaluation/benchmark_runner's skip handling).
                _bench_transforms = ([Log1p()] if getattr(self, 'skip_normalize', False)
                                     else [NormalizePerCell(self.target_sum), Log1p()])
                wrapper = _EncoderWrapper(
                    self._encoder_model,
                    self.device,
                    self.encoder_gene_names,
                    _bench_transforms,
                )
                run_end_of_training_benchmark(wrapper, wandb_module=wandb_mod)
            except Exception as e:
                print(f"Encoder benchmark failed: {e}")
                import traceback
                traceback.print_exc()

        # Finish wandb
        if wandb_mod is not None:
            wandb_mod.finish()

        print("\nDecoder training completed.")
        print(f"Checkpoints saved to: {ckpt_dir}")
        if self.best_combined_score > float('-inf'):
            print(f"Best combined score: {self.best_combined_score:.2f}")
        else:
            print(f"Best validation loss: {self.best_val_loss:.6f}")
