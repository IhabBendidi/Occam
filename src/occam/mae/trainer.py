"""The MAE trainer: pretraining and finetuning loops, eval hooks, checkpointing.

Extracted from ``training_pipeline.py:581-2242``, where it was
the research tree's trainer class. Behaviour is unchanged; only names and
imports moved. The full old-to-new symbol map is in docs/reproducing_chain_a.md.

Notes that matter when reading this:

* ``evaluation`` imports stay **lazy, inside methods**. ``occam.evaluation``
  imports ``occam.data.datasets`` and ``occam.transforms``, so importing it at
  module scope creates a cycle.
* ``train()`` and ``finetune()`` call ``dist.barrier()`` then
  ``cleanup_distributed()`` before their end-of-training benchmark, so every
  rank check after that point uses :func:`occam.ddp.is_rank0_env` rather than
  :func:`occam.ddp.is_main_process`.
* In ``optim`` data mode the NormalizePerCell and Log1p transforms are deferred
  to the GPU (``_defer_transforms``); datasets yield raw rows and
  ``_apply_gpu_transforms`` runs after ``.to(device)`` but before masking. Any
  new code path consuming optim-mode data must call it too.
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional, Tuple, Any
import pickle
from tqdm import tqdm
import warnings


from occam.model import AutoEncoder, create_occam_model
from occam.util import AverageMeter
from occam.data.datasets import (
    apply_noise, OccamAdataDataset, OccamChunkedDataset,
    optim_train_val_chunk_split, compute_global_chunk_info,
)
from occam.transforms import NormalizePerCell, Log1p, Scale, compute_gene_stats
from occam.run_dir import get_output_manager, ExperimentManager
from occam.evaluation.perturbation import tvn_on_controls, aggregate, known_relationship_benchmark, pert_signal_consistency_benchmark, DEFAULT_RECALL_THR_PAIRS

try:
    from occam.evaluation import run_end_of_training_benchmark
    _BENCHMARK_AVAILABLE = True
except ImportError:
    _BENCHMARK_AVAILABLE = False

from occam._wandb_guard import WANDB_AVAILABLE, wandb
from occam.ddp import (
    cleanup_distributed, get_rank, is_distributed, is_main_process,
    is_rank0_env, setup_distributed, setup_multi_gpu,
    _clean_state_dict, _inner_model_for_weights,
)
from occam.mae.config import PretrainConfig
from occam.mae.finetune_config import FinetuneConfig
from occam.mae.schedules import (
    MaskLossWeightSchedule, UnmaskingSchedule, build_mlw_schedule,
)

__all__ = ["OccamTrainer"]


class OccamTrainer:
    """Complete training pipeline for Occam."""
    
    def __init__(self, 
                 config: PretrainConfig,
                 device: str = 'cuda',
                 experiment_dir_name: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration object
            device: Device to use for training
        """
        self.config = config
        
        # Setup multi-GPU configuration
        self.device, self.use_parallel, self.num_gpus, self.distributed_info = setup_multi_gpu(
            device=device, 
            use_multi_gpu=config.use_multi_gpu,
            use_ddp=config.use_ddp
        )
        
        # Distributed training info
        self.is_distributed = self.distributed_info is not None
        self.use_ddp = self.is_distributed
        self.use_data_parallel = self.use_parallel and not self.is_distributed
        
        # Will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []

        # Gene name tracking (set by run_pretrain before training)
        self._gene_names = None
        self._dataset_info = None
        self._transforms = None
        self._defer_transforms = False  # When True, transforms run on GPU in train/val loops
        self.amp_enabled = getattr(config, 'enable_amp', False)

        # GPL auxiliary loss
        _gpl_w = getattr(config, 'gpl_weight', 0.0)
        if _gpl_w > 0:
            from occam.gpl_loss import GeometricPlackettLuceLoss
            self._gpl_fn = GeometricPlackettLuceLoss(reduction="mean")
        else:
            self._gpl_fn = None

        # Wandb initialization (only on main process for DDP)
        self.wandb_run = None
        # Optional experiment directory name to include in wandb notes
        self._wandb_experiment_dir_name = experiment_dir_name
        if self.config.use_wandb and WANDB_AVAILABLE and (not self.is_distributed or is_main_process()):
            self._init_wandb()
    
    def _apply_gpu_transforms(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply NormalizePerCell + Log1p + Scale on GPU tensors (optim mode)."""
        if self._transforms is None:
            return batch
        for t in self._transforms:
            if isinstance(t, NormalizePerCell):
                row_sums = batch.sum(dim=1, keepdim=True)
                row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
                batch = batch / row_sums * t.target_sum
            elif isinstance(t, Log1p):
                batch = torch.log1p(batch)
            elif isinstance(t, Scale):
                batch = (batch - t.mean.to(batch.device).unsqueeze(0)) / t.std.to(batch.device).unsqueeze(0)
        return batch

    def get_effective_batch_size(self):
        """Get the effective total batch size across all GPUs."""
        if self.use_parallel:  # Both DDP and DataParallel
            return self.config.batch_size * self.num_gpus
        else:
            return self.config.batch_size
    
    def get_batch_size_info(self):
        """Get formatted batch size information for logging."""
        if self.use_parallel:
            parallel_type = "DDP" if self.use_ddp else "DataParallel"
            return f"Per-GPU: {self.config.batch_size}, Total: {self.get_effective_batch_size()} ({parallel_type})"
        else:
            return f"{self.config.batch_size}"
        
    def _init_wandb(self):
        """Initialize wandb for experiment tracking."""
        if not WANDB_AVAILABLE:
            print("⚠️ Wandb not available - install with: pip install wandb")
            self.wandb_run = None
            return
            
        # Only initialize wandb on rank 0 in distributed training
        if self.is_distributed:
            print(f"🔄 Distributed training detected - initializing wandb on rank 0 only")
            
        # Prepare a safe config dict that reflects actual runtime values to avoid immutable changes later
        safe_config = self.config.to_dict()
        safe_config['use_multi_gpu'] = self.use_parallel
        safe_config['use_ddp'] = self.use_ddp
        safe_config['use_data_parallel'] = self.use_data_parallel
        safe_config['num_gpus'] = self.num_gpus
        safe_config['device'] = self.device
        safe_config['batch_size_total'] = self.get_effective_batch_size()
        notes_str = None
        try:
            if isinstance(self._wandb_experiment_dir_name, str) and len(self._wandb_experiment_dir_name) > 0:
                notes_str = f"exp: {self._wandb_experiment_dir_name}"
        except Exception:
            notes_str = None

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                timeout = 90 + 30 * (attempt - 1)
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    tags=self.config.wandb_tags,
                    config=safe_config,
                    notes=notes_str,
                    settings=wandb.Settings(init_timeout=timeout)
                    )
                if self.is_distributed:
                    print(f"✅ Wandb initialized successfully on rank 0 (GPU {self.distributed_info[2]})")
                else:
                    print("✅ Wandb initialized successfully")
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * attempt
                    print(f"⚠️ Wandb init attempt {attempt}/{max_retries} failed: {e}")
                    print(f"   Retrying in {wait}s (next timeout: {90 + 30 * attempt}s)...")
                    import time
                    time.sleep(wait)
                else:
                    print(f"⚠️ Wandb initialization failed after {max_retries} attempts: {e}")
                    print(f"💡 Training will not continue without wandb logging")
                    exit(1)
        
    def _setup_model(self, num_genes: int):
        """Setup model based on configuration."""
        # Prepare model arguments
        model_args = {
            'num_genes': num_genes,
            'model_size': self.config.model_size,
            'dropout': self.config.dropout,
            'masked_data_weight': self.config.masked_data_weight,
            'mask_loss_weight': self.config.mask_loss_weight,
            'reconstruction_loss': self.config.reconstruction_loss,
            'w_mse': getattr(self.config, 'w_mse', 1.0),
            'w_pois': getattr(self.config, 'w_pois', 0.1),
            'clamp_max_log1p': getattr(self.config, 'clamp_max_log1p', 5.5),
            'recon_loss_scale': getattr(self.config, 'recon_loss_scale', 1.0),
        }
        
        # Override hidden size if specified
        if self.config.hidden_size is not None:
            model_args['hidden_size'] = self.config.hidden_size

        # Override MAE decoder width if specified; else MODEL_CONFIGS[model_size] decides
        if getattr(self.config, 'mae_decoder_hidden', None) is not None:
            model_args['decoder_hidden'] = self.config.mae_decoder_hidden

        # Enable TF32 tensor cores for float32 matmuls (Ampere+ GPUs)
        torch.set_float32_matmul_precision('high')

        # Create model and move to primary device
        self.model = create_occam_model(**model_args).to(self.device)

        # Record the *resolved* MAE decoder width so the checkpoint config is explicit
        # and reloads never have to guess (see model.resolve_decoder_hidden).
        self.config.mae_decoder_hidden = int(self.model.decoder_hidden)

        # Compile model if requested (before DDP wrapping)
        if getattr(self.config, 'compile', False):
            if not self.is_distributed or is_main_process():
                print("⚡ Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Wrap with appropriate parallel wrapper
        if self.use_ddp:
            if is_main_process():
                print(f"🔀 Wrapping model with DistributedDataParallel for {self.num_gpus} GPUs")
            _find_unused = getattr(self.config, 'freeze_mask_predictor', False)
            self.model = DDP(self.model, device_ids=[self.distributed_info[2]],
                            broadcast_buffers=False, gradient_as_bucket_view=True,
                            find_unused_parameters=_find_unused)
        elif self.use_data_parallel:
            print(f"🔀 Wrapping model with DataParallel for {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        
        # Log model info to wandb
        if self.wandb_run is not None:
            # For parallel models (DDP or DataParallel), access the underlying module
            base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
            update_dict = {
                'model_parameters': base_model.get_parameter_count(),
                'model_hidden_size': base_model.hidden_size,
                'batch_size_per_gpu': self.config.batch_size,
            }
            # Log per-file dataset breakdown if available (optim folder mode)
            if hasattr(self, '_folder_summary') and self._folder_summary is not None:
                update_dict['training_data'] = self._folder_summary
            wandb.config.update(update_dict, allow_val_change=True)
        
        self._setup_optimizer_and_scheduler(self.config.epochs)

    def _setup_optimizer_and_scheduler(self, num_epochs: int):
        """Create optimizer and LR scheduler from current config."""
        # Freeze mask predictor if requested
        if self.config.freeze_mask_predictor:
            base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
            for param in base_model.mask_predictor.parameters():
                param.requires_grad = False
        
        # Create optimizer with only trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt_cls = torch.optim.AdamW if self.config.optimizer == 'adamw' else torch.optim.Adam
        self.optimizer = opt_cls(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        warmup_epochs = getattr(self.config, 'warmup_epochs', 0)
        main_scheduler = None
        if self.config.lr_scheduler == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs - warmup_epochs
            )
        elif self.config.lr_scheduler == 'step':
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

        if main_scheduler is not None and warmup_epochs > 0:
            start_factor = getattr(self.config, 'warmup_start_factor', 1e-3)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = main_scheduler
    
    def _get_mask_probabilities(self, num_genes: int):
        """Get masking probabilities for genes."""
        if self.config.variable_masking:
            # Could implement variable masking strategy here
            # For now, return uniform probability
            return [self.config.mask_probability] * num_genes
        else:
            return [self.config.mask_probability] * num_genes
    
    def train_epoch(self, train_loader: DataLoader,
                    unmasking_schedule: 'Optional[UnmaskingSchedule]' = None,
                    ft_epoch: int = 0) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            unmasking_schedule: Optional schedule that overrides mask_probability
                per batch (used during finetuning).
            ft_epoch: 0-indexed finetuning epoch (passed to unmasking_schedule).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        # Initialize component metric aggregators
        metric_keys = [
            'mse_pre', 'mse_post', 'poisson_pre', 'poisson_post',
            'recon_pre', 'recon_post', 'mask_pre', 'mask_post',
            'gpl_pre',
        ]
        metric_sums = {k: 0.0 for k in metric_keys}
        metric_counts = {k: 0.0 for k in metric_keys}

        # Get mask probabilities
        base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
        mask_probas = self._get_mask_probabilities(base_model.num_genes)

        _is_rank0 = not self.is_distributed or is_main_process()
        if _is_rank0:
            progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        else:
            progress_bar = train_loader

        _track_presence = getattr(self.config, 'gene_presence_mask', False)
        for batch_idx, batch in enumerate(progress_bar):
            gene_present = None
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0].to(self.device)
                if _track_presence and batch[1] is not None:
                    gene_present = batch[1].to(self.device)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            if self._defer_transforms:
                x = self._apply_gpu_transforms(x)
            # Presence-as-input: add scaled presence to expression in log1p space
            if gene_present is not None:
                x = x + gene_present * self.config.presence_shift
            target = x  # target is the shifted signal (or unshifted if no presence)

            # Per-batch mask probability override from unmasking schedule
            if unmasking_schedule is not None:
                current_mask_prob = unmasking_schedule.get_mask_probability(ft_epoch, batch_idx)
                mask_probas = [current_mask_prob] * base_model.num_genes
            else:
                current_mask_prob = self.config.mask_probability

            # Apply corruption
            corruption_mode = self.config.corruption_mode
            if corruption_mode == 'none':
                x_corrupted = x.clone()
                mask = torch.zeros_like(x)
                current_mask_prob = 0.0
            else:
                x_corrupted, mask = apply_noise(x, mask_probas,
                                                mode=corruption_mode,
                                                gaussian_sigma=self.config.gaussian_sigma)

            # Forward pass
            self.optimizer.zero_grad()
            # Call through the DDP/DP wrapper so gradient sync hooks fire
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp_enabled):
                latent, loss = self.model(x_corrupted, target, mask, mask_probability=current_mask_prob)
                # GPL auxiliary loss
                if self._gpl_fn is not None:
                    base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
                    _recon = base_model._last_reconstruction
                    _gpl_mask = gene_present.bool() if gene_present is not None else None
                    _gpl_loss = self._gpl_fn(_recon, target, mask=_gpl_mask)
                    loss = loss + self.config.gpl_weight * _gpl_loss
            # Collect per-batch component metrics if available
            base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
            try:
                last_metrics = getattr(base_model, '_last_loss_metrics', None)
                if isinstance(last_metrics, dict):
                    for k in metric_keys:
                        v = last_metrics.get(k, None)
                        if v is not None and not (isinstance(v, float) and (v != v)):
                            metric_sums[k] += float(v)
                            metric_counts[k] += 1.0
                if self._gpl_fn is not None:
                    metric_sums['gpl_pre'] += _gpl_loss.item()
                    metric_counts['gpl_pre'] += 1.0
            except Exception:
                pass

            if batch_idx < 5 and _is_rank0:  # Debug loss
                print(f"Loss: {loss.item():.6f}, requires_grad: {loss.requires_grad}")
                if loss.item() == 0.0:
                    print("⚠️ WARNING: Loss is exactly zero!")
                print("=" * 50)

            # Check for problematic loss values
            if torch.isnan(loss) or torch.isinf(loss):
                if _is_rank0:
                    print(f"❌ Invalid loss detected at batch {batch_idx}: {loss.item()}")
                break

            # Backward pass
            loss.backward()

            # Loss diagnostics: gradient norms (once per epoch, first batch only)
            if getattr(self.config, 'loss_diagnostics', False) and batch_idx == 0 and _is_rank0:
                try:
                    from occam.losses import compute_gradient_diagnostics
                    self._diag_grad = compute_gradient_diagnostics(self.model)
                except Exception:
                    pass

            # Gradient clipping if specified
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Early break for debugging if all losses are zero
            if batch_idx > 10 and total_loss == 0.0:
                if _is_rank0:
                    print("❌ All losses are zero - stopping for debugging")
                break

        # Step-based schedulers are now stepped in the main training loop
        
        # Compute global (across ranks) average loss and metrics for logging
        if self.is_distributed:
            # Reduce loss
            stats = torch.tensor([total_loss, float(num_batches)], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_total_loss = stats[0].item()
            global_num_batches = int(stats[1].item())
            avg_loss = global_total_loss / max(global_num_batches, 1)
            # Reduce metrics
            sums_tensor = torch.tensor([metric_sums[k] for k in metric_keys], dtype=torch.float64, device=self.device)
            cnts_tensor = torch.tensor([metric_counts[k] for k in metric_keys], dtype=torch.float64, device=self.device)
            dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnts_tensor, op=dist.ReduceOp.SUM)
            metric_avgs = {}
            for idx, k in enumerate(metric_keys):
                c = cnts_tensor[idx].item()
                metric_avgs[k] = (sums_tensor[idx].item() / c) if c > 0 else None
        else:
            avg_loss = total_loss / max(num_batches, 1)
            metric_avgs = {k: (metric_sums[k] / metric_counts[k]) if metric_counts[k] > 0 else None for k in metric_keys}
        # Expose epoch metrics
        try:
            self._train_metrics_epoch = metric_avgs
        except Exception:
            pass
        return avg_loss
    
    def validate(self, val_loader: DataLoader,
                 mask_probability_override: Optional[float] = None) -> float:
        """Validate the model.

        Args:
            val_loader: Validation data loader.
            mask_probability_override: If provided, use this mask probability
                instead of ``self.config.mask_probability`` (used during
                finetuning to validate on unmasked inputs).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        # Initialize component metric aggregators
        metric_keys = [
            'mse_pre', 'mse_post', 'poisson_pre', 'poisson_post',
            'recon_pre', 'recon_post', 'mask_pre', 'mask_post',
            'gpl_pre',
        ]
        metric_sums = {k: 0.0 for k in metric_keys}
        metric_counts = {k: 0.0 for k in metric_keys}

        # Get mask probabilities
        base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
        effective_mask_prob = mask_probability_override if mask_probability_override is not None else self.config.mask_probability
        mask_probas = [effective_mask_prob] * base_model.num_genes

        _track_presence = getattr(self.config, 'gene_presence_mask', False)
        with torch.no_grad():
            for batch in val_loader:
                gene_present = None
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x = batch[0].to(self.device)
                    if _track_presence and batch[1] is not None:
                        gene_present = batch[1].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                if self._defer_transforms:
                    x = self._apply_gpu_transforms(x)
                # Presence-as-input: add scaled presence to expression in log1p space
                if gene_present is not None:
                    x = x + gene_present * self.config.presence_shift
                target = x

                # Apply corruption
                corruption_mode = self.config.corruption_mode
                if corruption_mode == 'none':
                    x_corrupted = x.clone()
                    mask = torch.zeros_like(x)
                    effective_mask_prob = 0.0
                else:
                    x_corrupted, mask = apply_noise(x, mask_probas,
                                                    mode=corruption_mode,
                                                    gaussian_sigma=self.config.gaussian_sigma)

                # Forward pass
                # Access underlying model for DataParallel/DDP compatibility
                model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp_enabled):
                    latent, loss = model.loss_mask(x_corrupted, target, mask, mask_probability=effective_mask_prob)
                    # GPL auxiliary loss
                    if self._gpl_fn is not None:
                        _recon = model._last_reconstruction
                        _gpl_mask = gene_present.bool() if gene_present is not None else None
                        _gpl_loss_val = self._gpl_fn(_recon, target, mask=_gpl_mask)
                        loss = loss + self.config.gpl_weight * _gpl_loss_val
                # Collect per-batch component metrics if available
                try:
                    last_metrics = getattr(model, '_last_loss_metrics', None)
                    if isinstance(last_metrics, dict):
                        for k in metric_keys:
                            v = last_metrics.get(k, None)
                            if v is not None and not (isinstance(v, float) and (v != v)):
                                metric_sums[k] += float(v)
                                metric_counts[k] += 1.0
                    if self._gpl_fn is not None:
                        metric_sums['gpl_pre'] += _gpl_loss_val.item()
                        metric_counts['gpl_pre'] += 1.0
                except Exception:
                    pass

                # Loss diagnostics: prediction/reconstruction analysis (first batch only)
                if getattr(self.config, 'loss_diagnostics', False) and num_batches == 0:
                    _is_rank0_val = not self.is_distributed or is_main_process()
                    if _is_rank0_val:
                        try:
                            from occam.losses import compute_loss_diagnostics
                            _recon_diag = getattr(model, '_last_reconstruction', None)
                            if _recon_diag is not None:
                                self._diag_loss = compute_loss_diagnostics(_recon_diag, target, model)
                        except Exception:
                            pass

                total_loss += loss.item()
                num_batches += 1

        # Compute global (across ranks) average validation loss and metrics for logging
        if self.is_distributed:
            stats = torch.tensor([total_loss, float(num_batches)], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_total_loss = stats[0].item()
            global_num_batches = int(stats[1].item())
            avg_loss = global_total_loss / max(global_num_batches, 1)
            # Reduce metrics
            sums_tensor = torch.tensor([metric_sums[k] for k in metric_keys], dtype=torch.float64, device=self.device)
            cnts_tensor = torch.tensor([metric_counts[k] for k in metric_keys], dtype=torch.float64, device=self.device)
            dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnts_tensor, op=dist.ReduceOp.SUM)
            metric_avgs = {}
            for idx, k in enumerate(metric_keys):
                c = cnts_tensor[idx].item()
                metric_avgs[k] = (sums_tensor[idx].item() / c) if c > 0 else None
        else:
            avg_loss = total_loss / max(num_batches, 1)
            metric_avgs = {k: (metric_sums[k] / metric_counts[k]) if metric_counts[k] > 0 else None for k in metric_keys}
        # Expose epoch metrics
        try:
            self._val_metrics_epoch = metric_avgs
        except Exception:
            pass
        return avg_loss

    def _run_external_evaluation(self, epoch: int, save_dir: str):
        """Run evaluation on a fixed external dataset (rank 0 only).

        Returns:
            Tuple of (metric1, metric2) on success, or None on failure/skip.
        """
        try:
            import scanpy as sc
        except Exception as e:
            print(f"[Eval] scanpy import failed: {e}. Skipping evaluation.")
            return

        if self._gene_names is None:
            print("[Eval] No gene names set on trainer; cannot align genes. Skipping evaluation.")
            return

        eval_path = self.config.eval_h5ad_path
        if not eval_path or not os.path.exists(eval_path):
            print(f"[Eval] Evaluation file not found: {eval_path}. Skipping evaluation.")
            return

        print("\n========== EVALUATION START ==========")
        print(f"[Eval] Epoch {epoch+1} | File: {eval_path}")

        was_training = self.model.training
        try:
            return self._run_external_evaluation_inner(epoch, save_dir, eval_path, sc)
        finally:
            if was_training:
                self.model.train()
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("========== EVALUATION END =========\n")

    def _run_external_evaluation_inner(self, epoch, save_dir, eval_path, sc):
        """Inner evaluation logic, called within a try/finally that restores train mode."""
        import time as _time
        import psutil as _psutil
        _proc = _psutil.Process()
        def _mem_gb():
            return _proc.memory_info().rss / 1024**3
        _t0 = _time.time()
        _phase_t = _t0
        def _log_phase(name):
            nonlocal _phase_t
            now = _time.time()
            elapsed = now - _phase_t
            total = now - _t0
            gpu_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            gpu_res_mb = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
            print(f"[Eval⏱] {name}: {elapsed:.1f}s (total {total:.1f}s) | RSS {_mem_gb():.1f}GB | GPU alloc {gpu_mb:.0f}MB res {gpu_res_mb:.0f}MB")
            _phase_t = now

        print(f"[Eval⏱] START | RSS {_mem_gb():.1f}GB", flush=True)
        try:
            adata = sc.read_h5ad(eval_path)
        except Exception as e:
            print(f"[Eval] Failed to read h5ad: {e}. Skipping evaluation.")
            return
        _log_phase("read_h5ad")

        print(f"[Eval] AnnData loaded: {adata.n_obs} cells x {adata.n_vars} genes")

        # Find the best gene column by overlap with training gene names
        train_genes_upper = set(str(g).strip().upper() for g in self._gene_names)
        var_cols = list(adata.var.columns)
        candidate_cols = ['gene_name', 'gene_symbol', 'gene_symbols', 'symbol',
                          'feature_name', 'feature_names', 'Gene', 'ensembl_gene_id', 'gene_id']
        ds_info = self._dataset_info or {}
        train_col = ds_info.get('gene_name_col')
        if train_col and train_col != 'index' and train_col not in candidate_cols:
            candidate_cols.insert(0, train_col)

        best_col, best_overlap = None, -1
        for col in candidate_cols:
            if col not in var_cols:
                continue
            try:
                eval_names = set(str(x).strip().upper() for x in adata.var[col].tolist())
            except Exception:
                continue
            overlap = len(train_genes_upper & eval_names)
            print(f"[Eval] Overlap with '{col}': {overlap}")
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col

        used_gene_col = best_col if best_overlap > 0 else None
        if used_gene_col is None:
            print("[Eval] No positive-overlap gene column found; falling back to auto-detect")

        # Save metadata before freeing adata (OccamAdataDataset reads h5ad again)
        metadata = adata.obs.copy()
        del adata
        import gc as _gc_early
        _gc_early.collect()
        _log_phase("gene_detection_and_metadata_copy")

        print("[Eval] Creating aligned evaluation dataset...")
        try:
            _skip_norm = getattr(self.config, 'skip_normalize', False)
            if _skip_norm:
                eval_transforms = [Log1p()]
            elif self._transforms is not None:
                eval_transforms = self._transforms
            else:
                eval_transforms = [NormalizePerCell(1e4), Log1p()]
            eval_ds = OccamAdataDataset(
                adata_path=eval_path,
                mode="in_memory",
                transforms=eval_transforms,
                gene_name_col=used_gene_col,
                gene_vocabulary=self._gene_names,
            )
        except Exception as e:
            print(f"[Eval] Failed to create eval dataset: {e}. Skipping evaluation.")
            return
        _log_phase("create_dataset")

        print(f"[Eval] Eval dataset: {len(eval_ds)} cells, {eval_ds.num_genes} genes")

        batch_size = self.config.eval_batch_size or (self.config.batch_size * 2)
        try:
            loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                                pin_memory=False)
        except Exception as e:
            print(f"[Eval] Failed to create DataLoader: {e}. Skipping evaluation.")
            return

        # Free stale CUDA cache + force CPU memory return before heavy allocation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        import ctypes
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

        print(f"[Eval] Generating embeddings on device {self.device} with batch_size={batch_size} ...")
        base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
        self.model.eval()
        _gpm = getattr(self.config, 'gene_presence_mask', False)
        _pshift = getattr(self.config, 'presence_shift', 1.0)
        embeddings_list = []
        try:
            _batch_t = _time.time()
            with torch.no_grad():
                for b_idx, batch in enumerate(loader):
                    bx = batch[0].to(self.device)
                    if _gpm:
                        bx = bx + _pshift  # All genes present in single-file eval
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp_enabled):
                        be = base_model.feature(bx)
                    embeddings_list.append(be.detach().cpu().numpy())
                    if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(loader):
                        _elapsed = _time.time() - _batch_t
                        print(f"[Eval]   Processed {b_idx+1}/{len(loader)} batches ({_elapsed:.1f}s for last 10)", flush=True)
                        _batch_t = _time.time()
        except Exception as e:
            print(f"[Eval] Embedding generation failed: {e}. Skipping evaluation.")
            return

        occam_embeddings = np.concatenate(embeddings_list, axis=0)
        _log_phase("generate_embeddings")
        print(f"[Eval] Embeddings shape: {occam_embeddings.shape}")

        print("[Eval] Applying TVN alignment on controls...")
        try:
            pert_col_tvn = "is_control"
            batch_col = "dataset_batch_num"
            control_key = True
            tvn_embeddings = tvn_on_controls(
                embeddings=occam_embeddings,
                metadata=metadata,
                pert_col=pert_col_tvn,
                control_key=control_key,
                batch_col=batch_col,
            )
            print(f"[Eval] TVN embeddings shape: {tvn_embeddings.shape}")
        except Exception as e:
            print(f"[Eval] TVN failed: {e}. Skipping evaluation.")
            return
        _log_phase("tvn")

        print("[Eval] Computing perturbation signal consistency benchmark...")
        try:
            pert_colname = "gene_name"
            control_col = "is_control"
            control_key = True
            vc = metadata[pert_colname].value_counts()
            keys_to_drop = vc[vc == 1].index.tolist()
            neg_ctrl_perts = (
                metadata.loc[metadata[control_col] == control_key, pert_colname].unique().tolist()
            )
            np.random.seed(42)
            cons_res = pert_signal_consistency_benchmark(
                tvn_embeddings,
                metadata,
                pert_col=pert_colname,
                neg_ctrl_perts=neg_ctrl_perts,
                keys_to_drop=keys_to_drop,
                stat="cossim",
                num_null_cells=min(50000, tvn_embeddings.shape[0])
            )
            denom = np.sum(~pd.isna(cons_res["pval"]))
            metric1 = float(np.sum(cons_res.pval <= 0.01)) / float(denom if denom > 0 else 1) * 100.0
            print(f"[Eval] Metric1 (percent significant p<=0.01): {metric1:.3f}")
        except Exception as e:
            print(f"[Eval] Consistency benchmark failed: {e}")
            metric1 = float('nan')
        _log_phase("consistency_benchmark")

        print("[Eval] Computing known-relationship benchmark (recall_0.05_0.95 across sources)...")
        try:
            all_controls = []
            map_data = aggregate(tvn_embeddings, metadata, pert_col=pert_colname, keys_to_remove=all_controls)
            env_bmdb = os.environ.get('OCCAM_BMDB_DIR') or os.environ.get('OCCAM_BMDB_DIR')
            candidates = [env_bmdb, os.path.join(os.getcwd(), 'bmdb_database'), os.path.join(os.path.dirname(__file__), 'bmdb_database')]
            benchmark_dir = None
            for c in candidates:
                if c and os.path.isdir(c):
                    benchmark_dir = c
                    break
            if benchmark_dir is None:
                raise FileNotFoundError("bmdb_database directory not found. Set OCCAM_BMDB_DIR (or OCCAM_BMDB_DIR) or ensure ./bmdb_database exists.")
            sources = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"]
            metrics_df = known_relationship_benchmark(
                map_data,
                pert_col=pert_colname,
                benchmark_sources=sources,
                recall_thr_pairs=[(0.05, 0.95)],
                benchmark_data_dir=benchmark_dir,
                min_req_entity_cnt=20,
                log_stats=True,
            )
            if metrics_df is not None and not metrics_df.empty and 'recall_0.05_0.95' in metrics_df.columns:
                metric2 = float(metrics_df['recall_0.05_0.95'].sum()) / 5.0 * 100.0
            else:
                metric2 = float('nan')
            print(f"[Eval] Metric2 (mean recall@5-95 across sources): {metric2:.5f}")
        except Exception as e:
            print(f"[Eval] Relationship benchmark failed: {e}")
            metric2 = float('nan')
        _log_phase("relationship_benchmark")

        try:
            if self.wandb_run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'eval_metric_consistency_pct': metric1,
                    'eval_metric_knownrels_recall_mean': metric2,
                })
        except Exception as e:
            print(f"[Eval] Wandb logging failed: {e}")

        # Eagerly free large objects before returning
        import gc as _gc
        del eval_ds, loader, embeddings_list, occam_embeddings
        del tvn_embeddings, metadata
        try:
            del cons_res, map_data, metrics_df
        except NameError:
            pass
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import ctypes as _ctypes
            _ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass
        _log_phase("cleanup")
        print(f"[Eval⏱] TOTAL eval time: {_time.time() - _t0:.1f}s | RSS after cleanup {_mem_gb():.1f}GB", flush=True)

        return (metric1, metric2)
    
    def extract_embeddings(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from the trained model."""
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    x = batch[0].to(self.device)
                    y = batch[1] if len(batch) > 1 else torch.zeros(x.shape[0])
                else:
                    x = batch.to(self.device)
                    y = torch.zeros(x.shape[0])
                embedding = base_model.feature(x)
                embeddings.append(embedding.cpu().numpy())
                labels.append(y.numpy())

        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)

        return embeddings, labels
    
    def create_data_loader(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0):
        """Create a DataLoader with proper distributed settings.

        The import is deferred because ``occam.mae.stages`` imports this module;
        a top-level import here would be a cycle.
        """
        from occam.mae.stages import create_dataloader

        return create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            is_distributed=self.is_distributed,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              save_dir: str = './checkpoints',
              save_best: bool = True,
              save_final: bool = True,
              experiment_description: str = "",
              use_output_manager: bool = True,
              existing_experiment: Optional[ExperimentManager] = None,
              skip_wandb_finish: bool = False) -> Dict[str, Any]:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            save_dir: Directory to save checkpoints (ignored if use_output_manager=True)
            save_best: Whether to save best model
            save_final: Whether to save final model
            experiment_description: Description for the experiment
            use_output_manager: Whether to use centralized output management
            skip_wandb_finish: If True, do not call wandb.finish() at the end
                (used in tandem mode so finetuning can continue logging)

        Returns:
            Training history dictionary
        """
        # Setup output management (rank 0 only)
        if use_output_manager and (not self.is_distributed or is_main_process() or is_rank0_env()):
            # Use provided experiment if available; otherwise create a new one
            if existing_experiment is not None:
                experiment = existing_experiment
            else:
                output_manager = get_output_manager()
                experiment = output_manager.create_experiment(
                    experiment_type="training",
                    description=experiment_description,
                    config=self.config.to_dict(),
                    tags=["Occam", "training"]
                )
            save_dir = str(experiment.checkpoints_dir)
            experiment.log_progress("Training experiment initialized")

            # Update wandb notes with the experiment directory name if possible
            try:
                if self.wandb_run is not None and experiment is not None:
                    exp_dir_name = os.path.basename(str(experiment.experiment_dir))
                    prev_notes = getattr(self.wandb_run, 'notes', '') or ''
                    if isinstance(prev_notes, str):
                        addition = f"exp: {exp_dir_name}"
                        if addition not in prev_notes:
                            new_notes = f"{prev_notes} | {addition}" if prev_notes else addition
                            try:
                                self.wandb_run.notes = new_notes
                            except Exception:
                                pass
            except Exception as _e:
                print(f"⚠️ Wandb notes update failed: {_e}")
        else:
            experiment = None
            # Only rank0 should create directories
            if not self.is_distributed or is_main_process() or is_rank0_env():
                os.makedirs(save_dir, exist_ok=True)

        # Expose experiment handle on the trainer for downstream consumers (e.g., CLI prints)
        try:
            self.experiment = experiment
        except Exception:
            pass
        
        # Setup model (if not already done)
        if self.model is None:
            # Infer number of genes from data loader
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)):
                num_genes = sample_batch[0].shape[1]
            else:
                num_genes = sample_batch.shape[1]
            self._setup_model(num_genes)
        
        if not self.is_distributed or is_main_process() or is_rank0_env():
            print(f"Starting training for {self.config.epochs} epochs")
        # For DataParallel models, access the underlying module
        base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
        if not self.is_distributed or is_main_process() or is_rank0_env():
            print("\n" + "=" * 70)
            print("MODEL SUMMARY")
            print("=" * 70)
            base_model.print_architecture()
            print("=" * 70)
            print(f"Device: {self.device}")
            print(f"Batch size: {self.get_batch_size_info()}")
            print(f"Validation split: {'External' if val_loader else 'None provided'}")
            print("=" * 70 + "\n")
        
        # Print configuration
        self.config.print_config()
        
        early_stopping_counter = 0
        mlw_schedule = build_mlw_schedule(self.config)
        if mlw_schedule is not None and (not self.is_distributed or is_main_process() or is_rank0_env()):
            print(f"  MLW schedule: {mlw_schedule.start} -> {mlw_schedule.end} "
                  f"(warmup {mlw_schedule.warmup_fraction:.0%}, cooldown {mlw_schedule.cooldown_fraction:.0%} "
                  f"of {self.config.epochs} epochs)")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            if mlw_schedule is not None:
                base_model.mask_loss_weight = mlw_schedule.get_mask_loss_weight(epoch)

            # Ensure proper shuffling with DistributedSampler or iterable datasets
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            elif hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if val_loader is not None and hasattr(val_loader, "sampler") and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            # Periodic evaluation BEFORE training this epoch (includes epoch 0)
            if self.config.eval_enabled and (epoch % self.config.eval_frequency == 0) and epoch != 0:
                if self.is_distributed:
                    dist.barrier()
                if (not self.is_distributed) or is_main_process() or is_rank0_env():
                    try:
                        print(f"[Eval] Running external evaluation at epoch {epoch+1} on rank0 only...")
                        eval_result = self._run_external_evaluation(epoch, save_dir)
                        # Save checkpoint with the epoch label used for this eval
                        ckpt_path = os.path.join(save_dir, f'checkpoint_eval_epoch_{epoch + 1}.pth')
                        self.save_checkpoint(ckpt_path)
                        print(f"💾 Eval checkpoint saved: {ckpt_path}")
                        # Track best eval performance
                        if eval_result is not None:
                            m1, m2 = eval_result
                            eval_score = (m1 if not np.isnan(m1) else 0.0) + (m2 if not np.isnan(m2) else 0.0)
                            if not hasattr(self, '_best_eval_score') or eval_score > self._best_eval_score:
                                self._best_eval_score = eval_score
                                self._best_eval_epoch = epoch + 1
                                import shutil
                                best_perf_path = os.path.join(save_dir, 'best_perf_checkpoint.pth')
                                shutil.copy2(ckpt_path, best_perf_path)
                                print(f"🏆 New best eval performance (score={eval_score:.3f}) at epoch {epoch + 1}")
                    except Exception as e:
                        print(f"[Eval] Unexpected error during evaluation at epoch {epoch+1}: {e}")
                    if self.config.scib_eval_enabled:
                        try:
                            from occam.evaluation.scib_metrics import run_scib_fast
                            run_scib_fast(self, wandb_module=wandb if WANDB_AVAILABLE else None)
                        except Exception as e:
                            print(f"[scib] Fast eval failed at epoch {epoch+1}: {e}")
                if self.is_distributed:
                    dist.barrier()

            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate if validation loader provided and it's validation time
            val_loss = None
            if val_loader is not None and (epoch + 1) % self.config.validation_freq == 0:
                val_loss = self.validate(val_loader)
            
            # Step scheduler after epoch
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to wandb
            if self.wandb_run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'learning_rate': current_lr
                }
                if mlw_schedule is not None:
                    log_dict['mask_loss_weight'] = base_model.mask_loss_weight
                if val_loss is not None:
                    log_dict['val_loss'] = val_loss
                # Include component losses (averaged across epoch and ranks)
                try:
                    train_metrics = getattr(self, '_train_metrics_epoch', None)
                    if isinstance(train_metrics, dict):
                        for k, v in train_metrics.items():
                            if v is not None:
                                log_dict[f'train_{k}'] = float(v)
                except Exception:
                    pass
                try:
                    val_metrics = getattr(self, '_val_metrics_epoch', None)
                    if isinstance(val_metrics, dict):
                        for k, v in val_metrics.items():
                            if v is not None:
                                log_dict[f'val_{k}'] = float(v)
                except Exception:
                    pass
                # Loss diagnostics (gradient norms + reconstruction analysis)
                if getattr(self.config, 'loss_diagnostics', False):
                    try:
                        diag_grad = getattr(self, '_diag_grad', None)
                        if isinstance(diag_grad, dict):
                            for k, v in diag_grad.items():
                                log_dict[f'diagnostics/{k}'] = v
                        diag_loss = getattr(self, '_diag_loss', None)
                        if isinstance(diag_loss, dict):
                            for k, v in diag_loss.items():
                                log_dict[f'diagnostics/{k}'] = v
                    except Exception:
                        pass
                wandb.log(log_dict)

            # Log progress (only rank 0 prints). Losses are already global averages.
            log_msg = f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            log_msg += f" - LR: {current_lr:.6f}"
            if mlw_schedule is not None:
                log_msg += f" - MLW: {base_model.mask_loss_weight:.4f}"
            if not self.is_distributed or is_main_process() or is_rank0_env():
                print(log_msg)

            # Log to experiment if using output manager
            if experiment:
                experiment.log_progress(log_msg)
            
            # Save training history
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }
            self.training_history.append(epoch_info)
            
            # Save best model based on validation loss (or train loss if no validation)
            current_loss = val_loss if val_loss is not None else train_loss
            if save_best and current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                best_checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                # Only rank0 writes checkpoints
                if not self.is_distributed or is_main_process() or is_rank0_env():
                    self.save_checkpoint(best_checkpoint_path)
                    print(f"💾 New best model saved (val_loss: {current_loss:.4f})")
                early_stopping_counter = 0
                
                # Log to experiment if using output manager
                if experiment:
                    experiment.log_progress(f"New best model saved with loss: {current_loss:.4f}")
                    experiment.update_metadata({"best_val_loss": current_loss, "best_epoch": epoch + 1})
                
                # Log best model to wandb
                if self.wandb_run is not None:
                    wandb.log({'best_val_loss': current_loss, 'best_epoch': epoch + 1})
            else:
                early_stopping_counter += 1
            
            # Early stopping check
            if (self.config.early_stopping_patience is not None and 
                early_stopping_counter >= self.config.early_stopping_patience):
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
            

        # Final evaluation and checkpoint on the last model
        if save_final:
            if not self.is_distributed or is_main_process() or is_rank0_env():
                final_epoch = len(self.training_history)
                self.save_checkpoint(os.path.join(save_dir, 'final_model.pth'))
                if self.config.eval_enabled:
                    print(f"[Eval] Running final evaluation on last model (epoch {final_epoch})...")
                    eval_result = self._run_external_evaluation(final_epoch - 1, save_dir)
                    ckpt_path = os.path.join(save_dir, f'checkpoint_eval_epoch_{final_epoch}.pth')
                    self.save_checkpoint(ckpt_path)
                    print(f"💾 Final eval checkpoint saved: {ckpt_path}")
                    if eval_result is not None:
                        m1, m2 = eval_result
                        eval_score = (m1 if not np.isnan(m1) else 0.0) + (m2 if not np.isnan(m2) else 0.0)
                        if not hasattr(self, '_best_eval_score') or eval_score > self._best_eval_score:
                            self._best_eval_score = eval_score
                            self._best_eval_epoch = final_epoch
                            import shutil
                            shutil.copy2(ckpt_path, os.path.join(save_dir, 'best_perf_checkpoint.pth'))
                            print(f"🏆 New best eval performance (score={eval_score:.3f}) at epoch {final_epoch}")
                if hasattr(self, '_best_eval_score'):
                    print(f"📊 Best eval performance: score={self._best_eval_score:.3f} at epoch {self._best_eval_epoch}")
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        if experiment:
            # Save to experiment logs directory (rank0 only implicit by experiment existence)
            history_path = experiment.get_path('logs', 'training_history.csv')
            history_df.to_csv(history_path, index=False)
            experiment.save_config(self.config.to_dict(), 'training_config.json')
            experiment.update_metadata({
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "total_epochs": len(self.training_history),
                "status": "completed"
            })
            experiment.log_progress("Training completed successfully!")
        else:
            # Legacy mode (rank0 only)
            if not self.is_distributed or is_main_process() or is_rank0_env():
                history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
        
        # Barrier + cleanup before benchmark so non-rank-0 processes can exit
        # cleanly while rank 0 runs the lengthy benchmark (~10+ min).
        # Without this, NCCL watchdog kills rank 1 after 600s timeout.
        if self.is_distributed:
            dist.barrier()
            # Unwrap DDP so benchmark & post-training code can access model attributes
            if hasattr(self.model, 'module'):
                self.model = self.model.module
            cleanup_distributed()

        # Run end-of-training benchmark (training-only mode)
        if _BENCHMARK_AVAILABLE and self.wandb_run is not None and not skip_wandb_finish:
            if is_rank0_env():
                try:
                    run_end_of_training_benchmark(self, wandb_module=wandb if WANDB_AVAILABLE else None)
                except Exception as e:
                    print(f"[Benchmark] End-of-training benchmark failed: {e}")

        # Run scib cell-type benchmark (end of training)
        if self.config.scib_eval_enabled and self.wandb_run is not None and not skip_wandb_finish:
            if is_rank0_env():
                try:
                    from occam.evaluation.scib_metrics import run_scib_full
                    run_scib_full(self, wandb_module=wandb if WANDB_AVAILABLE else None)
                except Exception as e:
                    print(f"[scib] End-of-training cell-type benchmark failed: {e}")

        # Finish wandb run (unless deferred for tandem finetuning)
        if self.wandb_run is not None and not skip_wandb_finish:
            wandb.finish()

        if is_rank0_env():
            print("Training completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Print experiment summary if using output manager
        if experiment:
            print(experiment.get_summary())
        
        return {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'experiment': experiment
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint including gene names and dataset info."""
        base_model = _inner_model_for_weights(self.model)
        actual_num_genes = base_model.num_genes if hasattr(base_model, 'num_genes') else None

        config_dict = self.config.to_dict()
        if actual_num_genes is not None:
            config_dict['num_genes'] = actual_num_genes

        checkpoint = {
            'model_state_dict': _clean_state_dict(base_model.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config_dict,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        # Include gene names and dataset info for inference alignment
        if hasattr(self, '_gene_names') and self._gene_names is not None:
            checkpoint['gene_names'] = self._gene_names
        if hasattr(self, '_dataset_info') and self._dataset_info is not None:
            checkpoint['dataset_info'] = self._dataset_info

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config and setup model
        self.config = PretrainConfig.from_dict(checkpoint['config'])
        
        # Setup model with config
        sample_num_genes = None
        if 'model_state_dict' in checkpoint:
            # Infer num_genes from model state
            for key in checkpoint['model_state_dict'].keys():
                if 'encoder.0.weight' in key:  # First linear layer
                    sample_num_genes = checkpoint['model_state_dict'][key].shape[1]
                    break
        
        if sample_num_genes is not None:
            self._setup_model(sample_num_genes)
            state_dict = _clean_state_dict(checkpoint['model_state_dict'])
            base_model = _inner_model_for_weights(self.model)
            base_model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint

    # ------------------------------------------------------------------
    # Finetuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = './checkpoints',
        ft_config: 'Optional[FinetuneConfig]' = None,
        training_config: 'Optional[PretrainConfig]' = None,
        epoch_offset: int = 0,
        wandb_prefix: str = 'finetune_',
        checkpoint_prefix: str = 'finetuned_',
        experiment: 'Optional[ExperimentManager]' = None,
    ) -> Dict[str, Any]:
        """Run finetuning on an already-trained model.

        The method reuses the existing model weights but creates a fresh
        optimizer/scheduler.  It supports all three unmasking schedule modes
        and logs metrics with a configurable prefix so that they are clearly
        separated from training metrics in wandb.

        Args:
            train_loader: Training data loader (same data as training).
            val_loader: Optional validation loader.
            save_dir: Directory to save finetuned checkpoints.
            ft_config: Finetuning configuration.
            training_config: Original training configuration (used to resolve
                defaults and to seed the unmasking schedule).
            epoch_offset: Epoch offset for wandb logging (= number of
                training epochs already completed, for tandem mode).
            wandb_prefix: Prefix for all wandb metric keys.
            checkpoint_prefix: Prefix for checkpoint filenames.
            experiment: Optional experiment manager handle.

        Returns:
            Dict with finetuning history and best loss.
        """
        if ft_config is None:
            ft_config = FinetuneConfig()
        if training_config is None:
            training_config = self.config

        # Resolve finetuning config on top of training config
        original_config = self.config
        resolved = ft_config.resolve(training_config)
        self.config = resolved

        if not self.is_distributed or is_main_process() or is_rank0_env():
            os.makedirs(save_dir, exist_ok=True)
            print(f"\nFinetuning Configuration:")
            print(f"  Epochs: {resolved.epochs}")
            print(f"  Learning rate: {resolved.learning_rate}")
            print(f"  Mask probability: {resolved.mask_probability}")
            print(f"  Mask loss weight: {resolved.mask_loss_weight}")
            print(f"  Masked data weight: {resolved.masked_data_weight}")
            print(f"  Unmasking mode: {ft_config.unmasking_mode}")
            if ft_config.unmasking_mode == 'gradual':
                print(f"  Unmasking zero at progress: {ft_config.unmasking_zero_at}")
            if ft_config.unmasking_mode == 'mixed':
                print(f"  Unmasking final ratio: {ft_config.unmasking_final_ratio}")

        # Update model loss parameters
        base_model = self.model.module if (self.use_ddp or self.use_data_parallel) else self.model
        base_model.masked_data_weight = resolved.masked_data_weight
        base_model.mask_loss_weight = resolved.mask_loss_weight
        base_model.reconstruction_loss = resolved.reconstruction_loss
        # Recreate loss function if finetuning changes the loss type
        from occam.losses import create_loss
        base_model._recon_loss_fn = create_loss(
            resolved.reconstruction_loss,
            num_genes=base_model.num_genes,
            clamp_max_log1p=resolved.clamp_max_log1p,
            w_mse=getattr(resolved, 'w_mse', 1.0),
            w_pois=getattr(resolved, 'w_pois', 0.1),
        ).to(self.device)
        if 'poisson' in resolved.reconstruction_loss:
            base_model._poisson_loss_fn = base_model._recon_loss_fn
        if hasattr(base_model._poisson_loss_fn, 'clamp_max_log1p'):
            base_model._poisson_loss_fn.clamp_max_log1p = resolved.clamp_max_log1p
        # Reset cached mask probability so pos_weight gets recomputed
        base_model._cached_mask_probability = None

        # Fresh optimizer and scheduler for finetuning
        self._setup_optimizer_and_scheduler(resolved.epochs)

        # Unmasking schedule
        schedule = UnmaskingSchedule(
            mode=ft_config.unmasking_mode,
            training_mask_prob=training_config.mask_probability,
            ft_mask_prob=resolved.mask_probability,
            total_epochs=resolved.epochs,
            final_ratio=ft_config.unmasking_final_ratio,
            zero_at=ft_config.unmasking_zero_at,
        )

        # Mask loss weight schedule for finetuning
        ft_mlw_schedule = build_mlw_schedule(resolved)
        if ft_mlw_schedule is not None and (not self.is_distributed or is_main_process() or is_rank0_env()):
            print(f"  MLW schedule (finetuning): {ft_mlw_schedule.start} -> {ft_mlw_schedule.end} "
                  f"(warmup {ft_mlw_schedule.warmup_fraction:.0%}, cooldown {ft_mlw_schedule.cooldown_fraction:.0%} "
                  f"of {resolved.epochs} epochs)")

        # Reset tracking for finetuning
        self.best_val_loss = float('inf')
        ft_history = []
        early_stopping_counter = 0

        if not self.is_distributed or is_main_process() or is_rank0_env():
            print(f"\nStarting finetuning for {resolved.epochs} epochs "
                  f"(epoch offset={epoch_offset})")

        for ft_epoch in range(resolved.epochs):
            self.current_epoch = epoch_offset + ft_epoch
            display_epoch = epoch_offset + ft_epoch + 1
            if ft_mlw_schedule is not None:
                base_model.mask_loss_weight = ft_mlw_schedule.get_mask_loss_weight(ft_epoch)

            # DDP sampler / iterable dataset epoch
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(ft_epoch)
            elif hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(ft_epoch)
            if val_loader is not None and hasattr(val_loader, "sampler") and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(ft_epoch)

            # Periodic evaluation
            if resolved.eval_enabled and resolved.eval_frequency > 0 and ft_epoch > 0 and (ft_epoch % resolved.eval_frequency == 0):
                if self.is_distributed:
                    dist.barrier()
                if (not self.is_distributed) or is_main_process() or is_rank0_env():
                    try:
                        print(f"[FT Eval] Running evaluation at finetuning epoch {ft_epoch + 1} (display epoch {display_epoch})...")
                        eval_result = self._run_external_evaluation(self.current_epoch, save_dir)
                        ckpt_path = os.path.join(save_dir, f'{checkpoint_prefix}checkpoint_eval_epoch_{display_epoch}.pth')
                        self._save_finetuned_checkpoint(ckpt_path, ft_config, epoch_offset)
                        print(f"Eval checkpoint saved: {ckpt_path}")
                        if eval_result is not None:
                            m1, m2 = eval_result
                            eval_score = (m1 if not np.isnan(m1) else 0.0) + (m2 if not np.isnan(m2) else 0.0)
                            if not hasattr(self, '_ft_best_eval_score') or eval_score > self._ft_best_eval_score:
                                self._ft_best_eval_score = eval_score
                                self._ft_best_eval_epoch = display_epoch
                                import shutil
                                shutil.copy2(ckpt_path, os.path.join(save_dir, f'{checkpoint_prefix}best_perf_checkpoint.pth'))
                                print(f"New best finetuning eval (score={eval_score:.3f}) at epoch {display_epoch}")
                            if self.wandb_run is not None:
                                wandb.log({
                                    'epoch': display_epoch,
                                    f'{wandb_prefix}eval_metric_consistency_pct': m1,
                                    f'{wandb_prefix}eval_metric_knownrels_recall_mean': m2,
                                })
                    except Exception as e:
                        print(f"[FT Eval] Error during evaluation at ft epoch {ft_epoch + 1}: {e}")
                    if resolved.scib_eval_enabled:
                        try:
                            from occam.evaluation.scib_metrics import run_scib_fast
                            run_scib_fast(self, wandb_module=wandb if WANDB_AVAILABLE else None)
                        except Exception as e:
                            print(f"[scib] Fast eval failed at ft epoch {ft_epoch + 1}: {e}")
                if self.is_distributed:
                    dist.barrier()

            # Train one epoch with unmasking schedule
            train_loss = self.train_epoch(train_loader, unmasking_schedule=schedule, ft_epoch=ft_epoch)

            # Validate
            val_loss = None
            if val_loader is not None and (ft_epoch + 1) % resolved.validation_freq == 0:
                val_loss = self.validate(val_loader, mask_probability_override=0.0)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # WandB logging with prefix
            if self.wandb_run is not None:
                log_dict = {
                    'epoch': display_epoch,
                    f'{wandb_prefix}train_loss': train_loss,
                    f'{wandb_prefix}learning_rate': current_lr,
                }
                if ft_mlw_schedule is not None:
                    log_dict[f'{wandb_prefix}mask_loss_weight'] = base_model.mask_loss_weight
                if val_loss is not None:
                    log_dict[f'{wandb_prefix}val_loss'] = val_loss
                # Component metrics
                try:
                    train_metrics = getattr(self, '_train_metrics_epoch', None)
                    if isinstance(train_metrics, dict):
                        for k, v in train_metrics.items():
                            if v is not None:
                                log_dict[f'{wandb_prefix}train_{k}'] = float(v)
                except Exception:
                    pass
                try:
                    val_metrics = getattr(self, '_val_metrics_epoch', None)
                    if isinstance(val_metrics, dict):
                        for k, v in val_metrics.items():
                            if v is not None:
                                log_dict[f'{wandb_prefix}val_{k}'] = float(v)
                except Exception:
                    pass
                # Loss diagnostics (finetuning)
                if getattr(self.config, 'loss_diagnostics', False):
                    try:
                        diag_grad = getattr(self, '_diag_grad', None)
                        if isinstance(diag_grad, dict):
                            for k, v in diag_grad.items():
                                log_dict[f'diagnostics/{k}'] = v
                        diag_loss = getattr(self, '_diag_loss', None)
                        if isinstance(diag_loss, dict):
                            for k, v in diag_loss.items():
                                log_dict[f'diagnostics/{k}'] = v
                    except Exception:
                        pass
                wandb.log(log_dict)

            # Console logging
            log_msg = f"FT Epoch {ft_epoch + 1}/{resolved.epochs} (#{display_epoch}) - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            log_msg += f" - LR: {current_lr:.6f}"
            # Show effective mask prob for the epoch
            eff_p = schedule.get_mask_probability(ft_epoch)
            log_msg += f" - mask_p: {eff_p:.3f}"
            if ft_mlw_schedule is not None:
                log_msg += f" - MLW: {base_model.mask_loss_weight:.4f}"
            if not self.is_distributed or is_main_process() or is_rank0_env():
                print(log_msg)

            if experiment:
                experiment.log_progress(log_msg)

            # History
            ft_history.append({
                'epoch': display_epoch,
                'ft_epoch': ft_epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
            })

            # Save best
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                if not self.is_distributed or is_main_process() or is_rank0_env():
                    best_path = os.path.join(save_dir, f'{checkpoint_prefix}best_model.pth')
                    self._save_finetuned_checkpoint(best_path, ft_config, epoch_offset)
                    print(f"New best finetuned model saved (loss: {current_loss:.4f})")
                early_stopping_counter = 0
                if self.wandb_run is not None:
                    wandb.log({f'{wandb_prefix}best_val_loss': current_loss, f'{wandb_prefix}best_epoch': display_epoch})
            else:
                early_stopping_counter += 1

            # Early stopping
            if (resolved.early_stopping_patience is not None and
                    early_stopping_counter >= resolved.early_stopping_patience):
                if not self.is_distributed or is_main_process() or is_rank0_env():
                    print(f"Finetuning early stopping after {ft_epoch + 1} epochs")
                break

        # Final checkpoint & evaluation
        if not self.is_distributed or is_main_process() or is_rank0_env():
            final_path = os.path.join(save_dir, f'{checkpoint_prefix}final_model.pth')
            self._save_finetuned_checkpoint(final_path, ft_config, epoch_offset)
            print(f"Finetuned final model saved: {final_path}")

            if resolved.eval_enabled:
                final_display = epoch_offset + len(ft_history)
                print(f"[FT Eval] Running final evaluation (epoch {final_display})...")
                eval_result = self._run_external_evaluation(final_display - 1, save_dir)
                ckpt_path = os.path.join(save_dir, f'{checkpoint_prefix}checkpoint_eval_epoch_{final_display}.pth')
                self._save_finetuned_checkpoint(ckpt_path, ft_config, epoch_offset)
                if eval_result is not None:
                    m1, m2 = eval_result
                    eval_score = (m1 if not np.isnan(m1) else 0.0) + (m2 if not np.isnan(m2) else 0.0)
                    if not hasattr(self, '_ft_best_eval_score') or eval_score > self._ft_best_eval_score:
                        self._ft_best_eval_score = eval_score
                        self._ft_best_eval_epoch = final_display
                        import shutil
                        shutil.copy2(ckpt_path, os.path.join(save_dir, f'{checkpoint_prefix}best_perf_checkpoint.pth'))
                    if self.wandb_run is not None:
                        wandb.log({
                            'epoch': final_display,
                            f'{wandb_prefix}eval_metric_consistency_pct': m1,
                            f'{wandb_prefix}eval_metric_knownrels_recall_mean': m2,
                        })

        # Save finetuning history
        if not self.is_distributed or is_main_process() or is_rank0_env():
            history_df = pd.DataFrame(ft_history)
            history_path = os.path.join(save_dir, 'finetuning_history.csv')
            if experiment:
                history_path = experiment.get_path('logs', 'finetuning_history.csv')
            history_df.to_csv(history_path, index=False)

        # Barrier + cleanup before benchmark so non-rank-0 processes can exit
        # cleanly while rank 0 runs the lengthy benchmark (~10+ min).
        if self.is_distributed:
            dist.barrier()
            if hasattr(self.model, 'module'):
                self.model = self.model.module
            cleanup_distributed()

        # Run end-of-training benchmark (finetuning mode)
        if _BENCHMARK_AVAILABLE and self.wandb_run is not None:
            if is_rank0_env():
                try:
                    run_end_of_training_benchmark(self, wandb_module=wandb if WANDB_AVAILABLE else None)
                except Exception as e:
                    print(f"[Benchmark] End-of-finetuning benchmark failed: {e}")

        # Run scib cell-type benchmark (end of finetuning)
        if self.config.scib_eval_enabled and self.wandb_run is not None:
            if is_rank0_env():
                try:
                    from occam.evaluation.scib_metrics import run_scib_full
                    run_scib_full(self, wandb_module=wandb if WANDB_AVAILABLE else None)
                except Exception as e:
                    print(f"[scib] End-of-finetuning cell-type benchmark failed: {e}")

        # Finish wandb
        if self.wandb_run is not None:
            wandb.finish()

        if is_rank0_env():
            print(f"Finetuning completed! Best loss: {self.best_val_loss:.4f}")
            if hasattr(self, '_ft_best_eval_score'):
                print(f"Best finetuning eval: score={self._ft_best_eval_score:.3f} at epoch {self._ft_best_eval_epoch}")

        # Restore original config
        self.config = original_config

        return {
            'finetuning_history': ft_history,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': ft_history[-1]['train_loss'] if ft_history else None,
            'experiment': experiment,
        }

    def _save_finetuned_checkpoint(self, path: str, ft_config: 'FinetuneConfig',
                                   training_epochs: int = 0):
        """Save a finetuned checkpoint with extra metadata."""
        base_model = _inner_model_for_weights(self.model)
        actual_num_genes = base_model.num_genes if hasattr(base_model, 'num_genes') else None

        config_dict = self.config.to_dict()
        if actual_num_genes is not None:
            config_dict['num_genes'] = actual_num_genes

        checkpoint = {
            'model_state_dict': _clean_state_dict(base_model.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config_dict,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'is_finetuned': True,
            'finetuning_config': ft_config.to_dict(),
            'training_epochs': training_epochs,
        }

        if hasattr(self, '_gene_names') and self._gene_names is not None:
            checkpoint['gene_names'] = self._gene_names
        if hasattr(self, '_dataset_info') and self._dataset_info is not None:
            checkpoint['dataset_info'] = self._dataset_info
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
