"""Pretraining configuration (stage 1).

Extracted from ``training_pipeline.py:269-371``, where it was ``TrainingConfig``.

Attribute names differ from the CLI flags that used to set them
(``--gradient_clip`` -> ``gradient_clip_val``, ``--early_stopping`` ->
``early_stopping_patience``, ``--eval_freq`` -> ``eval_frequency``,
``--recon_loss`` -> ``reconstruction_loss``). The YAML schema in
``occam.config.schema`` uses the attribute names, so the mapping is visible in
one place instead of buried in an argparse block.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = ["PretrainConfig"]


class PretrainConfig:
    """Training configuration class with all hyperparameters."""
    
    def __init__(self):
        # Training hyperparameters (from original paper analysis)
        self.epochs = 100
        self.learning_rate = 1e-3  
        self.batch_size = 256  # Per-GPU batch size (total = batch_size * num_gpus)
        self.weight_decay = 0.0
        self.optimizer = 'adam'  # 'adam' or 'adamw'
        self.lr_scheduler = None  # 'cosine', 'step', or None
        self.warmup_epochs = 0
        self.warmup_start_factor = 1e-3

        # Masking hyperparameters
        self.mask_probability = 0.4  # 40% masking as in original
        self.corruption_mode = 'swap'  # 'swap', 'zero_mask', 'gaussian', 'none'
        self.gaussian_sigma = 0.5  # Std dev for gaussian corruption (log1p space)
        self.variable_masking = False  # Whether to use different masking per gene
        self.gene_presence_mask = False  # Add presence signal to input
        self.presence_shift = 1.0  # Magnitude of presence shift
        
        # Model hyperparameters
        self.model_size = 'small'  # 'small', 'base', 'large', 'xlarge'
        self.hidden_size = None  # Override model_size default if provided
        # Width of the MAE's *own* decoder hidden layer. None ->
        # MODEL_CONFIGS[model_size]['decoder_hidden'] (small=1024 == legacy, base=1536,
        # large=2048). NOT the external HybridNormGlobalSkipDecoder (--decoder_hidden_dim).
        self.mae_decoder_hidden = None
        self.dropout = 0.0
        self.masked_data_weight = 0.75  # From original paper
        self.mask_loss_weight = 0.7     # From original paper
        self.mlw_schedule_start = None   # mask_loss_weight schedule start (None = disabled)
        self.mlw_schedule_end = None     # mask_loss_weight schedule end
        self.mlw_schedule_warmup = 0.0   # fraction of epochs to hold at start before ramp
        self.mlw_schedule_cooldown = 0.0 # fraction of epochs to hold at end after ramp
        self.reconstruction_loss = 'mse'  # 'mse', 'poisson', or 'mse+poisson'
        self.w_mse = 1.0
        self.w_pois = 0.1
        self.clamp_max_log1p = 5.5
        self.recon_loss_scale = 1.0
        self.gpl_weight = 0.0  # GPL ranking loss weight (0 = disabled)
        self.loss_diagnostics = False

        # Data hyperparameters
        self.data_mode = 'in_memory'  # 'backed', 'in_memory', or 'optim'
        self.scale = False  # Whether to z-score scale (requires stats precomputation)
        self.skip_normalize = False  # Skip NormalizePerCell (data already library-normalized)
        self.skip_log1p = False  # Skip Log1p during training (data already log1p-transformed)
        self.val_split = 0.1  # 10% validation split if no validation data provided
        self.target_sum = 1e4  # Target sum for library size normalization (default: 10000)
        self.optim_chunk_size = 8192  # Rows per HDF5 read for optim mode
        self.optim_buffer_chunks = 4  # Chunks combined into shuffle buffer for optim mode
        self.min_gene_overlap_pct = None  # Exclude files below this % gene overlap (None = no filtering)

        # Training settings
        self.gradient_clip_val = None
        self.early_stopping_patience = None
        self.validation_freq = 1  # Validate every N epochs
        
        # Wandb settings
        self.use_wandb = True
        self.wandb_project = "Occam"
        self.wandb_run_name = None
        self.wandb_tags = None
        
        # Multi-GPU settings
        self.use_multi_gpu = True  # Enable multi-GPU when available
        self.use_ddp = True  # Use DistributedDataParallel instead of DataParallel
        
        # Model freezing
        self.freeze_mask_predictor = False

        # Evaluation settings
        self.eval_enabled = True
        self.eval_frequency = 100  # Run at epoch 0, 100, 200, ...
        # Relative to the repo root; part of the evaluation bundle.
        self.eval_h5ad_path = "data/eval/pert/replogle_2022_rpe1_e.h5ad"
        self.eval_batch_size = 1024
        self.enable_amp = False  # AMP bf16 mixed precision (H100 native support)
        self.compile = False  # torch.compile the model (fuses ops, reduces HBM traffic)
        self.scib_eval_enabled = True  # scib cell-type benchmarks (fast periodic + full end-of-training)
        
    def to_dict(self):
        """Convert to dictionary for saving."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        config = cls()
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config
    
    def print_config(self):
        """Print current configuration."""
        print("Training Configuration:")
        print("-" * 40)
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")
        print("-" * 40)


    # -- construction from persisted configs -----------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainConfig":
        """Rebuild from a config dict read off a checkpoint.

        Uses ``setattr``-over-defaults rather than ``**data`` into the
        constructor, deliberately: the released configs predate about ten
        fields, and their absence must resolve to the values that were in force
        at the time. Those are pinned in ``occam._compat.LEGACY_DEFAULTS``.
        """
        from occam._compat import apply_legacy_defaults, migrate_config_dict

        merged = apply_legacy_defaults(migrate_config_dict(data))
        obj = cls()
        for key, value in merged.items():
            setattr(obj, key, value)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}
