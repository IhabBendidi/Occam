"""Finetuning configuration (stage 2), as an overlay on stage 1.

Extracted from ``training_pipeline.py:373-477``, where it was
``FinetuningConfig``.

``None`` means **inherit**, and ``resolve()`` overlays the non-None fields onto
a :class:`~occam.mae.config.PretrainConfig` via ``_FIELD_MAP``.

One thing to know before changing anything here: ``resolve()`` is called with a
config built from the *parent checkpoint*, not from the YAML's stage-1 block. So
stage 2 inherits ``target_sum``, ``corruption_mode``, ``model_size``,
``mae_decoder_hidden`` and ``scale`` from the weights it is finetuning. The
loader diffs the two and writes ``overlay_audit.json`` recording, per field,
what the YAML said, what was inherited, and which won.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# Safe as a top-level import: occam.mae.config does not import this module.
from occam.mae.config import PretrainConfig

__all__ = ["FinetuneConfig"]


class FinetuneConfig:
    """Configuration for finetuning phase.

    Values of ``None`` mean 'inherit from training config'.  Call
    :meth:`resolve` to produce a concrete :class:`PretrainConfig` with
    all overrides applied.
    """

    def __init__(self):
        self.enabled = False
        self.checkpoint = None  # path to pretrained checkpoint (standalone mode)

        # Unmasking schedule
        self.unmasking_mode = 'none'  # 'none', 'gradual', or 'mixed'
        self.unmasking_final_ratio = 0.5  # final proportion of unmasked batches for 'mixed'
        self.unmasking_zero_at = 1.0  # progress fraction at which mask prob hits 0 in 'gradual'

        # All training hyperparameters (None = inherit from training config)
        self.epochs = None
        self.learning_rate = None
        self.weight_decay = None
        self.batch_size = None
        self.optimizer = None
        self.lr_scheduler = None
        self.warmup_epochs = None
        self.warmup_start_factor = None
        self.mask_probability = None
        self.corruption_mode = None
        self.gaussian_sigma = None
        self.mask_loss_weight = None
        self.mlw_schedule_start = None
        self.mlw_schedule_end = None
        self.mlw_schedule_warmup = None
        self.mlw_schedule_cooldown = None
        self.masked_data_weight = None
        self.recon_loss = None
        self.clamp_max_log1p = None
        self.gradient_clip = None
        self.early_stopping = None
        self.validation_freq = None
        self.eval_freq = None
        self.freeze_mask_predictor = None

        # Runtime / logging overrides (None = inherit from checkpoint training config)
        self.data_mode = None
        self.use_wandb = None
        self.wandb_run_name = None
        self.enable_amp = None
        self.compile = None

    # Maps FinetuneConfig field names -> PretrainConfig field names
    _FIELD_MAP = {
        'epochs': 'epochs',
        'learning_rate': 'learning_rate',
        'weight_decay': 'weight_decay',
        'batch_size': 'batch_size',
        'optimizer': 'optimizer',
        'lr_scheduler': 'lr_scheduler',
        'warmup_epochs': 'warmup_epochs',
        'warmup_start_factor': 'warmup_start_factor',
        'mask_probability': 'mask_probability',
        'corruption_mode': 'corruption_mode',
        'gaussian_sigma': 'gaussian_sigma',
        'mask_loss_weight': 'mask_loss_weight',
        'mlw_schedule_start': 'mlw_schedule_start',
        'mlw_schedule_end': 'mlw_schedule_end',
        'mlw_schedule_warmup': 'mlw_schedule_warmup',
        'mlw_schedule_cooldown': 'mlw_schedule_cooldown',
        'masked_data_weight': 'masked_data_weight',
        'recon_loss': 'reconstruction_loss',
        'clamp_max_log1p': 'clamp_max_log1p',
        'gradient_clip': 'gradient_clip_val',
        'early_stopping': 'early_stopping_patience',
        'validation_freq': 'validation_freq',
        'eval_freq': 'eval_frequency',
        'freeze_mask_predictor': 'freeze_mask_predictor',
        'data_mode': 'data_mode',
        'use_wandb': 'use_wandb',
        'wandb_run_name': 'wandb_run_name',
        'enable_amp': 'enable_amp',
        'compile': 'compile',
    }

    def resolve(self, training_config: PretrainConfig) -> PretrainConfig:
        """Create a resolved PretrainConfig by overlaying ft values on training defaults."""
        resolved = PretrainConfig.from_dict(training_config.to_dict())
        for ft_field, config_field in self._FIELD_MAP.items():
            ft_val = getattr(self, ft_field)
            if ft_val is not None:
                setattr(resolved, config_field, ft_val)
        return resolved

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
