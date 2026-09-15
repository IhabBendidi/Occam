"""Masking and mask-loss-weight schedules.

Extracted from ``training_pipeline.py:479-579``.

⚠️ ``UnmaskingSchedule`` in ``gradual`` mode ramps down from the **parent
checkpoint's** ``mask_probability``, not from the finetuning config's
``mask_probability``. Stage 2 of the released chain sets
``ft_mask_probability = 0.0``, which is inert: the schedule reads 0.1 off the
stage-1 config, which is why ``exp_062``'s log shows ``mask_p: 0.100`` at epoch
1. This is preserved deliberately -- it is what produced the released weights --
and asserted in ``tests/test_schedules.py``.

Both schedules are pure functions of ``(epoch, total_epochs)``, so mid-stage
resume reproduces them exactly.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["UnmaskingSchedule", "MaskLossWeightSchedule", "build_mlw_schedule"]


class UnmaskingSchedule:
    """Computes effective mask_probability per epoch/batch during finetuning.

    Modes:
      - ``none``:  constant ``ft_mask_prob``
      - ``gradual``: linear interpolation from ``training_mask_prob`` to 0
      - ``mixed``:  per-batch random choice between ``training_mask_prob`` and 0,
                    proportion of 0-mask batches grows from 0% to ``final_ratio``
    """

    def __init__(self, mode: str, training_mask_prob: float,
                 ft_mask_prob: float, total_epochs: int,
                 final_ratio: float = 0.5, zero_at: float = 1.0,
                 seed: int = 42):
        self.mode = mode
        self.training_mask_prob = training_mask_prob
        self.ft_mask_prob = ft_mask_prob
        self.total_epochs = max(total_epochs, 1)
        self.final_ratio = final_ratio
        self.zero_at = max(min(zero_at, 1.0), 0.01)
        self.seed = seed

    def get_mask_probability(self, epoch: int, batch_idx: int = 0) -> float:
        """Return effective mask probability for a given finetuning epoch/batch.

        Args:
            epoch: 0-indexed epoch within finetuning.
            batch_idx: batch index within the epoch (used by 'mixed' mode).
        """
        if self.mode == 'none':
            return self.ft_mask_prob

        elif self.mode == 'gradual':
            progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
            return self.training_mask_prob * max(1.0 - progress / self.zero_at, 0.0)

        elif self.mode == 'mixed':
            progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
            no_mask_prob = self.final_ratio * progress
            # Deterministic per (epoch, batch) for DDP consistency across ranks
            import random as _random
            rng = _random.Random(self.seed + epoch * 100000 + batch_idx)
            if rng.random() < no_mask_prob:
                return 0.0
            else:
                return self.training_mask_prob

        return self.ft_mask_prob


class MaskLossWeightSchedule:
    """Schedules mask_loss_weight to ramp linearly from start to end.

    Three phases:
      1. Warmup: hold at ``start`` for ``warmup_fraction`` of total epochs
      2. Ramp: linearly interpolate from ``start`` to ``end``
      3. Cooldown: hold at ``end`` for ``cooldown_fraction`` of total epochs
    """

    def __init__(self, start: float, end: float, total_epochs: int,
                 warmup_fraction: float = 0.0, cooldown_fraction: float = 0.0):
        self.start = start
        self.end = end
        self.total_epochs = max(total_epochs, 1)
        self.warmup_fraction = max(min(warmup_fraction, 0.99), 0.0)
        self.cooldown_fraction = max(min(cooldown_fraction, 0.99), 0.0)

    @property
    def is_active(self) -> bool:
        return self.start != self.end

    def get_mask_loss_weight(self, epoch: int) -> float:
        if not self.is_active:
            return self.start
        warmup_epochs = int(self.warmup_fraction * self.total_epochs)
        cooldown_epochs = int(self.cooldown_fraction * self.total_epochs)
        if epoch < warmup_epochs:
            return self.start
        if epoch >= self.total_epochs - cooldown_epochs:
            return self.end
        ramp_epochs = self.total_epochs - warmup_epochs - cooldown_epochs
        if ramp_epochs <= 1:
            return self.end
        progress = min((epoch - warmup_epochs) / (ramp_epochs - 1), 1.0)
        return self.start + (self.end - self.start) * progress


def build_mlw_schedule(config: 'PretrainConfig') -> 'Optional[MaskLossWeightSchedule]':
    start = getattr(config, 'mlw_schedule_start', None)
    end = getattr(config, 'mlw_schedule_end', None)
    if start is None or end is None or start == end:
        return None
    warmup = getattr(config, 'mlw_schedule_warmup', 0.0) or 0.0
    cooldown = getattr(config, 'mlw_schedule_cooldown', 0.0) or 0.0
    return MaskLossWeightSchedule(
        start=start, end=end,
        total_epochs=config.epochs,
        warmup_fraction=warmup,
        cooldown_fraction=cooldown,
    )
