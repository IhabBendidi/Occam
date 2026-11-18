"""Schedule behaviour, including the two traps that shaped the released weights.

These are the cheapest checks in the suite and they catch the most confusing
class of regression, so they run first.
"""

from __future__ import annotations

import pytest

from occam.mae.schedules import MaskLossWeightSchedule, UnmaskingSchedule

# The released stage-1 / stage-2 values, read from the checkpoints themselves.
STAGE1_MASK_PROB = 0.1
STAGE2_FT_MASK_PROB = 0.0
STAGE2_EPOCHS = 100
STAGE2_ZERO_AT = 0.9
STAGE2_MLW_START = 0.7
STAGE2_MLW_END = 0.0
STAGE2_MLW_WARMUP = 0.1
STAGE2_MLW_COOLDOWN = 0.1


def _stage2_unmasking() -> UnmaskingSchedule:
    return UnmaskingSchedule(
        mode="gradual",
        training_mask_prob=STAGE1_MASK_PROB,   # from the PARENT checkpoint
        ft_mask_prob=STAGE2_FT_MASK_PROB,      # 0.0, and inert under 'gradual'
        total_epochs=STAGE2_EPOCHS,
        zero_at=STAGE2_ZERO_AT,
    )


class TestGradualUnmaskingReadsTheParent:
    """The single most confusing thing about stage 2.

    Under ``mode='gradual'`` the schedule ramps down from the *parent
    checkpoint's* ``mask_probability`` (0.1), NOT from the finetuning config's
    ``mask_probability`` (0.0). So stage 2 trains with masking live for most of
    the run even though its own config says 0.0.

    Ground truth: ``exp_062``'s experiment.log records ``FT Epoch 1 ...
    mask_p: 0.100``. If this test fails, stage 2 is no longer reproducing the
    released weights.
    """

    def test_epoch_zero_uses_parent_not_ft_value(self):
        sched = _stage2_unmasking()
        assert sched.get_mask_probability(0) == pytest.approx(STAGE1_MASK_PROB)
        # and emphatically not the finetuning config's 0.0
        assert sched.get_mask_probability(0) != pytest.approx(STAGE2_FT_MASK_PROB)

    def test_ft_mask_probability_is_inert_under_gradual(self):
        """Changing ft_mask_prob must not move the gradual schedule at all."""
        a = _stage2_unmasking()
        b = UnmaskingSchedule(
            mode="gradual", training_mask_prob=STAGE1_MASK_PROB,
            ft_mask_prob=0.99,  # wildly different
            total_epochs=STAGE2_EPOCHS, zero_at=STAGE2_ZERO_AT,
        )
        for epoch in (0, 1, 25, 50, 89, 90, 99):
            assert a.get_mask_probability(epoch) == b.get_mask_probability(epoch)

    def test_reaches_zero_at_zero_at_fraction(self):
        sched = _stage2_unmasking()
        # zero_at=0.9 over 100 epochs -> progress hits 0.9 at epoch 89.1
        assert sched.get_mask_probability(89) > 0.0
        assert sched.get_mask_probability(90) == pytest.approx(0.0)
        assert sched.get_mask_probability(99) == pytest.approx(0.0)

    def test_monotone_non_increasing(self):
        sched = _stage2_unmasking()
        values = [sched.get_mask_probability(e) for e in range(STAGE2_EPOCHS)]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))

    def test_none_mode_does_use_the_ft_value(self):
        """Contrast: under 'none' the finetuning value is what applies."""
        sched = UnmaskingSchedule(
            mode="none", training_mask_prob=STAGE1_MASK_PROB,
            ft_mask_prob=0.42, total_epochs=STAGE2_EPOCHS,
        )
        assert sched.get_mask_probability(0) == pytest.approx(0.42)
        assert sched.get_mask_probability(99) == pytest.approx(0.42)


class TestUnmaskingIsPureAndDeterministic:
    """Required for exact mid-stage resume, and for DDP rank agreement."""

    def test_gradual_is_a_pure_function_of_epoch(self):
        sched = _stage2_unmasking()
        for epoch in (0, 7, 50, 99):
            first = sched.get_mask_probability(epoch)
            assert all(sched.get_mask_probability(epoch) == first for _ in range(5))

    def test_mixed_is_deterministic_per_epoch_and_batch(self):
        kwargs = dict(mode="mixed", training_mask_prob=0.1, ft_mask_prob=0.0,
                      total_epochs=50, final_ratio=0.5, seed=42)
        a, b = UnmaskingSchedule(**kwargs), UnmaskingSchedule(**kwargs)
        for epoch in (0, 10, 49):
            for batch in (0, 1, 137):
                assert (a.get_mask_probability(epoch, batch)
                        == b.get_mask_probability(epoch, batch))


class TestMaskLossWeightSchedule:
    """Stage 2 ramps the mask-loss weight 0.7 -> 0.0; stage 1 ramps 0.7 -> 0.9."""

    def test_stage2_endpoints(self):
        sched = MaskLossWeightSchedule(
            start=STAGE2_MLW_START, end=STAGE2_MLW_END,
            total_epochs=STAGE2_EPOCHS,
            warmup_fraction=STAGE2_MLW_WARMUP, cooldown_fraction=STAGE2_MLW_COOLDOWN,
        )
        assert sched.is_active
        assert sched.get_mask_loss_weight(0) == pytest.approx(STAGE2_MLW_START)
        assert sched.get_mask_loss_weight(STAGE2_EPOCHS - 1) == pytest.approx(STAGE2_MLW_END)

    def test_stage1_ramps_upward(self):
        """Stage 1 is nb_softplus with mlw 0.7 -> 0.9, not the 0.0 -> 0.7 the
        Chain A doc claimed. Both wandb runs claimed the same experiment
        directory; the weights on disk are the 0.7 -> 0.9 arm. See
        docs/reproducing_chain_a.md."""
        sched = MaskLossWeightSchedule(
            start=0.7, end=0.9, total_epochs=100,
            warmup_fraction=0.1, cooldown_fraction=0.0,
        )
        assert sched.get_mask_loss_weight(0) == pytest.approx(0.7)
        assert sched.get_mask_loss_weight(99) == pytest.approx(0.9)
        assert sched.get_mask_loss_weight(50) > sched.get_mask_loss_weight(0)

    def test_warmup_holds_at_start(self):
        sched = MaskLossWeightSchedule(start=0.7, end=0.0, total_epochs=100,
                                       warmup_fraction=0.1, cooldown_fraction=0.1)
        # warmup_epochs = int(0.1 * 100) = 10, so epochs 0..9 are held at start.
        for epoch in range(10):
            assert sched.get_mask_loss_weight(epoch) == pytest.approx(0.7)
        # Epoch 10 is the ramp's first step, where progress == 0, so it still
        # returns `start` exactly. The decrease begins at epoch 11.
        assert sched.get_mask_loss_weight(10) == pytest.approx(0.7)
        assert sched.get_mask_loss_weight(11) < 0.7

    def test_cooldown_holds_at_end(self):
        sched = MaskLossWeightSchedule(start=0.7, end=0.0, total_epochs=100,
                                       warmup_fraction=0.1, cooldown_fraction=0.1)
        for epoch in range(90, 100):
            assert sched.get_mask_loss_weight(epoch) == pytest.approx(0.0)

    def test_weight_stays_within_endpoints(self):
        sched = MaskLossWeightSchedule(start=0.7, end=0.0, total_epochs=100,
                                       warmup_fraction=0.1, cooldown_fraction=0.1)
        for epoch in range(100):
            assert 0.0 - 1e-9 <= sched.get_mask_loss_weight(epoch) <= 0.7 + 1e-9

    def test_inactive_when_endpoints_equal(self):
        sched = MaskLossWeightSchedule(start=0.7, end=0.7, total_epochs=100)
        assert not sched.is_active
        assert sched.get_mask_loss_weight(50) == pytest.approx(0.7)
