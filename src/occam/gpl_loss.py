"""Geometric Plackett-Luce (GPL) ranking loss for gene expression prediction.

Standalone module — importable by virtual_cell, pretraining, and decoder pipelines.

Mathematical framework:
  - Parameterization: phi_g = prediction_g, theta_g = sigmoid(-phi_g)
  - Binary term: expressed (count > 0) vs unexpressed (count == 0)
  - Ranking term: correct ordering among expressed genes via GPL stage-wise formula
  - log(1 - theta) = logsigmoid(phi), log(theta) = logsigmoid(-phi)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x < 0.

    Two branches (Maclaurin 2012):
      x > -ln(2): log(-expm1(x))   -- avoids catastrophic cancellation
      x <= -ln(2): log1p(-exp(x))  -- avoids overflow in expm1
    """
    out = torch.empty_like(x)
    mask = x > -0.6931471805599453  # -ln(2)
    out[mask] = torch.log(-torch.expm1(x[mask]))
    out[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return out


class _CumsumNonDeterministic(torch.autograd.Function):
    """cumsum that bypasses torch.use_deterministic_algorithms for both
    forward and backward, keeping everything on GPU."""

    @staticmethod
    def forward(ctx, x, dim):
        _det = torch.are_deterministic_algorithms_enabled()
        if _det:
            torch.use_deterministic_algorithms(False)
        out = torch.cumsum(x, dim=dim)
        if _det:
            torch.use_deterministic_algorithms(True)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        _det = torch.are_deterministic_algorithms_enabled()
        if _det:
            torch.use_deterministic_algorithms(False)
        grad_input = torch.flip(
            torch.cumsum(torch.flip(grad_output, [ctx.dim]), dim=ctx.dim),
            [ctx.dim],
        )
        if _det:
            torch.use_deterministic_algorithms(True)
        return grad_input, None


def _cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """cumsum compatible with torch.use_deterministic_algorithms(True) on CUDA."""
    if x.is_cuda and torch.are_deterministic_algorithms_enabled():
        return _CumsumNonDeterministic.apply(x, dim)
    return torch.cumsum(x, dim=dim)


class GeometricPlackettLuceLoss(nn.Module):
    """Geometric Plackett-Luce ranking loss for gene expression.

    Operates in log1p space. Factorizes into:
      - Binary term: which genes are expressed (count > 0) vs unexpressed (== 0)
      - Ranking term: correct ordering among expressed genes by expression level

    Parameterization:
      phi_g = prediction_g (raw decoder output in log1p space)
      theta_g = sigmoid(-phi_g)  (Geometric distribution parameter)
      log(1 - theta) = logsigmoid(phi)
      log(theta) = logsigmoid(-phi)

    GPL stage-wise formula for position j (sorted by target descending):
      log P_j = logsigmoid(-phi_j) + suffix_after[j] - log1mexp(suffix[j])
    where suffix[j] = sum_{l>=j} logsigmoid(phi_l)

    Args:
        reduction: 'mean' averages over batch, 'none' returns per-sample (B,).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute GPL loss (fully vectorized, no per-sample loop).

        Args:
            prediction: (B, G) raw decoder output in log1p space.
            target: (B, G) target expression in log1p space.
            mask: (B, G) bool tensor, True = gene is valid/measured. None = all valid.

        Returns:
            Scalar (reduction='mean') or (B,) tensor (reduction='none').
        """
        # Ensure float32 for numerical precision in logsigmoid cumsum chain
        prediction = prediction.float()
        target = target.float()

        B, G = prediction.shape

        # --- Determine expressed / unexpressed per gene ---
        if mask is not None:
            expressed = (target > 0) & mask  # (B, G)
            unexpressed = (target == 0.0) & mask  # (B, G)
            n_valid = mask.sum(dim=1).float().clamp(min=1)  # (B,)
        else:
            expressed = target > 0  # (B, G)
            unexpressed = target == 0.0  # (B, G)
            n_valid = torch.full(
                (B,), float(G), device=prediction.device, dtype=prediction.dtype
            )

        # --- Binary term (fully vectorized) ---
        ls_phi = F.logsigmoid(prediction)  # (B, G) -- log(1-theta)
        ls_neg_phi = F.logsigmoid(-prediction)  # (B, G) -- log(theta)
        L_binary = (
            (expressed.float() * ls_phi).sum(dim=1)
            + (unexpressed.float() * ls_neg_phi).sum(dim=1)
        )  # (B,)

        # --- Ranking term (batched sort + pad + mask) ---
        n_expr = expressed.sum(dim=1)  # (B,)
        max_expr = int(n_expr.max().item())

        if max_expr <= 1:
            # No ranking needed for any sample
            losses = -L_binary / n_valid
            return losses.mean() if self.reduction == "mean" else losses

        # Sort: set unexpressed/invalid genes to -inf target so they sort last
        t_sortable = target.clone()
        t_sortable[~expressed] = -float("inf")
        _, sort_idx = torch.sort(t_sortable, dim=1, descending=True)  # (B, G)

        # Gather prediction in sorted order, keep only first max_expr columns
        phi_sorted = torch.gather(prediction, 1, sort_idx[:, :max_expr])  # (B, max_expr)

        # Position validity mask: position j valid if j < n_expr[i]
        positions = torch.arange(max_expr, device=prediction.device)  # (max_expr,)
        valid_pos = positions.unsqueeze(0) < n_expr.unsqueeze(1)  # (B, max_expr)

        ls = F.logsigmoid(phi_sorted)  # (B, max_expr)
        ls_neg = F.logsigmoid(-phi_sorted)  # (B, max_expr)

        # Zero invalid positions before cumsum so they don't contribute
        ls_masked = ls * valid_pos.float()  # (B, max_expr)

        # Reverse suffix cumsum: suffix[j] = sum_{l>=j} ls_masked[l]
        suffix = torch.flip(
            _cumsum(torch.flip(ls_masked, [1]), dim=1), [1]
        )  # (B, max_expr)

        # suffix_after[j] = suffix[j+1], suffix_after[last] = 0
        suffix_after = torch.zeros_like(suffix)
        suffix_after[:, :-1] = suffix[:, 1:]

        # Protect padding positions from log1mexp(0) = -inf → NaN when multiplied by 0
        suffix_safe = torch.where(valid_pos, suffix, torch.full_like(suffix, -1.0))

        # GPL stage-wise: log P_j = log(theta_j) + suffix_after[j] - log1mexp(suffix[j])
        log_P = ls_neg + suffix_after - log1mexp(suffix_safe)

        # Mask invalid positions and sum
        L_rank = (log_P * valid_pos.float()).sum(dim=1)  # (B,)

        # --- Combine ---
        losses = -(L_binary + L_rank) / n_valid

        return losses.mean() if self.reduction == "mean" else losses
