"""Reconstruction loss functions for Occam.

All losses share the same interface:
    forward(prediction, target) -> per_element_loss  (no reduction)

Both prediction and target are in log1p space (log(1 + raw_counts)).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY = {}


class _RectifiedSigmoid(nn.Module):
    def __init__(self, upper_bound: int = 2000):
        super().__init__()
        self.upper_bound = math.log1p(upper_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upper_bound * torch.relu(2 * torch.sigmoid(x / (2 * math.e)) - 1)


def register_loss(name):
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


_LOSS_ALIASES = {
    "poisson_clean": "poisson",
}


def create_loss(name, **kwargs):
    name = _LOSS_ALIASES.get(name, name)
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)


@register_loss("mse")
class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction, target):
        return F.mse_loss(prediction, target, reduction="none")


@register_loss("smoothl1")
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__()
        self.beta = beta

    def forward(self, prediction, target):
        return F.smooth_l1_loss(prediction, target, reduction="none", beta=self.beta)


@register_loss("basic_poisson")
class BasicPoissonNLLLogSpace(nn.Module):
    """Naive Poisson NLL: treats raw prediction as log(mu), target as counts."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, log1p_input, log1p_target):
        return log1p_input.exp() - log1p_target.exp() * log1p_input


@register_loss("poisson")
class PoissonNLLLogSpace(nn.Module):
    """Poisson NLL in log1p space. Softplus on prediction, clean expm1 on target."""

    def __init__(self, clamp_max_log1p=5.5, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.clamp_max_log1p = clamp_max_log1p

    def forward(self, log1p_input, log1p_target):
        z = F.softplus(log1p_input)
        z = torch.clamp(z, max=self.clamp_max_log1p)
        mu_hat = torch.expm1(z)
        mu_hat = torch.clamp(mu_hat, min=self.eps)
        log_mu_hat = torch.log(mu_hat)

        target_counts = torch.expm1(log1p_target)
        target_counts = torch.clamp(target_counts, min=0.0)

        return mu_hat - target_counts * log_mu_hat


@register_loss("poisson_no_softplus")
class PoissonNoSoftplus(nn.Module):
    """ReLU instead of softplus — tests whether gradient smoothness matters."""

    def __init__(self, clamp_max_log1p=5.5, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.clamp_max_log1p = clamp_max_log1p

    def forward(self, log1p_input, log1p_target):
        z = F.relu(log1p_input)
        z = torch.clamp(z, max=self.clamp_max_log1p)
        mu_hat = torch.expm1(z)
        mu_hat = torch.clamp(mu_hat, min=self.eps)
        log_mu_hat = torch.log(mu_hat)

        target_counts = torch.expm1(log1p_target)
        target_counts = torch.clamp(target_counts, min=0.0)

        return mu_hat - target_counts * log_mu_hat


@register_loss("poisson_no_clamp")
class PoissonNoClamp(nn.Module):
    """Softplus but no max clamp — tests whether outlier truncation matters."""

    def __init__(self, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(self, log1p_input, log1p_target):
        z = F.softplus(log1p_input)
        mu_hat = torch.expm1(z)
        mu_hat = torch.clamp(mu_hat, min=self.eps)
        log_mu_hat = torch.log(mu_hat)

        target_counts = torch.expm1(log1p_target)
        target_counts = torch.clamp(target_counts, min=0.0)

        return mu_hat - target_counts * log_mu_hat


@register_loss("nb")
class NegativeBinomialLoss(nn.Module):
    """Negative Binomial with learnable per-gene dispersion theta.

    Proper distributional assumption for overdispersed count data (variance > mean).
    theta is learned as log(theta) for unconstrained optimization.
    """

    def __init__(self, num_genes=1, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))

    def forward(self, log1p_input, log1p_target):
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        mu = torch.expm1(log1p_input.clamp(min=0)).clamp(min=self.eps)
        y = torch.expm1(log1p_target.clamp(min=0))

        # NB NLL: -log Gamma(y+theta) + log Gamma(theta) + theta*log(theta/(mu+theta)) + y*log(mu/(mu+theta))
        # We negate the log-likelihood terms that depend on mu (the ones the model can affect)
        loss = (
            torch.lgamma(theta)
            - torch.lgamma(y + theta)
            + theta * torch.log1p(mu / theta)
            + y * torch.log1p(theta / mu)
        )
        return loss


@register_loss("zinb")
class ZeroInflatedNBLoss(nn.Module):
    """Zero-Inflated Negative Binomial with learnable per-gene dispersion and zero-inflation.

    Models both overdispersion (theta) and excess zeros (pi).
    Both parameters are per-gene and learned jointly with the model.
    """

    def __init__(self, num_genes=1, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.logit_pi = nn.Parameter(torch.full((num_genes,), -2.0))

    def forward(self, log1p_input, log1p_target):
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        pi = torch.sigmoid(self.logit_pi)
        mu = torch.expm1(log1p_input.clamp(min=0)).clamp(min=self.eps)
        y = torch.expm1(log1p_target.clamp(min=0))

        # NB log-prob (full, including y-dependent terms needed for ZINB mixture)
        nb_log_prob = (
            torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
            + theta * torch.log(theta / (mu + theta))
            + y * torch.log(mu / (mu + theta) + self.eps)
        )

        # NB log-prob at y=0
        nb_log_zero = theta * torch.log(theta / (mu + theta))

        zero_mask = (y < 0.5).float()

        # y=0: log(pi + (1-pi)*NB(0))
        zero_case = torch.log(pi + (1 - pi) * nb_log_zero.exp() + self.eps)
        # y>0: log(1-pi) + log NB(y)
        nonzero_case = torch.log(1 - pi + self.eps) + nb_log_prob

        nll = -(zero_mask * zero_case + (1 - zero_mask) * nonzero_case)
        return nll


@register_loss("nb_softplus")
class NBSoftplus(nn.Module):
    """NB with softplus pred->count conversion (fixes dead-gradient zone)."""

    def __init__(self, num_genes=1, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))

    def forward(self, log1p_input, log1p_target):
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        z = F.softplus(log1p_input)
        mu = torch.expm1(z).clamp(min=self.eps)
        y = torch.expm1(log1p_target.clamp(min=0))

        loss = (
            torch.lgamma(theta)
            - torch.lgamma(y + theta)
            + theta * torch.log1p(mu / theta)
            + y * torch.log1p(theta / mu)
        )
        return loss


@register_loss("zinb_softplus")
class ZINBSoftplus(nn.Module):
    """ZINB with softplus pred->count conversion (fixes dead-gradient zone)."""

    def __init__(self, num_genes=1, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.logit_pi = nn.Parameter(torch.full((num_genes,), -2.0))

    def forward(self, log1p_input, log1p_target):
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        pi = torch.sigmoid(self.logit_pi)
        z = F.softplus(log1p_input)
        mu = torch.expm1(z).clamp(min=self.eps)
        y = torch.expm1(log1p_target.clamp(min=0))

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
        return nll


@register_loss("nb_rectsig")
class NBRectSig(nn.Module):
    """NB with RectifiedSigmoid pred->count conversion (bounded activation)."""

    def __init__(self, num_genes=1, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.rectsig = _RectifiedSigmoid(upper_bound=2000)

    def forward(self, log1p_input, log1p_target):
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        log1p_mu = self.rectsig(log1p_input)
        mu = torch.expm1(log1p_mu).clamp(min=self.eps)
        y = torch.expm1(log1p_target.clamp(min=0))

        loss = (
            torch.lgamma(theta)
            - torch.lgamma(y + theta)
            + theta * torch.log1p(mu / theta)
            + y * torch.log1p(theta / mu)
        )
        return loss


@register_loss("zinb_rectsig")
class ZINBRectSig(nn.Module):
    """ZINB with RectifiedSigmoid pred->count conversion (bounded activation)."""

    def __init__(self, num_genes=1, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.logit_pi = nn.Parameter(torch.full((num_genes,), -2.0))
        self.rectsig = _RectifiedSigmoid(upper_bound=2000)

    def forward(self, log1p_input, log1p_target):
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        pi = torch.sigmoid(self.logit_pi)
        log1p_mu = self.rectsig(log1p_input)
        mu = torch.expm1(log1p_mu).clamp(min=self.eps)
        y = torch.expm1(log1p_target.clamp(min=0))

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
        return nll


@register_loss("smoothl1_b2")
class SmoothL1Beta2(nn.Module):
    """SmoothL1 with beta=2.0 (matching decoder training pipeline)."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction, target):
        return F.smooth_l1_loss(prediction, target, reduction="none", beta=2.0)


@register_loss("poisson_rectsig")
class PoissonRectSig(nn.Module):
    """Poisson NLL with RectifiedSigmoid instead of softplus."""

    def __init__(self, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.rectsig = _RectifiedSigmoid(upper_bound=2000)

    def forward(self, log1p_input, log1p_target):
        log1p_mu = self.rectsig(log1p_input)
        mu_hat = torch.expm1(log1p_mu).clamp(min=self.eps)
        log_mu_hat = torch.log(mu_hat)

        target_counts = torch.expm1(log1p_target)
        target_counts = torch.clamp(target_counts, min=0.0)

        return mu_hat - target_counts * log_mu_hat


@register_loss("smoothl1_rectsig")
class SmoothL1RectSig(nn.Module):
    """SmoothL1 on RectifiedSigmoid(prediction) vs target — bounded prediction space."""

    def __init__(self, **kwargs):
        super().__init__()
        self.rectsig = _RectifiedSigmoid(upper_bound=2000)

    def forward(self, prediction, target):
        return F.smooth_l1_loss(self.rectsig(prediction), target, reduction="none")


@register_loss("smoothl1_softplus")
class SmoothL1Softplus(nn.Module):
    """SmoothL1 on softplus(prediction) vs target — smooth non-negative prediction space."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction, target):
        return F.smooth_l1_loss(F.softplus(prediction), target, reduction="none")


@register_loss("smoothl1_relu")
class SmoothL1ReLU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction, target):
        return F.smooth_l1_loss(F.relu(prediction), target, reduction="none")


@register_loss("mse_softplus")
class MSESoftplus(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction, target):
        return F.mse_loss(F.softplus(prediction), target, reduction="none")


@register_loss("mse_relu")
class MSEReLU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction, target):
        return F.mse_loss(F.relu(prediction), target, reduction="none")


@register_loss("mse_rectsig")
class MSERectSig(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rectsig = _RectifiedSigmoid(upper_bound=2000)

    def forward(self, prediction, target):
        return F.mse_loss(self.rectsig(prediction), target, reduction="none")


# Combined losses (delegating to registered components)
@register_loss("mse+poisson")
class MSEPlusPoissonLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_pois=0.1, clamp_max_log1p=5.5, **kwargs):
        super().__init__()
        self.w_mse = w_mse
        self.w_pois = w_pois
        self._mse = MSELoss()
        self._poisson = PoissonNLLLogSpace(clamp_max_log1p=clamp_max_log1p)

    def forward(self, prediction, target):
        return self.w_mse * self._mse(prediction, target) + self.w_pois * self._poisson(prediction, target)


@register_loss("mse+basic_poisson")
class MSEPlusBasicPoissonLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_pois=0.1, **kwargs):
        super().__init__()
        self.w_mse = w_mse
        self.w_pois = w_pois
        self._mse = MSELoss()
        self._basic_poisson = BasicPoissonNLLLogSpace()

    def forward(self, prediction, target):
        return self.w_mse * self._mse(prediction, target) + self.w_pois * self._basic_poisson(prediction, target)


# ---------------------------------------------------------------------------
# Diagnostics helpers (used when --loss_diagnostics is on)
# ---------------------------------------------------------------------------

def compute_loss_diagnostics(prediction, target, base_model):
    """Compute lightweight diagnostics on a single batch.

    Returns a dict of scalar values ready for wandb logging.
    Called every eval_freq epochs on the first validation batch.
    """
    diag = {}
    with torch.no_grad():
        raw_decoder_out = prediction  # raw decoder output (before any activation in loss)
        diag["pred_mean"] = raw_decoder_out.mean().item()
        diag["pred_std"] = raw_decoder_out.std().item()
        diag["pred_min"] = raw_decoder_out.min().item()
        diag["pred_max"] = raw_decoder_out.max().item()
        diag["negative_pred_frac"] = (raw_decoder_out < 0).float().mean().item()

        # Reconstruction MAE by expression bin (in log1p space)
        diff = (prediction - target).abs()
        bins = [
            ("zero", target < 1e-6),
            ("low_0_to_0.5", (target >= 1e-6) & (target < 0.5)),
            ("med_0.5_to_2", (target >= 0.5) & (target < 2.0)),
            ("high_2_to_4", (target >= 2.0) & (target < 4.0)),
            ("very_high_4_plus", target >= 4.0),
        ]
        for name, mask in bins:
            if mask.any():
                diag[f"recon_mae_{name}"] = diff[mask].mean().item()
            else:
                diag[f"recon_mae_{name}"] = 0.0

        # Per-gene loss variance (how concentrated is the loss across genes)
        per_gene_mae = diff.mean(dim=0)
        diag["per_gene_loss_std"] = per_gene_mae.std().item()
        diag["per_gene_loss_max"] = per_gene_mae.max().item()

        # NB/ZINB specific diagnostics
        loss_fn = getattr(base_model, '_recon_loss_fn', None)
        if loss_fn is None:
            loss_fn = getattr(base_model, '_poisson_loss_fn', None)

        if hasattr(loss_fn, 'log_theta'):
            theta = loss_fn.log_theta.exp()
            diag["nb_theta_mean"] = theta.mean().item()
            diag["nb_theta_std"] = theta.std().item()
            diag["nb_theta_min"] = theta.min().item()
            diag["nb_theta_max"] = theta.max().item()
        if hasattr(loss_fn, 'logit_pi'):
            pi = torch.sigmoid(loss_fn.logit_pi)
            diag["zinb_pi_mean"] = pi.mean().item()
            diag["zinb_pi_std"] = pi.std().item()
            diag["zinb_pi_min"] = pi.min().item()
            diag["zinb_pi_max"] = pi.max().item()

    return diag


def compute_gradient_diagnostics(model):
    """Compute gradient norm per component. Call after loss.backward(), before optimizer.step()."""
    diag = {}
    base = model.module if hasattr(model, 'module') else model
    for name, module in [("encoder", base.encoder), ("decoder", base.decoder),
                         ("mask_predictor", base.mask_predictor)]:
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        diag[f"grad_norm_{name}"] = total_norm ** 0.5

    # Loss function params (NB/ZINB)
    loss_fn = getattr(base, '_recon_loss_fn', None)
    if loss_fn is not None and any(p.requires_grad for p in loss_fn.parameters()):
        total_norm = 0.0
        for p in loss_fn.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        diag["grad_norm_loss_params"] = total_norm ** 0.5

    return diag


ALL_LOSS_NAMES = list(LOSS_REGISTRY.keys())

# ===========================================================================
# Decoder-side losses
# ===========================================================================
# Extracted from ``decoder_training_pipeline.py:122-275``. These belong with the
# other losses rather than in the trainer module.
#
# Caveat carried verbatim from the source: on library-normalised + log1p data
# the "counts" recovered by expm1 are non-integer, so the discrete NB/ZINB
# assumption is only approximate. Both losses learn a per-gene ``log_theta``.

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


class BasicPoissonLoss(nn.Module):
    """Simplified Poisson NLL (parity with model.py BasicPoissonNLLLogSpace).

    Treats the raw model output as a log-rate and uses ``.exp()`` to map the
    log1p-space pred/target to a rate: ``loss = exp(pred) - exp(target) * pred``.
    NOTE: ``exp()`` of a log1p value recovers ``(1 + count)``, not ``count`` — so
    both the rate and the target carry a systematic +1 offset. Kept for parity
    with the encoder's ``basic_poisson``; see ``CorrectPoissonLoss`` for the
    expm1 variant that recovers true counts.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = input.exp() - target.exp() * input
        if self.reduction == "none":
            return loss
        return loss.mean()


class CorrectPoissonLoss(nn.Module):
    """``basic_poisson`` but with ``expm1`` instead of ``exp``.

    Identical structure to ``BasicPoissonLoss`` except log1p-space values are
    mapped to true counts via ``expm1`` (``expm1(log1p(c)) = c``) instead of
    ``exp`` (which yields ``1 + c``):
    ``loss = expm1(pred) - expm1(target) * pred``. The ``* pred`` term still uses
    the raw output as a log-rate proxy (as basic_poisson does).
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.expm1(input) - torch.expm1(target) * input
        if self.reduction == "none":
            return loss
        return loss.mean()


class NBDecoderLoss(nn.Module):
    """Negative Binomial NLL with learnable per-gene dispersion (theta).

    Mirrors ``losses.NegativeBinomialLoss``. Inputs/targets are log1p-space and
    converted to counts via ``expm1``. ``theta`` is learned as ``log_theta`` and
    optimized jointly (added to the optimizer in ``_setup_optimizer``).
    NOTE: on library-normalized + log1p data the recovered "counts" are
    non-integer (normalized), so the discrete NB assumption is only approximate.
    """
    def __init__(self, num_genes: int, reduction: str = "mean", eps: float = 1e-8,
                 theta_init: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        # theta = exp(log_theta); theta_init is the log-space init (0.0 -> theta=1).
        self.log_theta = nn.Parameter(torch.full((num_genes,), float(theta_init)))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        mu = torch.expm1(input.clamp(min=0)).clamp(min=self.eps)
        y = torch.expm1(target.clamp(min=0))
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
    """Zero-Inflated NB NLL with learnable per-gene theta + zero-inflation pi.

    Mirrors ``losses.ZeroInflatedNBLoss``. Both ``log_theta`` and ``logit_pi``
    are per-gene and optimized jointly. Same count-assumption caveat as
    ``NBDecoderLoss`` on normalized+log1p data.
    """
    def __init__(self, num_genes: int, reduction: str = "mean", eps: float = 1e-8,
                 theta_init: float = 0.0, pi_init: float = -2.0):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        # theta = exp(log_theta); pi = sigmoid(logit_pi). theta_init/pi_init are the
        # respective pre-activation inits (defaults: theta=1, pi=sigmoid(-2)=0.12).
        self.log_theta = nn.Parameter(torch.full((num_genes,), float(theta_init)))
        self.logit_pi = nn.Parameter(torch.full((num_genes,), float(pi_init)))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        theta = self.log_theta.exp().clamp(min=self.eps, max=1e6)
        pi = torch.sigmoid(self.logit_pi)
        mu = torch.expm1(input.clamp(min=0)).clamp(min=self.eps)
        y = torch.expm1(target.clamp(min=0))
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


# ---- Configuration ----
