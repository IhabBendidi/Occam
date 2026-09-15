import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
import torch.nn.functional as F

import math

from occam.losses import create_loss, LOSS_REGISTRY, _LOSS_ALIASES


class RectifiedSigmoid(nn.Module):
    """This activation function enables more straightforward optimziation landscape due to its boundedness.

    It could also be worth considering this paper https://www.alphaxiv.org/abs/2412.20851
    which defines rectified sigmoid as Re-σ(x) = 1/2 [ReLU(x + 1) − ReLU(x − 1)].

    Nonetheless, we have found that the current implementation facilitates stable and effective training.
    """

    def __init__(self, upper_bound: int = 2000):
        super().__init__()
        self.upper_bound = math.log1p(upper_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upper_bound * torch.relu(2 * torch.sigmoid(x / (2 * math.e)) - 1)  # type: ignore



class BasicPoissonNLLLogSpace(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction
        self.rectified_sigmoid = RectifiedSigmoid()

    def forward(self, log1p_input, log1p_target):

        # we must transform target out of log space to have valid poisson loss
        loss = log1p_input.exp() - log1p_target.exp() * log1p_input
        return loss



# Poisson NLL operating in log1p space
class PoissonNLLLogSpace(nn.Module):
    def __init__(self, reduction: str = "none", eps: float = 1e-8, clamp_max_log1p: float = 5.5):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.clamp_max_log1p = clamp_max_log1p

    def forward(self, log1p_input: torch.Tensor, log1p_target: torch.Tensor) -> torch.Tensor:
        z = F.softplus(log1p_input)
        z = torch.clamp(z, max=self.clamp_max_log1p)
        mu_hat = torch.expm1(z)
        mu_hat = torch.clamp(mu_hat, min=self.eps)
        log_mu_hat = torch.log(mu_hat)

        target_counts = torch.expm1(log1p_target)
        target_counts = torch.clamp(target_counts, min=0.0)

        loss = mu_hat - target_counts * log_mu_hat
        return loss


# Model size configurations
#
# `decoder_hidden` is the width of the decoder's single hidden layer:
#     Linear(hidden_size + num_genes, decoder_hidden) -> LN -> Mish -> Linear(decoder_hidden, num_genes)
#
# Rule: decoder_hidden = 2 * hidden_size. For 'small' this evaluates to 1024, which is
# exactly the historical hardcoded value -- so 'small' models are bit-identical to every
# checkpoint trained before decoder scaling existed (Chain A / Chain B included).
# 'base' and 'large' now widen the decoder in step with the encoder instead of forcing
# a 4.3x-larger encoder through the same fixed 1024-unit reconstruction bottleneck.
MODEL_CONFIGS = {
    'small': {
        'hidden_size': 512,
        'encoder_layers': [2048, 1024, 512],  # gene_dim -> 2048 -> 1024 -> 512
        'decoder_hidden': 1024,               # == legacy hardcoded width
        'description': 'Small model (512-dim embeddings)'
    },
    'base': {
        'hidden_size': 768,
        'encoder_layers': [4096, 2048, 768],  # gene_dim -> 4096 -> 2048 -> 768
        'decoder_hidden': 1536,
        'description': 'Base model (768-dim embeddings)'
    },
    'large': {
        'hidden_size': 1024,
        'encoder_layers': [8192, 4096, 1024],  # gene_dim -> 8192 -> 4096 -> 1024
        'decoder_hidden': 2048,
        'description': 'Large model (1024-dim embeddings)'
    },
}

# Width used by checkpoints written before `decoder_hidden` was configurable.
LEGACY_DECODER_HIDDEN = 1024


def resolve_decoder_hidden(ckpt_config=None, state_dict=None, model_size='small'):
    """Work out the MAE decoder hidden width for a checkpoint being reloaded.

    NOTE: this is the *MAE's own* decoder (``AutoEncoder.decoder``), NOT the external
    ``HybridNormGlobalSkipDecoder`` used by the decoder-training and virtual-cell
    pipelines -- that one is sized by ``--decoder_hidden_dim`` and is unrelated.

    Resolution order:
      1. explicit ``mae_decoder_hidden`` recorded in the checkpoint config
      2. inferred from the decoder's first weight matrix in ``state_dict``
      3. ``LEGACY_DECODER_HIDDEN`` (1024) -- what every pre-scaling checkpoint used

    We deliberately do NOT fall back to ``MODEL_CONFIGS[model_size]``: a checkpoint that
    predates this feature used 1024 regardless of its size, and guessing the new default
    would build a decoder that cannot load those weights.
    """
    if ckpt_config:
        dh = ckpt_config.get('mae_decoder_hidden') or ckpt_config.get('decoder_hidden')
        if dh:
            return int(dh)
    if state_dict:
        for key in ('decoder.0.weight', 'module.decoder.0.weight'):
            w = state_dict.get(key)
            if w is not None and hasattr(w, 'shape'):
                return int(w.shape[0])
    return LEGACY_DECODER_HIDDEN


def build_encoder_modules(num_genes, encoder_layers, hidden_size, dropout=0.0):
    """Build the encoder stack shared by :class:`AutoEncoder` and :class:`OccamEncoder`.

    Module order is load-bearing: it fixes the integer indices that appear in
    every checkpoint's state-dict keys. For the released `small` encoder
    (`encoder_layers=[2048, 1024, 512]`, `hidden_size=512`) the parametric
    indices are 1, 2, 4, 5, 7, 8 -- which is exactly what
    ``encoder_state_dict`` contains. Do not reorder or insert modules.
    """
    modules = [
        nn.Dropout(p=dropout),
        nn.Linear(num_genes, encoder_layers[0]),
        nn.LayerNorm(encoder_layers[0]),
        nn.Mish(inplace=True),
    ]
    for i in range(len(encoder_layers) - 1):
        modules.extend([
            nn.Linear(encoder_layers[i], encoder_layers[i + 1]),
            nn.LayerNorm(encoder_layers[i + 1]),
            nn.Mish(inplace=True),
        ])
    # Final projection only when the last encoder width differs from hidden_size.
    if encoder_layers[-1] != hidden_size:
        modules.extend([
            nn.Linear(encoder_layers[-1], hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
        ])
    return modules


class OccamEncoder(torch.nn.Module):
    """The encoder half of a released checkpoint, and nothing else.

    Exists so that ``encoder_state_dict`` can be loaded with ``strict=True``.
    Loading it into a full :class:`AutoEncoder` cannot be strict -- the
    AutoEncoder also owns ``decoder.*``, ``mask_predictor.*`` and possibly
    ``_recon_loss_fn.*``, which a unified checkpoint does not carry -- so the
    research tree used ``strict=False`` and could not distinguish "these keys
    are legitimately absent" from "the architecture is wrong".

    This module's ``state_dict()`` keys are exactly the 13 in the released
    checkpoint: ``pos_weight`` plus ``encoder.{1,2,4,5,7,8}.{weight,bias}``.

    Use :meth:`from_checkpoint` rather than constructing it by hand; it reads
    the geometry out of ``encoder_config`` so there is no default to get wrong.
    """

    def __init__(self, num_genes, model_size='small', hidden_size=None, dropout=0.0):
        super().__init__()
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")
        config = MODEL_CONFIGS[model_size]

        self.num_genes = int(num_genes)
        self.model_size = model_size
        self.hidden_size = int(hidden_size if hidden_size is not None else config['hidden_size'])

        self.encoder = nn.Sequential(*build_encoder_modules(
            num_genes=self.num_genes,
            encoder_layers=config['encoder_layers'],
            hidden_size=self.hidden_size,
            dropout=dropout,
        ))
        # Present in every checkpoint; carried so the key inventory matches.
        self.register_buffer('pos_weight', torch.ones(self.num_genes))

    def forward(self, x):
        return self.encoder(x)

    def feature(self, x):
        """Embed a batch. Named to match :class:`AutoEncoder`'s interface.

        The evaluation suite calls ``model.feature(x)``, so providing it here
        lets a released unified checkpoint be benchmarked directly, without
        rebuilding the full autoencoder around an encoder-only state dict.
        """
        return self.encoder(x)

    @classmethod
    def from_checkpoint(cls, checkpoint, strict=True):
        """Build from a checkpoint dict and load its encoder weights.

        Accepts a unified encoder+decoder checkpoint (``encoder_state_dict`` +
        ``encoder_config``) or a full MAE checkpoint (``model_state_dict``, from
        which only the ``encoder.*`` and ``pos_weight`` entries are taken).

        Returns ``(model, info)`` where ``info`` records what was loaded, so the
        caller can assert on it instead of trusting a silent ``strict=False``.
        """
        from occam._compat import clean_state_dict

        cfg = checkpoint.get('encoder_config')
        if cfg is None:
            raw_cfg = checkpoint.get('config') or checkpoint.get('training_config') or {}
            if not isinstance(raw_cfg, dict):
                raw_cfg = getattr(raw_cfg, '__dict__', {}) or {}
            cfg = {
                'num_genes': len(checkpoint.get('gene_names', ())) or raw_cfg.get('num_genes'),
                'model_size': raw_cfg.get('model_size', 'small'),
                'hidden_size': raw_cfg.get('hidden_size'),
            }
        if not cfg.get('num_genes'):
            raise ValueError("checkpoint carries no encoder geometry (num_genes)")

        model = cls(
            num_genes=cfg['num_genes'],
            model_size=cfg.get('model_size', 'small'),
            hidden_size=cfg.get('hidden_size'),
            dropout=0.0,
        )

        if 'encoder_state_dict' in checkpoint:
            source, origin = checkpoint['encoder_state_dict'], 'encoder_state_dict'
        elif 'model_state_dict' in checkpoint:
            source, origin = checkpoint['model_state_dict'], 'model_state_dict'
        else:
            raise ValueError("checkpoint has neither encoder_state_dict nor model_state_dict")

        cleaned = clean_state_dict(source)
        wanted = set(model.state_dict().keys())
        subset = {k: v for k, v in cleaned.items() if k in wanted}
        dropped = sorted(set(cleaned) - wanted)

        missing, unexpected = model.load_state_dict(subset, strict=strict)
        info = {
            'origin': origin,
            'loaded': sorted(subset),
            'dropped_from_source': dropped,
            'missing': sorted(missing),
            'unexpected': sorted(unexpected),
        }
        model.eval()
        return model, info


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_genes,
        model_size='small',  # 'small', 'base', 'large'
        hidden_size=None,   # Override if you want custom size
        decoder_hidden=None,  # Override decoder width; None -> MODEL_CONFIGS[model_size]
        dropout=0,
        masked_data_weight=.75,
        mask_loss_weight=0.7,
        reconstruction_loss: str = 'mse',
        w_mse: float = 1.0,
        w_pois: float = 0.1,
        clamp_max_log1p: float = 5.5,
        custom_encoder_layers=None,  # Custom architecture
        recon_loss_scale: float = 1.0,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight
        self.reconstruction_loss = reconstruction_loss.lower() if isinstance(reconstruction_loss, str) else 'mse'
        self.reconstruction_loss = _LOSS_ALIASES.get(self.reconstruction_loss, self.reconstruction_loss)
        if self.reconstruction_loss not in LOSS_REGISTRY:
            raise ValueError(f"reconstruction_loss must be one of {list(LOSS_REGISTRY.keys())}")
        self.recon_loss_scale = recon_loss_scale
        self._recon_loss_fn = create_loss(
            self.reconstruction_loss,
            num_genes=num_genes,
            clamp_max_log1p=clamp_max_log1p,
            w_mse=w_mse,
            w_pois=w_pois,
        )
        # Keep legacy attributes for backward compat (finetuning code references these)
        self._poisson_loss_fn = self._recon_loss_fn if 'poisson' in self.reconstruction_loss else PoissonNLLLogSpace(clamp_max_log1p=clamp_max_log1p)
        self._basic_poisson_loss_fn = BasicPoissonNLLLogSpace()
        self.w_mse = float(w_mse)
        self.w_pois = float(w_pois)
        
        # Get model configuration
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_size]
        
        # Use provided hidden_size or config default
        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = config['hidden_size']
        
        # Use custom layers or config default
        if custom_encoder_layers is not None:
            encoder_layers = custom_encoder_layers
        else:
            encoder_layers = config['encoder_layers']

        # Decoder width: explicit override, else scale with model size
        if decoder_hidden is not None:
            self.decoder_hidden = int(decoder_hidden)
        else:
            self.decoder_hidden = config.get('decoder_hidden', LEGACY_DECODER_HIDDEN)

        print(f"Creating {model_size} Occam model:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Encoder layers: {encoder_layers}")
        print(f"  Decoder hidden: {self.decoder_hidden}")
        print(f"  Description: {config['description']}")
        
        # Build encoder with configurable layers
        
        # Built by the shared helper so AutoEncoder and OccamEncoder cannot
        # drift apart -- OccamEncoder is what makes a strict=True load of a
        # released encoder half possible.
        encoder_modules = build_encoder_modules(
            num_genes=self.num_genes,
            encoder_layers=encoder_layers,
            hidden_size=self.hidden_size,
            dropout=dropout,
        )
        self.encoder = nn.Sequential(*encoder_modules)

        self.mask_predictor = nn.Linear(self.hidden_size, num_genes)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size + num_genes, self.decoder_hidden),
            nn.LayerNorm(self.decoder_hidden),
            nn.Mish(inplace=True),
            nn.Linear(self.decoder_hidden, num_genes)
        )
        
        # Register pos_weight as a buffer to avoid recreating it every forward pass
        # Will be updated in loss_mask when mask_probability changes
        self.register_buffer('pos_weight', torch.ones(num_genes))
        self._cached_mask_probability = None

    def forward_mask(self, x):
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        predicted_mask = torch.sigmoid(predicted_mask)
        reconstruction = self.decoder(
            torch.cat([latent, predicted_mask], dim=1))

            
        return latent, predicted_mask, reconstruction

    def forward(self, x, y, mask, mask_probability=0.4):
        """Forward pass used by DDP — delegates to loss_mask."""
        return self.loss_mask(x, y, mask, mask_probability)

    def _compute_loss(self, x, y, mask, mask_probability):
        """Pure tensor computation — no .item(), no dict, no buffer mutation.

        Safe for torch.compile (no graph breaks).
        """
        latent, predicted_mask, reconstruction = self.forward_mask(x)
        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)

        per_elem_loss = self._recon_loss_fn(reconstruction, y)

        recon_pre = torch.mul(w_nums, per_elem_loss)

        reconstruction_loss = (1 - self.mask_loss_weight) * self.recon_loss_scale * recon_pre
        reconstruction_loss = reconstruction_loss.mean()

        mask_bce_pre = bce_logits(predicted_mask, mask, reduction="mean",
                                  pos_weight=self.pos_weight)

        mask_loss = self.mask_loss_weight * mask_bce_pre

        loss = reconstruction_loss + mask_loss

        return latent, loss, per_elem_loss, recon_pre, reconstruction_loss, mask_bce_pre, mask_loss, reconstruction

    def loss_mask(self, x, y, mask, mask_probability=0.4):
        # Update pos_weight buffer BEFORE compiled forward (not inside compiled graph)
        if self._cached_mask_probability != mask_probability:
            if mask_probability <= 0:
                self.pos_weight.fill_(1.0)
            else:
                raw_pw = (1 - mask_probability) / mask_probability
                self.pos_weight.fill_(min(raw_pw, 100.0))
            self._cached_mask_probability = mask_probability

        latent, loss, per_elem_loss, recon_pre, reconstruction_loss, mask_bce_pre, mask_loss, reconstruction = \
            self._compute_loss(x, y, mask, mask_probability)
        # Expose reconstruction for external consumers (e.g. GPL auxiliary loss)
        self._last_reconstruction = reconstruction

        # Metrics extraction (outside compiled graph, uses .item())
        metrics = {}
        metrics['recon_pre'] = recon_pre.detach().mean().item()
        metrics['recon_post'] = reconstruction_loss.detach().item()
        metrics['mask_pre'] = mask_bce_pre.detach().item()
        metrics['mask_post'] = mask_loss.detach().item()
        if self.reconstruction_loss == 'mse':
            metrics['mse_pre'] = per_elem_loss.detach().mean().item()
            metrics['mse_post'] = metrics['mse_pre']
        else:
            metrics['mse_pre'] = None
            metrics['mse_post'] = None
        if 'poisson' in self.reconstruction_loss or self.reconstruction_loss in ('nb', 'zinb'):
            metrics['poisson_pre'] = per_elem_loss.detach().mean().item()
            metrics['poisson_post'] = metrics['poisson_pre']
        else:
            metrics['poisson_pre'] = None
            metrics['poisson_post'] = None
        try:
            self._last_loss_metrics = metrics
        except Exception:
            pass

        return latent, loss

    def feature(self, x):
        latent = self.encoder(x)
        return latent
    
    @staticmethod
    def get_model_info():
        """Get information about available model sizes."""
        print("\nAvailable Occam model sizes:")
        print("-" * 50)
        for size, config in MODEL_CONFIGS.items():
            print(f"{size.upper()}:")
            print(f"  Hidden size: {config['hidden_size']}")
            print(f"  Encoder layers: {config['encoder_layers']}")
            print(f"  Decoder hidden: {config['decoder_hidden']}")
            print(f"  Description: {config['description']}")
            print()
    
    def get_parameter_count(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """Print detailed model architecture."""
        # Calculate parameters for each component
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        mask_predictor_params = sum(p.numel() for p in self.mask_predictor.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = self.get_parameter_count()
        
        print(f"\nOccam Architecture:")
        print(f"Input genes: {self.num_genes}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Decoder hidden: {self.decoder_hidden}")
        print(f"\nParameter Count:")
        print(f"  Encoder:        {encoder_params:>15,} ({100*encoder_params/total_params:>5.1f}%)")
        print(f"  Mask Predictor: {mask_predictor_params:>15,} ({100*mask_predictor_params/total_params:>5.1f}%)")
        print(f"  Decoder:        {decoder_params:>15,} ({100*decoder_params/total_params:>5.1f}%)")
        print(f"  {'─' * 42}")
        print(f"  Total:          {total_params:>15,}")
        print(f"\nTraining Configuration:")
        print(f"  Masked data weight: {self.masked_data_weight}")
        print(f"  Mask loss weight: {self.mask_loss_weight}")
        print(f"  Reconstruction loss: {self.reconstruction_loss}")
        print("\nEncoder architecture:")
        for i, layer in enumerate(self.encoder):
            print(f"  {i}: {layer}")
        print(f"\nMask predictor: {self.mask_predictor}")
        print("Decoder architecture:")
        for i, layer in enumerate(self.decoder):
            print(f"  {i}: {layer}")


def create_occam_model(num_genes, model_size='small', **kwargs):
    """
    Convenience function to create Occam model.

    Args:
        num_genes: Number of input genes
        model_size: 'small' (512), 'base' (768), or 'large' (1024)
        **kwargs: Additional arguments for AutoEncoder

    Returns:
        AutoEncoder model
    """
    model = AutoEncoder(num_genes=num_genes, model_size=model_size, **kwargs)
    model.print_architecture()
    return model


if __name__ == "__main__":
    AutoEncoder.get_model_info()

    num_genes = 2000

    print("Creating Small model:")
    small_model = create_occam_model(num_genes, 'small')

    print("\nCreating Base model:")
    base_model = create_occam_model(num_genes, 'base')

    print("\nCreating Large model:")
    large_model = create_occam_model(num_genes, 'large')


# Backward compatibility alias
create_occam_model = create_occam_model
