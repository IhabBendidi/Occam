"""Discriminator and feature extraction for adversarial/perceptual decoder losses.

Uses a frozen or trainable Occam encoder to extract features from gene expression
data. Supports frozen, trainable-pretrained, and trainable-scratch encoder modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List


class OccamPreprocessor(nn.Module):
    """Differentiable preprocessing that converts decoder output (log1p space)
    into the format expected by a feature-extractor Occam encoder.

    Pipeline:
        decoder_output (log1p of counts/total*target_sum)
        -> expm1 -> pseudo-counts
        -> gene reorder (decoder order -> encoder order, zero-fill missing)
        -> NormalizePerCell(encoder_target_sum)
        -> log1p
        -> ready for encoder
    """

    def __init__(
        self,
        encoder_target_sum: float,
        decoder_genes: List[str],
        encoder_genes: List[str],
    ):
        super().__init__()
        self.encoder_target_sum = encoder_target_sum
        self.n_encoder_genes = len(encoder_genes)

        # Build gene alignment mapping
        decoder_gene_to_idx = {g: i for i, g in enumerate(decoder_genes)}
        valid_encoder = []
        valid_decoder = []
        for j, gene in enumerate(encoder_genes):
            if gene in decoder_gene_to_idx:
                valid_encoder.append(j)
                valid_decoder.append(decoder_gene_to_idx[gene])

        self.register_buffer(
            "valid_encoder_indices", torch.tensor(valid_encoder, dtype=torch.long)
        )
        self.register_buffer(
            "valid_decoder_indices", torch.tensor(valid_decoder, dtype=torch.long)
        )

        matched = len(valid_encoder)
        logging.info(
            f"Gene alignment: {matched}/{len(encoder_genes)} encoder genes found "
            f"in decoder output ({len(decoder_genes)} decoder genes)"
        )
        if matched < len(encoder_genes) * 0.5:
            logging.warning(
                f"Low gene overlap ({matched}/{len(encoder_genes)}). "
                "Check that decoder and encoder use compatible gene names."
            )

    def forward(self, log1p_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log1p_counts: (batch, n_decoder_genes) decoder output in log1p space.

        Returns:
            (batch, n_encoder_genes) preprocessed for encoder.
        """
        batch_size = log1p_counts.shape[0]

        # Invert log1p to get pseudo-counts
        pseudo_counts = torch.expm1(log1p_counts)
        pseudo_counts = pseudo_counts.clamp(min=0.0)

        # Gene reorder: decoder order -> encoder order (zero-fill missing)
        encoder_input = pseudo_counts.new_zeros(batch_size, self.n_encoder_genes)
        encoder_input[:, self.valid_encoder_indices] = pseudo_counts[
            :, self.valid_decoder_indices
        ]

        # NormalizePerCell
        row_sums = encoder_input.sum(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = encoder_input / row_sums * self.encoder_target_sum

        # Log1p
        return torch.log1p(normalized)


class OccamFeatureExtractor(nn.Module):
    """Combines preprocessing + Occam encoder into a single differentiable module.

    Usage:
        extractor = OccamFeatureExtractor(encoder_model, target_sum, dec_genes, enc_genes)
        features = extractor(decoder_output)  # (batch, hidden_size)
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        encoder_target_sum: float,
        decoder_genes: List[str],
        encoder_genes: List[str],
    ):
        super().__init__()
        self.preprocessor = OccamPreprocessor(
            encoder_target_sum, decoder_genes, encoder_genes
        )
        self.encoder_model = encoder_model

    def forward(self, log1p_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log1p_counts: (batch, n_decoder_genes) in log1p space.

        Returns:
            (batch, hidden_size) encoder embeddings.
        """
        preprocessed = self.preprocessor(log1p_counts)
        return self.encoder_model.feature(preprocessed)


class DiscriminatorHead(nn.Module):
    """Simple linear discriminator head on top of encoder features."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) raw logits (no sigmoid)."""
        return self.head(features)


# ---- Loss functions ----

def hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss."""
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss (decoder wants D to think fakes are real)."""
    return -d_fake.mean()


def perceptual_loss(
    features_real: torch.Tensor, features_fake: torch.Tensor
) -> torch.Tensor:
    """Perceptual loss: MSE between encoder embeddings of real vs reconstructed."""
    return F.mse_loss(features_fake, features_real)
