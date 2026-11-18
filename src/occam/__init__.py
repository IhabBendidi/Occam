"""Occam - a masked autoencoder for single-cell transcriptomes.

An encoder-decoder MLP with a parallel mask-prediction branch, trained on all
genes of the input (no HVG selection) with swap-noise masking, plus a
high-capacity external decoder for expression reconstruction.

Importing this package has no side effects: no filesystem writes, no config
loading, no CUDA initialisation. The heavy submodules are imported lazily.

Entry points:
    occam-train       three-stage pipeline (pretrain -> finetune -> decoder)
    occam-embed       embed .h5ad files with a released checkpoint
    occam-reconstruct reconstruct expression from embeddings
    occam-benchmark   encoder perturbation benchmark
    occam-scib        batch-integration eval
    occam-verify      verify a released checkpoint against golden fixtures
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
