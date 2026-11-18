"""Load a checkpoint into something the evaluation suite can score.

Occam ships two shapes of checkpoint and they are easy to confuse:

*plain MAE* (stages 1 and 2)
    ``config`` + ``model_state_dict`` holding the whole autoencoder --
    encoder, the MAE's own decoder, and the mask predictor.

*unified* (stage 3, and the released artifact)
    ``encoder_config`` + ``encoder_state_dict`` for the encoder half, and
    ``decoder_config`` + ``decoder_state_dict`` for the external decoder.
    ⚠️ Its ``model_state_dict`` is byte-identical to ``decoder_state_dict``,
    NOT the whole model -- code that assumes otherwise silently ends up with
    an encoder-shaped hole. That is why this function exists rather than each
    caller reaching into the dict.

:func:`load_encoder_wrapper` returns an object satisfying the interface the
evaluation code expects: ``.model`` (with ``.feature(x)``), ``.device``,
``._gene_names``, ``._transforms`` and ``.config``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from occam._compat import clean_state_dict
from occam.transforms import Log1p, NormalizePerCell

__all__ = ["EncoderWrapper", "load_encoder_wrapper", "checkpoint_kind"]


class EncoderWrapper:
    """Minimal object satisfying the evaluation suite's trainer interface."""

    def __init__(self, model, device, gene_names, target_sum, config=None,
                 amp_enabled: bool = False):
        self.model = model
        self.device = device
        self._gene_names = list(gene_names)
        self.config = config
        self.amp_enabled = amp_enabled
        skip_normalize = bool(getattr(config, "skip_normalize", False)) if config else False
        skip_log1p = bool(getattr(config, "skip_log1p", False)) if config else False
        transforms = []
        if not skip_normalize:
            transforms.append(NormalizePerCell(target_sum))
        if not skip_log1p:
            transforms.append(Log1p())
        self._transforms = transforms
        self.target_sum = target_sum


def checkpoint_kind(checkpoint: Dict[str, Any]) -> str:
    """``'unified'`` or ``'mae'``."""
    if checkpoint.get("is_unified_checkpoint") or "encoder_state_dict" in checkpoint:
        return "unified"
    return "mae"


def load_encoder_wrapper(checkpoint_path: str, device: str = "cuda",
                         amp_enabled: bool = False) -> EncoderWrapper:
    """Load either checkpoint shape and return a scorable wrapper."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    kind = checkpoint_kind(ck)
    print(f"[load] {os.path.basename(checkpoint_path)}: {kind} checkpoint")

    gene_names = (ck.get("gene_names")
                  or (ck.get("dataset_info") or {}).get("gene_names"))
    if not gene_names:
        raise ValueError(
            f"{checkpoint_path} carries no gene_names. Occam cannot align input "
            f"genes without them; the vocabulary is not recoverable from weights."
        )

    if kind == "unified":
        from occam.model import OccamEncoder

        model, info = OccamEncoder.from_checkpoint(ck, strict=True)
        model.to(device).eval()
        cfg_source = ck.get("training_config") or {}
        target_sum = ck.get("target_sum") or cfg_source.get("target_sum") or 1e4
        config = _as_pretrain_config({
            "num_genes": ck["encoder_config"]["num_genes"],
            "model_size": ck["encoder_config"].get("model_size", "small"),
            "hidden_size": ck["encoder_config"].get("hidden_size"),
            "target_sum": target_sum,
            "skip_normalize": cfg_source.get("skip_normalize", False),
            "skip_log1p": cfg_source.get("skip_log1p", False),
        })
        print(f"[load]   encoder {config.model_size}, {config.num_genes} genes, "
              f"target_sum={target_sum}, {len(info['loaded'])} tensors strict=True")
    else:
        from occam.model import AutoEncoder, resolve_decoder_hidden

        raw_cfg = ck["config"]
        raw_cfg = raw_cfg if isinstance(raw_cfg, dict) else dict(vars(raw_cfg))
        state_dict = clean_state_dict(ck["model_state_dict"])
        model_size = raw_cfg.get("model_size", "small")
        model = AutoEncoder(
            num_genes=raw_cfg["num_genes"],
            model_size=model_size,
            decoder_hidden=resolve_decoder_hidden(raw_cfg, state_dict, model_size),
            dropout=raw_cfg.get("dropout", 0),
            reconstruction_loss=raw_cfg.get("reconstruction_loss", "mse"),
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        target_sum = ((ck.get("dataset_info") or {}).get("target_sum")
                      or raw_cfg.get("target_sum") or 1e4)
        config = _as_pretrain_config(raw_cfg)
        print(f"[load]   model {model_size}, {raw_cfg['num_genes']} genes, "
              f"target_sum={target_sum}")

    return EncoderWrapper(model, device, gene_names, target_sum,
                          config=config, amp_enabled=amp_enabled)


def _as_pretrain_config(data: Dict[str, Any]):
    from occam.mae.config import PretrainConfig

    return PretrainConfig.from_dict(data)
