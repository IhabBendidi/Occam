"""External decoder architectures (separate from the MAE's own decoder)."""

from __future__ import annotations

__all__ = ["HybridNormGlobalSkipDecoder", "SEResNetDecoder",
           "resolve_decoder_size", "DECODER_SIZE_CONFIGS", "load_decoder"]


def __getattr__(name):
    if name in __all__:
        from occam.decoder import external as _e
        return getattr(_e, name)
    raise AttributeError(f"module 'occam.decoder' has no attribute {name!r}")
