"""Backward compatibility with checkpoints and configs written by the research tree.

Occam renamed the model, which touched three things that were *persisted* to
disk: a config key, a checkpoint key, and a set of environment variables. This
module reads the old forms and never writes them, with one deliberate
exception noted in :func:`checkpoint_write_keys`.

Everything here is one-directional (old -> new). Class and function aliases from
the old naming are deliberately **not** provided -- keeping them would defeat
the rename -- so importing them raises ``ImportError``. See CHANGELOG.md.

Four shims, and only four:

1. :data:`LEGACY_CONFIG_KEYS` -- renamed config/checkpoint keys.
2. :data:`LEGACY_DEFAULTS` -- fields the released configs predate. These values
   are load-bearing: changing one changes how the released checkpoints load.
3. :func:`legacy_env` -- the old environment variable names.
4. :func:`clean_state_dict` -- DDP / ``torch.compile`` prefix stripping. Carried
   verbatim from the research tree; it is why the released checkpoints load at
   all.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "LEGACY_CONFIG_KEYS",
    "LEGACY_DEFAULTS",
    "migrate_config_dict",
    "legacy_env",
    "clean_state_dict",
]


# ---------------------------------------------------------------------------
# 1. Renamed keys
# ---------------------------------------------------------------------------
# `ihabnet_checkpoint` appears in every run_config.json on disk and in
# best_decoder.pth['training_config']. `ihabnet_checkpoint_path` is a top-level
# key of every unified checkpoint. `recon_loss` is the FinetuneConfig field name
# for what PretrainConfig calls `reconstruction_loss`.
LEGACY_CONFIG_KEYS: Dict[str, str] = {
    "ihabnet_checkpoint": "encoder_checkpoint",
    "ihabnet_checkpoint_path": "parent_checkpoint_path",
    "scmae_checkpoint": "encoder_checkpoint",
}

_WARNED_KEYS: set[str] = set()


def migrate_config_dict(config: Mapping[str, Any], *, warn: bool = True) -> Dict[str, Any]:
    """Rename legacy keys in a config dict read from disk.

    An explicit new-name key always wins over a legacy one, so re-migrating an
    already-migrated dict is a no-op.
    """
    out: Dict[str, Any] = dict(config)
    for old, new in LEGACY_CONFIG_KEYS.items():
        if old not in out:
            continue
        value = out.pop(old)
        if new in out:
            continue  # the new name is already present and authoritative
        out[new] = value
        if warn and old not in _WARNED_KEYS:
            _WARNED_KEYS.add(old)
            warnings.warn(
                f"config key '{old}' is deprecated; read as '{new}'. "
                f"Written by the pre-Occam research tree.",
                DeprecationWarning,
                stacklevel=3,
            )
    return out


# ---------------------------------------------------------------------------
# 2. Defaults the released configs predate
# ---------------------------------------------------------------------------
# The released stage-1 config (2026-04-28) and stage-3 config (2026-05-01) were
# written before these fields existed. Their absence must resolve to exactly the
# values that were in force at load time, or the released checkpoints load
# differently than they did.
#
# DO NOT change a value here to "improve" a default. Add a new field instead.
LEGACY_DEFAULTS: Dict[str, Any] = {
    # PretrainConfig / FinetuneConfig
    "corruption_mode": "swap",
    "gaussian_sigma": 0.5,
    "skip_normalize": False,
    "skip_log1p": False,
    "recon_loss_scale": 1.0,
    "mae_decoder_hidden": None,
    "min_gene_overlap_pct": None,
    "variable_masking": False,
    "gene_presence_mask": False,
    "presence_shift": 1.0,
    "gpl_weight": 0.0,
    # DecoderTrainConfig. `rectified_sigmoid` is the value in force for the
    # released decoder -- the softplus option postdates it. See
    # scripts/verify_checkpoint.py, which determines this empirically rather
    # than trusting the default.
    "decoder_output_activation": "rectified_sigmoid",
    "decoder_weight_tie": False,
    "decoder_bottleneck_k": 1,
    "decoder_monarch_blocks": 0,
    "decoder_kron_rank": 0,
    "decoder_glu": False,
    "decoder_sparsity_24": False,
    "dispersion_lr": None,
    "disp_theta_init": 0.0,
    "zinb_pi_init": -2.0,
}


def apply_legacy_defaults(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Fill in fields the released configs predate, without overwriting anything set."""
    out = dict(config)
    for key, value in LEGACY_DEFAULTS.items():
        out.setdefault(key, value)
    return out


# ---------------------------------------------------------------------------
# 3. Environment variables
# ---------------------------------------------------------------------------
_LEGACY_ENV_PREFIXES = ("IHABNET_", "SCMAE_")


def legacy_env(suffix: str) -> Optional[str]:
    """Read a deprecated ``IHABNET_*`` / ``SCMAE_*`` environment variable.

    Callers should check ``OCCAM_<suffix>`` first; this is the fallback.
    """
    for prefix in _LEGACY_ENV_PREFIXES:
        name = prefix + suffix
        value = os.environ.get(name)
        if value:
            if name not in _WARNED_KEYS:
                _WARNED_KEYS.add(name)
                warnings.warn(
                    f"environment variable {name} is deprecated; use OCCAM_{suffix}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            return value
    return None


# ---------------------------------------------------------------------------
# 4. State-dict prefix stripping
# ---------------------------------------------------------------------------
def clean_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip DDP ``module.`` and ``torch.compile`` ``_orig_mod.`` key prefixes.

    Verbatim from ``training_pipeline.py:160`` of the research tree.
    """
    cleaned: Dict[str, Any] = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = v
    return cleaned


# ---------------------------------------------------------------------------
# Write side: there isn't one
# ---------------------------------------------------------------------------
# Occam writes only new names. Checkpoints it produces carry
# `parent_checkpoint_path`, never the old `ihabnet_checkpoint_path`, and the
# renames above are strictly read-only.
#
# An earlier draft of this module exported a helper that wrote both names for
# one release cycle. It was removed once nothing used it: an unused
# compatibility path is worse than none, because a reader cannot tell whether
# the dual-write is happening. If an external consumer turns out to need the
# old key, add it back at the single call site that writes the checkpoint, not
# here.
