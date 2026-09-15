"""The YAML schema, and how each block maps onto a stage config object.

One table per stage, listing every key the YAML may contain and which attribute
of the underlying config it sets. The loader uses these tables to **reject
unknown keys** -- a typo'd hyperparameter in a published recipe is a
silent-wrong-results bug, not a warning.

The YAML groups keys (``optim``, ``masking``, ``loss``, ``data``, ...) purely
for readability. The groups carry no meaning beyond that; a key's group is
recorded here so the mapping is visible in one place instead of being inferred.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "PRETRAIN_SCHEMA", "FINETUNE_SCHEMA", "DECODER_SCHEMA",
    "TOP_LEVEL_KEYS", "RUN_KEYS", "DATA_KEYS",
    "SchemaError", "flatten_stage", "validate_stage",
]


class SchemaError(ValueError):
    """A config key is unknown, missing, or the wrong shape."""


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
TOP_LEVEL_KEYS = frozenset({
    "version", "name", "seed", "device", "paths", "run", "data", "stages",
})

RUN_KEYS = frozenset({
    "root", "stages", "runner", "resume", "wandb", "run_suffix",
})

DATA_KEYS = frozenset({
    "train", "val", "train_gene_col", "val_gene_col", "expect",
})

# ---------------------------------------------------------------------------
# Stage schemas: {group: {yaml_key: config_attr}}
# ---------------------------------------------------------------------------
# `None` as the attr means "consumed by the loader / orchestrator, not by the
# stage config object" -- e.g. `init_from`, or the read-only annotations that
# exist to make an implicit behaviour visible to a reader.

PRETRAIN_SCHEMA: Dict[str, Dict[str, Optional[str]]] = {
    "model": {
        "model_size": "model_size",
        "hidden_size": "hidden_size",
        "mae_decoder_hidden": "mae_decoder_hidden",
        "dropout": "dropout",
    },
    "optim": {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "optimizer": "optimizer",
        "lr_scheduler": "lr_scheduler",
        "warmup_epochs": "warmup_epochs",
        "warmup_start_factor": "warmup_start_factor",
        "gradient_clip_val": "gradient_clip_val",
        "early_stopping_patience": "early_stopping_patience",
        "validation_freq": "validation_freq",
    },
    "masking": {
        "mask_probability": "mask_probability",
        "corruption_mode": "corruption_mode",
        "gaussian_sigma": "gaussian_sigma",
        "variable_masking": "variable_masking",
        "gene_presence_mask": "gene_presence_mask",
        "presence_shift": "presence_shift",
    },
    "loss": {
        "reconstruction_loss": "reconstruction_loss",
        "masked_data_weight": "masked_data_weight",
        "mask_loss_weight": "mask_loss_weight",
        "mlw_schedule_start": "mlw_schedule_start",
        "mlw_schedule_end": "mlw_schedule_end",
        "mlw_schedule_warmup": "mlw_schedule_warmup",
        "mlw_schedule_cooldown": "mlw_schedule_cooldown",
        "clamp_max_log1p": "clamp_max_log1p",
        "w_mse": "w_mse",
        "w_pois": "w_pois",
        "recon_loss_scale": "recon_loss_scale",
        "gpl_weight": "gpl_weight",
        "loss_diagnostics": "loss_diagnostics",
    },
    "data": {
        "data_mode": "data_mode",
        "target_sum": "target_sum",
        "val_split": "val_split",
        "scale": "scale",
        "skip_normalize": "skip_normalize",
        "skip_log1p": "skip_log1p",
        "optim_chunk_size": "optim_chunk_size",
        "optim_buffer_chunks": "optim_buffer_chunks",
        "min_gene_overlap_pct": "min_gene_overlap_pct",
    },
    "runtime": {
        "enable_amp": "enable_amp",
        "compile": "compile",
        "use_multi_gpu": "use_multi_gpu",
        "use_ddp": "use_ddp",
        "freeze_mask_predictor": "freeze_mask_predictor",
    },
    "eval": {
        "enabled": "eval_enabled",
        "frequency": "eval_frequency",
        "batch_size": "eval_batch_size",
        "h5ad_path": "eval_h5ad_path",
        "scib": "scib_eval_enabled",
    },
}

FINETUNE_SCHEMA: Dict[str, Dict[str, Optional[str]]] = {
    # Not a group: a bare key at the stage root.
    "": {
        "init_from": None,          # resolved by the loader
    },
    "unmasking": {
        "mode": "unmasking_mode",
        "zero_at": "unmasking_zero_at",
        "final_ratio": "unmasking_final_ratio",
        # Read-only annotation. Exists so a reader can see that under
        # mode=gradual the mask probability comes from the PARENT checkpoint,
        # not from masking.mask_probability below. Asserted by the loader.
        "effective_mask_prob_source": None,
    },
    "optim": {
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "optimizer": "optimizer",
        "lr_scheduler": "lr_scheduler",
        "warmup_epochs": "warmup_epochs",
        "warmup_start_factor": "warmup_start_factor",
        "gradient_clip": "gradient_clip",
        "early_stopping": "early_stopping",
        "validation_freq": "validation_freq",
    },
    "masking": {
        "mask_probability": "mask_probability",
        "corruption_mode": "corruption_mode",
        "gaussian_sigma": "gaussian_sigma",
    },
    "loss": {
        "recon_loss": "recon_loss",
        "masked_data_weight": "masked_data_weight",
        "mask_loss_weight": "mask_loss_weight",
        "mlw_schedule_start": "mlw_schedule_start",
        "mlw_schedule_end": "mlw_schedule_end",
        "mlw_schedule_warmup": "mlw_schedule_warmup",
        "mlw_schedule_cooldown": "mlw_schedule_cooldown",
        "clamp_max_log1p": "clamp_max_log1p",
    },
    "runtime": {
        "data_mode": "data_mode",
        "freeze_mask_predictor": "freeze_mask_predictor",
        "enable_amp": "enable_amp",
        "compile": "compile",
    },
    "eval": {
        "eval_freq": "eval_freq",
    },
    "load": {
        "strict": None,
        "expect_unexpected": None,
        "expect_missing": None,
    },
}

DECODER_SCHEMA: Dict[str, Dict[str, Optional[str]]] = {
    "": {
        "init_from": None,
    },
    "encoder": {
        "freeze_encoder": "freeze_encoder",
        "encoder_lr": "encoder_lr",
        "skip_normalize": "skip_normalize",
        "skip_log1p": "skip_log1p",
    },
    "architecture": {
        "decoder_size": "decoder_size",
        "decoder_hidden_dim": "decoder_hidden_dim",
        "decoder_num_blocks": "decoder_num_blocks",
        "decoder_dropout": "decoder_dropout",
        "decoder_se_reduction": "decoder_se_reduction",
        "decoder_output_activation": "decoder_output_activation",
        "decoder_weight_tie": "decoder_weight_tie",
        "decoder_bottleneck_k": "decoder_bottleneck_k",
        "decoder_monarch_blocks": "decoder_monarch_blocks",
        "decoder_kron_rank": "decoder_kron_rank",
        "decoder_glu": "decoder_glu",
        "decoder_sparsity_24": "decoder_sparsity_24",
    },
    "optim": {
        "lr": "lr",
        "eta_min": "eta_min",
        "weight_decay": "weight_decay",
        "gradient_clip": "gradient_clip",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "loss_type": "loss_type",
        "smooth_l1_beta": "smooth_l1_beta",
        "weight_init": "weight_init",
        "dispersion_lr": "dispersion_lr",
        "disp_theta_init": "disp_theta_init",
        "zinb_pi_init": "zinb_pi_init",
    },
    "data": {
        "data_mode": "data_mode",
        "val_split": "val_split",
        "optim_chunk_size": "optim_chunk_size",
        "optim_buffer_chunks": "optim_buffer_chunks",
        "gene_presence_mask": "gene_presence_mask",
        "presence_shift": "presence_shift",
        "min_gene_overlap_pct": "min_gene_overlap_pct",
        "train_gene_col": "train_gene_col",
        "eval_gene_col": "eval_gene_col",
    },
    "augmentation": {
        "no_augmentation": "no_augmentation",
        "gvf_cache_dir": "gvf_cache_dir",
        "aug_hparams": "aug_hparams",
    },
    "adversarial": {
        "adv_loss": "adv_loss",
        "perc_loss": "perc_loss",
        "feature_extractor_checkpoint": "feature_extractor_checkpoint",
        "adv_mode": "adv_mode",
        "perc_mode": "perc_mode",
        "lambda_adv": "lambda_adv",
        "lambda_perc": "lambda_perc",
        "disc_lr": "disc_lr",
    },
    "eval": {
        "decoder_eval_freq": "decoder_eval_freq",
        "bmdb_eval_freq": "bmdb_eval_freq",
        "run_encoder_benchmark": "run_encoder_benchmark",
        "log_interval": "log_interval",
        "cell_eval_enabled": "cell_eval_enabled",
        "eval_data": "eval_data",
        "eval_present_genes_only": "eval_present_genes_only",
        "select_on": None,          # consumed by the trainer's selection logic
    },
    # Flags added after the released run, listed at their no-op values so a
    # reader can see they were not in play. Applied like any other field.
    "postdates_release": {
        "no_amp": "no_amp",
        "eval_present_genes_only": "eval_present_genes_only",
        "eval_zero_threshold": "eval_zero_threshold",
        "eval_calibrate": "eval_calibrate",
        "eval_libnorm": "eval_libnorm",
        "celleval_calibrate": "celleval_calibrate",
        "celleval_zero_mode": "celleval_zero_mode",
        "celleval_sample_predictive": "celleval_sample_predictive",
    },
}

STAGE_SCHEMAS = {
    "pretrain": PRETRAIN_SCHEMA,
    "finetune": FINETUNE_SCHEMA,
    "decoder": DECODER_SCHEMA,
}


def validate_stage(stage: str, block: Mapping[str, Any]) -> None:
    """Reject unknown groups and unknown keys within a stage block."""
    schema = STAGE_SCHEMAS.get(stage)
    if schema is None:
        raise SchemaError(
            f"unknown stage {stage!r}; expected one of {sorted(STAGE_SCHEMAS)}"
        )
    root_keys = set(schema.get("", {}))
    groups = {g for g in schema if g}

    problems = []
    for key, value in block.items():
        if key in root_keys:
            continue
        if key not in groups:
            near = _suggest(key, sorted(groups | root_keys))
            problems.append(f"unknown group '{stage}.{key}'{near}")
            continue
        if not isinstance(value, Mapping):
            problems.append(f"'{stage}.{key}' must be a mapping, got {type(value).__name__}")
            continue
        allowed = set(schema[key])
        for sub in value:
            if sub not in allowed:
                near = _suggest(sub, sorted(allowed))
                problems.append(f"unknown key '{stage}.{key}.{sub}'{near}")
    if problems:
        raise SchemaError(
            f"{len(problems)} problem(s) in stage '{stage}':\n  " + "\n  ".join(problems)
        )


def _suggest(key: str, candidates) -> str:
    """A 'did you mean' hint, because a typo'd hyperparameter is a real bug."""
    import difflib
    match = difflib.get_close_matches(key, candidates, n=1, cutoff=0.7)
    return f" (did you mean '{match[0]}'?)" if match else ""


def flatten_stage(stage: str, block: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Flatten a grouped stage block into config attributes.

    Returns ``(attrs, meta)``: ``attrs`` are the config-object attributes,
    ``meta`` the loader-consumed keys (those mapped to ``None`` in the schema).
    """
    validate_stage(stage, block)
    schema = STAGE_SCHEMAS[stage]
    attrs: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    for key, value in block.items():
        if key in schema.get("", {}):
            meta[key] = value
            continue
        mapping = schema[key]
        for sub, sub_value in value.items():
            attr = mapping[sub]
            if attr is None:
                meta[f"{key}.{sub}"] = sub_value
            else:
                attrs[attr] = sub_value
    return attrs, meta
