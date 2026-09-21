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
    "PRETRAIN_SCHEMA", "FINETUNE_SCHEMA", "DECODER_SCHEMA", "VCELL_SCHEMA",
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
        # cell-eval settings. `celleval_pert_col` in particular has no usable
        # default: it is the obs column naming the perturbation, it differs per
        # dataset, and a wrong value makes cell-eval SKIP the file with a
        # printed message rather than fail -- which reads as "no metrics" hours
        # later. Set it explicitly for every eval file.
        "celleval_pert_col": "celleval_pert_col",
        "celleval_control_label": "celleval_control_label",
        "celleval_calibrate": "celleval_calibrate",
        "celleval_modes": "celleval_modes",
        "celleval_num_threads": "celleval_num_threads",
        "celleval_de_batch_size": "celleval_de_batch_size",
        "celleval_de_real_cache": "celleval_de_real_cache",
        "celleval_emission": "celleval_emission",
        "celleval_timeout": "celleval_timeout",
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

VCELL_SCHEMA: Dict[str, Dict[str, Optional[str]]] = {
    "": {"init_from": None},                     # the unified encoder+decoder checkpoint
    "benchmark": {
        "dir": "bench_dir", "id": "bench_id", "sha256": "bench_sha256",
        "axis_scope": "bench_axis_scope",
        "eval_cell_lines": "bench_eval_cell_lines",
        "build_test_datasets": "bench_build_test_datasets",
    },
    "data": {
        "paths": "data_paths", "gene_col": "gene_name_col", "pert_col": "pert_gene_col",
        "batch_col": "batch_col", "target_sum": "target_sum",
        "skip_normalize": "skip_normalize", "skip_log1p": "skip_log1p",
        "n_controls_per_sample": "n_controls_per_sample",
        "include_controls_as_training": "include_controls_as_training",
        "min_controls_per_cell_line": "min_controls_per_cell_line",
    },
    "stream": {
        "stream_v2": "stream_v2", "interleave": "stream_interleave",
        "balance_sources": "balance_sources", "fixed_pairs": "stream_fixed_pairs",
        "chunk_size": "optim_chunk_size", "buffer_chunks": "optim_buffer_chunks",
        "num_workers": "num_workers", "control_cache_dir": "control_cache_dir",
        "control_scan_threads": "control_scan_threads", "amp_dtype": "amp_dtype",
        "target_pb_k": "target_pb_k", "ctrl_avg_k": "ctrl_avg_k",
        "max_cells_per_pert": "max_cells_per_pert_stream",
        "min_cells_per_pert": "min_cells_per_pert_stream",
    },
    "encoder": {"freeze_encoder": "freeze_encoder", "freeze_decoder": "freeze_decoder"},
    "graph": {
        # Perturbation-model selection. `adjacency_map` is SimpleAdjacencyMapModel: each
        # gene's embedding is MLP(its adjacency row), with no learned message passing, so
        # it costs a fraction of the 4-layer GATv2's per-step time. `no_gnn` swaps in a
        # plain learnable embedding table -- the falsifier for the graph itself.
        "pert_model_type": "pert_model_type", "no_gnn": "no_gnn",
        "adj_hidden_dim": "adj_hidden_dim", "adj_dropout": "adj_dropout",
        "adj_deg": "adj_deg", "adj_normalize": "adj_normalize",
        "adj_self_loops": "adj_self_loops", "adj_only_reverse": "adj_only_reverse",
        "adj_make_undirected": "adj_make_undirected", "adj_use_ffn": "adj_use_ffn",
        "adj_activation": "adj_activation",
        "graph_type": "graph_type", "graph_cache_dir": "graph_cache_dir",
        "master_gene_set_path": "master_gene_set_path",
        "graph_types": "graph_types", "graph_edge_filter_mode": "graph_edge_filter_mode",
        "gnn_layer_type": "gnn_layer_type", "gnn_hidden_dim": "gnn_hidden_dim",
        "gnn_num_layers": "gnn_num_layers", "gnn_num_heads": "gnn_num_heads",
        "gnn_dropout": "gnn_dropout", "gnn_activation": "gnn_activation",
        "pert_latent_dim": "pert_latent_dim", "use_edge_weight": "use_edge_weight",
        "graph_reduce2perts": "graph_reduce2perts", "data_perts_only": "data_perts_only",
    },
    # ---- the ablation axes, kept adjacent so a grid row is readable ----
    "interaction": {
        "interaction_type": "interaction_type", "film_mode": "film_mode",
        "film_scale": "film_scale", "film_hidden_dim": "film_hidden_dim",
        "film_activation": "film_activation", "film_mean_free": "film_mean_free",
    },
    "optim": {
        "lr": "lr", "decoder_lr": "decoder_lr", "weight_decay": "weight_decay",
        "batch_size": "batch_size", "epochs": "epochs",
        "warmup_epochs": "warmup_epochs", "eta_min": "eta_min",
        "save_every_epochs": "save_every_epochs",
        "gradient_clip": "gradient_clip",
        "decoder_lr_warmup_epochs": "decoder_lr_warmup_epochs",
    },
    "loss": {
        "loss_type": "loss_type", "smooth_l1_beta": "smooth_l1_beta",
        "cosine_loss_weight": "cosine_loss_weight",
        "recon_loss_scale": "recon_loss_scale",
        "loss_norm": "loss_norm", "source_weight": "source_weight",
        "latent_loss_type": "latent_loss_type", "delta_reg_type": "delta_reg_type",
        # ⚠️ LATENT-SPACE LOSSES. `latent_loss_type` alone was routable; its WEIGHT was not,
        # so the term could be switched on and never scaled -- the F-loss1 shape. All of them
        # are routed now.
        #   latent_loss_weight : scales the MSE latent loss. Default in config.py is 1.0, which
        #                        the measured scales (F-latent2) put at ~99.5% of the total
        #                        loss -- never use it unexamined.
        #   latent_loss_alpha  : INERT. It blends `delta_z` and `z_hat`, which are the SAME
        #                        function (F-latent2), so any alpha gives the same number.
        #   latent_cos_weight  : 1 - cos(...) in latent space, the direction-only counterpart.
        #   latent_cos_target  : delta_z (effect direction) | z_hat (absolute state). Unlike
        #                        the MSE these are NOT equivalent -- cosine is not
        #                        translation-invariant.
        "latent_loss_weight": "latent_loss_weight",
        "latent_loss_alpha": "latent_loss_alpha",
        "latent_cos_weight": "latent_cos_weight",
        "latent_cos_target": "latent_cos_target",
        "vicreg_var_weight": "vicreg_var_weight",
        "vicreg_cov_weight": "vicreg_cov_weight",
        "vicreg_gamma": "vicreg_gamma",
        "vicreg_target": "vicreg_target",
        # F-recon1 split reconstruction. Routed here too, or --set cannot reach them:
        # dataclass + YAML + THIS TABLE are three independent layers.
        "recon_split": "recon_split",
        "recon_split_mode": "recon_split_mode",
        "recon_delta_target": "recon_delta_target",
        "recon_ctrl_weight": "recon_ctrl_weight",
        "recon_delta_weight": "recon_delta_weight",
        "expression_loss_mode": "expression_loss_mode",
        "delta_mag_weight": "delta_mag_weight",
        # DCOS -- the direction of the EFFECT, not of the profile. Distinct from
        # `cosine_loss_weight`, which is cos(prediction, target) and is dominated by the
        # baseline expression every perturbation shares.
        "delta_cos_weight": "delta_cos_weight",
        # nb_smooth_l1: the two recon terms differ by a measured ~6.5x in scale, so the blend
        # weight and the pre-scaling are both sweep axes. Without `nb_scale`, `nb_weight`
        # sweeps over one order of magnitude of the NB term's own size rather than over the
        # trade-off -- which is the confound F6 hit.
        "nb_weight": "nb_weight", "nb_scale": "nb_scale",
        "vc_load_dispersion": "vc_load_dispersion",
    },
    "guards": {
        # >0 stops the first epoch after N batches and prints samples/s. The throughput
        # measurement knob; it existed on the config but was unreachable from a stage file.
        "bench_batches": "bench_batches",
        "nan_guard_max_frac": "nan_guard_max_frac",
        "weight_check_every": "weight_check_every",
    },
    "eval": {
        "eval_freq": "eval_freq", "run_baselines": "run_baselines",
    },
}

STAGE_SCHEMAS = {
    "pretrain": PRETRAIN_SCHEMA,
    "finetune": FINETUNE_SCHEMA,
    "decoder": DECODER_SCHEMA,
    "vcell": VCELL_SCHEMA,
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
