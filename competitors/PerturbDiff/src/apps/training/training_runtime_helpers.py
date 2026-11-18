"""Helpers for trainer runtime assembly."""

import os
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import ListConfig
from torch.profiler import schedule as profiler_schedule, tensorboard_trace_handler

try:
    from pytorch_lightning.profiler import PyTorchProfiler
except ImportError:
    from pytorch_lightning.profilers import PyTorchProfiler

from src.models.lightning.lightning_factories import EMAWeightAveraging


def resolve_trainer_strategy(cfg):
    """Execute `resolve_trainer_strategy` and return values used by downstream logic."""
    devices = getattr(cfg.trainer, "devices", None)
    accelerator = str(getattr(cfg.trainer, "accelerator", "")).lower()
    use_ddp = False
    if isinstance(devices, int):
        use_ddp = devices > 1
    elif isinstance(devices, (list, tuple, ListConfig)):
        use_ddp = len(devices) > 1
    elif isinstance(devices, str):
        if devices not in ("auto", "1"):
            use_ddp = True
    if accelerator in ("cpu",):
        use_ddp = False
    return "ddp_find_unused_parameters_true" if use_ddp else "auto"


def build_callbacks(cfg, trainer_logger):
    """Execute `build_callbacks` and return values used by downstream logic."""
    callbacks = [
        hydra.utils.instantiate(cfg.lightning.callbacks.checkpoint),
        hydra.utils.instantiate(cfg.lightning.callbacks.progress_bar),
    ]
    if trainer_logger is not False:
        callbacks.append(hydra.utils.instantiate(cfg.lightning.callbacks.lr_monitor))
    # Validation cell-eval, so a checkpoint can be selected on the validation split rather than on
    # the test split (which is what the empty-validation benchmark split forced before). Off unless
    # explicitly enabled, so existing configs are unchanged.
    val_ce = getattr(cfg.lightning.callbacks, "validation_celleval", None)
    if val_ce is not None and bool(getattr(val_ce, "enabled", False)):
        from src.apps.training.validation_celleval_callback import ValidationCellEvalCallback

        callbacks.append(
            ValidationCellEvalCallback(
                cfg=cfg,
                interval=int(getattr(val_ce, "interval", 5)),
                num_sampled_batches=getattr(val_ce, "num_sampled_batches", None),
            )
        )
    if getattr(cfg.lightning.ema, "decay", 0) > 0:
        callbacks.append(
            EMAWeightAveraging(
                decay=cfg.lightning.ema.decay,
                update_steps=cfg.lightning.ema.update_steps,
            )
        )
    return callbacks


def build_profiler(cfg):
    """
    Build profiler.

    :param cfg: Runtime configuration object.
    :return: Requested object(s) for downstream use.
    """
    profiler = None
    profiler_cfg = getattr(cfg, "profiler", None)
    if profiler_cfg is not None:
        profiler_cfg = omegaconf.OmegaConf.to_container(profiler_cfg, resolve=True)
    if profiler_cfg and profiler_cfg.get("enabled", False):
        profile_dir = profiler_cfg.get("output_dir") or os.path.join(cfg.save_dir_path, "profiler_traces")
        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        schedule_kwargs = dict(
            wait=profiler_cfg.get("wait_steps", 1),
            warmup=profiler_cfg.get("warmup_steps", 1),
            active=profiler_cfg.get("active_steps", 3),
            repeat=profiler_cfg.get("repeat", 1),
        )
        profiler_kwargs = dict(
            dirpath=str(profile_dir),
            filename=profiler_cfg.get("filename", "training_profiler"),
            schedule=profiler_schedule(**schedule_kwargs),
            on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
            record_shapes=profiler_cfg.get("record_shapes", True),
            profile_memory=profiler_cfg.get("profile_memory", True),
            with_stack=profiler_cfg.get("with_stack", False),
            with_flops=profiler_cfg.get("with_flops", False),
            row_limit=profiler_cfg.get("row_limit", 20),
        )
        activities = profiler_cfg.get("activities")
        if activities:
            activity_objects = []
            for activity in activities:
                try:
                    activity_objects.append(getattr(torch.profiler.ProfilerActivity, activity.upper()))
                except AttributeError as err:
                    raise ValueError(f"Unsupported profiler activity '{activity}'.") from err
            profiler_kwargs["activities"] = activity_objects
        profiler = PyTorchProfiler(**profiler_kwargs)
    return profiler
