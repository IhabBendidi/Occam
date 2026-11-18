"""Train/validate run pipeline split from rawdata_diffusion_training.py (logic-preserving)."""

from pathlib import Path

import torch


def run_pipeline(trainer, model, datamodule, cfg, logger):
    """
    Run pipeline.

    :param trainer: Input `trainer` value.
    :param model: Model instance for forward/sampling/training.
    :param datamodule: Data module providing datasets and loaders.
    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :return: Computed output(s) for this function.
    """
    # torch >= 2.6 defaults torch.load to weights_only=True, which cannot
    # unpickle the OmegaConf configs that save_hyperparameters() writes into
    # every checkpoint. load_plmodel_checkpoint() has its own fallback, but
    # Lightning's *internal* reloads (resume from cfg.model.ckpt_path, and the
    # final validate below) do not, so tell it these checkpoints are ours and
    # safe. Requires pytorch-lightning >= 2.5.
    if not getattr(cfg, "validate_only", False) and not getattr(cfg, "test_only", False):
        logger.info("*********** start training ***********\n\n")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path,
            weights_only=False,
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(f"Saving final model weights to {cfg.save_dir_path}")
        trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
        logger.info("Finished saving final model")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # Prefer the checkpoint ModelCheckpoint selected on VALIDATION; fall back to the final
        # weights we just wrote.
        #
        # Asking Lightning for ckpt_path="best" directly is what this used to avoid, for a good
        # reason: "best" raises after training has already finished when the monitored metric never
        # fired -- an empty validation split, or a run shorter than every_n_train_steps. Reading
        # best_model_path ourselves keeps that protection while still using the selected checkpoint
        # when one exists, which is what makes this comparable to the other model families.
        #
        # `best_ckpt_path_file` records the choice so the downstream eval job scores the same
        # weights instead of re-deriving them.
        best_ckpt = None
        for cb in getattr(trainer, "checkpoint_callbacks", []) or []:
            candidate = getattr(cb, "best_model_path", "") or ""
            if candidate and Path(candidate).exists():
                best_ckpt = candidate
                monitor = getattr(cb, "monitor", None)
                score = getattr(cb, "best_model_score", None)
                logger.info(
                    f"ModelCheckpoint selected {candidate} (monitor={monitor}, score={score})"
                )
                break

        eval_ckpt = best_ckpt or cfg.save_dir_path
        if best_ckpt is None:
            logger.warning(
                "No best-validation checkpoint available; falling back to the FINAL weights at "
                f"{cfg.save_dir_path}. Set lightning.callbacks.checkpoint.monitor to a metric that "
                "actually fires (e.g. validation/celleval/selection_score with "
                "lightning.callbacks.validation_celleval.enabled=true) to select on validation."
            )

        marker = Path(cfg.save_dir_path).parent / f"{Path(cfg.save_dir_path).name}.best_ckpt_path"
        try:
            marker.write_text(str(eval_ckpt) + "\n")
            logger.info(f"Recorded the evaluated checkpoint path in {marker}")
        except OSError:
            logger.warning(f"Could not write {marker}; the eval job will need the path explicitly.")

        logger.info(f"Validating checkpoint: {eval_ckpt}")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=eval_ckpt,
            weights_only=False,
        )
    else:
        logger.info("*********** start validation ***********\n\n")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path,
            weights_only=False,
        )
