"""Training entrypoint split from rawdata_diffusion_training.py (logic-preserving)."""

import os
import sys
import logging
import hydra
import omegaconf
import pytorch_lightning as pl

# ─── make a hang diagnosable ───────────────────────────────────────────────────────────────
# A run of this script sat RUNNING for 10 HOURS at Epoch 0 step 1120 with a 0% GPU: 27 threads
# in futex_wait_queue_me, the 4 dataloader workers parked in do_poll, 10 minutes of CPU in
# 9h53m. Shared memory and HDF5 fork-unsafety were ruled out by measurement and the identical
# twin run completed, so it was a race -- but the blocking Python frame could NOT be captured
# (`py-spy` needs ptrace and the node has `ptrace_scope=1` with no sudo), so the cause is still
# unproven and the 10 hours of GPU bought nothing.
#
# `faulthandler` closes that hole with no dependency: SIGUSR1 dumps every thread's stack to
# stderr, i.e. into the job log. On the next hang:
#
#     scancel --signal=USR1 <jobid>      # then read runs/slurm/cmp_pdiff_<id>_0.err
#
# `register(..., chain=False)` keeps the process running, so this is a probe and not a kill.
import faulthandler
import signal

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

# ⚠️ OCCAM INFRA (2026-09-23): DataLoader workers passing tensors through /dev/shm file
# descriptors died on "could not unlink the shared memory file /torch_..." and then hung
# (norman_combo s44, s45 -- watchdog-killed after ~2.5 h). `file_system` sharing avoids the
# fd-passing path. It changes only how workers hand batches over, not what is computed.
# Opt-in via TORCH_SHARING_STRATEGY (set by scripts/slurm/_competitor_common.sh).
if os.environ.get("TORCH_SHARING_STRATEGY"):
    import torch.multiprocessing as _tmp
    _tmp.set_sharing_strategy(os.environ["TORCH_SHARING_STRATEGY"])

# local imports
exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(exc_dir)
module_logger = logging.getLogger(__name__)
module_logger.info("exc_dir: %s", exc_dir)


from src.common.utils import setup_loggings
from src.apps.training.training_datamodule_builder import build_datamodule, populate_covariate_cfg
from src.apps.training.training_model_builder import (
    build_model,
    maybe_compare_model_representation,
    maybe_load_and_patch_checkpoint_model,
    maybe_reinitialize_from_scratch,
)
from src.apps.training.training_pipeline import run_pipeline
from src.apps.training.training_runtime import setup_trainer

@hydra.main(version_base=None, config_path="../../../configs", config_name="rawdata_diffusion_training")
def main(cfg):
    """
    Launch supervised fine-tuning using a hydra config, for protein classification
    """
    omegaconf.OmegaConf.resolve(cfg)

    logger = setup_loggings(cfg)

    if cfg.model.ckpt_path:
        logger.info(f"Resuming from checkpoint {cfg.model.ckpt_path}. ")

    pl.seed_everything(cfg.optimization.seed)

    trainer = setup_trainer(cfg)

    datamodule = build_datamodule(cfg, logger)
    populate_covariate_cfg(cfg, datamodule)

    model = build_model(cfg, logger, datamodule)
    maybe_compare_model_representation(cfg, logger, datamodule)
    model = maybe_load_and_patch_checkpoint_model(cfg, model, logger)
    maybe_reinitialize_from_scratch(cfg, model, logger)

    logger.info("Model:\n%s", model)
    run_pipeline(trainer, model, datamodule, cfg, logger)

if __name__ == "__main__":
    main()
