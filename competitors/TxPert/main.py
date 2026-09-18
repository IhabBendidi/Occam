import os
import sys
from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf

import torch

import wandb

import traceback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from gspp.predictor import PertPredictor
from gspp.data.datamodule_hybrid import HybridDataModuleWrapper
from gspp.data.graphmodule import GSPGraph
from gspp.data.datamodule import PertDataModule
from gspp.constants import SCHEDULER_DICT

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from gspp.utils import set_seed, modify_config, tmp_cfg_sanity_checks

## Where wandb writes its run files. Upstream hardcoded /rxrx/data/user/$USER/outgoing;
## respect an externally-set WANDB_DIR first so SLURM scripts can point it at scratch.
os.environ.setdefault(
    "WANDB_DIR", str(Path(__file__).resolve().parent / "outputs" / "wandb")
)
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)


# Main function to initialize Hydra and parse the configuration
@hydra.main(
    config_name="config-xcell-v2-best-adj-map", config_path="configs", version_base="1.3"
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    try:
        train(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

def get_checkpoint_dir(cfg, dm):
    """
    Create checkpoint directory path based on config and model parameters
    
    Args:
        cfg: The config object containing datamodule settings
        model: The GSPP model instance containing pert_model_type
    """
    # Get dataset identifier from datamodule
    dataset_id = dm.base_dm.get_dataset_identifier()
    
    # Get model info
    model_type = cfg.model.pert_model.model_type
    
    checkpoint_dir = os.path.join(
        dm.cache_dir,
        "checkpoints",
        dataset_id,
        model_type
    )
    
    # Add wandb run ID
    if wandb.run:
        checkpoint_dir = os.path.join(checkpoint_dir, wandb.run.dir.split('/')[-1])
    
    return checkpoint_dir

def train(cfg: DictConfig):
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"

    # Set seed (TODO: fix deterministic behavior)
    set_seed(cfg.training.seed)

    run = wandb.init(**cfg.wandb)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Modify the configuration
    cfg = modify_config(cfg)

    # Check if config is valid
    tmp_cfg_sanity_checks(cfg)

    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)
    added_data_args = data_args.pop("added_data_args", {})

    datamodule = HybridDataModuleWrapper(
        base_dm_args=data_args,
        **added_data_args,
    )
    datamodule.prepare_data()

    # Instatiate graph(s) used by the GNN
    graph = GSPGraph(
        pert2id=datamodule.pert2id,
        gene2id=datamodule.gene2id,
        **cfg.graph,
    )

    # Instantiate predictor
    GSP_model = PertPredictor(
        input_dim=datamodule.input_dim,
        output_dim=datamodule.output_dim,
        pert_input_dim=sum([dim[0] for dim in datamodule.base_dm.mol_fp_dim.values()]) + datamodule.base_dm.num_doses if datamodule.base_dm.molecular_perts else None, # TODO: Fix this
        graph=graph,
        model_args=OmegaConf.to_container(cfg.model, resolve=True),
        lr=cfg.training.lr,
        min_lr=cfg.training.get("min_lr", 0.0),
        lr_scheduler_args=cfg.training.get("lr_scheduler", {}),
        run_val_on_train_data=cfg.training.get("run_val_on_train_data", False),
        run_test_during_validation=cfg.training.get("run_test_during_validation", True),
        adata_output_dim=datamodule.adata_output_dim,
        weight_decay=cfg.training.weight_decay,
        device=device,
        match_cntr_for_eval=cfg.training.get("match_cntr_for_eval", True),
        save_results=OmegaConf.to_container(cfg.training.save_results, resolve=True) if "save_results" in cfg.training else {},
    ).to(device)

    if cfg.mode == "retrain" and cfg.checkpoint_path:
        GSP_model = PertPredictor.load_from_checkpoint(
            cfg.checkpoint_path,
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            num_perts=len(datamodule.pert2id),
            graph=graph,
            pert_names=list(datamodule.pert2id.keys()),
            model_args=OmegaConf.to_container(cfg.model, resolve=True),
            device=device,
            match_cntr_for_eval=cfg.training.get("match_cntr_for_eval", True),
            save_results=OmegaConf.to_container(cfg.training.save_results, resolve=True) if "save_results" in cfg.training else {},
        ).to(device)

    # Instantiate the trainer
    logger = WandbLogger()
    
    # Both the zarr benchmark and xcell_v2 hold out perturbations without carving a validation
    # split, so `val_pearson_delta` is never logged. Anything monitoring it must be skipped --
    # Lightning raises rather than warning when a monitored metric is absent.
    has_val_split = not (
        cfg.datamodule.get("zarr", False) or cfg.datamodule.get("xcell_v2", False)
    )

    callbacks = []
    if has_val_split:
        callbacks.append(
            EarlyStopping(monitor="val_pearson_delta", patience=100, mode="max"),
        )
    
    # Get the checkpoint directory path
    if "checkpoint" in cfg.training:
        # dirpath was hardcoded to another user's outgoing dir, which is not writable.
        # Default under this checkout's cache/ instead; training.checkpoint.dirpath still wins.
        _ckpt_cfg = dict(cfg.training.checkpoint)
        _default_dirpath = str(
            Path(__file__).resolve().parent
            / "cache"
            / "checkpoints"
            / datamodule.base_dm.get_dataset_identifier()
            / cfg.model.pert_model.model_type
            / str(cfg.training.seed)
        )
        ckpt_kwargs = dict(
            dirpath=_ckpt_cfg.pop("dirpath", _default_dirpath),
            **_ckpt_cfg,
        )
        if "monitor" in ckpt_kwargs:
            # An explicitly configured monitor wins, untouched. This is how the fewshot arm
            # selects on `val/celleval/selection_score` -- a metric published by
            # ValidationCellEvalCallback, not by compute_metrics, so none of the branches below
            # could name it.
            #
            # The filename is deliberately plain: the monitor name contains '/' characters, which
            # ModelCheckpoint would read as path separators, and auto_insert_metric_name would put
            # an '=' in the filename, which Hydra cannot then accept as an override value (that
            # cost PerturbDiff a whole eval array). best_model_score carries the value instead.
            ckpt_kwargs.setdefault("filename", "best-ep{epoch}")
            ckpt_kwargs.setdefault("auto_insert_metric_name", False)
        elif has_val_split:
            ckpt_kwargs.update(
                filename="best-{epoch}-{val_pearson_delta:.2f}",
                monitor="val_pearson_delta",
                mode="max",
            )
        elif ckpt_kwargs.get("save_last"):
            # No validation split exists in this mode, so upstream monitored
            # `test_pearson_delta` -- selecting the checkpoint on the test set. When the config
            # asks for save_last we take the final epoch instead and monitor nothing.
            ckpt_kwargs.setdefault("filename", "{epoch}")
        else:
            ckpt_kwargs.update(
                filename="best-{epoch}-{test_pearson_delta:.2f}",
                monitor="test_pearson_delta",
                mode="max",
            )
        callbacks.append(ModelCheckpoint(**ckpt_kwargs))

    # Validation cell-eval, which is what makes `monitor: val/celleval/selection_score` resolvable.
    # gspp's own compute_metrics publishes val_pearson etc., but the cross-arm selection metric is
    # a cell-eval composite, and cell-eval is not otherwise run during training.
    _val_celleval_cfg = cfg.training.get("validation_celleval", {})
    if _val_celleval_cfg.get("enabled", False):
        from gspp.validation_celleval_callback import ValidationCellEvalCallback

        _val_root = _val_celleval_cfg.get("output_dir") or os.path.join(
            os.environ.get("WANDB_DIR", "."), "valscore"
        )
        callbacks.append(
            ValidationCellEvalCallback(
                output_root=_val_root,
                interval=int(_val_celleval_cfg.get("interval", 1)),
                profile=_val_celleval_cfg.get("profile", "full"),
            )
        )

    trainer = Trainer(
        accelerator="gpu" if gpu_available else "cpu",
        logger=logger,
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )

    if cfg.mode == "train" or cfg.mode == "retrain":
        if cfg.mode == "retrain" and cfg.checkpoint_path:
            trainer.fit(
                GSP_model, 
                datamodule,
                ckpt_path=cfg.checkpoint_path
            )
        else:
            trainer.fit(GSP_model, datamodule)

    elif cfg.mode == "baseline":
        datamodule.setup("fit")
        GSP_model.model.prepare_baseline(**GSP_model.model.parse_inputs(datamodule, graph))

    elif cfg.mode == "test":
        # trainer.fit() normally triggers datamodule.setup(); mode=test skips it, so
        # condition_group / the predefined splits are never populated and downstream
        # code raises AttributeError. The `baseline` branch calls setup() for the
        # same reason.
        datamodule.setup("fit")
        GSP_model = PertPredictor.load_from_checkpoint(
            cfg.checkpoint_path,
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            num_perts=len(datamodule.pert2id),
            graph=graph,
            pert_names=list(datamodule.pert2id.keys()),
            model_args=OmegaConf.to_container(cfg.model, resolve=True),
            device=device,
            save_results=OmegaConf.to_container(cfg.training.save_results, resolve=True) if "save_results" in cfg.training else {},
        ).to(device)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    # Record the checkpoint ModelCheckpoint actually selected.
    #
    # `trainer.test` below runs the IN-MEMORY weights and takes no ckpt_path, so nothing here ever
    # loads the best checkpoint. Under the select-then-test protocol the test leg is a separate job
    # that must be handed the selected file, and a marker is the only durable record of it --
    # best_model_path lives in the ModelCheckpoint object and dies with the process.
    for _cb in trainer.callbacks:
        if isinstance(_cb, ModelCheckpoint) and getattr(_cb, "best_model_path", ""):
            _marker = Path(_cb.dirpath) / "best_ckpt_path.txt"
            _marker.write_text(_cb.best_model_path + "\n")
            _score = _cb.best_model_score
            print(
                f"[checkpoint] selected {_cb.best_model_path} "
                f"(monitor={_cb.monitor}, score={float(_score) if _score is not None else None}) "
                f"-> {_marker}"
            )
            break

    # Testing. Off for runs that select a checkpoint and then test it in a separate job: this call
    # would score the FINAL weights on the test split, which is both a wasted pass and a test
    # number that is not the one being reported.
    if cfg.training.get("test_after_fit", True):
        trainer.test(GSP_model, datamodule)

    # Score with cell-eval, using the same protocol as ../STATE and ../PerturbDiff.
    # Upstream instead had `export_norman_eval` (gspp/norman_export.py, hardcoded to
    # scMAE-main/ paths) and two h5ad exporters that renormalized predictions to a magic
    # target_sum; all three are gone.
    celleval_cfg = cfg.get("celleval", {})
    if celleval_cfg.get("enable", False):
        from gspp.evaluation import export_for_celleval

        output_dir = celleval_cfg.get(
            "output_dir", os.path.join(os.environ.get("WANDB_DIR", "."), "celleval")
        )
        if wandb.run is not None:
            output_dir = os.path.join(output_dir, wandb.run.id)
        os.makedirs(output_dir, exist_ok=True)

        export_for_celleval(
            model=GSP_model,
            datamodule=datamodule,
            device=device,
            output_dir=output_dir,
            profile=celleval_cfg.get("profile", "full"),
        )
        wandb.log({"celleval_output_dir": output_dir})

    # Write pred/real AnnData instead of scoring in-process, so ONE forward pass can be scored
    # under several predicted-control conventions by the shared scorer
    # (../../the upstream tree/eval/celleval_artifacts.py). export_for_celleval above can only ever produce
    # the inject-real convention, because it injects the real controls into both objects itself.
    #
    # Used by launch/txpert_fewshot_eval.sh together with `mode=test checkpoint_path=...`, so the
    # scored weights are the ones validation selected rather than the final epoch.
    artifact_cfg = cfg.get("export_artifacts", {})
    if artifact_cfg.get("enable", False):
        from gspp.export_artifacts import export_artifacts

        artifact_dir = artifact_cfg.get("dir") or os.path.join(
            os.environ.get("WANDB_DIR", "."), "artifacts"
        )
        export_artifacts(
            model=GSP_model,
            datamodule=datamodule,
            device=device,
            artifact_dir=artifact_dir,
            split=artifact_cfg.get("split", "test"),
        )

    wandb.run.finish()


if __name__ == "__main__":
    main()
