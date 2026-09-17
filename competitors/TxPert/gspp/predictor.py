from typing import Any, List, Dict, Union
import os
import pickle

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pandas as pd

from lightning import LightningModule
from torch.utils.data import DataLoader

import wandb

from gspp.data.graphmodule import GSPGraph
from gspp.models.gears_derivative import GSPP
from gspp.evaluation import evaluate
from gspp.metrics import log_metrics, compute_metrics

import gspp.constants as cs

# Upstream also registers mean / experimental / graph-neighbor baselines. Those live in
# hooke-tx's own baseline code (hooke_tx/eval/baselines/) for this benchmark, so the
# competitor arm keeps only the trained model.
MODEL_DICT = {
    "gears_derivate": GSPP,
}


class PertPredictor(LightningModule):
    """
    Predictor class that manages training, testing, logging and instantiate the model. To allow for different model types,
    the model is selected based on the `model_type` argument in the `model_args` dictionary and the loss function is defined
    in the model class rather than here.

    Args:
        input_dim (int): The input dimension of the data.
        graph (GSPGraph): Graph object containing the edge index and edge weight.
        model_args (Dict[str, Any]): The arguments for the model.
        no_pert (bool): Whether to use perturbations.
        basal_mode (str): The basal mode (how to smaple controls for inference).
        lr (float): The learning rate.
        min_lr (float): Minimum learning rate for scheduler
        weight_decay (float): The weight decay.
        device (str): The device.
        run_val_on_train_data (bool): Whether to run validation on the train data.
        run_test_during_validation (bool): Whether every validation pass also evaluates the TEST
            loader. True reproduces the historical behaviour (upstream had no validation split, so
            the validation hook scored test). Set False whenever test is meant to be scored once,
            from a validation-selected checkpoint.
        visualize (bool): Whether to visualize the data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adata_output_dim: int,
        graph: GSPGraph,
        model_args: Dict[str, Any],
        pert_input_dim: int = None,
        no_pert: bool = False,
        lr: float = 0.001,
        min_lr: float = 0.0,
        lr_scheduler_args: Dict[str, Any] = {},
        weight_decay: float = 0.0,
        device: str = "cpu",
        run_val_on_train_data: bool = False,
        run_test_during_validation: bool = True,
        visualize: bool = False,
        match_cntr_for_eval: bool = True,
        save_results: Dict[str, Any] = None,
    ):
        super(PertPredictor, self).__init__()

        self.no_pert = no_pert

        self.slow_benchmark = model_args.pop("slow_benchmark", False)
        self.run_val_on_train_data = run_val_on_train_data
        self.run_test_during_validation = run_test_during_validation
        self.visualize = visualize
        self.match_cntr_for_eval = match_cntr_for_eval
        self.save_results = save_results if save_results is not None else {}

        self.lr = lr
        self.min_lr = min_lr
        self.lr_scheduler_args = lr_scheduler_args
        self.weight_decay = weight_decay

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adata_output_dim = adata_output_dim

        self.model_type = model_args.pop("model_type", "gears_derivate")

        if "baseline" in self.model_type:
            self.model = MODEL_DICT[self.model_type](**model_args)
        else:
            self.latent_dim = model_args["latent_dim"]
            cntr_model_args = model_args.pop("cntr_model")
            pert_model_args = model_args.pop("pert_model")
            self.basal_mode = model_args.pop("basal_mode", "construct_control")

            self.model = MODEL_DICT[self.model_type](
                input_dim=input_dim,
                output_dim=output_dim,
                adata_output_dim=adata_output_dim,
                graph=graph,
                cntr_model_args=cntr_model_args,
                pert_model_args=pert_model_args,
                pert_input_dim=pert_input_dim,
                device=device,
                **model_args,
            ).to(device)

    def forward(self, cntr, pert_idxs, p_emb):
        cntr = cntr.to(self.device)
        p_emb = p_emb.to(self.device)
        
        return self.model.forward(cntr, pert_idxs, p_emb)

    def loss(self, prediction, intrinsic_mean, intrinsic_log_var, target):
        # Call loss function provided through the model
        loss, logs = self.model.loss(
            prediction, intrinsic_mean, intrinsic_log_var, target
        )

        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        prediction, intrinsic_mean, intrinsic_log_var = self.forward(
            batch.control, batch.pert_idxs, batch.p
        )

        loss = self.loss(
            prediction,
            intrinsic_mean,
            intrinsic_log_var,
            target=batch.x[:, : self.adata_output_dim],
        )

        return loss

    def sample_inference(self, cntr, pert_idxs, p_emb, cell_types):
        """
        Sample from the model for inference. The basal mode determines different ways how the control is sampled.
        """
        size = cntr.size(0)
        prediction, _, _ = self.forward(cntr, pert_idxs, p_emb)

        if self.no_pert:
            return cntr

        if self.basal_mode == "latent":
            zs = torch.randn(size, self.latent_dim, device=self.device)
            prediction = self.model.decoder(
                torch.cat(
                    [zs + self.model.pert_z, cntr[:, self.model.output_dim :]], dim=-1
                )
            )

        elif self.basal_mode == "rand_cntr":
            cntr_data = self.trainer.datamodule.control_data
            intrinsic_recon = torch.Tensor(
                cntr_data.X[np.random.randint(0, cntr_data.shape[0], size)].toarray()
            )
            prediction = self.model.decoder(self.model.pert_z) + intrinsic_recon

        elif self.basal_mode == "construct_control":
            prediction = prediction

        elif self.basal_mode == "matched_control":
            prediction = self.model.decoder(self.model.pert_z) + cntr

        else:
            prediction = None
            raise ValueError("Invalid basal mode")

        return prediction

    def general_validation_epoch(
        self, stage: str, loader: DataLoader, metric_stage: str = cs.FAST
    ):
        """
        Run a general validation epoch. This function is used to run the validation on the train, val, and test data.

        Args:
            stage (str): The stage of the experiment (train, val, test).
            loader (DataLoader): The data loader.
            metric_stage (str): The metric stage (fast, extended).
        """
        test_adata = self.trainer.datamodule.base_dm.adata

        if "baseline" in self.model_type:
            results = self.model.apply_baseline(
                self.trainer.datamodule.base_dm.test_data
            )
        else:
            results = evaluate(
                loader,
                self,
                self.device,
                test_adata,
                self.trainer.datamodule.base_dm.id2pert,
            )

        metrics, test_pert_res = compute_metrics(results, test_adata, metric_stage, match_cntr=self.match_cntr_for_eval)

        log_metrics(
            metrics, test_pert_res, stage, self.trainer.datamodule.subgroup, self
        )

    def _has_validation_data(self) -> bool:
        """Whether the datamodule actually has a validation dataset.

        ``val_dataloader()`` raises rather than silently returning the test loader, so callers ask
        this first. The wrapper delegates to ``base_dm`` for public-only runs, hence both lookups.
        """
        dm = self.trainer.datamodule
        for holder in (dm, getattr(dm, "base_dm", None)):
            if holder is not None and getattr(holder, "val_data", None) is not None:
                return True
        return False

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        if self.run_val_on_train_data:
            self.general_validation_epoch(
                cs.TRAIN, self.trainer.datamodule.train_dataloader()
            )

        # Score VAL whenever a validation dataset exists. The old gate was `not xcell_v2`, which
        # skipped it unconditionally on the zarr arm -- necessary back when val_dataloader() fell
        # back to the test loader, and wrong now that a real validation split can be present.
        if self._has_validation_data():
            self.general_validation_epoch(cs.VAL, self.trainer.datamodule.val_dataloader())

        # Scoring TEST on every validation pass is upstream's behaviour and is how it monitored
        # `test_pearson_delta` without a validation split. Under the select-then-test protocol the
        # test split must be touched exactly once, from the selected checkpoint, so this is off for
        # those runs.
        if self.run_test_during_validation:
            self.general_validation_epoch(
                cs.TEST, self.trainer.datamodule.test_dataloader()
            )

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        metric_set = cs.SLOW if self.slow_benchmark else cs.EXTENDED
        self.general_validation_epoch(
            cs.TEST, self.trainer.datamodule.test_dataloader(), metric_set
        )

    def configure_optimizers(self):
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)

        schedulers = []
        milestones = []

        warmup_epochs = self.lr_scheduler_args.get("warmup_epochs", 0)
        if warmup_epochs > 0:
            schedulers.append(cs.SCHEDULER_DICT["linear"](optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs))
            milestones.append(warmup_epochs)
        
        scheduler_type = self.lr_scheduler_args.get("type", None)
        if scheduler_type is not None:
            total_epochs = self.lr_scheduler_args["total_epochs"]
            schedulers.append(cs.SCHEDULER_DICT[scheduler_type](optimizer, T_max=total_epochs - warmup_epochs, eta_min=self.min_lr))

        scheduler = cs.SCHEDULER_DICT["sequential"](optimizer, schedulers, milestones=milestones) if len(schedulers) > 0 else None

        return optimizer if scheduler is None else {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        if not self.visualize:
            return

        for dic, name in [
            (self.trainer.datamodule.condition_group, "group"),
            (self.trainer.datamodule.subgroup["test_subgroup"], "subgroup"),
        ]:

            all_perts = []
            labels = []
            for group, perts in dic.items():
                all_perts.extend([str(i) for i in perts])
                labels.extend([group] * len(perts))
            df = pd.DataFrame({"pert": all_perts, "label": labels})
            table = wandb.Table(data=df)
            wandb.log({f"Perturbation {name} table": table})

    def on_train_epoch_end(self):
        # Get the current learning rate from the first optimizer parameter group
        current_lr = self.optimizers().param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})

    def save_results_for_split(self, loader: DataLoader, split_name: str, save_dir: str):
        """
        Collect and save inputs and outputs for a given data split.
        
        Args:
            loader: DataLoader for the split
            split_name: Name of the split (train, val, test)
            save_dir: Directory to save the pickle file
        """
        self.eval()
        self.to(self.device)
        
        all_perturbations = []
        all_base_states = []
        all_outputs = []
        all_pert_idxs = []
        all_pert_cond_names = []
        all_indices = []
        all_cell_types = []
        all_experimental_batches = []
        
        with torch.no_grad():
            for batch in loader:
                # Get predictions
                if self.no_pert:
                    predictions = batch.control[:, : self.adata_output_dim]
                else:
                    predictions = self.sample_inference(
                        batch.control, batch.pert_idxs, batch.p, batch.cell_types
                    )
                
                # Store data (move to CPU for saving)
                all_base_states.append(batch.control.cpu())
                all_outputs.append(predictions.cpu())
                all_perturbations.append(batch.p.cpu())
                
                # Store metadata
                all_pert_idxs.extend(batch.pert_idxs)
                all_pert_cond_names.extend(batch.pert_cond_names)
                all_indices.extend(batch.idx)
                all_cell_types.extend(batch.cell_types)
                all_experimental_batches.extend(batch.experimental_batches)
        
        # Concatenate tensors
        base_states = torch.cat(all_base_states).numpy()
        outputs = torch.cat(all_outputs).numpy()
        perturbations = torch.cat(all_perturbations).numpy()
        
        # Prepare data dictionary
        results_data = {
            'input_perturbation': perturbations,
            'base_state': base_states,
            'output': outputs,
            'pert_idxs': all_pert_idxs,
            'pert_cond_names': all_pert_cond_names,
            'indices': all_indices,
            'cell_types': all_cell_types,
            'experimental_batches': all_experimental_batches,
        }
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as pickle file
        save_path = os.path.join(save_dir, f"{split_name}_results.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results_data, f)
        
        logger.info(f"Saved {split_name} results to {save_path} ({len(all_indices)} samples)")

    def on_train_end(self):
        """
        Called at the end of training. Save results if configured.
        """
        if self.save_results.get("enabled", False):
            save_dir = self.save_results.get("save_dir", None)
            if save_dir is None:
                logger.warning("save_results.enabled is True but save_dir is not specified. Skipping result saving.")
                return
            
            logger.info(f"Saving training results to {save_dir}")
            
            # Save results for each split
            if self.save_results.get("save_train", True):
                self.save_results_for_split(
                    self.trainer.datamodule.train_dataloader(),
                    "train",
                    save_dir
                )
            
            if self.save_results.get("save_val", True) and self._has_validation_data():
                self.save_results_for_split(
                    self.trainer.datamodule.val_dataloader(),
                    "val",
                    save_dir
                )
            
            if self.save_results.get("save_test", True):
                self.save_results_for_split(
                    self.trainer.datamodule.test_dataloader(),
                    "test",
                    save_dir
                )
            
            logger.info("Finished saving all results")