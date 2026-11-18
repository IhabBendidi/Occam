from typing import Any, List, Dict, Union

import os

import joblib

from tqdm import tqdm

from loguru import logger

import scanpy as sc
import anndata as ad
import zarr

import torch
import random
import numpy as np
import pandas as pd

import lightning as L

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import anndata as ad

from gspp.constants import PUBLIC_CELL_TYPE_DATASETS, CELL_TYPE

from gspp.data.data_utils import (
    collate_fn,
    calculate_de_for_eval,
)
from gspp.data.datamodule import PertDataModule


# Upstream registered an internal HUVEC Perturb-seq dataset here, loaded from
# /rxrx/data/scrnaseq/internal/ with obs/var patched in from a third user's home directory.
# The benchmark runs with `global.use_added_data: false` (modify_config then blanks
# `added_data_args`), so this arm never takes that path and the dependency is dropped.
INTERNAL_CELL_TYPE_DATASETS = {}


class HybridDataModuleWrapper(L.LightningDataModule):
    """
    Lightning datamodule for Perturb-seq data that does not require loading the entire dataset into memory.

    Args:
        base_dm_args (Dict[str, str]): The arguments for the base datamodule
        added_cell_lines (Union[str, List[str]]): The cell lines to add to the public cell lines
        p_added_data_usage (float): The percentage of added data to use
        p_sampling_per_epoch (float): The percentage of added data to sample per epoch
        cells_per_chunk (int): The number of cells per chunk in the zarr file
        compressed (bool): Whether to use compression in the zarr file
        cache_dir (str): The directory to save the zarr files and metadata
        target_sum (int): The target sum for the total counts per cell
        min_cells (int): The minimum number of cells required for a perturbation to be included in the dataset
        min_de_genes (int): The minimum number of differentially expressed genes required for a perturbation to be included in the dataset
        knockdown_threshold (float): The threshold for the knockdown ratio required for a perturbation to be included in the dataset
    """

    def __init__(
        self,
        base_dm_args: Dict[str, str],
        added_cell_lines: Union[str, List[str]] = None,
        p_added_data_usage: float = None,
        p_sampling_per_epoch: float = None,
        cells_per_chunk: int = 1,
        compressed: bool = False,
        cache_dir: str = str(Path(__file__).resolve().parents[2] / "cache"),
        target_sum: int = 4e3,
        min_cells: int = 25,
        min_de_genes: int = 50,
        knockdown_threshold: float = -0.3,
    ):
        super().__init__()
        if isinstance(added_cell_lines, str):
            added_cell_lines = [added_cell_lines]

        if added_cell_lines is not None:
            self.added_datasets = {
                cell_type: INTERNAL_CELL_TYPE_DATASETS[cell_type]
                for cell_type in added_cell_lines
            }
        else:
            self.added_datasets = {}

        self.only_public_datasets = False if len(self.added_datasets) > 0 else True

        self.perts_obs, self.genes_obs, self.gene_to_col, self.cntr_indices = (
            {},
            {},
            {},
            {},
        )
        gene_prior = None
        for dataset_name, info in self.added_datasets.items():
            # Load data
            path, self.clean_fn = info
            adata = ad.read_h5ad(path, backed="r")
            self.clean_fn(adata)

            # Identify controls and un-labelled genes
            nan_identifier = pd.isna(adata.obs["gene_name"])
            cntr_identifier = adata.obs["is_control"] == 1
            self.cntr_indices[dataset_name] = np.where(cntr_identifier)[0]

            adata.obs["gene_name"] = adata.obs["gene_name"].cat.add_categories(
                ["non-targeting"]
            )
            adata.obs.loc[nan_identifier, "gene_name"] = "non-targeting"

            self.perts_obs[dataset_name] = (
                adata.obs["gene_name"][~nan_identifier].unique().tolist()
            )
            self.genes_obs[dataset_name] = adata.var["gene_name"].tolist()
            self.gene_to_col[dataset_name] = {
                g: idx for idx, g in enumerate(self.genes_obs[dataset_name])
            }

            gene_prior = list(
                set.intersection(*map(set, list(self.genes_obs.values())))
            )

        # Now process base datasets in memory
        self.base_dm = PertDataModule(
            **base_dm_args,
            gene_prior=gene_prior,
            num_additional_cell_types=len(self.added_datasets),
        )
        self.base_dm.prepare_data()

        self.input_dim = self.base_dm.input_dim
        self.output_dim = self.base_dm.output_dim
        self.adata_output_dim = self.base_dm.adata_output_dim
        self.pert2id = self.base_dm.pert2id
        self.gene2id = self.base_dm.gene2id

        # self.genes = self.base_dm.adata.var.gene_name.values.tolist()
        self.genes = self.base_dm.adata.var.index.tolist()
        self.xcell = getattr(self.base_dm, 'xcell', False)
        self.xcell_v2 = getattr(self.base_dm, 'xcell_v2', False)
        self.perts = list(self.base_dm.pert2id.keys()) if not self.base_dm.molecular_perts else list(
           set([p[0] for p in self.base_dm.conditions])
        )
        
        if "ctrl" in self.perts:
            self.perts.remove("ctrl")

        self.perts_calc = {}

        for key, perts_obs in self.perts_obs.items():
            self.perts_calc[key] = [p for p in perts_obs if p in self.perts]

        self.cache_dir = cache_dir
        self.cells_per_chunk = cells_per_chunk

        self.p_added_data_usage = p_added_data_usage
        self.p_sampling_per_epoch = p_sampling_per_epoch

        self.target_sum = target_sum
        self.min_cells = min_cells
        self.compressed = compressed
        self.min_de_genes = min_de_genes
        self.knockdown_threshold = knockdown_threshold

        self.batch_size = self.base_dm.batch_size

    def prepare_data(self) -> None:
        """
        Prepares the added datasets by creating metadata and zarr files
        """
        self.metadata = {}

        for dataset_name, info in self.added_datasets.items():
            # Check if metadata already exists; TODO: Make this more robust
            if os.path.exists(f"{self.cache_dir}/{dataset_name}"):
                if sum(
                    1
                    for entry in os.scandir(f"{self.cache_dir}/{dataset_name}")
                    if entry.is_file()
                ) == len(self.perts_calc[dataset_name]):
                    continue

            # Load data
            path, self.clean_fn = info
            adata = ad.read_h5ad(path, backed="r")

            # Create zarr datasets (if not already created)
            self.save_adata_to_zarr(dataset_name, adata)

            self.clean_fn(adata)

            # Identify controls and un-labelled genes
            nan_identifier = pd.isna(adata.obs["gene_name"])

            adata.obs["gene_name"] = adata.obs["gene_name"].cat.add_categories(
                ["non-targeting"]
            )
            adata.obs.loc[nan_identifier, "gene_name"] = "non-targeting"

            adata_cntr = adata[self.cntr_indices].to_memory()

            for p in tqdm(self.perts_calc[dataset_name], desc="Processing genes"):
                self.create_metadata(
                    dataset_name, p, adata=adata, adata_cntr=adata_cntr.copy()
                )

    def create_metadata(
        self,
        dataset_name: str,
        p: str,
        adata: sc.AnnData = None,
        adata_cntr: sc.AnnData = None,
    ) -> None:
        """
        This function creates metadata for a batch of genes, namely
        - total_counts: the total counts for each perturbation
        - pert_indices: the indices of the perturbations in the raw data
        - valids: whether the perturbation is valid (i.e. has enough DEGs)
        - knockdowns: the knockdown ratio for the perturbation (to be used for further filtering)

        Args:
            dataset_name (str): The name of the dataset
            p (str): The name of the perturbation
            adata (sc.AnnData): The AnnData object for the dataset
            adata_cntr (sc.AnnData): The AnnData object for the controls
        """

        # Check if perturbation metadata is already cached
        if os.path.exists(f"{self.cache_dir}/{dataset_name}/{p}.pkl"):
            return

        if adata is None:
            adata = ad.read_h5ad(
                f"{self.raw_dir}/raw_gex_and_guide_counts.h5ad", backed="r"
            )
            self.clean_fn(adata)

            adata.obs["gene_name"] = adata.obs["gene_name"].cat.add_categories(
                ["non-targeting"]
            )
            adata.obs.loc[pd.isna(adata.obs["gene_name"]), "gene_name"] = (
                "non-targeting"
            )

        if adata_cntr is None:
            adata_cntr = adata[self.cntr_indices].to_memory().copy()

        sc.pp.normalize_total(adata_cntr, target_sum=self.target_sum)
        sc.pp.log1p(adata_cntr)

        p_identifier = adata.obs["gene_name"] == p
        logger.info(f"started {p} ({p_identifier.sum()} cells)")

        adata_p = adata[p_identifier].to_memory().copy()

        indices = np.where(p_identifier)[0].tolist()

        valid_perturbations = (
            (adata_p.obs["gene_name"].value_counts() >= self.min_cells)
            .loc[lambda x: x]
            .index.tolist()
        )

        sc.pp.normalize_total(adata_p, target_sum=self.target_sum)
        sc.pp.log1p(adata_p)

        adata_combined = ad.concat([adata_p, adata_cntr], axis=0, join="inner").copy()
        adata_combined.var = adata_p.var

        # Differential expression analysis
        if p_identifier.sum() > 1:
            sc.tl.rank_genes_groups(
                adata_combined,
                groupby="gene_name",
                reference="non-targeting",
            )
            # Extract DE results
            de_results = adata_combined.uns["rank_genes_groups"]

            # Count DE genes for each perturbation (p-value < 0.05)
            de_genes_counts = {
                group: sum(pd.DataFrame(de_results["pvals_adj"])[group] < 0.05)
                for group in de_results["names"].dtype.names
            }

            # Filter perturbations with ≥ min_de_genes DEGs
            valid_perturbations = [
                group
                for group, count in de_genes_counts.items()
                if count >= self.min_de_genes and group != "non-targeting"
            ]
            valid = len(valid_perturbations) > 0
        else:
            valid = np.nan

        # Calculate knockdown ratio if perturbation is among observed genes
        if p in adata_combined.var.gene_name.tolist():
            cntr_cells = adata_combined[adata_combined.obs["is_control"]]
            pert_cells = adata_combined[adata_combined.obs["gene_name"] == p]
            p_idx = adata_combined.var.gene_name.tolist().index(p)

            cntr_expr = cntr_cells[:, p_idx].X.mean()
            pert_expr = pert_cells[:, p_idx].X.mean()
            p_knockdown = (pert_expr / cntr_expr) - 1 if cntr_expr > 0 else np.nan
        else:
            p_knockdown = np.nan

        metadata = {
            "pert_indices": indices,
            "valid": valid,
            "knockdown": p_knockdown,
        }

        torch.save(metadata, f"{self.cache_dir}/{dataset_name}/{p}.pkl")

        logger.info(f"Finished {p}")

    def save_adata_to_zarr(self, dataset_name: str, adata: sc.AnnData) -> None:
        """
        This function saves the AnnData object to a zarr file in chunks
        """
        # Check if zarr file already exists
        if os.path.exists(f"{self.cache_dir}/{dataset_name}.zarr"):
            return

        # Create zarr file (this can take few hours for large datasets)
        num_cells = len(adata)
        num_genes = len(adata.var)

        zarr_file = zarr.open(
            f"{self.cache_dir}/{dataset_name}.zarr",
            mode="w",
            shape=(num_cells, num_genes),
            chunks=(self.cells_per_chunk, num_genes),
            dtype="float32",
            compressor=zarr.Blosc(cname="zstd", clevel=1) if self.compressed else None,
        )

        zarr_file.attrs["gene_name"] = list(adata.obs.target_gene_name)

        batch_size = 1_000
        for i in tqdm(range(0, num_cells, batch_size)):
            j = i + batch_size
            X = np.array(adata.X[i:j, :].todense(), dtype=np.float32)
            zarr_file[i:j, :] = X

        # Important to reduce impact on filesystem
        zarr.consolidate_metadata(f"{self.cache_dir}/{dataset_name}.zarr")

    def train_dataloader(self):
        """
        Define train data depending on if only public datasets are used or not
        """
        if self.only_public_datasets:
            return self.base_dm.train_dataloader()

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            sampler=CustomRandomSampler(
                self.train_data.num_samples_base_data,
                self.train_data.num_samples_added_data,
                samples_per_epoch=min(
                    int(
                        self.train_data.num_samples_base_data
                        / (1 - self.p_sampling_per_epoch)
                    ),
                    self.train_data.num_samples_base_data
                    + self.train_data.num_samples_added_data,
                ),
            ),
        )

    def val_dataloader(self):
        """
        Define validation data depending on if only public datasets are used or not
        """
        if self.only_public_datasets:
            return self.base_dm.val_dataloader()

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Define test data depending on if only public datasets are used or not
        """
        if self.only_public_datasets:
            return self.base_dm.test_dataloader()

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def setup(self, stage=None):
        # Revert to base datamodule if no data is added
        if len(self.added_datasets) == 0:
            self.base_dm.setup(stage)
            self.condition_group, self.subgroup = (
                self.base_dm.condition_group,
                self.base_dm.subgroup,
            )

            return

        # Collect metadata for added datasets
        add_datasets_metadata = {}
        dataset_pert_idx_tuples = []
        for dataset_name in self.added_datasets.keys():
            files = [
                entry.name
                for entry in os.scandir(f"{self.cache_dir}/{dataset_name}")
                if entry.is_file()
            ]
            add_datasets_metadata[dataset_name] = {
                file.split(".")[0]: torch.load(
                    f"{self.cache_dir}/{dataset_name}/{file}"
                )
                for file in files
            }

            add_datasets_metadata[dataset_name] = {
                p: metadata["pert_indices"]
                for p, metadata in add_datasets_metadata[dataset_name].items()
                if metadata["valid"]
                and metadata["knockdown"]
                <= self.knockdown_threshold  # TODO: Investigate this threshold; probably don't want this filtering
            }

            # Add the perturbation indices
            for p, indices in add_datasets_metadata[dataset_name].items():
                dataset_pert_idx_tuples += [
                    (dataset_name, [self.pert2id[p], -1], f"{p}+ctrl", idx)
                    for idx in indices
                ]

            # Also add the control indices
            dataset_pert_idx_tuples += [
                (dataset_name, [-1], "ctrl", idx)
                for idx in self.cntr_indices[dataset_name]
            ]

        if stage == "fit" or stage is None:
            self.train_cell_type_mask = self.base_dm.treatment_data.obs[
                CELL_TYPE
            ].isin(self.base_dm.train_cell_types)
            self.test_cell_type_mask = (
                self.base_dm.treatment_data.obs[CELL_TYPE]
                == self.base_dm.test_cell_type
            )
            # TODO change to our random splits instead of using theirs
            if self.base_dm.use_predefined_splits:
                if not os.path.exists(
                    self.base_dm.data_path
                    / PUBLIC_CELL_TYPE_DATASETS[self.base_dm.test_cell_type]
                    / f"splits/{PUBLIC_CELL_TYPE_DATASETS[self.base_dm.test_cell_type]}_simulation_1_0.75.pkl"
                ):
                    raise ValueError("Predefined splits not found")
                self.condition_group = joblib.load(
                    self.base_dm.data_path
                    / PUBLIC_CELL_TYPE_DATASETS[self.base_dm.test_cell_type]
                    / f"splits/{PUBLIC_CELL_TYPE_DATASETS[self.base_dm.test_cell_type]}_simulation_1_0.75.pkl"
                )
                self.subgroup = joblib.load(
                    self.base_dm.data_path
                    / PUBLIC_CELL_TYPE_DATASETS[self.base_dm.test_cell_type]
                    / f"splits/{PUBLIC_CELL_TYPE_DATASETS[self.base_dm.test_cell_type]}_simulation_1_0.75_subgroup.pkl"
                )
                logger.info("Using predefined splits")
            else:
                self.condition_group, self.subgroup = (
                    self.base_dm.get_condition_groups_and_subgroups(
                        np.unique(
                            self.base_dm.treatment_data[self.train_cell_type_mask]
                            .obs["condition"]
                            .values
                        ),
                        np.unique(
                            self.base_dm.treatment_data[self.test_cell_type_mask]
                            .obs["condition"]
                            .values
                        ),
                        test_size=0.25,
                    )
                )

            held_out_perts = set(
                self.condition_group["val"] + self.condition_group["test"]
            )

            dataset_pert_idx_tuples = [
                item
                for item in dataset_pert_idx_tuples
                if item[2] not in held_out_perts
            ]

            train_indices = (
                self.base_dm.treatment_data.obs["condition"]
                .isin(self.condition_group["train"])
                .values
            ) & self.train_cell_type_mask

            self.train_adata = self.base_dm.treatment_data[train_indices]
            self.train_conditions = self.base_dm.conditions[train_indices]
            self.train_condition_names = self.base_dm.condition_names[train_indices]

            added_data_args = (
                {
                    "dataset_names": list(self.added_datasets.keys()),
                    "zarr_path": self.cache_dir,
                    "dataset_pert_idx_tuples": dataset_pert_idx_tuples,
                    "cntr_indices": self.cntr_indices,
                    "gene_indices": {
                        dataset_name: [
                            self.gene_to_col[dataset_name][g] for g in self.genes
                        ]
                        for dataset_name in self.added_datasets.keys()
                    },
                    "num_base_cell_types": len(self.base_dm.total_cell_types),
                }
                if self.added_datasets is not None
                else None
            )

            self.train_data = HybridXcelltypePerturbDataset(
                self.train_adata,
                self.base_dm.control_data_train,
                self.train_conditions,
                self.train_condition_names,
                self.base_dm.device,
                train_cond_names=self.base_dm.control_train_condition_names,
                added_data_args=added_data_args,
                target_sum=self.target_sum,
                p_added_data_usage=self.p_added_data_usage,
                p_sampling_per_epoch=self.p_sampling_per_epoch,
            )

            val_indices = (
                self.base_dm.treatment_data.obs["condition"]
                .isin(self.condition_group["val"])
                .values
            ) & self.train_cell_type_mask

            self.val_adata = self.base_dm.treatment_data[val_indices]
            self.val_conditions = self.base_dm.conditions[val_indices]
            self.val_condition_names = self.base_dm.condition_names[val_indices]

            self.val_data = HybridXcelltypePerturbDataset(
                self.val_adata,
                self.base_dm.control_data_train,
                self.val_conditions,
                self.val_condition_names,
                self.base_dm.device,
            )

            train_cell_type_de_indices = self.base_dm.adata.obs[CELL_TYPE].isin(
                self.base_dm.train_cell_types
            )

            train_de_indices = (
                self.base_dm.adata.obs["condition"]
                .isin(self.condition_group["train"])
                .values
            ) & train_cell_type_de_indices

            val_de_indices = (
                self.base_dm.adata.obs["condition"]
                .isin(self.condition_group["val"])
                .values
            ) & train_cell_type_de_indices

            test_indices = (
                self.base_dm.treatment_data.obs["condition"]
                .isin(self.condition_group["test"])
                .values
            ) & self.test_cell_type_mask

            self.test_adata = self.base_dm.treatment_data[test_indices]
            self.test_conditions = self.base_dm.conditions[test_indices]
            self.test_condition_names = self.base_dm.condition_names[test_indices]

            self.test_data = HybridXcelltypePerturbDataset(
                self.test_adata,
                self.base_dm.control_data_test,
                self.test_conditions,
                self.test_condition_names,
                self.base_dm.device,
            )

            cache_file = (
                f"{self.base_dm.cache_path}/de_adata_fit_predefined.h5ad"
                if self.base_dm.use_predefined_splits
                else f"{self.base_dm.cache_path}/de_adata_fit.h5ad"
            )

            if (
                os.path.exists(cache_file)
                and not self.base_dm.debug
                and self.base_dm.use_cache
            ):
                logger.info(f"de_adata cached, loading from {cache_file}")
                de_adata = sc.read_h5ad(cache_file)
            else:
                test_cell_type_de_indices = (
                    self.base_dm.adata.obs[CELL_TYPE] == self.base_dm.test_cell_type
                )
                test_de_indices = test_cell_type_de_indices
                de_adata = self.base_dm.adata[
                    (train_de_indices) | (val_de_indices) | (test_de_indices)
                ]
                de_adata = calculate_de_for_eval(de_adata)

                if not self.base_dm.debug:

                    de_adata.write(cache_file)
                    logger.info(f"de_adata saved to {cache_file}")

            self.base_dm.adata.uns.update(de_adata.uns)

        elif stage == "validate":
            val_indices = (
                self.base_dm.treatment_data.obs["condition"]
                .isin(self.condition_group["val"])
                .values
            ) & self.train_cell_type_mask

            self.val_adata = self.base_dm.treatment_data[val_indices]
            self.val_conditions = self.base_dm.conditions[val_indices]
            self.val_condition_names = self.base_dm.condition_names[val_indices]

            self.val_data = HybridXcelltypePerturbDataset(
                self.val_adata,
                self.base_dm.control_data_train,
                self.val_conditions,
                self.val_condition_names,
                self.base_dm.device,
            )

        elif stage == "test":
            test_indices = (
                self.base_dm.treatment_data.obs["condition"]
                .isin(self.condition_group["test"])
                .values
            ) & self.test_cell_type_mask

            self.test_adata = self.base_dm.treatment_data[test_indices]
            self.test_conditions = self.base_dm.conditions[test_indices]
            self.test_condition_names = self.base_dm.condition_names[test_indices]

            self.test_data = HybridXcelltypePerturbDataset(
                self.test_adata,
                self.base_dm.control_data_test,
                self.test_conditions,
                self.test_condition_names,
                self.base_dm.device,
            )

            cache_file = (
                f"{self.base_dm.cache_path}/de_adata_test_predefined.h5ad"
                if self.base_dm.use_predefined_splits
                else f"{self.base_dm.cache_path}/de_adata_test.h5ad"
            )

            if (
                os.path.exists(cache_file)
                and not self.base_dm.debug
                and self.base_dm.use_cache
            ):
                logger.info(f"de_adata cached, loading from {cache_file}")
                de_adata = sc.read_h5ad(cache_file)
            else:
                test_cell_type_de_indices = (
                    self.base_dm.adata.obs[CELL_TYPE] == self.base_dm.test_cell_type
                )
                test_de_indices = test_cell_type_de_indices
                de_adata = self.base_dm.adata[test_de_indices]
                de_adata = calculate_de_for_eval(de_adata)

                if not self.base_dm.debug:

                    de_adata.write(cache_file)
                    logger.info(f"de_adata saved to {cache_file}")

            self.base_dm.adata.uns.update(de_adata.uns)


class HybridXcelltypePerturbDataset(Dataset):
    def __init__(
        self,
        base_adata: sc.AnnData,
        base_cntr_adata: sc.AnnData,
        pert_conditions,
        condition_names,
        device: str = "cuda",
        train_cond_names: pd.Series = None,
        added_data_args: Dict[str, Any] = None,
        target_sum: int = 4e4,
        p_added_data_usage: float = 1.0,
        p_sampling_per_epoch: float = 0.5,
    ):
        """
        Args:
            base_adata (sc.AnnData): The AnnData object
            base_cntr_adata (sc.AnnData): The AnnData object for the control data
            pert_conditions (pd.Series): The conditions for each perturbation
            condition_names (pd.Series): The names of the conditions
            device (str): The device to use for the data
            train_cond_names (pd.Series): The names of the conditions for the training data
            added_data_args (dict): The arguments for the added data
            target_sum (int): The target sum for the total counts per cell
            p_added_data_usage (float): The percentage of added data to use
            p_sampling_per_epoch (float): The percentage of added data to sample per epoch
        """
        if added_data_args is not None:
            self.added_data = True
            self.p_sampling_per_epoch = p_sampling_per_epoch

            (
                dataset_names,
                zarr_path,
                self.dataset_pert_idx_tuples,
                self.cntr_indices,
                self.gene_indices,
                num_base_cell_types,
            ) = added_data_args.values()

            self.zarr_file_dict = {
                dataset_name: zarr.open_consolidated(
                    f"{zarr_path}/{dataset_name}.zarr", mode="r"
                )  # zarr should be saved with consolidate_metadata
                for dataset_name in dataset_names
            }

            self.cx_to_onehot = {
                dataset_name: torch.Tensor(
                    (num_base_cell_types + idx) * [0.0]
                    + [1.0]
                    + (len(dataset_names) - 1 - idx) * [0.0]
                )
                for idx, dataset_name in enumerate(dataset_names)
            }

            # Shuffle added samples
            np.random.shuffle(self.dataset_pert_idx_tuples)

            # Reduce number of samples if desired
            if p_added_data_usage is not None:
                assert 0 < p_added_data_usage <= 1

                num_samples = int(
                    p_added_data_usage * len(self.dataset_pert_idx_tuples)
                )
                self.dataset_pert_idx_tuples = self.dataset_pert_idx_tuples[
                    :num_samples
                ]
        else:
            self.added_data = False

        self.full_ctrl = torch.Tensor(base_cntr_adata.X.toarray()).to(device)
        self.pert_data = torch.Tensor(base_adata.X.toarray()).to(device)

        self.treatment_cell_types = base_adata.obs[CELL_TYPE]
        self.control_cell_types = base_cntr_adata.obs[CELL_TYPE]
        self.all_cell_types = pd.concat(
            [self.treatment_cell_types, self.control_cell_types]
        )

        self.control_cell_type_indices = {}
        for cell_type in self.all_cell_types.unique():
            cell_type_indices = base_adata.obs[CELL_TYPE] == cell_type
            control_indices = base_cntr_adata.obs[CELL_TYPE] == cell_type
            self.control_cell_type_indices[cell_type] = (
                control_indices.values.nonzero()[0]
            )

        self.pert_condition = pert_conditions
        self.condition_names = condition_names
        self.device = device
        self.index = base_adata.obs.index.to_series()

        if not train_cond_names is None:
            train_ctrl = self.full_ctrl
            self.pert_data = torch.cat([self.pert_data, train_ctrl])
            self.pert_condition = pd.concat(
                [self.pert_condition, pd.Series([[-1]] * self.full_ctrl.shape[0])],
            )
            self.condition_names = pd.concat([self.condition_names, train_cond_names])
            self.index = pd.concat([self.index, base_cntr_adata.obs.index.to_series()])
            self.treatment_cell_types = self.all_cell_types

        self.num_samples_base_data = self.pert_data.shape[0]
        self.num_samples_added_data = (
            len(self.dataset_pert_idx_tuples) if self.added_data else 0
        )
        self.target_sum = target_sum

        if self.added_data:
            base_data_percentage = round(
                self.num_samples_base_data
                / (self.num_samples_base_data + self.num_samples_added_data)
                * 100,
                1,
            )
            base_data_percentage_epoch = round(
                self.num_samples_base_data
                / (self.num_samples_base_data / (1 - self.p_sampling_per_epoch))
                * 100,
                1,
            )

            logger.info(
                f"Dataset composition: {self.num_samples_base_data}/{self.num_samples_added_data} public samples ({base_data_percentage} % globally; {base_data_percentage_epoch} % per epoch)"
            )

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the base dataset.
        """
        num_samples = self.num_samples_base_data

        if self.added_data:
            num_samples = int(num_samples / (1 - self.p_sampling_per_epoch))
            num_samples = min(
                num_samples, self.num_samples_base_data + self.num_samples_added_data
            )

        return num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (sample, label) where sample is the data sample and label is the corresponding label.
        """
        if idx < self.num_samples_base_data:
            cell_type = self.treatment_cell_types.iloc[idx]

            control = self.full_ctrl[
                np.random.choice(self.control_cell_type_indices[cell_type])
            ]

            pert_data = self.pert_data[idx]

            return (
                pert_data,
                control,
                self.pert_condition.iloc[idx],
                self.condition_names.iloc[idx],
                self.index.iloc[idx],
                cell_type,
            )

        else:
            idx -= self.num_samples_base_data
            dataset_name, pert_id, pert_name, pert_idx = self.dataset_pert_idx_tuples[
                idx
            ]
            pert_data = torch.Tensor(self.zarr_file_dict[dataset_name][pert_idx, :])[
                self.gene_indices[dataset_name]
            ]

            # Normalize total counts
            pert_data = pert_data / pert_data.sum() * self.target_sum

            # Transform log1p
            pert_data = np.log1p(pert_data)

            # Concatenate one-hot encoded cell type
            pert_data = torch.cat([pert_data, self.cx_to_onehot[dataset_name]]).to(
                self.device
            )

            cntr_idx = np.random.choice(self.cntr_indices[dataset_name])
            control = torch.Tensor(self.zarr_file_dict[dataset_name][cntr_idx, :])[
                self.gene_indices[dataset_name]
            ]
            control = control / control.sum() * self.target_sum
            control = np.log1p(control)
            control = torch.cat([control, self.cx_to_onehot[dataset_name]]).to(
                self.device
            )

            return (
                pert_data,
                control,
                pert_id,
                f"{dataset_name}_{pert_name}_n/a",
                idx,
                dataset_name,
            )


class CustomRandomSampler(Sampler):
    """
    Custom random sampler that samples a fixed number of base samples and a custom number of added samples per epoch
    """

    def __init__(
        self,
        num_base_samples: int,
        num_added_samples: int,
        samples_per_epoch: int,
    ):
        self.num_base_samples = num_base_samples
        self.num_added_samples = num_added_samples
        self.samples_per_epoch = samples_per_epoch
        self.delta_samples = samples_per_epoch - num_base_samples

    def __iter__(self):
        # Sample without replacement
        base_indices = list(range(self.num_base_samples))
        added_indices = random.sample(
            range(
                self.num_base_samples, self.num_base_samples + self.num_added_samples
            ),
            self.delta_samples,
        )
        combined_indices = base_indices + added_indices
        random.shuffle(combined_indices)
        return iter(combined_indices)

    def __len__(self):
        return self.samples_per_epoch
