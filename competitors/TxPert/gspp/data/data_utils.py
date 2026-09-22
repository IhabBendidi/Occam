from typing import List, Dict, Tuple, Union, Optional
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm

import torch
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from loguru import logger

from gspp.constants import (
    GENE_NAME,
    MOL_NAME,
    CONDITION_NAME,
    CONTROL,
    CONDITION,
    DOSE,
    DOSE_RAW,
    CELL_TYPE,
    IS_CONTROL,
    BATCH,
    BATCH_RAW,
    CONTROL_LABEL,
    DATA_DIR    
)

warnings.filterwarnings("ignore")

logger.remove()
logger.add(sys.stdout, level="INFO")

sc.settings.verbosity = 0


@dataclass
class Payload:
    """
    Dataclass for collate function
    """
    x: torch.Tensor
    control: torch.Tensor
    p: torch.Tensor
    pert_idxs: pd.Series
    pert_cond_names: pd.Series
    pert_dose: pd.Series
    idx: pd.Series
    cell_types: pd.Series
    experimental_batches: pd.Series


def collate_fn(batch) -> Payload:
    """
    Custom collate function for DataLoader

    Args:
        batch: Batch of data

    Returns:
        Payload: Collated data
    """
    x, control, p, pert_idxs, pert_cond_names, pert_dose, idx, cell_types, experimental_batches = zip(*batch)
    return Payload(
        x=torch.stack(x),
        control=torch.stack(control),
        p=torch.stack(p),
        pert_idxs=pert_idxs,
        pert_cond_names=pert_cond_names,
        pert_dose=pert_dose,
        idx=idx,
        cell_types=cell_types,
        experimental_batches=experimental_batches,
    )




def calculate_de_for_eval(adata: sc.AnnData, molecular_perts: bool = False) -> sc.AnnData:
    """
    Calculate DEGs for evaluation

    Args:
        adata: AnnData object

    Returns:
        AnnData: Updated AnnData object with DEGs
    """
    adata = get_DE_genes(adata, skip_calc_de=False, molecular_perts=molecular_perts)
    adata = get_dropout_non_zero_genes(adata)
    return adata


def get_DE_genes(adata: sc.AnnData, skip_calc_de: bool = False, molecular_perts: bool = False) -> sc.AnnData:
    """
    Get DE genes

    Args:
        adata: AnnData object
        skip_calc_de: Whether to skip DE calculation
        molecular_perts: Whether perturbations are molecules

    Returns:
        AnnData: Updated AnnData object with DE genes
    """
    if not skip_calc_de:
        _rank_genes_groups_by_cov(
            adata,
            groupby=CONDITION_NAME,
            covariate=CELL_TYPE,
            control_group=f"{CONTROL_LABEL}_0" if molecular_perts else f"{CONTROL_LABEL}_1",
            n_genes=len(adata.var),
            key_added="rank_genes_groups_cov_all",
        )
    return adata


def get_dropout_non_zero_genes(adata: sc.AnnData) -> sc.AnnData:
    """
    Get dropout and non-zero genes

    Args:
        adata: AnnData object

    Returns:
        AnnData: Updated AnnData object with dropout and non-zero genes
    """
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis=0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(
        len(adata.obs.condition.unique()), adata.X.toarray().shape[1]
    )
    ctrl = mean_expression[np.where(pert_list == CONTROL_LABEL)[0]]

    # in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[[CONDITION, CONDITION_NAME]].values)
    pert_full_id2pert = dict(adata.obs[[CONDITION_NAME, CONDITION]].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in tqdm(adata.uns["rank_genes_groups_cov_all"].keys()):
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis=0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns["rank_genes_groups_cov_all"][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)

    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))

    adata.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
    adata.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
    adata.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
    adata.uns["top_non_zero_de_20"] = top_non_zero_de_20

    return adata














def normalize_and_log_transform(
    adata: sc.AnnData, target_sum: Optional[int] = None
) -> sc.AnnData:
    """
    Normalize and log transform AnnData object

    Args:
        adata: AnnData object
    """
    if target_sum is not None:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata








def _ensure_required_columns(adata: sc.AnnData, pert_name: str) -> sc.AnnData:
    """
    Ensure the input AnnData object has the required columns

    Args:
        adata: AnnData object
    """
    required_columns = [pert_name, IS_CONTROL, BATCH_RAW, CELL_TYPE]
    
    if pert_name == MOL_NAME:
        required_columns.append(DOSE_RAW)
    
    for col in required_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Required column '{col}' is missing from adata.obs")


def _standardize_obs_format(adata: sc.AnnData, pert_name: str) -> sc.AnnData:
    """
    Standardize the AnnData obs object to what our model uses.
    Requires the origial obs to have the following columns (check their actual columns in constants.py):
    - GENE_NAME (or MOL_NAME for molecular perturbations)
    - IS_CONTROL
    - BATCH_RAW
    - CELL_LINE
    - DOSE (only for molecular perturbations)
    Args:
        adata: AnnData object
    """

    _ensure_required_columns(adata, pert_name)

    adata.obs[CONDITION] = adata.obs[pert_name].astype(str)

    # This ensures that the molecular perturbation datasets get the same format as the genetic ones
    # Can and should be removed once input data format is curated to be consistent
    #
    # ⚠️ PER ROW, not globally. The original was
    #     if not all("+ctrl" in c for c in <all perturbed conditions>):
    #         <append "+ctrl" to EVERY perturbed row>
    # which is correct only when every perturbation is single-gene. On a COMBINATORIAL
    # corpus the singles ("AHR") fail the test, so the whole column is rewritten and a
    # pair "AHR+FEV" becomes "AHR+FEV+ctrl" -- THREE slots in a two-slot format. The
    # held-out pairs then match nothing and `_load_zarr_splits` refuses with "131
    # held-out perturbation(s) ... absent from the data after vocabulary filtering".
    # Measured on norman_combo 2026-09-22.
    #
    # The suffix exists to fill the UNUSED combinatorial slot of a single-gene
    # perturbation -- GEARS format is "A+B", and a single gene is "A+ctrl". A row that
    # already carries a "+" has both slots used and must be left alone. Applying it row
    # by row is therefore both the fix and the original intent.
    perturbed = ~adata.obs[IS_CONTROL]
    cond = adata.obs.loc[perturbed, CONDITION].astype(str)
    needs_slot = ~cond.str.contains(r"\+", regex=True)
    if needs_slot.any():
        adata.obs.loc[cond.index[needs_slot], CONDITION] = (
            cond[needs_slot] + f"+{CONTROL_LABEL}"
        )

    adata.obs.loc[adata.obs[IS_CONTROL], CONDITION] = CONTROL_LABEL
    adata.obs[CONDITION] = adata.obs[CONDITION].astype("category")
    adata.obs[CONTROL] = adata.obs[IS_CONTROL].astype(int).astype("category")
    adata.obs[BATCH] = adata.obs[BATCH_RAW].astype("category")
    adata.obs[CELL_TYPE] = adata.obs[CELL_TYPE]

    adata.var = adata.var[[GENE_NAME]]
    adata.var_names = list(adata.var[GENE_NAME])
    adata.var_names_make_unique()

    if pert_name == MOL_NAME:
        adata.obs[DOSE] = adata.obs[DOSE_RAW].astype(str)
        adata.obs[DOSE][adata.obs[IS_CONTROL]] = "0"

        # Drop samples with missing dose
        adata = adata[~(adata.obs[DOSE] == "nan")]
    else:
        adata.obs.loc[:, DOSE] = adata.obs.condition.apply(
            lambda x: "1+1" if len(x.split("+")) == 2 else "1"
        )
    if pert_name == MOL_NAME:
        adata.obs[CONDITION_NAME] = adata.obs.apply(
            lambda x: "_".join([x[CELL_TYPE], x[CONDITION], x[DOSE]]), axis=1
        )
    else:
        adata.obs.loc[:, CONDITION_NAME] = adata.obs.apply(
            lambda x: "_".join([x[CELL_TYPE], x[CONDITION], x[DOSE]]), axis=1
        )
    #@TODO gene name is technically not needed
    adata.obs = adata.obs[[CONDITION, DOSE, CONTROL, BATCH, CELL_TYPE, CONDITION_NAME, pert_name]]

    adata.obs = adata.obs.astype("category")
    adata.obs_names_make_unique()

    

    adata.X = sparse.csr_matrix(adata.X)

    return adata


def _rank_genes_groups_by_cov(
    adata: sc.AnnData,
    groupby: str,
    control_group: str,
    covariate: str,
    n_genes: int = 50,
    rankby_abs: bool = True,
    key_added: str = "rank_genes_groups_cov",
    return_dict: bool = False,
) -> Union[Dict, None]:
    """
    Rank genes groups by covariate

    Args:
        adata: AnnData object
        groupby: groupby column
        control_group: control group
        covariate: covariate column
        n_genes: number of genes
        rankby_abs: rank by absolute value
        key_added: key added
        return_dict: return dictionary
    """
    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        min_samples = 2
        valid_groups = adata_cov.obs[groupby].value_counts()[lambda x: x >= min_samples].index
        adata_cov = adata_cov[adata_cov.obs[groupby].isin(valid_groups)]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict
    return None
