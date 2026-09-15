"""In-memory train/test1/test2 split for benchmark evaluation.

Adapted from biomodal_codebase/biomodalities/data/data_split.py.
Returns boolean masks instead of AnnData copies — no disk writes,
safe for concurrent training runs.
"""

import numpy as np
import pandas as pd


def train_test1_test2_split(obs_df, batch_column, perturb_column, control_column, control_key,
                            train_size=0.7, random_state=42):
    """Split observation metadata into train/test1/test2 using boolean masks.

    Uses the same splitting logic as the biomodal benchmark:
    - Train: train_size fraction of batches AND perturbations
    - Test1: remaining batches, SAME perturbations as train
    - Test2: remaining batches, NEW perturbations not in train
    - Controls are distributed to each split by batch membership

    Args:
        obs_df: DataFrame with observation metadata (adata.obs)
        batch_column: Column name for batch differentiation
        perturb_column: Column name for perturbation differentiation
        control_column: Column name for control flag
        control_key: Value indicating control samples
        train_size: Proportion of batches and perturbations for training
        random_state: Random seed for reproducibility

    Returns:
        (train_mask, test1_mask, test2_mask): boolean numpy arrays indexing into obs_df rows
    """
    np.random.seed(random_state)

    n = len(obs_df)
    is_control = (obs_df[control_column] == control_key).values
    is_not_control = ~is_control

    # Get batches and perturbations from non-control data
    non_ctrl_obs = obs_df[is_not_control]
    batches = np.random.permutation(non_ctrl_obs[batch_column].unique())
    perturbations = np.random.permutation(non_ctrl_obs[perturb_column].unique())

    num_train_batches = int(len(batches) * train_size)
    train_batches = set(batches[:num_train_batches])
    test_batches = set(batches[num_train_batches:])

    num_train_perturbations = int(len(perturbations) * train_size)
    train_perturbations = set(perturbations[:num_train_perturbations])
    test2_perturbations = set(perturbations[num_train_perturbations:])

    batch_vals = obs_df[batch_column].values
    pert_vals = obs_df[perturb_column].values

    # Non-control masks
    in_train_batch = np.array([b in train_batches for b in batch_vals])
    in_test_batch = np.array([b in test_batches for b in batch_vals])
    in_train_pert = np.array([p in train_perturbations for p in pert_vals])
    in_test2_pert = np.array([p in test2_perturbations for p in pert_vals])

    # Non-control assignment
    nc_train = is_not_control & in_train_batch & in_train_pert
    nc_test1 = is_not_control & in_test_batch & in_train_pert
    nc_test2 = is_not_control & in_test_batch & in_test2_pert

    # Ensure exclusivity
    nc_test1 = nc_test1 & ~nc_train
    nc_test2 = nc_test2 & ~nc_train & ~nc_test1

    # Determine which batches ended up in each split
    train_batch_set = set(obs_df.loc[nc_train, batch_column].unique())
    test1_batch_set = set(obs_df.loc[nc_test1, batch_column].unique())
    test2_batch_set = set(obs_df.loc[nc_test2, batch_column].unique())

    # Assign controls by batch membership
    ctrl_train = is_control & np.array([b in train_batch_set for b in batch_vals])
    ctrl_test1 = is_control & np.array([b in test1_batch_set for b in batch_vals])
    ctrl_test2 = is_control & np.array([b in test2_batch_set for b in batch_vals])

    # Combine non-control and control masks
    train_mask = nc_train | ctrl_train
    test1_mask = nc_test1 | ctrl_test1
    test2_mask = nc_test2 | ctrl_test2

    return train_mask, test1_mask, test2_mask
