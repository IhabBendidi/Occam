"""Helper utilities extracted from sampling_generation.py."""

import pickle
import logging

import anndata
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def resolve_sampling_runner(cfg, diffusion, use_ddim):
    """
    Resolve sampling runner.

    :param cfg: Runtime configuration object.
    :param diffusion: Diffusion process instance.
    :param use_ddim: Boolean flag controlling this behavior.
    :return: Requested object(s) for downstream use.
    """
    start_time_steps = cfg.sampling.start_time if cfg.sampling.start_time is not None else cfg.model.steps
    start_time_steps = min(start_time_steps, cfg.model.steps)

    logger.info("start_time_steps: %s", start_time_steps)

    if use_ddim:
        sample_fn = diffusion.ddim_sample_loop
        sampling_kwargs = {
            "start_time": start_time_steps,
            "eta": cfg.sampling.eta,
            "guidance_strength": cfg.sampling.guidance_strength,
        }
    else:
        sample_fn = diffusion.p_sample_loop
        sampling_kwargs = {
            "start_time": start_time_steps,
            "nw": cfg.sampling.nw,
            "start_guide_steps": cfg.sampling.start_guide_steps,
            "guidance_strength": cfg.sampling.guidance_strength,
        }

    sampling_kwargs["sample_kwargs"] = {}
    return sample_fn, sampling_kwargs


def load_selected_genes(cfg):
    """Execute `load_selected_genes` and return values used by downstream logic."""
    if hasattr(cfg.data, "sample_pbmc_only") and cfg.data.sample_pbmc_only:
        cfg.data.selected_gene_file = cfg.data.pbmc.selected_gene_file
    if hasattr(cfg.data, "sample_tahoe100m_only") and cfg.data.sample_tahoe100m_only:
        cfg.data.selected_gene_file = cfg.data.tahoe100m.selected_gene_file
    if hasattr(cfg.data, "sample_replogle_only") and cfg.data.sample_replogle_only:
        cfg.data.selected_gene_file = cfg.data.replogle.selected_gene_file

    # The zarr benchmark's panel is a plain .txt of Ensembl ids (the same file
    # STATE passes as selected_ensembl_gene_ids_path); upstream ships pickles.
    from src.data.dataset.dataset_io_zarr import load_selected_genes as _load_panel

    genes = _load_panel(cfg.data.selected_gene_file)

    expected = getattr(cfg.data, "pad_length", 2000)
    if len(genes) != expected:
        raise ValueError(
            f"selected_gene_file has {len(genes)} genes but data.pad_length={expected}; "
            f"the panel and the model input dimension must agree."
        )
    return genes


def build_zarr_controls(cfg, datamodule, genes, logger):
    """Real control cells for the test split, read straight from X.zarr.

    Upstream loads controls from a separate per-dataset ``.h5ad``
    (``load_ctrl_adata``); on the zarr benchmark the controls are already in the
    test split, so read them from there. Returns ``(X_ctrl, obs_dict)`` with the
    expression in selected-gene order and obs columns named exactly as the
    perturbed rows' are.

    These real controls are concatenated into *both* the predicted and the true
    adata, which is what STATE's ``--inject-real-controls`` does: cell-eval
    computes each perturbation's differential expression against the controls
    found inside each adata, so both sides must share one real baseline.
    """
    import numpy as np

    test_ds = datamodule.test_dataset
    x_rows, pert, celltype, batch = [], [], [], []

    for ds_name in test_ds._names:
        indices = test_ds.data_indices[ds_name]
        ctrl_rows = indices["control"] if isinstance(indices, dict) else []
        if len(ctrl_rows) == 0:
            continue
        cache = test_ds.meta_cache._cache[test_ds.dataset_path_map[ds_name]]
        handle = test_ds.store.dataset_file(ds_name)
        for row in ctrl_rows:
            row = int(row)
            x_rows.append(test_ds._retrieve_counts(handle, ds_name, row).numpy().reshape(-1))
            pert.append(cfg.data.control_pert)
            celltype.append(str(cache.cell_type_categories[cache.cell_type_codes[row]]))
            batch.append(str(cache.batch_categories[cache.batch_codes[row]]))

    if not x_rows:
        raise ValueError(
            "No control cells in the test split; cell-eval needs a control "
            "baseline inside each adata."
        )
    X_ctrl = np.stack(x_rows).astype(np.float32)
    if X_ctrl.shape[1] != len(genes):
        raise ValueError(
            f"control matrix has {X_ctrl.shape[1]} genes, panel has {len(genes)}"
        )
    logger.info(
        "Control cells from the zarr test split: %d x %d", *X_ctrl.shape
    )
    return X_ctrl, {"pert": pert, "celltype": celltype, "batch": batch}


def build_obs_data_zarr(cfg, all_pert, all_celltype, all_batch, ctrl_obs):
    """obs for the zarr benchmark, using STATE's column names.

    ``pert_col`` / ``cell_line_key`` / ``perturbseq_batch_col`` come from the
    data config, so the frame carries the same column names STATE's
    ``tx predict`` writes (``gene`` / ``cell_type`` / ``batch_center``) and the
    same cell-eval invocation works on both models' outputs.
    """
    import numpy as np

    return pd.DataFrame(
        {
            str(cfg.data.pert_col): np.concatenate([all_pert, np.asarray(ctrl_obs["pert"])]),
            str(cfg.data.cell_line_key): np.concatenate(
                [all_celltype, np.asarray(ctrl_obs["celltype"])]
            ),
            str(cfg.data.perturbseq_batch_col): np.concatenate(
                [all_batch, np.asarray(ctrl_obs["batch"])]
            ),
        }
    )


def build_gene_embedding_cache(model, batch_data, device):
    """Execute `build_gene_embedding_cache` and return values used by downstream logic."""
    if getattr(model, "gene_embedding", None) is None:
        return None

    gene_emb = []
    for i, dataset_name in enumerate(batch_data["ds_name"]):
        dataset_name = dataset_name[0]
        if dataset_name not in model.gene_name_embedding_cache:
            model.gene_name_embedding_cache[dataset_name] = torch.stack(
                [model.gene_embedding.get(idx, torch.zeros(5120)) for idx in batch_data["col_genes"][i]]
            )
        gene_emb.append(model.gene_name_embedding_cache[dataset_name])
    return torch.stack(gene_emb).to(device)


def build_self_condition(cfg, model, batch_data, gene_emb):
    """Execute `build_self_condition` and return values used by downstream logic."""
    _ = (cfg, model)
    return {
        "batch_emb": batch_data["batch_emb"],
        "cont_emb": batch_data["cont_emb"],
        "gene_emb": gene_emb,
        "ds_name": batch_data["ds_name"],
    }


def collect_batch_covariates(batch_data, dataloader, datamodule, mask):
    """
    Collect batch covariates.

    :param batch_data: Current batch dictionary.
    :param dataloader: DataLoader yielding input batches.
    :param datamodule: Data module providing datasets and loaders.
    :param mask: Boolean mask for selecting valid elements.
    :return: Requested object(s) for downstream use.
    """
    ret = []
    mask = mask.cpu().detach().numpy()
    for item_idx, ds_name in enumerate(batch_data["ds_name"]):
        data_cov_cache = dataloader.dataset.meta_cache._cache[dataloader.dataset.dataset_path_map[ds_name]]
        local_idx_mapping = {datamodule.pert_dict[v]: i for i, v in enumerate(data_cov_cache.pert_categories)}
        local_idx = np.vectorize(local_idx_mapping.get)(batch_data["cov_pert"].cpu().detach().numpy()[item_idx])
        cov1 = data_cov_cache.pert_categories[local_idx]

        local_idx_mapping = {datamodule.cell_type_dict[v]: i for i, v in enumerate(data_cov_cache.cell_type_categories)}
        local_idx = np.vectorize(local_idx_mapping.get)(batch_data["cov_celltype"].cpu().detach().numpy()[item_idx])
        cov2 = data_cov_cache.cell_type_categories[local_idx]

        local_idx_mapping = {datamodule.batch_dict[ds_name + "_" + str(v)]: i for i, v in enumerate(data_cov_cache.batch_categories)}
        local_idx = np.vectorize(local_idx_mapping.get)(batch_data["cov_batch"].cpu().detach().numpy()[item_idx])
        cov3 = data_cov_cache.batch_categories[local_idx]
        ret.append((cov1[mask[item_idx]], cov2[mask[item_idx]], cov3[mask[item_idx]]))
    return ret


def load_ctrl_adata(cfg):
    """Execute `load_ctrl_adata` and return values used by downstream logic."""
    var_index = None
    if "replogle" in cfg.data.data_name.lower() and (not cfg.data.sample_pbmc_only):
        ctrl_adata = anndata.read_h5ad(cfg.path.replogle_ctrl_h5ad)
        ctrl_adata = ctrl_adata[ctrl_adata.obs.cell_line == "hepg2"]
        ctrl_adata = ctrl_adata[ctrl_adata.obs.gene == "non-targeting"]
        assert len(ctrl_adata) == 4976
    elif "pbmc" in cfg.data.data_name.lower() and (not cfg.data.sample_replogle_only):
        ctrl_adata = anndata.read_h5ad(cfg.path.pbmc_ctrl_h5ad)
    elif "tahoe" in cfg.data.data_name.lower():
        ctrl_adata = anndata.read_h5ad(cfg.path.tahoe100m_ctrl_h5ad)
    else:
        raise ValueError(
            f"Unsupported data_name for control adata loading: {cfg.data.data_name}. "
            "Expected one of replogle/pbmc/tahoe."
        )
    return ctrl_adata, var_index


def build_x_ctrl(ctrl_adata, genes, cfg):
    """
    Build x ctrl.

    :param ctrl_adata: Input `ctrl_adata` value.
    :param genes: Input `genes` value.
    :param cfg: Runtime configuration object.
    :return: Requested object(s) for downstream use.
    """
    all_genes = ctrl_adata.var.index.tolist()
    mask = np.isin(all_genes, genes)

    try:
        assert mask.sum() == len(genes)
    except Exception:
        assert cfg.data.sample_tahoe100m_only
        assert ctrl_adata.X.shape[1] == 2000

    if mask.sum() == len(genes):
        x_ctrl = ctrl_adata.X[:, mask].toarray()
        cur_genes = np.array(all_genes)[mask].tolist()
        cur_index = {g: i for i, g in enumerate(cur_genes)}
        column_map = np.array([cur_index.get(g, -1) for g in genes])
        new_length = len(genes)

        existing_mask = column_map >= 0
        real_counts = np.zeros((x_ctrl.shape[0], new_length), dtype=x_ctrl.dtype)
        real_counts[:, existing_mask] = x_ctrl[:, column_map[existing_mask]]
        x_ctrl = real_counts
    else:
        x_ctrl = ctrl_adata.X
    return x_ctrl


def build_obs_data(cfg, all_pert, all_celltype, all_batch, ctrl_adata):
    """
    Build obs data.

    :param cfg: Runtime configuration object.
    :param all_pert: Input `all_pert` value.
    :param all_celltype: Input `all_celltype` value.
    :param all_batch: Input `all_batch` value.
    :param ctrl_adata: Input `ctrl_adata` value.
    :return: Requested object(s) for downstream use.
    """
    if "replogle" in cfg.data.data_name.lower() and (not cfg.data.sample_pbmc_only) and (not cfg.data.sample_tahoe100m_only):
        obs_data = {
            "gene": np.concatenate([all_pert, np.array(ctrl_adata.obs.gene.values)]),
            "cell_line": np.concatenate([all_celltype, np.array(ctrl_adata.obs.cell_line.values)]),
            "gem_group": np.concatenate([all_batch, np.array(ctrl_adata.obs.gem_group.values)]),
        }
    elif "pbmc" in cfg.data.data_name.lower() and (not cfg.data.sample_replogle_only) and (not cfg.data.sample_tahoe100m_only):
        obs_data = {
            "cytokine": np.concatenate([all_pert, np.array(ctrl_adata.obs.cytokine.values)]),
            "cell_type": np.concatenate([all_celltype, np.array(ctrl_adata.obs.cell_type.values)]),
            "donor": np.concatenate([all_batch, np.array(ctrl_adata.obs.donor.values)]),
        }
    elif "tahoe100m" in cfg.data.data_name.lower() and (not cfg.data.sample_replogle_only) and (not cfg.data.sample_pbmc_only):
        with open(cfg.path.tahoe100m_cellname_to_cellline_pkl, "rb") as fin:
            all_cellname_to_cellline = pickle.load(fin)
        obs_cell_line = ctrl_adata.obs["cell_name"].map(all_cellname_to_cellline)
        plate_dict = {
            x: f"plate{i+1}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab"
            for i, x in enumerate(sorted(set(ctrl_adata.obs.plate.values))[::-1])
        }
        obs_plate = ctrl_adata.obs["plate"].map(plate_dict)

        obs_data = {
            "drugname_drugconc": np.concatenate([all_pert, np.array(ctrl_adata.obs.drugname_drugconc.values)]),
            "cell_line": np.concatenate([all_celltype, np.array(obs_cell_line)]),
            "plate": np.concatenate([all_batch, np.array(obs_plate)]),
        }
    else:
        raise ValueError(
            f"Unsupported data_name for obs building: {cfg.data.data_name}. "
            "Expected one of replogle/pbmc/tahoe100m."
        )
    return pd.DataFrame(obs_data)
