"""DataModule builders split from rawdata_diffusion_training.py (logic-preserving)."""

from src.data import data_module

def build_datamodule(cfg, logger):
    """
    Build datamodule.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :return: Requested object(s) for downstream use.
    """
    if cfg.data.data_name in ["PBMCFinetune"]:
        datamodule = data_module.PBMCPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in ["Tahoe100mFinetune", "ReplogleFinetune"]:
        datamodule = data_module.Tahoe100mPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in ["Tahoe100mPBMCReplogleCellxGenePretrain"]:
        datamodule = data_module.PerturbationPretrainingDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in ["IhabxcellZarrFinetune"]:
        # the upstream benchmark canonical zarr benchmark; see configs/data/ihabxcell_zarr.yaml
        datamodule = data_module.ZarrPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in ["OccamH5adFinetune"]:
        # Occam's rhj_fewshot: OUR three h5ads, OUR frozen 6,983-gene axis, OUR splits, read
        # in place. See configs/data/occam_h5ad.yaml and src/data/occam_bridge.py.
        datamodule = data_module.OccamPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    else:
        raise ValueError(
            f"Unknown data.data_name={cfg.data.data_name!r}. Known values: "
            "PBMCFinetune, Tahoe100mFinetune, ReplogleFinetune, "
            "Tahoe100mPBMCReplogleCellxGenePretrain, IhabxcellZarrFinetune, "
            "OccamH5adFinetune."
        )

    datamodule.replace_pert_dict = cfg.cov_encoding.get("replace_pert_dict", False)
    datamodule.setup()
    return datamodule

def populate_covariate_cfg(cfg, datamodule):
    """Execute `populate_covariate_cfg` and return values used by downstream logic."""
    cfg.cov_encoding.num_pert = len(datamodule.pert_dict)
    cfg.cov_encoding.num_celltype = len(datamodule.cell_type_dict)
    cfg.cov_encoding.num_batch = len(datamodule.batch_dict)

    cfg.cov_encoding.pert_dict = datamodule.pert_dict
    cfg.cov_encoding.cell_type_dict = datamodule.cell_type_dict
    cfg.cov_encoding.batch_dict = datamodule.batch_dict
    cfg.model.dataset_dict = datamodule.original_dataset_name_list
