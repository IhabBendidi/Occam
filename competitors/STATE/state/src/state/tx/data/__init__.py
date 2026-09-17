"""STATE tx data package.

Provides a small dispatcher over data modules so the existing cell_load h5ad path
stays untouched while a native-zarr loader can be selected via ``data.name``.
"""

from __future__ import annotations

# Name of the native-zarr data module (matches configs/data/perturbation_zarr.yaml).
ZARR_DATAMODULE_NAME = "ZarrPerturbationDataModule"


def _to_plain_dict(kwargs):
    """Convert an OmegaConf DictConfig to a plain dict (no-op for dicts)."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(kwargs, DictConfig):
            return OmegaConf.to_container(kwargs, resolve=True)
    except ImportError:
        pass
    return dict(kwargs)


def get_datamodule(name, kwargs, batch_size=None, cell_sentence_len=1):
    """Dispatch to the native-zarr module or fall back to cell_load.

    Mirrors ``cell_load.utils.modules.get_datamodule``'s signature so it is a
    drop-in replacement in ``_train.py``.
    """
    if name == ZARR_DATAMODULE_NAME:
        from .zarr_perturbation import ZarrPerturbationDataModule

        plain = _to_plain_dict(kwargs)
        if batch_size is not None:
            plain["batch_size"] = batch_size
            plain["cell_sentence_len"] = cell_sentence_len
        if "toml_config_path" not in plain or plain["toml_config_path"] is None:
            raise ValueError("toml_config_path must be provided in data.kwargs")
        return ZarrPerturbationDataModule(**plain)

    # Default: delegate to cell_load (unchanged h5ad path).
    from cell_load.utils.modules import get_datamodule as _cell_load_get_datamodule

    return _cell_load_get_datamodule(
        name, kwargs, batch_size=batch_size, cell_sentence_len=cell_sentence_len
    )


def load_data_module_state(name, filepath, toml_config_path=None):
    """Load a saved data module, dispatching on its config name.

    Used by ``_predict.py`` to reload either the zarr or cell_load module.
    ``toml_config_path`` overrides the path stored in the checkpoint so a
    predict job can restage data without the original training TOML.
    """
    if name == ZARR_DATAMODULE_NAME:
        from .zarr_perturbation import ZarrPerturbationDataModule

        return ZarrPerturbationDataModule.load_state(
            filepath, toml_config_path=toml_config_path
        )

    from cell_load.data_modules import PerturbationDataModule

    return PerturbationDataModule.load_state(filepath)


__all__ = ["get_datamodule", "load_data_module_state", "ZARR_DATAMODULE_NAME"]
