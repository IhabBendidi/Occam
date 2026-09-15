"""Virtual Cell Mode — perturbation effect prediction for Occam."""

from occam.vcell.config import VirtualCellConfig


def __getattr__(name):
    """Lazy import VirtualCellModel to avoid requiring torch_geometric at import time."""
    if name == "VirtualCellModel":
        from occam.vcell.model import VirtualCellModel
        return VirtualCellModel
    raise AttributeError(f"module 'virtual_cell' has no attribute {name}")
