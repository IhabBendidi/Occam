"""Panel evaluation: single cell in, single cell out, cell-eval v1.

One predicted cell per real perturbed cell, each generated from a single real
control cell drawn from a pool the scoring reference does not contain. Nothing
is averaged anywhere except inside the deliberately zero-signal ``ctrlmean``
anchor, where averaging is the definition of the baseline.

    from occam.vcell.panel import load_panel, predict_panel, run_panel_evaluation

    panel = load_panel(config, encoder_gene_names, skip_normalize, skip_log1p)
    pred, labels, ctrl_used, cin_idx = predict_panel(
        model, panel, pert2id, device, seed=42)
    results = run_panel_evaluation(panel, pred, labels, ctrl_used, out_dir)

The zero-signal anchors are mandatory. See :mod:`occam.vcell.panel.arms`.
"""

from __future__ import annotations

__all__ = [
    "Panel", "load_panel", "assert_disjoint_controls",
    "predict_panel", "predict_null",
    "fit_control_quantile_map", "apply_quantile_map", "estimate_gamma",
    "write_celleval_v1", "run_celleval_v1",
    "build_arms", "ANCHOR_TAGS",
    "run_panel_evaluation", "write_report", "HEADLINE_PANEL",
    "build_control_identity_samples", "build_panel_val_dataset",
]

_SOURCES = {
    "Panel": "panel_io", "load_panel": "panel_io",
    "assert_disjoint_controls": "panel_io",
    "predict_panel": "predict", "predict_null": "predict",
    "fit_control_quantile_map": "calibrate", "apply_quantile_map": "calibrate",
    "estimate_gamma": "calibrate",
    "write_celleval_v1": "celleval_v1", "run_celleval_v1": "celleval_v1",
    "build_arms": "arms", "ANCHOR_TAGS": "arms",
    "run_panel_evaluation": "run", "write_report": "run", "HEADLINE_PANEL": "run",
    "build_control_identity_samples": "train_hooks",
    "build_panel_val_dataset": "train_hooks",
}


def __getattr__(name):
    """Lazy re-export, so importing the package does not pull in torch."""
    mod = _SOURCES.get(name)
    if mod is None:
        raise AttributeError(f"module 'occam.vcell.panel' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"occam.vcell.panel.{mod}"), name)
