"""Label-free finetuning of the released encoder (docs/finetuning.md).

Guards the three things that went wrong, or would have, the first time it was run:

1. `configs/finetune_ssl.yaml` claims to be stage 2 of the published recipe with only
   the length changed. If it drifts anywhere else, that claim is false.
2. `run.wandb.enabled: false` must reach the finetune stage. It did not: the stage
   inherited the parent checkpoint's `use_wandb` and logged under the parent's run name.
3. `make_finetune_init.py` must change `config.mask_probability` and nothing else --
   gradual unmasking ramps from it, and stage 2 records 0.0 (no masking at all).
   `densify_h5ad.py` must reproduce X exactly.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT_CFG = os.path.join(REPO, "configs", "finetune_ssl.yaml")
RECIPE = os.path.join(REPO, "configs", "occam_small.yaml")


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!include", lambda loader, node: None)


def _raw(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_Loader)


def _script(name):
    path = os.path.join(REPO, "scripts", "finetune", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finetune_ssl_is_stage_two_of_the_recipe_with_only_the_length_changed():
    ft = _raw(FT_CFG)["stages"]["finetune"]
    ref = _raw(RECIPE)["stages"]["finetune"]
    assert ft["optim"]["epochs"] == 5
    for group in ("unmasking", "optim", "masking", "loss", "runtime"):
        for key, want in ref[group].items():
            if (group, key) == ("optim", "epochs"):
                continue
            assert ft[group][key] == want, f"{group}.{key}: {ft[group][key]!r} != recipe {want!r}"
    assert ft["loss"]["recon_loss"] == "poisson"
    assert ft["load"]["strict"] is True          # poisson -> poisson: nothing to excuse


def test_run_wandb_disabled_reaches_the_finetune_stage():
    from occam.config.loader import load_config

    cfg = load_config(FT_CFG)
    ft_cfg, _meta = cfg.build_finetune_config()
    assert ft_cfg.use_wandb is False, (
        "run.wandb.enabled: false did not reach stage 2; resolve() would inherit the "
        "parent checkpoint's use_wandb and log under the parent's run name")


def test_finetune_init_changes_only_the_mask_start():
    import torch

    mod = _script("make_finetune_init")
    with tempfile.TemporaryDirectory() as tmp:
        parent = os.path.join(tmp, "stage2.pth")
        sd = {"encoder.1.weight": torch.randn(4, 3), "decoder.0.weight": torch.randn(3, 4)}
        torch.save({"model_state_dict": sd, "config": {"mask_probability": 0.0, "target_sum": 2000.0},
                    "gene_names": ["a", "b", "c"]}, parent)
        out = os.path.join(tmp, "init.pth")
        info = mod.make_init(parent, out, mask_start=0.1)
        ck = torch.load(out, map_location="cpu", weights_only=False)
        assert info["mask_probability_before"] == 0.0
        assert ck["config"]["mask_probability"] == 0.1
        assert ck["config"]["target_sum"] == 2000.0
        assert ck["gene_names"] == ["a", "b", "c"]
        assert all(torch.equal(ck["model_state_dict"][k], sd[k]) for k in sd)


def test_finetune_init_refuses_a_unified_checkpoint():
    import torch

    mod = _script("make_finetune_init")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "unified.pth")
        torch.save({"model_state_dict": {}, "config": {}, "is_unified_checkpoint": True}, p)
        try:
            mod.make_init(p, os.path.join(tmp, "x.pth"))
        except ValueError as err:
            assert "unified" in str(err)
        else:
            raise AssertionError("a unified checkpoint was accepted as a finetune parent")


def test_densify_reproduces_a_sparse_x_exactly():
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    mod = _script("densify_h5ad")
    rng = np.random.default_rng(0)
    dense = rng.poisson(0.3, size=(37, 11)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "s.h5ad"), os.path.join(tmp, "d.h5ad")
        ad.AnnData(X=sp.csr_matrix(dense)).write_h5ad(src)
        assert mod.densify(src, dst, block=8, check_rows=37)
        with h5py.File(dst, "r") as f:
            assert not isinstance(f["X"], h5py.Group)
            np.testing.assert_array_equal(f["X"][:], dense)
        assert ad.read_h5ad(dst).shape == (37, 11)
