import torch
import numpy as np
import scanpy as sc

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

from pathlib import Path
import wandb

import warnings
import gspp.constants as cs


def evaluate(loader, model, device, adata, id2pert):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    dose_cat = []
    pred = []
    truth = []
    cell_types = []
    experimental_batches = []
    results = {}

    pert_name_cache = {}
    # head obs is, e.g.
    # condition        control batch cell_type dose_val      condition_name
    # UBL5+ctrl        0       25    K562      1+1            K562_UBL5+ctrl_1+1
    # so the following makes a mapping from e.g. 'K562_UBL5+ctrl_1+1' to 'UBL5+ctrl' style naming
    for condition_name in np.unique(adata.obs[["condition_name"]]):
        pert_name_cache[condition_name] = (condition_name.split("_")[1], condition_name.split("_")[-1])
    
    for batch in loader:
        # batch.to(device)
        with torch.no_grad():
            if model.no_pert:
                p = batch.control[:, : model.adata_output_dim]
            else:
                p = model.sample_inference(
                    batch.control, batch.pert_idxs, batch.p, batch.cell_types
                )
            t = batch.x[:, : model.adata_output_dim]
            pred.append(p.cpu())
            truth.append(t.cpu())
            for itr, perts in enumerate(batch.pert_cond_names):
                # perts is a gears-style perturbation name, e.g. K562_CACTIN+ctrl_1+1
                # we will extract the perturbation +ctrl, e.g. CACTIN+ctrl
                pert_name = pert_name_cache[perts][0]
                dose = pert_name_cache[perts][1]
                pert_cat.append(pert_name)
                dose_cat.append(dose)
                cell_types.append(batch.cell_types[itr])
                experimental_batches.append(batch.experimental_batches[itr])

    print(f"A total of {len(pert_cat)} perturbations were evaluated", flush=True)
    results[cs.RES_PERT_CAT] = np.array(pert_cat)
    results[cs.RES_DOSE_CAT] = np.array(dose_cat)
    results[cs.RES_CELL_TYPE] = np.array(cell_types)
    results[cs.RES_PRED] = torch.cat(pred).numpy()
    results[cs.RES_TRUTH] = torch.cat(truth).numpy()
    results[cs.RES_EXP_BATCH] = np.array(experimental_batches)

    return results








def _split_dataloader(datamodule, split: str):
    """The dataloader for ``split``, by name.

    Exists so the caller can score the VALIDATION split during training without a second code
    path. ``val_dataloader()`` raises when no validation dataset exists, which is the intended
    behaviour -- it used to return the test loader, and scoring "validation" that way is scoring
    test.
    """
    loaders = {
        "train": datamodule.train_dataloader,
        "validation": datamodule.val_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }
    if split not in loaders:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(loaders)}")
    return loaders[split]()


def export_for_celleval(
    model, datamodule, device, output_dir, profile: str = "full", split: str = "test"
):
    """Score the held-out predictions with cell-eval, matching STATE and PerturbDiff.

    Replaces upstream's ``save_xcellv2_predictions``, which wrote an h5ad renormalized to a
    magic ``target_sum=823`` and remapped genes by symbol -- both artefacts of the old merged
    h5ad. Here predictions already live in the zarr's own ``log1p(1e4)`` space and the gene axis
    is the benchmark's 2,000 genes in order, so neither step is needed.

    The layout follows STATE's ``tx predict`` (see gspp/celleval_eval.py, a port of
    PerturbDiff's port of it):

    * ``pred`` and ``real`` are row-aligned, one row per evaluated cell.
    * Real control cells are appended to *both*, unchanged, so differential expression in the
      two objects is measured against exactly the same baseline. cell-eval refuses to run
      without control rows.
    * ``pert_col='target_gene'`` carries Ensembl-free gspp symbols with the ``+ctrl`` suffix
      stripped; controls carry ``control_pert``.
    """
    import anndata as ad
    from gspp.celleval_eval import CONTROL_OUT, run_cell_eval

    base_dm = datamodule.base_dm
    adata = base_dm.adata

    res = evaluate(_split_dataloader(datamodule, split), model, device, adata, base_dm.id2pert)
    pred = np.asarray(res[cs.RES_PRED], dtype=np.float32)
    real = np.asarray(res[cs.RES_TRUTH], dtype=np.float32)
    perts = np.array([str(p).replace("+ctrl", "") for p in res[cs.RES_PERT_CAT]])
    ctypes = np.array([str(c) for c in res[cs.RES_CELL_TYPE]])
    print(f"[celleval] evaluate({split}) -> {pred.shape}, {len(pd.unique(perts))} perturbations")

    genes = [str(g) for g in adata.var_names[: pred.shape[1]]]
    if len(genes) != pred.shape[1]:
        raise ValueError(f"gene count {len(genes)} != prediction width {pred.shape[1]}")

    # Real controls from the test cell type, in the same gene space and on the same scale.
    ctrl_mask = (
        (adata.obs[cs.CONTROL].astype(int) == 1)
        & (adata.obs[cs.CELL_TYPE] == base_dm.test_cell_type)
    ).values
    ctrl_x = adata.X[ctrl_mask]
    ctrl_x = np.asarray(
        ctrl_x.todense() if hasattr(ctrl_x, "todense") else ctrl_x, dtype=np.float32
    )[:, : pred.shape[1]]
    print(f"[celleval] injecting {len(ctrl_x):,} real controls into pred AND real")

    var = pd.DataFrame(index=pd.Index(genes))
    ctrl_types = np.full(len(ctrl_x), base_dm.test_cell_type, dtype=object)

    def build(matrix):
        obs = pd.DataFrame(
            {
                "target_gene": np.concatenate([perts, np.full(len(ctrl_x), CONTROL_OUT, dtype=object)]),
                "cell_type": np.concatenate([ctypes, ctrl_types]),
            }
        )
        obs.index = obs.index.astype(str)
        return ad.AnnData(
            X=np.vstack([matrix, ctrl_x]).astype(np.float32), obs=obs, var=var.copy()
        )

    return run_cell_eval(
        build(pred),
        build(real),
        output_dir=output_dir,
        pert_col="target_gene",
        cell_type_key="cell_type",
        control_pert=CONTROL_OUT,
        profile=profile,
    )
