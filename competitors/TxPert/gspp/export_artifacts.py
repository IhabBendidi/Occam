"""Write row-aligned pred/real AnnData once, so one forward pass can be scored twice.

WHY THIS EXISTS. ``export_for_celleval`` scores in-process and injects the REAL control cells into
both pred and real, so it can only ever produce the inject-real convention. Reporting a second
convention that way would mean a second forward pass writing into the same directory -- which is
how ``test_replogle_zarr/`` ended up with two agg CSVs that survive only because someone renamed
them by hand.

Instead: predict once, write ``adata_pred.h5ad`` + ``adata_real.h5ad``, then let the SHARED scorer
(``the upstream tree/eval/celleval_artifacts.py``) score those artifacts twice into separate results dirs.
That is exactly how STATE's two arms are produced, so the two competitors' conventions mean the
same thing and are computed by the same code.

THE TWO CONVENTIONS FOR TXPERT:

  decoded ("reconstruct controls")  -- the control rows on the PRED side are model outputs. TxPert
      already trains for this: ``embed_cntr: true`` puts ``ctrl -> -1`` in ``pert2id``, control
      cells are injected as training targets with condition ``['ctrl']``, and ``gears_derivative``
      maps that to ``z_p[-1]`` -- a row no real gene owns, isolated in every knowledge graph, so
      its embedding is the MLP bias, i.e. a learned control token. Predicting a control is
      therefore in-distribution, not an extrapolation, and needs no training change.
  real ("inject controls")          -- the measured control cells, which the scorer swaps in.

Note this is NOT the PCA-decode sense of "reconstruct" used for the upstream benchmark; like STATE, TxPert
genuinely emits a predicted control. There is no codec here.

HOW THE CONTROLS ARE PREDICTED. Through ``XcelltypePerturbDataset`` and ``evaluate()``, the same
classes that build the test batches -- deliberately, rather than assembling tensors by hand, so
the basal input, dose encoding and ``p`` embedding cannot drift from what the perturbed rows get.
With ``match_cntr``/``avg_cntr`` true, each control cell's basal input is the mean of the controls
in its own (cell_type, batch), which is the same basal input a perturbed cell of that batch
receives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import gspp.constants as cs
from gspp.evaluation import _split_dataloader, evaluate

PRED_NAME = "adata_pred.h5ad"
REAL_NAME = "adata_real.h5ad"

# Column names and the control label the shared scorer is pointed at. Keep in step with
# launch/txpert_fewshot_eval.sh, which passes them on the command line.
PERT_COL = "target_gene"
CELL_TYPE_COL = "cell_type"


def _predict_controls(model, datamodule, device):
    """Model-predicted control cells for the test cell type, plus the real ones.

    Returns ``(pred, real, cell_types)``. The control rows are not members of any split loader --
    ``treatment_data`` excludes controls by construction -- so a dataset is built for them here
    with perturbation id ``-1``, the control token.
    """
    from torch.utils.data import DataLoader

    from gspp.data.datamodule import XcelltypePerturbDataset, collate_fn

    base_dm = datamodule.base_dm
    adata = base_dm.adata

    ctrl_mask = (
        (adata.obs[cs.CONTROL].astype(int) == 1)
        & (adata.obs[cs.CELL_TYPE] == base_dm.test_cell_type)
    ).values
    controls = adata[ctrl_mask]
    if controls.n_obs == 0:
        raise ValueError(f"no control cells for cell type {base_dm.test_cell_type!r}")

    # `-1` rather than the string 'ctrl': gears_derivative accepts both (it maps 'ctrl' to -1) but
    # the int is the documented form, and it is what XcelltypePerturbDatasetFineTune uses.
    conditions = pd.Series([[-1]] * controls.n_obs, index=controls.obs.index)

    dataset = XcelltypePerturbDataset(
        controls,
        controls,               # basal input comes from the controls themselves, batch-matched
        conditions,
        controls.obs[cs.CONDITION_NAME],
        base_dm.device,
        train_cond_names=None,  # do NOT re-inject controls as extra rows; they are the rows here
        match_cntr=base_dm.match_cntr,
        avg_cntr=base_dm.avg_cntr,
        smiles2vec=None,
        unique_doses=base_dm.unique_doses,
        obsm_key=base_dm.obsm_key,
    )
    loader = DataLoader(dataset, batch_size=base_dm.batch_size, collate_fn=collate_fn)

    res = evaluate(loader, model, device, adata, base_dm.id2pert)
    names = {str(p) for p in res[cs.RES_PERT_CAT]}
    if names != {cs.CONTROL_LABEL}:
        # If this fires, `evaluate` resolved the control rows to something other than 'ctrl' and
        # the "predicted controls" would not be controls at all.
        raise ValueError(f"control pass resolved to perturbations {sorted(names)[:5]}, not 'ctrl'")

    pred = np.asarray(res[cs.RES_PRED], dtype=np.float32)
    real = np.asarray(res[cs.RES_TRUTH], dtype=np.float32)
    logger.info(f"predicted {len(pred):,} control cells with pert id -1")
    return pred, real, np.array([str(c) for c in res[cs.RES_CELL_TYPE]])


def export_artifacts(
    model,
    datamodule,
    device,
    artifact_dir: str | Path,
    split: str = "test",
) -> Path:
    """Write ``adata_pred.h5ad`` and ``adata_real.h5ad`` into ``artifact_dir``.

    Pred holds model outputs for both the perturbed rows and the control rows; real holds the
    measured values for both. The two are row-aligned and share an obs index, which the shared
    scorer requires in order to swap the real controls in for the inject-real arm.
    """
    import anndata as ad

    from gspp.celleval_eval import CLIP_MAX, CLIP_MIN, CONTROL_OUT, clip_anndata_values

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base_dm = datamodule.base_dm
    adata = base_dm.adata

    res = evaluate(_split_dataloader(datamodule, split), model, device, adata, base_dm.id2pert)
    pert_pred = np.asarray(res[cs.RES_PRED], dtype=np.float32)
    pert_real = np.asarray(res[cs.RES_TRUTH], dtype=np.float32)
    perts = np.array([str(p).replace("+ctrl", "") for p in res[cs.RES_PERT_CAT]])
    ptypes = np.array([str(c) for c in res[cs.RES_CELL_TYPE]])
    logger.info(
        f"{split}: {pert_pred.shape[0]:,} cells, {len(pd.unique(perts)):,} perturbations"
    )

    ctrl_pred, ctrl_real, ctypes = _predict_controls(model, datamodule, device)

    n_genes = pert_pred.shape[1]
    genes = [str(g) for g in adata.var_names[:n_genes]]
    if len(genes) != n_genes:
        raise ValueError(f"gene count {len(genes)} != prediction width {n_genes}")
    for name, block in (("control pred", ctrl_pred), ("control real", ctrl_real)):
        if block.shape[1] != n_genes:
            raise ValueError(f"{name} has {block.shape[1]} genes, expected {n_genes}")

    var = pd.DataFrame(index=pd.Index(genes))
    obs = pd.DataFrame(
        {
            PERT_COL: np.concatenate(
                [perts, np.full(len(ctrl_pred), CONTROL_OUT, dtype=object)]
            ),
            CELL_TYPE_COL: np.concatenate([ptypes, ctypes]),
        }
    )
    # A shared, explicit index. The scorer matches control rows between pred and real by obs name,
    # so a default RangeIndex written twice would work only by accident of ordering.
    obs.index = pd.Index(
        [f"{split}_{i}" for i in range(len(perts))]
        + [f"ctrl_{i}" for i in range(len(ctrl_pred))]
    )

    paths = {}
    for name, pert_block, ctrl_block in (
        (PRED_NAME, pert_pred, ctrl_pred),
        (REAL_NAME, pert_real, ctrl_real),
    ):
        out = ad.AnnData(
            X=np.vstack([pert_block, ctrl_block]).astype(np.float32),
            obs=obs.copy(),
            var=var.copy(),
        )
        # Clip exactly as run_cell_eval does to BOTH objects (celleval_eval.py:115-116), and with
        # its function rather than a second np.clip, so the artifact path and the in-process path
        # cannot drift on preprocessing.
        #
        # This is load-bearing, not hygiene: the model emits small negative values (observed min
        # -0.16) and cell-eval rejects them outright -- "Invalid scale: min value -0.16 is
        # negative". Writing the artifacts unclipped made the scorer fail before computing a single
        # metric, and had it not failed, the numbers would not have been comparable to the
        # already-recorded ones, which were all produced through the clip.
        clip_anndata_values(out)
        path = artifact_dir / name
        out.write_h5ad(path)
        paths[name] = path
        logger.info(
            f"wrote {path} {out.shape} "
            f"clipped to [{CLIP_MIN}, {CLIP_MAX}], range "
            f"[{float(np.min(out.X)):.2f}, {float(np.max(out.X)):.2f}]"
        )

    logger.info(
        f"artifacts ready in {artifact_dir}: {len(perts):,} perturbed + {len(ctrl_pred):,} "
        f"control rows. Pred controls are MODEL OUTPUTS (the 'decoded' convention); the scorer "
        f"swaps in the real ones for the 'real' convention."
    )
    return artifact_dir
