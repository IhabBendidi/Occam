"""Score the VALIDATION split with cell-eval during training, so checkpoints can be selected on it.

Why this exists. STATE's only training-time validation metric is ``val_loss``
(``state_transition.py``), and on a split whose ``validation`` list is empty ``val_dataloader``
used to fall back to the TEST loader -- so ``ModelCheckpoint(monitor="val_loss")`` has been
selecting its "best" checkpoint on test for every run on the original replogle split. This
publishes a real validation cell-eval score so selection can use one.

Contract shared with the other model families, so the arms stay comparable:

  * publishes ``val/celleval/selection_score`` = mean(pearson_delta, discrimination_score_l1),
    higher is better, plus every raw metric.
  * published via ``pl_module.log(...)``, which is what puts a value in Lightning's
    ``callback_metrics`` where ``ModelCheckpoint(monitor=...)`` can see it. cell-eval's own CSV
    output is invisible to Lightning. Unlike PerturbDiff's copy this uses ``logger=True`` as well,
    because STATE *does* log to W&B and the reproducibility collector reads run summaries.
  * scores the WHOLE validation split, a handful of times per run -- cost is controlled by
    validating rarely, not by truncating what is validated.

Validation always uses the INJECT-REAL-CONTROLS convention, even though STATE's default at predict
time is reconstruct. Selection must be on one fixed convention or checkpoints from different
validation passes would be ranked against differently-defined baselines; the test leg reports both.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
from lightning.pytorch import Callback

from state.tx.eval.celleval import (
    SELECTION_SCORE_COMPONENTS,
    SELECTION_SCORE_NAME,
    inject_real_controls,
    score_celleval_pair,
    selection_score,
)

logger = logging.getLogger(__name__)

METRIC_PREFIX = "val/celleval"


class ValidationCellEvalCallback(Callback):
    """Predict the validation split, score it with cell-eval, publish the score to Lightning.

    :param profile: cell-eval profile. ``minimal`` still contains pearson_delta and is much cheaper
        than ``full``; the selection score only needs pearson_delta and discrimination_score_l1.
    :param every_n_validations: run on every Nth validation pass. With ``val_check_interval`` set to
        ``max_steps / n_validations`` this should stay at 1, so every checkpoint step is scored.
    :param max_batches: optional cap on validation batches, for smoke tests only. ``None`` scores
        the whole split, which is what a real run must do.
    """

    def __init__(
        self,
        profile: str = "minimal",
        every_n_validations: int = 1,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.every_n_validations = max(1, int(every_n_validations))
        self.max_batches = max_batches
        self._n_calls = 0
        self.last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------ gating

    def _should_run(self, trainer) -> bool:
        """Run every Nth validation pass, plus the final one under either budget.

        STATE trains on a step budget (`max_steps`, `check_val_every_n_epoch=None`), so
        `current_epoch` stays 0 and an epoch-based final test would never fire.
        """
        if trainer.sanity_checking:
            return False
        self._n_calls += 1
        max_steps = int(getattr(trainer, "max_steps", -1) or -1)
        max_epochs = int(getattr(trainer, "max_epochs", -1) or -1)
        if max_steps > 0:
            is_final = int(trainer.global_step) >= max_steps
        elif max_epochs > 0:
            is_final = (int(trainer.current_epoch) + 1) >= max_epochs
        else:
            is_final = False
        return is_final or (self._n_calls % self.every_n_validations == 0)

    # ------------------------------------------------------------------ hook

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._should_run(trainer):
            return
        dm = trainer.datamodule
        if dm is None:
            logger.warning("no datamodule on trainer; skipping validation cell-eval")
            return

        was_training = pl_module.training
        try:
            adata_pred, adata_real = self._predict_validation(dm, pl_module)
            n = inject_real_controls(
                adata_pred, adata_real, dm.pert_col, dm.get_control_pert()
            )
            logger.info("validation cell-eval: injected %d real control rows", n)
            metrics = score_celleval_pair(
                adata_pred,
                adata_real,
                pert_col=dm.pert_col,
                cell_type_key=dm.cell_type_key,
                control_pert=dm.get_control_pert(),
                embed_key=getattr(dm, "embed_key", None),
                profile=self.profile,
                write_csv=False,   # ~120 MB of DE CSVs per pass otherwise
            )
        except Exception:
            # A failed validation score must not kill an otherwise healthy training run. The cost
            # is that ModelCheckpoint sees no new value and keeps its current best.
            logger.exception(
                "validation cell-eval failed at step %d; not publishing a score",
                int(trainer.global_step),
            )
            return
        finally:
            if was_training:
                pl_module.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._publish(pl_module, metrics)

    # ------------------------------------------------------------------ predict

    def _predict_validation(self, dm, pl_module):
        """Run the model over the validation split and build row-aligned pred/real AnnDatas.

        Mirrors the manual loop in ``_cli/_tx/_predict.py``, restricted to the gene-space case.
        """
        import anndata
        import pandas as pd

        embed_key = getattr(dm, "embed_key", None)
        if embed_key and embed_key != "X_hvg":
            # In a latent configuration `preds` live in the latent space and must be decoded before
            # scoring. Rather than half-implement that here, refuse: a silently latent-space
            # "pearson_delta" would look plausible and mean nothing.
            raise NotImplementedError(
                f"validation cell-eval supports gene-space models only (embed_key='X_hvg'); "
                f"got embed_key={embed_key!r}. Decode predictions before scoring."
            )

        loader = dm.val_dataloader_for_scoring()
        device = pl_module.device

        preds, reals, pert_names, celltypes = [], [], [], []
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    break
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                out = pl_module.predict_step(batch, batch_idx, padded=False)

                preds.append(out["preds"].cpu().numpy().astype(np.float32))
                reals.append(out["pert_cell_emb"].cpu().numpy().astype(np.float32))
                for key, sink in (("pert_name", pert_names), ("celltype_name", celltypes)):
                    v = out[key]
                    sink.extend(v) if isinstance(v, list) else sink.append(v)

        if not preds:
            raise RuntimeError("validation loader yielded no batches")

        X_pred = np.concatenate(preds, axis=0)
        X_real = np.concatenate(reals, axis=0)
        if len(pert_names) != X_pred.shape[0]:
            raise RuntimeError(
                f"metadata/prediction length mismatch: {len(pert_names)} names vs "
                f"{X_pred.shape[0]} rows -- the adatas would not be row-aligned."
            )

        obs = pd.DataFrame({dm.pert_col: pert_names, dm.cell_type_key: celltypes})
        gene_names = dm.get_var_dims().get("gene_names")
        var = pd.DataFrame(index=list(gene_names)) if gene_names is not None else None

        adata_pred = anndata.AnnData(X=X_pred, obs=obs, var=var)
        adata_real = anndata.AnnData(X=X_real, obs=obs.copy(), var=var)
        return adata_pred, adata_real

    # ------------------------------------------------------------------ publish

    def _publish(self, pl_module, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            if np.isfinite(value):
                pl_module.log(f"{METRIC_PREFIX}/{name}", float(value), sync_dist=False)

        score = selection_score(metrics)
        if score is None:
            missing = [n for n in SELECTION_SCORE_COMPONENTS
                       if n not in metrics or not np.isfinite(metrics.get(n, float("nan")))]
            logger.warning(
                "%s/%s not published: missing or non-finite %s. ModelCheckpoint cannot select on "
                "it this pass.", METRIC_PREFIX, SELECTION_SCORE_NAME, missing,
            )
            return

        pl_module.log(f"{METRIC_PREFIX}/{SELECTION_SCORE_NAME}", score, sync_dist=False)
        self.last_metrics = dict(metrics)
        logger.info(
            "validation cell-eval: %s=%.5f (%s)",
            SELECTION_SCORE_NAME, score,
            ", ".join(f"{n}={metrics[n]:.5f}" for n in SELECTION_SCORE_COMPONENTS),
        )
