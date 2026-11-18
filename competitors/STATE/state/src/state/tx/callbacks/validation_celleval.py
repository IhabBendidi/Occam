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
from pathlib import Path

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
            # ⚠️ outdir MUST be per-run. `write_csv=False` suppresses our own results
            # CSVs but NOT pdex's DE dumps, and `MetricsEvaluator`'s default is the
            # RELATIVE "./cell-eval-outdir" -- i.e. the launcher's cwd, which is the
            # checked-out repo. That put 133 MB of DE csv into `competitors/` and, worse,
            # would have two concurrent seeds writing the same filenames.
            outdir = Path(
                trainer.log_dir or getattr(trainer, "default_root_dir", None) or "."
            ) / "celleval_scratch"
            outdir.mkdir(parents=True, exist_ok=True)
            metrics = score_celleval_pair(
                adata_pred,
                adata_real,
                pert_col=dm.pert_col,
                cell_type_key=dm.cell_type_key,
                control_pert=dm.get_control_pert(),
                embed_key=getattr(dm, "embed_key", None),
                outdir=outdir,
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

        # ⛔ COMBINATORIAL BENCHMARK (norman_combo): pairs are predicted by sequential
        # composition, exactly as the export does, never through the untrained "A+B" one-hot.
        # On a single-perturbation benchmark this branch is not taken and the loop below runs
        # byte for byte as before.
        if self._has_pairs(dm):
            return self._predict_validation_composed(dm, pl_module)

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

    # ------------------------------------------------------------------ pairs

    @staticmethod
    def _has_pairs(dm) -> bool:
        """True when the validation split carries a combination label (`"A+B"`)."""
        from state.tx.pure_sentence import is_multi

        ds = (getattr(dm, "val_datasets", None) or [None])[0]
        if ds is None:
            return False
        control = str(dm.get_control_pert())
        return any(is_multi(p) for p in set(map(str, ds.pert_str)) if p != control)

    def _predict_validation_composed(self, dm, pl_module):
        """Validation predictions made the way `export_occam.py --compose decode_seq_mean` makes them.

        The export answers a pair `"A+B"` with `compose_sequential` (order `mean`): f(ctrl, A)
        fed back as the control of f(., B), the same with B first, and the two chains
        averaged -- every pass through the TRAINED single-gene one-hots. Validation used to
        send the pair through the one-hot of the literal `"A+B"`, a column that never trains,
        so `selection_score` ranked checkpoints on an untrained input (audit 2026-09-23).

        Same code as the export, not a copy of it: `state.tx.pure_sentence` holds both the
        pure-sentence predictor and the composition wrapper (which imports
        `competitors/shared/export_predictions.py::compose_sequential`).

        What matches the export:
          * pair rows are ordered by label with a stable sort, like the panel's
            `np.argsort(labels, kind="stable")`, and the WHOLE pair set goes through the
            composition in one call, so the chained passes group rows into pure sentences
            of the first (then second) gene across pairs exactly as the export does;
          * one pure sentence per `predict_step`, true length, `cell_sentence_len` cap;
          * per-cell batch covariate = the perturbed cell's own batch (the export's
            `split.batches` is the perturbed cell's batch too).
        What is necessarily different: the control input is the validation loader's drawn
        control (the export's is the frozen panel's drawn half-B control) -- data, not model.

        Single-perturbation rows in the same split (controls; singles on a mixed split) are
        predicted by `predict_pure_sentences` directly -- what `compose_sequential` does with
        them too (its pass-through call).
        """
        import anndata
        import pandas as pd

        from state.tx.pure_sentence import (is_multi, make_composed_pair_predict,
                                            predict_pure_sentences)

        loader = dm.val_dataloader_for_scoring()
        ctrl, real, names, ct_names, ct_oh, b_oh, b_names = [], [], [], [], [], [], []
        for batch_idx, batch in enumerate(loader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            ctrl.append(batch["ctrl_cell_emb"].reshape(-1, batch["ctrl_cell_emb"].shape[-1]))
            real.append(batch["pert_cell_emb"].reshape(-1, batch["pert_cell_emb"].shape[-1]))
            names.extend(map(str, batch["pert_name"]))
            ct_names.extend(map(str, batch["cell_type"]))
            ct_oh.append(batch["cell_type_onehot"])
            b_oh.append(batch["batch"])
            b_names.extend(map(str, batch["batch_name"]))
        if not ctrl:
            raise RuntimeError("validation loader yielded no batches")

        X_ctrl = torch.cat(ctrl).float().cpu().numpy().astype(np.float32)
        X_real = torch.cat(real).float().cpu().numpy().astype(np.float32)
        ct_oh_t = torch.cat(ct_oh).cpu()
        b_oh_t = torch.cat(b_oh).cpu()
        names_a = np.asarray(names, dtype=object)
        n = len(names_a)
        if not (X_ctrl.shape[0] == X_real.shape[0] == n == len(ct_names) == len(b_names)
                == len(ct_oh_t) == len(b_oh_t)):
            raise RuntimeError("validation batch fields are not row-aligned")

        # the export's row order: stable sort by label
        order = np.argsort(names_a.astype(str), kind="stable")
        X_ctrl, X_real, names_a = X_ctrl[order], X_real[order], names_a[order]
        ct_names = np.asarray(ct_names, dtype=object)[order]
        b_names = np.asarray(b_names, dtype=object)[order]
        ct_oh_t, b_oh_t = ct_oh_t[order], b_oh_t[order]

        S = int(getattr(dm, "cell_sentence_len", None)
                or getattr(pl_module, "cell_sentence_len"))
        device = pl_module.device
        pl_module.eval()

        def _single(rows):
            def _p(c, labels):
                return predict_pure_sentences(
                    pl_module, c, labels,
                    pert_onehot_map=dm.pert_onehot_map,
                    batch_onehot=b_oh_t[rows], batch_names=b_names[rows],
                    cell_type_onehot=ct_oh_t[rows], cell_type_names=ct_names[rows],
                    sentence_len=S, device=device, dataset_name="occam_val")
            return _p

        multi = np.array([is_multi(p) for p in names_a], dtype=bool)
        X_pred = np.full_like(X_ctrl, np.nan)
        pair_rows = np.flatnonzero(multi)
        one_rows = np.flatnonzero(~multi)
        if len(one_rows):
            X_pred[one_rows] = _single(one_rows)(X_ctrl[one_rows], list(names_a[one_rows]))
        if len(pair_rows):
            composed = make_composed_pair_predict(_single(pair_rows), n_rows=len(pair_rows),
                                                  order="mean")
            X_pred[pair_rows] = composed(X_ctrl[pair_rows], list(names_a[pair_rows]))
        if np.isnan(X_pred).any():
            raise RuntimeError("validation prediction left NaN rows")
        logger.info(
            "validation cell-eval: %d pair rows (%d pairs) by sequential composition "
            "decode_seq_mean, %d single/control rows by pure sentences",
            len(pair_rows), len(set(names_a[pair_rows])), len(one_rows))

        obs = pd.DataFrame({dm.pert_col: list(names_a), dm.cell_type_key: list(ct_names)})
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
