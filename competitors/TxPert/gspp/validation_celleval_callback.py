"""Score the VALIDATION split with cell-eval during training, so checkpoints can be selected on it.

Why this exists. Upstream TxPert had no validation split on this benchmark, so it monitored
`test_pearson_delta` -- it selected the checkpoint on the test set. The vendored arm instead kept
the last epoch (`save_last`), which is honest but leaves the epoch budget unchosen. With
`configs/manifests/ihabxcell_hepg2_fewshot_zarr.toml` the validation split is real (4,825 HepG2
cells over 60 perturbations), and this callback turns it into the same cell-eval metric the other
three model families are selected on.

Contract shared with the upstream benchmark, STATE and PerturbDiff, so the arms stay comparable:

  * metric published: `val/celleval/selection_score` = mean(pearson_delta,
    discrimination_score_l1), higher is better. Both raw components are published too.
  * published via `pl_module.log(...)`, which is what puts a value in Lightning's
    `callback_metrics` where `ModelCheckpoint(monitor=...)` can see it. cell-eval's own CSV output
    is invisible to Lightning, and `wandb.log` does not populate callback_metrics either.
  * runs every `interval` validation passes and always on the final epoch, since a final epoch
    that is never scored cannot be selected.
  * validation always scores with REAL controls injected, which is what `export_for_celleval`
    does. The predicted-control convention is a test-time reporting choice, not a selection one,
    so both test arms are selected by the same checkpoint.

This is a port of `../PerturbDiff/src/apps/training/validation_celleval_callback.py`. The
PerturbDiff shape was chosen over STATE's because TxPert's `run_cell_eval` returns an output
*directory* rather than a metrics dict, so metrics come back by reading the CSVs it wrote -- hence
`read_agg_metrics`, which is the same function, kept in lockstep by hand (each competitor has its
own uv venv and cannot import the others').

No modelling change. The only upstream edit this needed was making the scored split selectable by
name in `export_for_celleval`, which previously hardcoded `test_dataloader()`.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from lightning import Callback
from loguru import logger

# Keep in lockstep with the upstream tree.eval.evaluation.SELECTION_SCORE_COMPONENTS. Changing this changes
# what "best checkpoint" means, and silently makes arms non-comparable.
SELECTION_SCORE_COMPONENTS = ("pearson_delta", "discrimination_score_l1")
SELECTION_SCORE_NAME = "selection_score"
METRIC_PREFIX = "val/celleval"


def read_agg_metrics(eval_dir: Path) -> dict[str, float]:
    """Mean cell-eval metrics from the `<cell_type>_agg_results.csv` files in `eval_dir`.

    cell-eval writes one file per cell type; this benchmark has a single one (`hepg2`, lowercase
    in TxPert's outputs versus `HepG2` in STATE's), but averaging across whatever is present
    matches `merge_celltype_celleval_metrics` on the the upstream benchmark side.
    """
    import pandas as pd

    files = sorted(eval_dir.glob("*_agg_results.csv"))
    if not files:
        raise FileNotFoundError(f"no *_agg_results.csv in {eval_dir}")

    per_file: list[dict[str, float]] = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if "mean" not in df.index:
            raise ValueError(f"{f} has no 'mean' row (index: {list(df.index)})")
        row = df.loc["mean"]
        per_file.append({str(k): float(v) for k, v in row.items() if np.isfinite(float(v))})

    keys = set().union(*(set(d) for d in per_file))
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [d[k] for d in per_file if k in d]
        if vals:
            out[k] = float(np.mean(vals))
    return out


class ValidationCellEvalCallback(Callback):
    """Score the validation split with cell-eval and publish the result to Lightning.

    :param output_root: where per-validation cell-eval output goes. Each pass gets its own
        ``val_celleval_ep<E>/`` subdirectory so consecutive passes cannot read each other's CSVs
        and the run's test-time outputs are never overwritten.
    :param interval: run every N validation passes. TxPert's `check_val_every_n_epoch` already
        controls the cadence, so leave this at 1.
    :param profile: cell-eval profile. `full` matches what the test leg reports.
    """

    def __init__(self, output_root: str | Path, interval: int = 1, profile: str = "full") -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.interval = max(1, int(interval))
        self.profile = profile
        self._last: dict[str, float] = {}
        self._n_val_calls = 0

    def _should_run(self, trainer) -> bool:
        """Run every `interval`-th validation pass, and always on the final epoch.

        TxPert trains on an epoch budget, so the final-epoch test is the meaningful one: with
        `epochs=50` and `check_val_every_n_epoch=10` the last validation lands exactly on epoch 50
        and this is a no-op, but it keeps a budget that is not a multiple of the cadence from
        leaving the final weights unscored and therefore unselectable.
        """
        if trainer.sanity_checking:
            return False
        self._n_val_calls += 1

        max_epochs = int(getattr(trainer, "max_epochs", -1) or -1)
        is_final = max_epochs > 0 and (int(trainer.current_epoch) + 1) >= max_epochs
        return is_final or (self._n_val_calls % self.interval == 0)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._should_run(trainer):
            return
        if trainer.datamodule is None:
            logger.warning("no datamodule on trainer; skipping validation cell-eval")
            return

        from gspp.evaluation import export_for_celleval

        epoch = int(trainer.current_epoch)
        eval_dir = self.output_root / f"val_celleval_ep{epoch + 1}"
        was_training = pl_module.training
        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                export_for_celleval(
                    pl_module,
                    trainer.datamodule,
                    pl_module.device,
                    str(eval_dir),
                    profile=self.profile,
                    split="validation",
                )
            metrics = read_agg_metrics(eval_dir)
        except Exception:
            # A failed validation score must not kill a training run that is otherwise fine. The
            # cost is that ModelCheckpoint sees no new value this pass and keeps its current best.
            logger.exception(
                f"validation cell-eval failed at epoch {epoch + 1}; not publishing a score"
            )
            return
        finally:
            if was_training:
                pl_module.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._publish(pl_module, metrics)

    def _publish(self, pl_module, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            if np.isfinite(value):
                pl_module.log(f"{METRIC_PREFIX}/{name}", float(value), sync_dist=False)

        components = [metrics.get(n) for n in SELECTION_SCORE_COMPONENTS]
        if any(c is None or not np.isfinite(c) for c in components):
            missing = [
                n for n, c in zip(SELECTION_SCORE_COMPONENTS, components)
                if c is None or not np.isfinite(c)
            ]
            # Publishing a partial composite would let a half-computed score win selection, which
            # is worse than skipping the pass: ModelCheckpoint simply keeps its current best.
            logger.warning(
                f"{METRIC_PREFIX}/{SELECTION_SCORE_NAME} not published: missing or non-finite "
                f"{missing}. ModelCheckpoint cannot select on it this pass."
            )
            return

        score = float(np.mean(components))
        pl_module.log(f"{METRIC_PREFIX}/{SELECTION_SCORE_NAME}", score, sync_dist=False)
        self._last = dict(metrics)
        logger.info(
            f"validation cell-eval: {SELECTION_SCORE_NAME}={score:.5f} ("
            + ", ".join(f"{n}={metrics[n]:.5f}" for n in SELECTION_SCORE_COMPONENTS)
            + ")"
        )
