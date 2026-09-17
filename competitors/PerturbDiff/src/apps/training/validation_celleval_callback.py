"""Score the VALIDATION split with cell-eval during training, so checkpoints can be selected on it.

Why this exists. On the hooke-tx zarr benchmark PerturbDiff's only live validation metric was
`validation_test_loss_epoch` -- a loss, computed on the TEST split, because the benchmark's original
split file ships an empty `validation` list. Selecting a checkpoint on that is selecting on test.
With `configs/manifests/ihabxcell_hepg2_fewshot_zarr.toml` the validation split is real (4,825
perturbed cells), and this callback turns it into the same cell-eval metric the other model families
are selected on.

Contract shared with hooke-tx, so the arms stay comparable:

  * metric published: `validation/celleval/selection_score` = mean(pearson_delta,
    discrimination_score_l1), higher is better. Both raw components are published too.
  * published via `pl_module.log(..., logger=False)`, which is what puts a value in Lightning's
    `callback_metrics` where `ModelCheckpoint(monitor=...)` can see it. cell-eval's own CSV output
    is invisible to Lightning.
  * runs every `interval` validation epochs and always on the final epoch, since a final epoch that
    is never scored cannot be selected.

This adds no modelling change. It reuses `generate_samples`, which already samples and runs
cell-eval itself; the only thing that changed there was making the split selectable by name instead
of a hardcoded `val_dataloader()[1]`. PerturbDiff stays in gene space -- there is no PCA here, and
none is introduced.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback

logger = logging.getLogger(__name__)

# Keep in lockstep with hooke_tx.eval.evaluation.SELECTION_SCORE_COMPONENTS. Changing this changes
# what "best checkpoint" means, and silently makes arms non-comparable.
SELECTION_SCORE_COMPONENTS = ("pearson_delta", "discrimination_score_l1")
SELECTION_SCORE_NAME = "selection_score"
METRIC_PREFIX = "validation/celleval"


def read_agg_metrics(eval_dir: Path) -> dict[str, float]:
    """Mean cell-eval metrics from the `<cell_type>_agg_results.csv` files in `eval_dir`.

    cell-eval writes one file per cell type; the benchmark has a single one (HepG2), but averaging
    across whatever is present matches `merge_celltype_celleval_metrics` on the hooke-tx side.
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
    """Sample the validation split, score it with cell-eval, publish the score to Lightning.

    :param cfg: the Hydra config the run was built from; `generate_samples` needs it.
    :param interval: run every N validation passes. With `val_check_interval` aligned to the
        checkpoint step cadence, leave this at 1 so every checkpoint step is scored.
    :param num_sampled_batches: cap on validation batches sampled. ``None`` means the whole split.
        A cap makes the metric cheaper but noisier -- it is a *ranking* signal, not a reported
        number, so a cap is acceptable; just keep it fixed across seeds and arms.
    """

    def __init__(self, cfg, interval: int = 5, num_sampled_batches: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.interval = max(1, int(interval))
        self.num_sampled_batches = num_sampled_batches
        self._last: dict[str, float] = {}
        self._n_val_calls = 0

    def _should_run(self, trainer) -> bool:
        """Run every `interval`-th validation pass, and on the last one we can detect.

        PerturbDiff trains on a STEP budget (`trainer.max_steps`), so `trainer.max_epochs` is
        usually -1 here and an epoch-based "final epoch" test would never fire. Counting validation
        passes works under either budget; the final-pass test checks whichever bound is actually set.

        Note the reliable way to keep the monitored metric fresh for ModelCheckpoint is to align the
        cadences rather than rely on this: set `val_check_interval` to the checkpoint step cadence
        and leave `interval=1`, so every checkpoint step is also a scored step.
        """
        if trainer.sanity_checking:
            return False
        self._n_val_calls += 1

        max_steps = int(getattr(trainer, "max_steps", -1) or -1)
        max_epochs = int(getattr(trainer, "max_epochs", -1) or -1)
        is_final = False
        if max_steps > 0:
            is_final = int(trainer.global_step) >= max_steps
        elif max_epochs > 0:
            is_final = (int(trainer.current_epoch) + 1) >= max_epochs

        return is_final or (self._n_val_calls % self.interval == 0)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._should_run(trainer):
            return
        if trainer.datamodule is None:
            logger.warning("no datamodule on trainer; skipping validation cell-eval")
            return

        from src.apps.sampling.sampling_generation import generate_samples

        # Route sampling output somewhere per-epoch so concurrent/consecutive evals cannot read
        # each other's CSVs, and so the run's own test-time outputs are never overwritten.
        epoch = int(trainer.current_epoch)
        base = self.cfg.sampling.output_dir or str(Path(self.cfg.save_dir_path) / "samples")
        eval_root = Path(base) / f"val_celleval_ep{epoch + 1}_step{int(trainer.global_step)}"

        prev = {
            "output_dir": self.cfg.sampling.output_dir,
            "run_cell_eval": getattr(self.cfg.sampling, "run_cell_eval", False),
            "num_sampled_batches": getattr(self.cfg.sampling, "num_sampled_batches", None),
            "progress": getattr(self.cfg.sampling, "progress", False),
        }
        was_training = pl_module.training
        try:
            self.cfg.sampling.output_dir = str(eval_root)
            self.cfg.sampling.run_cell_eval = True
            self.cfg.sampling.num_sampled_batches = self.num_sampled_batches
            self.cfg.sampling.progress = False

            with torch.no_grad():
                generate_samples(
                    pl_module,
                    pl_module.diffusion,
                    self.cfg,
                    pl_module.device,
                    logger,
                    trainer.datamodule,
                    pca_for_decode=None,   # PerturbDiff works in gene space; no codec here
                    split="validation",
                )

            # generate_samples timestamps its own subdirectory, so find the newest one it made.
            made = sorted(
                (p for p in eval_root.glob("cell_eval_*") if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
            )
            if not made:
                raise FileNotFoundError(f"generate_samples wrote no cell_eval_* dir under {eval_root}")
            metrics = read_agg_metrics(made[-1])
        except Exception:
            # A failed validation score must not kill a training run that is otherwise fine. The
            # cost is that ModelCheckpoint sees no new value this epoch and keeps its current best.
            logger.exception("validation cell-eval failed at epoch %d; not publishing a score", epoch + 1)
            return
        finally:
            for k, v in prev.items():
                setattr(self.cfg.sampling, k, v)
            if was_training:
                pl_module.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._publish(pl_module, metrics)

    def _publish(self, pl_module, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            if np.isfinite(value):
                pl_module.log(f"{METRIC_PREFIX}/{name}", float(value), logger=False, sync_dist=False)

        components = [metrics.get(n) for n in SELECTION_SCORE_COMPONENTS]
        if any(c is None or not np.isfinite(c) for c in components):
            missing = [n for n, c in zip(SELECTION_SCORE_COMPONENTS, components)
                       if c is None or not np.isfinite(c)]
            logger.warning(
                "%s/%s not published: missing or non-finite %s. ModelCheckpoint cannot select "
                "on it this epoch.", METRIC_PREFIX, SELECTION_SCORE_NAME, missing,
            )
            return

        score = float(np.mean(components))
        pl_module.log(f"{METRIC_PREFIX}/{SELECTION_SCORE_NAME}", score, logger=False, sync_dist=False)
        self._last = dict(metrics)
        logger.info(
            "validation cell-eval: %s=%.5f (%s)",
            SELECTION_SCORE_NAME, score,
            ", ".join(f"{n}={metrics[n]:.5f}" for n in SELECTION_SCORE_COMPONENTS),
        )
