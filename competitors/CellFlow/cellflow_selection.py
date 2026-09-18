"""Score CellFlow's validation split with cell-eval during training, and keep the best state.

WHY THIS EXISTS. CellFlow already validates periodically
(`cellflow/src/cellflow/training/_trainer.py:127-147`) and already has a callback system whose
computation hooks receive the live solver (`.../training/_callbacks.py:102-109`). What it has no
notion of is a *best* state: `train()` returns whatever the last iteration produced, and the
vendored wrapper passed no callbacks at all. So the arm had no way to choose a checkpoint without
looking at test data — the one thing the protocol every other arm on this benchmark follows is
built to avoid.

WHY IT SUBPROCESSES INSTEAD OF SCORING IN-PROCESS. The selection metric has to be the same
quantity as the reported metric, computed by the same code. CellFlow's own `_run_celleval`
(`predict_cellflow.py:150-157`) passes no `pdex_kwargs`, where the shared scorer always passes
`{"exp_post_agg": True, "is_log1p": True}`; and this venv resolves pdex 0.3.0 against 0.2.2
elsewhere. Rather than reproduce those differences, every score — validation and test — is
produced by `the upstream tree/eval/celleval_artifacts.py` running under the upstream benchmark's own interpreter. The
same two-interpreter arrangement the STATE and TxPert eval legs use, and for the same reason.

WHY CHECKPOINTING COSTS NOTHING EXTRA. `CellFlow.save()` cloudpickles the whole instance
including the ~5 GB dense train AnnData, so saving at five validation points would be ~25 GB per
run. But inference reads only `solver.vf_state_inference.params`
(`cellflow/src/cellflow/solvers/_otfm.py:377`) — a pytree of tens of MB. This keeps the best
params in host memory and restores them in `on_train_end`, so the caller's single `cf.save()`
persists the selected state and `predict_cellflow.py` needs no changes at all.

Contract shared with the other four arms:

  * `selection_score = mean(pearson_delta, discrimination_score_l1)`, higher is better.
  * Validation scores with REAL controls injected. The predicted-control convention is a
    test-time reporting choice, so both test arms come from this one selected state.
  * The final state is always a candidate: the trainer's `on_train_end` pass fires
    unconditionally, so an early-peaking run is selectable and so is a still-improving one.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

# Keep in lockstep with the upstream tree.eval.evaluation.SELECTION_SCORE_COMPONENTS. Changing this changes
# what "best" means and silently makes the arms non-comparable.
SELECTION_SCORE_COMPONENTS = ("pearson_delta", "discrimination_score_l1")
SELECTION_SCORE_NAME = "selection_score"
METRIC_PREFIX = "valid_celleval"

# cell-eval rejects negative expression outright ("Invalid scale: min value ... is negative"), and
# the test path clips to this range, so selection must use the same bounds or it would be ranking
# differently-preprocessed data.
CLIP_MIN, CLIP_MAX = 0.0, 14.0

PERT_COL = "target_gene"
CELL_TYPE_COL = "cell_type"
CONTROL_PERT = "empty"

# ⚠️ SELECTION SCORES THROUGH *OUR* CELL-EVAL.
#
# It used to shell out to a sibling repo's `celleval_artifacts.py` in that repo's own venv,
# which pinned a DIFFERENT cell-eval (0.7.2 / pdex 0.2.2-0.3.0) than the 0.6.0 / pdex 0.1.24
# every reported number comes from. Two consequences, both gone: the arm depended on another
# repo existing at a hard-coded path, and its in-training selection metric was computed by a
# different scorer than its reported one.
#
# Still a subprocess: jax and cell-eval do not co-exist in one environment, which is the same
# constraint that makes the prediction bridge two-phase.
OCCAM_REPO = Path("/rxrx/scratch/sandbox/ihab.bendidi/tx_pretrained_models/Occam")
OCCAM_PY = Path("/mnt/ps/home/CORP/ihab.bendidi/.pyenv/versions/scmae/bin/python")  # noqa: legacy-name


def read_agg_metrics(results_dir: Path) -> dict[str, float]:
    """Mean cell-eval metrics from the ``*_agg_results.csv`` files in ``results_dir``.

    Duplicated from the PerturbDiff and TxPert callbacks by necessity — each competitor runs in
    its own uv venv and cannot import the others' code — so the four copies must be kept in
    lockstep by hand. They all read the same file format, so a divergence would show up as a
    mismatch between an arm's validation and test numbers.
    """
    import pandas as pd

    files = sorted(results_dir.glob("*_agg_results.csv"))
    if not files:
        raise FileNotFoundError(f"no *_agg_results.csv in {results_dir}")

    per_file: list[dict[str, float]] = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if "mean" not in df.index:
            raise ValueError(f"{f} has no 'mean' row (index: {list(df.index)})")
        per_file.append(
            {str(k): float(v) for k, v in df.loc["mean"].items() if np.isfinite(float(v))}
        )

    keys = set().union(*(set(d) for d in per_file))
    return {
        k: float(np.mean([d[k] for d in per_file if k in d]))
        for k in sorted(keys)
        if any(k in d for d in per_file)
    }


def _condition_label(key: Any) -> str:
    """One condition key from the validation sampler as a cell-eval perturbation label."""
    if isinstance(key, (tuple, list)):
        return "+".join(str(k) for k in key)
    return str(key)


class SelectionCallback:
    """Score validation with cell-eval, track the best params, restore them at the end.

    Subclasses ``ComputationCallback`` at construction time rather than by inheritance, so this
    module imports without pulling in jax at collection time; ``as_callback()`` returns the real
    thing.

    :param pcs: PCA components as ``varm["PCs"]``, ``(n_genes, n_comps)`` — ``pca_pcs.npy``.
    :param means: gene means, ``(n_genes,)`` — ``pca_means.npy``.
    :param genes: gene names in the order the PCA basis expects.
    :param control_x: real control cells in GENE space, injected into both objects so DE on each
        side is measured against the same baseline. The validation split has no controls of its
        own on this benchmark (all 4,976 are in finetune), so these come from the training
        controls the wrapper already injects.
    :param cell_type: the value written into the cell-type column; the shared scorer requires the
        column in both objects and slices per cell type.
    :param output_root: per-validation artifacts and results land under here.
    :param occam_py: interpreter that owns OUR cell-eval 0.6.0.
    :param occam_repo: repo root, put on ``PYTHONPATH`` so `occam.vcell.panel` imports.
    """

    def __init__(
        self,
        *,
        pcs: np.ndarray,
        means: np.ndarray,
        genes: list[str],
        control_x: np.ndarray,
        cell_type: str,
        output_root: Path | str,
        occam_py: Path | str = OCCAM_PY,
        occam_repo: Path | str = OCCAM_REPO,
        profile: str = "full",
        num_threads: int | None = None,
        keep_artifacts: bool = False,
    ) -> None:
        self.pcs = np.asarray(pcs)
        self.means = np.asarray(means).reshape(-1)
        self.genes = [str(g) for g in genes]
        self.control_x = np.asarray(control_x, dtype=np.float32)
        self.cell_type = str(cell_type)
        self.output_root = Path(output_root)
        self.occam_py = Path(occam_py)
        self.occam_repo = Path(occam_repo)
        if not self.occam_py.exists():
            raise ValueError(
                f"no interpreter at {self.occam_py}; selection scores through OUR cell-eval")
        self.profile = profile
        self.num_threads = num_threads or len(os_sched_affinity())
        self.keep_artifacts = keep_artifacts

        # `pcs` is varm["PCs"], i.e. (n_genes, n_comps) -- the orientation
        # occam_bridge.reconstruct_from_pca expects.
        if self.pcs.shape[0] != len(self.genes):
            raise ValueError(
                f"PCA basis covers {self.pcs.shape[0]} genes but {len(self.genes)} gene names "
                "were given"
            )
        if self.control_x.size == 0:
            raise ValueError(
                "no control cells for validation scoring; cell-eval cannot run without them"
            )

        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self._best_params = None
        self._n_calls = 0

    # -- the two things the trainer's CallbackRunner needs -------------------------------------

    def on_train_begin(self) -> Any:
        self.output_root.mkdir(parents=True, exist_ok=True)
        return None

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        return self._score_and_maybe_keep(valid_true_data, valid_pred_data, solver, final=False)

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        metrics = self._score_and_maybe_keep(valid_true_data, valid_pred_data, solver, final=True)
        self._restore_best(solver)
        return metrics

    # -- internals ----------------------------------------------------------------------------

    def _decode(self, x_pca: np.ndarray) -> np.ndarray:
        """PCA latent -> gene space, through the SAME function the test path uses.

        Calling `reconstruct_from_pca` rather than reimplementing `x @ pcs.T + means` is
        deliberate: selection and reporting must not be able to drift on preprocessing, and it
        also removes any question about the basis orientation.
        """
        from occam_bridge import reconstruct_from_pca

        return reconstruct_from_pca(
            x_pca, means=self.means, pcs=self.pcs, clip_min=CLIP_MIN, clip_max=CLIP_MAX
        )

    def _score_and_maybe_keep(self, valid_true_data, valid_pred_data, solver, *, final: bool):
        self._n_calls += 1
        tag = "final" if final else f"call{self._n_calls:02d}"
        work = self.output_root / f"valid_{tag}"
        try:
            metrics = self._score(valid_true_data, valid_pred_data, work)
        except Exception as exc:  # noqa: BLE001
            # A failed validation score must not kill an otherwise healthy run; the cost is that
            # this state simply is not a selection candidate.
            print(f"[selection] scoring failed at {tag}: {type(exc).__name__}: {exc}", flush=True)
            return {}

        components = [metrics.get(n) for n in SELECTION_SCORE_COMPONENTS]
        if any(c is None or not np.isfinite(c) for c in components):
            missing = [
                n for n, c in zip(SELECTION_SCORE_COMPONENTS, components, strict=True)
                if c is None or not np.isfinite(c)
            ]
            # Publishing a partial composite would let a half-computed score win selection, which
            # is worse than skipping this state.
            print(f"[selection] {tag}: no {SELECTION_SCORE_NAME}, missing {missing}", flush=True)
            return {}

        score = float(np.mean(components))
        entry = {"tag": tag, "final": final, SELECTION_SCORE_NAME: score, **metrics}
        self.history.append(entry)

        improved = self.best is None or score > self.best[SELECTION_SCORE_NAME]
        if improved:
            self.best = entry
            self._best_params = self._snapshot(solver)
        print(
            f"[selection] {tag}: {SELECTION_SCORE_NAME}={score:.5f} ("
            + ", ".join(f"{n}={metrics[n]:.5f}" for n in SELECTION_SCORE_COMPONENTS)
            + (")  <== new best" if improved else ")"),
            flush=True,
        )
        return {f"{METRIC_PREFIX}_{k}": v for k, v in metrics.items()} | {
            f"{METRIC_PREFIX}_{SELECTION_SCORE_NAME}": score
        }

    def _score(self, valid_true_data, valid_pred_data, work: Path) -> dict[str, float]:
        import anndata as ad
        import pandas as pd

        work.mkdir(parents=True, exist_ok=True)
        pred_blocks, pred_labels, real_blocks, real_labels = [], [], [], []
        for val_key, preds in valid_pred_data.items():
            truths = valid_true_data[val_key]
            for cond, pred in preds.items():
                label = _condition_label(cond)
                n_comps = self.pcs.shape[1]
                p = self._decode(np.asarray(pred).reshape(-1, n_comps))
                t = self._decode(np.asarray(truths[cond]).reshape(-1, n_comps))
                pred_blocks.append(p)
                pred_labels += [label] * len(p)
                real_blocks.append(t)
                real_labels += [label] * len(t)

        if not pred_blocks:
            raise ValueError("validation produced no conditions")

        var = pd.DataFrame(index=pd.Index(self.genes))
        n_ctrl = len(self.control_x)

        def build(blocks, labels):
            x = np.vstack(blocks + [self.control_x]).astype(np.float32)
            obs = pd.DataFrame(
                {
                    PERT_COL: list(labels) + [CONTROL_PERT] * n_ctrl,
                    CELL_TYPE_COL: [self.cell_type] * (len(labels) + n_ctrl),
                }
            )
            # The scorer matches control rows between the two objects by obs name, so the control
            # block must carry the same index in both.
            obs.index = pd.Index(
                [f"cell_{i}" for i in range(len(labels))] + [f"ctrl_{i}" for i in range(n_ctrl)]
            )
            return ad.AnnData(X=x, obs=obs, var=var.copy())

        build(pred_blocks, pred_labels).write_h5ad(work / "adata_pred.h5ad")
        build(real_blocks, real_labels).write_h5ad(work / "adata_real.h5ad")

        results = work / "results"
        metrics = self._score_with_occam_celleval(work, results)
        (work / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        if not self.keep_artifacts:
            # The h5ads are the bulk and are reproducible from the checkpoint; the metrics JSON
            # and the agg CSVs are the record.
            for name in ("adata_pred.h5ad", "adata_real.h5ad"):
                (work / name).unlink(missing_ok=True)
        return metrics

    def _score_with_occam_celleval(self, work: Path, results: Path) -> dict[str, float]:
        """Score `adata_pred.h5ad` against `adata_real.h5ad` through OUR cell-eval 0.6.0.

        A subprocess because jax and cell-eval do not co-exist in one environment -- the same
        constraint that makes the prediction bridge two-phase. The child does nothing but
        call `run_celleval_v1`, the single scorer every reported number comes from.
        """
        import os

        results.mkdir(parents=True, exist_ok=True)
        driver = work / "_score.py"
        driver.write_text(
            "import json, sys\n"
            "from occam.vcell.panel.celleval_v1 import run_celleval_v1\n"
            "m = run_celleval_v1(sys.argv[1], sys.argv[2], sys.argv[3],\n"
            f"                    num_threads={int(self.num_threads)},\n"
            f"                    profile={self.profile!r})\n"
            "json.dump(m, open(sys.argv[4], 'w'), indent=2, default=str)\n"
        )
        out_json = work / "occam_celleval.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.occam_repo / "src")
        env.setdefault("OMP_NUM_THREADS", "1")
        proc = subprocess.run(
            [str(self.occam_py), str(driver),
             str(work / "adata_pred.h5ad"), str(work / "adata_real.h5ad"),
             str(results), str(out_json)],
            env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"our cell-eval failed on the validation split: exit {proc.returncode}\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        if not out_json.exists():
            raise RuntimeError(f"cell-eval returned no metrics JSON at {out_json}")
        metrics = json.loads(out_json.read_text())
        missing = [k for k in SELECTION_SCORE_COMPONENTS if k not in metrics]
        if missing:
            raise RuntimeError(
                f"cell-eval returned no {missing} on the validation split, so "
                f"{SELECTION_SCORE_NAME} cannot be formed and no checkpoint could be "
                f"selected. Got {sorted(metrics)[:10]}")
        return {str(k): float(v) for k, v in metrics.items()
                if isinstance(v, (int, float)) and np.isfinite(float(v))}

    def _run(self, cmd: list[str], pythonpath: str) -> None:
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = pythonpath
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"scorer failed ({' '.join(cmd[2:5])}): exit {proc.returncode}\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )

    def _snapshot(self, solver):
        import jax

        state = getattr(solver, "vf_state_inference", None) or solver.vf_state
        return jax.device_get(state.params)

    def _restore_best(self, solver) -> None:
        if self._best_params is None:
            print("[selection] nothing to restore; keeping the final state", flush=True)
            return
        assert self.best is not None
        if self.best["final"]:
            print(
                f"[selection] best state IS the final one "
                f"({SELECTION_SCORE_NAME}={self.best[SELECTION_SCORE_NAME]:.5f}); "
                "no restore needed",
                flush=True,
            )
            return
        for attr in ("vf_state_inference", "vf_state"):
            state = getattr(solver, attr, None)
            if state is not None:
                setattr(solver, attr, state.replace(params=self._best_params))
        print(
            f"[selection] restored {self.best['tag']} "
            f"({SELECTION_SCORE_NAME}={self.best[SELECTION_SCORE_NAME]:.5f}) "
            "into the solver; this is what save() will persist",
            flush=True,
        )

    def as_callback(self):
        """Return an object the trainer's CallbackRunner classifies as a ComputationCallback."""
        from cellflow.training import ComputationCallback

        outer = self

        class _Bound(ComputationCallback):
            def on_train_begin(self, *a, **k):
                return outer.on_train_begin()

            def on_log_iteration(self, *a, **k):
                return outer.on_log_iteration(*a, **k)

            def on_train_end(self, *a, **k):
                return outer.on_train_end(*a, **k)

        return _Bound()

    def summary(self) -> dict[str, Any]:
        """What the caller records in run_meta.json."""
        return {
            "selection_metric": f"mean({', '.join(SELECTION_SCORE_COMPONENTS)})",
            "n_scored_states": len(self.history),
            "best": self.best,
            "history": self.history,
        }


def os_sched_affinity() -> set[int]:
    """CPUs this process may use.

    `os.cpu_count()` returns the host count and ignores the cgroup limit, and cell-eval's numba
    path then raises "The number of threads must be between 1 and N" after the artifacts are
    already written.
    """
    import os

    if hasattr(os, "sched_getaffinity"):
        return os.sched_getaffinity(0)
    return set(range(os.cpu_count() or 1))


__all__ = ["SelectionCallback", "read_agg_metrics", "SELECTION_SCORE_COMPONENTS"]
