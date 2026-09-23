"""Phase A of the bridge: a competitor emits a prediction matrix in OUR evaluation space.

WHY TWO PHASES
--------------
A competitor's checkpoint can only be loaded in that competitor's own interpreter, and our
scoring path (`occam.vcell.panel.protocol.run_split`) can only run in ours. Two interpreters
cannot share a Python callable, so the bridge is a file:

    Phase A (competitor venv, THIS module)
        read our frozen panel -> feed the model the drawn half-B control cells
        -> map the emission into our evaluation space -> write pred.npy

    Phase B (our venv, scripts/eval_competitor.py)
        wrap pred.npy in a matrix-backed predict_fn -> protocol.run_split

Everything that makes the comparison fair lives in Phase B, which the competitor never
touches: the frozen 611-perturbation cohort, the 5 splits, the mandatory zero-signal anchors,
`pred_ctrl = ctrl_B`, the sha256-keyed real-DE cache, and cell-eval 0.6.0.

WHAT AN ARM MUST PROVIDE
------------------------
One callable:

    predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray

`ctrl` is `(n, 6983)` float32 in **the arm's own training space** (by default
`log1p(target_sum=10_000)`, which is what `occam_data` hands it), one real half-B HepG2
control cell per row. `pert_names` is the perturbation to apply to each row, same length,
same order. The return must be `(n, 6983)` in the arm's **emission** space.

This module then converts that emission into our evaluation space and refuses anything that
is not in it. It does NOT ask the arm to know anything about our space -- that is the point:
their internal preprocessing is their business, the interface is ours.

⚠️ ROW ORDER IS A CONTRACT
--------------------------
The rows are the half-A perturbed cells of a split, ordered by
`np.argsort(labels, kind="stable")`, and `run_split` re-checks that order before scoring.
`labels.json` is written alongside so Phase B can verify rather than assume.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from emission_space import (  # noqa: E402
    EVAL_TARGET_SUM, TRAIN_TARGET_SUM, EmissionSpaceError,
    assert_representable, describe_space, space_diagnostics, to_eval_space,
)

__all__ = ["PanelSplit", "load_panel_split", "export_split", "draw_proxy_batches", "CONTROL"]

CONTROL = "non-targeting"


class PanelSplit:
    """One split of the frozen panel, in the arm's training space.

    Attributes
    ----------
    ctrl : (18575, 6983) float32
        The drawn half-B control cells, one per predicted row -- the model's input.
    labels : (18575,) str
        The perturbation for each row, in the order `run_split` will re-check.
    target : (18575, 6983) float32
        The half-A perturbed cells, in OUR evaluation space. Used only by
        `space_diagnostics`; never shown to the model.
    batches : (18575,) str
        The sequencing batch of each PERTURBED cell -- i.e. of the cell being predicted, not
        of the control drawn as its input.

        ⚠️ WHY THE PERTURBED CELL'S. Both set-based arms read this covariate off the
        perturbed row at training (STATE `zarr_perturbation.py:689` `batch_name =
        self.batch[i]`; PerturbDiff `dataset_core.py:189-196` `get_covarates`), so the
        faithful value here is `batches[a_rows]`. The control's batch sits one index away
        (`batches[ctrl_b][cin_idx]`) and is the easy thing to reach for by mistake.

        ⚠️ WHY IT EXISTS AT ALL. Both arms' exports used to pin this to a single constant --
        `sorted(...)[0]`, i.e. one of 56 HepG2 batches -- and broadcast it over all 18,575
        rows, while both models carry a TRAINED batch embedding that is added to every cell.
        Measured on STATE: `mean|delta pred| / mean|pred|` = 0.37, and the
        perturbation-specific component of the delta correlated only 0.73-0.86 with the
        faithful prediction. It does not cancel, because the scored control rows are the real
        half-B cells, not the model's own output. A real 30-cell sentence spans ~22 distinct
        batches; one label repeated 30x is off-distribution whichever label is chosen.

        The VALUES here are the panel's own (`hepg2-nad-49`). Each arm's vocabulary differs
        (STATE's map is keyed `HEPG2__hepg2-nad-49`), so every consumer must BUILD and ASSERT
        its mapping rather than assume it.
    """

    def __init__(self, index, ctrl, labels, target, ctrl_b_rows, cohort, manifest,
                 batches=None, model_cols=None):
        self.index = index
        self.ctrl = ctrl
        self.labels = labels
        self.target = target
        self.ctrl_b_rows = ctrl_b_rows
        self.cohort = cohort
        self.manifest = manifest
        self.batches = batches
        # OOD-context panels only (e.g. h1_seen): the panel axis is a SUBSET of the axis the
        # model was trained on. `ctrl` is then on the MODEL axis (zeros at the genes the
        # panel lacks) and `model_cols[j]` is the model-axis column of panel gene j, which
        # `export_split` gathers from the emission. None = the two axes are the same.
        self.model_cols = model_cols
        # set by `draw_proxy_batches` when the batch covariate is a proxy, recorded
        self.context_proxy = None

    def __len__(self) -> int:
        return len(self.labels)


def load_panel_split(panel_dir: str, split_index: int, *,
                     target_sum: float = TRAIN_TARGET_SUM) -> PanelSplit:
    """The frozen panel's split `k`, with the control input in the arm's training space.

    The panel on disk is in our **evaluation** space (`target_sum=2000`). The model expects
    its **training** space, so the control cells are converted with the same analytic map
    used in reverse on the way out. At the default 10,000 that is
    `log1p(expm1(x) * 5.0)` -- exact, because the per-cell denominator cancels.
    """
    z = np.load(os.path.join(panel_dir, "panel.npz"), allow_pickle=True)
    zs = np.load(os.path.join(panel_dir, "splits.npz"))
    with open(os.path.join(panel_dir, "cohort.json")) as fh:
        cohort = json.load(fh)
    with open(os.path.join(panel_dir, "splits_manifest.json")) as fh:
        man = json.load(fh)
    if man["cohort_sha256"] != cohort["cohort_sha256"]:
        raise RuntimeError("cohort.json and splits_manifest.json disagree; rebuild the panel")

    X = np.asarray(z["X"], dtype=np.float32)                 # our EVAL space
    labels_all = np.asarray(z["labels"], dtype=str)
    k = int(split_index)
    key = f"_{k}"
    if f"pert_A{key}" not in zs:
        raise RuntimeError(f"split {k} is not in splits.npz "
                           f"(built with n_splits={man['n_splits']})")

    a_rows = np.asarray(zs[f"pert_A{key}"], dtype=np.int64)
    ctrl_b = np.asarray(zs[f"ctrl_B{key}"], dtype=np.int64)
    # the same stable ordering run_split re-checks
    a_rows = a_rows[np.argsort(labels_all[a_rows], kind="stable")]
    labels = labels_all[a_rows]

    # The control DRAW must match predict_panel's, or `meanpert`, `transfer` and the model
    # arm stop sharing control noise and the paired comparison loses its point.
    # predict_panel: rng = default_rng(seed + 1000*split); cin_idx = rng.integers(0, n_ctrl, n)
    rng = np.random.default_rng(42 + 1000 * k)
    cin_idx = rng.integers(0, len(ctrl_b), size=len(a_rows))
    ctrl_eval = X[ctrl_b][cin_idx]

    ctrl_train = ctrl_eval
    if abs(target_sum - EVAL_TARGET_SUM) > 1e-9:
        # eval -> train: the inverse of to_eval_space, same cancellation
        ctrl_train = np.log1p(
            np.expm1(ctrl_eval.astype(np.float64)) * (target_sum / EVAL_TARGET_SUM)
        ).astype(np.float32)

    # the PERTURBED cell's batch -- see the PanelSplit docstring for why not the control's
    batches = None
    if "batches" in z.files:
        batches = np.asarray(z["batches"], dtype=str)[a_rows]

    # ---- OOD-context panel: scatter the panel axis onto the model's training axis ------
    model_cols = None
    if man.get("ood_context"):
        src = man.get("seen_from") or {}
        repo = os.path.dirname(os.path.dirname(_HERE))
        axis_csv = os.path.join(repo, "data", "benchmarks", str(src.get("benchmark_id")),
                                "gene_axis.csv")
        with open(axis_csv) as fh:
            model_axis = [ln.split(",")[0] for ln in fh.read().splitlines()[1:] if ln]
        import hashlib as _hl
        if _hl.sha256("\n".join(model_axis).encode()).hexdigest() != src.get(
                "gene_axis_sha256"):
            raise RuntimeError(f"{axis_csv} does not hash to the panel's seen_from axis")
        pos = {g: i for i, g in enumerate(model_axis)}
        panel_axis = [str(g) for g in z["axis_ensembl"]]
        outside = [g for g in panel_axis if g not in pos]
        if outside:
            raise RuntimeError(f"{len(outside)} panel genes are not on the model axis, "
                               f"e.g. {outside[:3]}")
        model_cols = np.array([pos[g] for g in panel_axis], dtype=np.int64)
        full = np.zeros((ctrl_train.shape[0], len(model_axis)), dtype=np.float32)
        full[:, model_cols] = ctrl_train
        ctrl_train = full
        print(f"[export]   OOD context {man.get('cell_line')!r}: panel axis "
              f"{len(panel_axis):,} scattered onto the {len(model_axis):,}-gene model axis "
              f"({len(model_axis) - len(panel_axis):,} input genes are zero)")

    print(f"[export] split {k}: {len(labels):,} rows · {len(ctrl_b):,} half-B controls · "
          f"control input at target_sum {target_sum:g}")
    if batches is None:
        print("[export]   ⚠️  panel.npz has no `batches`; arms with a trained batch embedding "
              "will have to fall back to a constant, which is off-distribution")
    else:
        print(f"[export]   batches     : {len(set(batches.tolist()))} distinct, "
              f"from the perturbed cells")
    print(f"[export]   input space : {describe_space(ctrl_train, target_sum=target_sum).summary()}")
    return PanelSplit(k, ctrl_train, labels, X[a_rows], ctrl_b, cohort, man, batches=batches,
                      model_cols=model_cols)


def draw_proxy_batches(split: PanelSplit, candidates: Sequence[str], *,
                       seed: int = 42, why: str = "") -> np.ndarray:
    """Replace `split.batches` by a seeded, uniform per-cell draw from `candidates`.

    For an OOD-context panel whose batches the model has never seen (h1_seen: `Flex_*`),
    an arm with a trained batch embedding still needs SOME batch per cell. One constant is
    off-distribution (measured 0.37 relative change on STATE), so each cell gets an
    independent draw from the model's own batches of the proxy cell line. Deterministic
    per split. Recorded in the manifest by `export_split` as `context_proxy`.
    """
    cands = sorted(str(c) for c in candidates)
    if not cands:
        raise RuntimeError("no candidate batches to draw a proxy from")
    rng = np.random.default_rng(int(seed) + 1000 * int(split.index) + 7)
    split.batches = np.asarray(cands, dtype=object)[rng.integers(0, len(cands), len(split))]
    split.context_proxy = {"batch": "uniform per-cell draw", "n_candidates": len(cands),
                           "candidates_example": cands[:3], "seed": int(seed), "why": why}
    print(f"[export]   PROXY batch covariate: uniform per-cell draw from {len(cands)} "
          f"model batches (e.g. {cands[0]!r})")
    return split.batches


def compose_sequential(predict: Callable[[np.ndarray, Sequence[str]], np.ndarray],
                      order: str = "mean", sep: str = "+"):
    """Wrap a single-perturbation `predict` so it can answer a PAIR, sequentially.

    WHY THIS EXISTS
        `norman_combo` asks for `"A+B"`, and two of the four ported arms have no
        multi-perturbation mechanism of their own:

          State        its paper's combinatorial experiment (on DRUGS) simulates a
                       combination "by running consecutive forward passes", which is
                       exactly this. Faithful to its own design; the genetic setting is
                       our extension of it and is labelled as one.
          PerturbDiff  has NO published combinatorial protocol at all. This rule is
                       OURS and must be reported as a deviation, not as their method.

        TxPert and CellFlow are NOT wrapped: TxPert sums the two perturbation embeddings
        natively (its condition is already `"A+B"`), and CellFlow aggregates a
        perturbation SET with attention or DeepSets. Wrapping either would measure our
        rule instead of theirs. See docs/competitors_on_doubles.md.

    THE RULE -- `decode_seq_mean`, Occam's own default
        prediction(A) = f(ctrl, A);  prediction(A then B) = f(prediction(A), B)
        and the same with the order reversed, then the mean of the two chains.

        Nothing inside the model changes: the second pass simply receives the first
        pass's OUTPUT as its control input. That is why this belongs in the export --
        the export is what decides what is fed in.

    ⚠️ ORDER IS NOT IMMATERIAL here. On Occam's own 72-cell matrix `decode_seq_ba` beats
    `decode_seq_ab` on all 6 arms and both panels, with max |ab - ba| = 8.96e-02, which
    is why `mean` is the default rather than either single order.

    ⚠️ SINGLE perturbations are passed straight through, untouched, in ONE call -- so an
    arm wrapped here produces bit-identical output to the unwrapped arm on any
    single-perturbation panel. That is what keeps `rhj_fewshot` numbers comparable.
    """
    if order not in ("ab", "ba", "mean"):
        raise ValueError(f"order must be ab|ba|mean, got {order!r}")

    def _chain(ctrl: np.ndarray, first: List[str], second: List[str]) -> np.ndarray:
        step1 = np.asarray(predict(ctrl, first), dtype=np.float32)
        if step1.shape != ctrl.shape:
            raise ValueError(
                f"predict returned {step1.shape} for a {ctrl.shape} input; the "
                f"sequential composition feeds the output back in, so the two must "
                f"have the same width")
        return np.asarray(predict(step1, second), dtype=np.float32)

    def _predict(ctrl: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        labels = [str(l) for l in labels]
        parts = [[p for p in l.split(sep) if p] for l in labels]
        multi = np.array([len(p) > 1 for p in parts], dtype=bool)
        out = np.asarray(predict(ctrl, labels), dtype=np.float32).copy()
        if not multi.any():
            return out
        idx = np.flatnonzero(multi)
        sub = ctrl[idx]
        a = [parts[i][0] for i in idx]
        b = [parts[i][1] for i in idx]
        if order in ("ab", "mean"):
            ab = _chain(sub, a, b)
        if order in ("ba", "mean"):
            ba = _chain(sub, b, a)
        out[idx] = ab if order == "ab" else ba if order == "ba" else 0.5 * (ab + ba)
        if len(idx) and len(idx) == len(labels):
            print(f"[export]   sequential composition ({order}) on all "
                  f"{len(idx):,} rows", flush=True)
        elif len(idx):
            print(f"[export]   sequential composition ({order}) on {len(idx):,} of "
                  f"{len(labels):,} rows; the rest are single perturbations",
                  flush=True)
        return out

    return _predict


def export_split(predict: Callable[[np.ndarray, Sequence[str]], np.ndarray],
                 split: PanelSplit, out_dir: str, *,
                 arm: str,
                 from_target_sum: float = TRAIN_TARGET_SUM,
                 from_log1p: bool = True,
                 batch_size: int = 2048,
                 extra_manifest: Optional[Dict[str, object]] = None) -> str:
    """Run an arm over one split and write `pred.npy` + `labels.json` + `manifest.json`.

    Returns the directory written. Raises `EmissionSpaceError` rather than writing a
    prediction that is not in our evaluation space -- a wrong-space matrix that reaches
    cell-eval produces a plausible number from a meaningless comparison.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(split)

    chunks: List[np.ndarray] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = np.asarray(predict(split.ctrl[s:e], list(split.labels[s:e])),
                         dtype=np.float32)
        if out.shape != (e - s, split.ctrl.shape[1]):
            raise ValueError(
                f"{arm}: predict returned {out.shape} for rows [{s}:{e}), expected "
                f"{(e - s, split.ctrl.shape[1])}. Every arm must emit one row per input "
                f"cell on the benchmark's {split.ctrl.shape[1]:,}-gene axis.")
        chunks.append(out)
        if (s // batch_size) % 4 == 0:
            print(f"[export]   {e:,}/{n:,} rows", flush=True)
    raw = np.concatenate(chunks, axis=0)
    if split.model_cols is not None:
        # OOD-context panel: emitted on the model axis, scored on the panel's subset of it
        raw = raw[:, split.model_cols]
        print(f"[export]   gathered {raw.shape[1]:,} panel genes from the model-axis emission")

    print(f"[export]   raw emission : "
          f"{describe_space(raw, target_sum=from_target_sum).summary()}")
    pred = to_eval_space(raw, from_target_sum=from_target_sum, from_log1p=from_log1p)
    print(f"[export]   eval space   : {describe_space(pred).summary()}")

    # ⛔ HARD: can this be scored at all? Model-agnostic -- a terrible arm passes.
    assert_representable(pred, n_genes=split.target.shape[1],
                         name=f"{arm} prediction", target_sum=EVAL_TARGET_SUM,
                         target=split.target)

    # DIAGNOSTIC: never a refusal. Library size, density and mean correlation are all
    # confounded with model quality, and refusing on them would exclude exactly the weak
    # arms this benchmark exists to measure. Recorded in the manifest, read beside the score.
    diag = space_diagnostics(pred, split.target, pred_name=f"{arm} prediction",
                             target_name="panel target")
    print(f"[export]   representable ✓   diagnostics: rowsum ratio "
          f"{diag['rowsum_ratio']:.4f} · gene-mean corr {diag['gene_mean_corr']:.4f} · "
          f"zero {diag['zero_fraction_pred']:.4f} (target "
          f"{diag['zero_fraction_target']:.4f})")
    print(f"[export]   for reference, OUR OWN arm: rowsum ratio 0.9399 · "
          f"gene-mean corr 0.9998 · zero 0.0000")

    np.save(os.path.join(out_dir, "pred.npy"), pred)
    with open(os.path.join(out_dir, "labels.json"), "w") as fh:
        json.dump(list(map(str, split.labels)), fh)
    manifest = {
        "arm": arm,
        "split": split.index,
        "n_rows": int(n),
        "n_genes": int(pred.shape[1]),
        "cohort_sha256": split.cohort["cohort_sha256"],
        "benchmark_sha256": split.manifest["benchmark_sha256"],
        "gene_axis_sha256": split.manifest["gene_axis_sha256"],
        "emission": {
            "from_target_sum": float(from_target_sum),
            "from_log1p": bool(from_log1p),
            "to_target_sum": float(EVAL_TARGET_SUM),
            "map": ("log1p(expm1(x) * to/from)" if from_log1p
                    else "log1p(x * to/from)"),
        },
        "representable": True,
        "ood_context": bool(split.manifest.get("ood_context")),
        "context_proxy": split.context_proxy,
        "space_diagnostics": diag,
    }
    if extra_manifest:
        # ⚠️ An adapter's `extra_manifest` must not be able to CLOBBER a field this writer
        # owns. CellFlow's adapter passed `emission` as a prose string, which silently
        # replaced the structured {from_target_sum, from_log1p, to_target_sum, map} dict that
        # every consumer reads -- and scoring then died five splits in with
        # `'str' object has no attribute 'get'`, after a 1h22m export. The adapter is the
        # wrong place to discover this, so refuse it here where the contract lives.
        _owned = set(manifest) & set(extra_manifest)
        _clashes = {k for k in _owned if manifest[k] != extra_manifest[k]}
        if _clashes:
            raise ValueError(
                f"extra_manifest from arm {arm!r} would overwrite field(s) this writer owns: "
                f"{sorted(_clashes)}. These are the declared, machine-read provenance fields "
                f"(emission space, shas, row counts); an adapter that needs to say more should "
                f"use a NEW key such as '<field>_note'. Existing values: "
                f"{ {k: manifest[k] for k in sorted(_clashes)} }")
        manifest.update(extra_manifest)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    print(f"[export] wrote {out_dir}/pred.npy  {pred.shape} {pred.dtype}")
    return out_dir
