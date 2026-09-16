"""Score a checkpoint on a panel: single cell in, single cell out, cell-eval v1.

The headline panel is ``all``. ``seen`` and ``unseen`` sub-panels are available
and are computed by **re-aggregating the ``all`` run's per-perturbation
results**, not by re-running cell-eval.

That shortcut is sound here and was not sound for the research tree's other
scorer. That one scaled each metric against a per-panel baseline and
replicate ceiling, so a sub-panel needed its own reference build; cell-eval v1
metrics are computed per perturbation and then averaged, so a sub-panel result
*is* a restriction of the full run. It turns "3 panels x N arms x ~60 min" into
"1 panel x N arms", and removes the whole reference-management layer.

Two metrics are panel-level rather than per-perturbation
(``clustering_agreement``, ``pearson_edistance``); they are reported for ``all``
only and never re-aggregated.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from occam.vcell.panel.arms import ANCHOR_TAGS, Arm, build_arms
from occam.vcell.panel.celleval_v1 import run_celleval_v1, write_celleval_v1
from occam.vcell.panel.panel_io import Panel

__all__ = [
    "run_panel_evaluation", "HEADLINE_PANEL", "REPORTED_METRICS",
    "PANEL_LEVEL_METRICS", "AnchorMissingError", "write_report",
]

HEADLINE_PANEL = "all"

# Promoted to the report. The rest of cell-eval v1's ~28 metrics stay in
# metrics.json.
REPORTED_METRICS = (
    "pearson_delta",              # field convention -- and the metric the anchor wins here
    "discrimination_score_l2",    # second convention metric
    "overlap_at_N",               # the metric the model does win; do not quietly drop it
    "de_direction_match",         # direction only, magnitude-free: the shrink sweep's control
    "mse_delta",                  # effect-size error, collapses under shrink
    "mae_delta",
)

# Computed over the whole panel, not per perturbation. Not re-aggregable.
PANEL_LEVEL_METRICS = ("clustering_agreement", "pearson_edistance")

# Metrics where a LOWER value is better. Reported with the direction stated,
# because a table mixing both without saying so is unreadable.
LOWER_IS_BETTER = ("mse_delta", "mae_delta", "mse", "mae")


class AnchorMissingError(RuntimeError):
    """A report was requested for a panel with no zero-signal anchor scored."""


def run_panel_evaluation(
    panel: Panel,
    pred: np.ndarray,
    labels: np.ndarray,
    ctrl_used: np.ndarray,
    out_dir: str,
    *,
    train_pert_names: Optional[Sequence[str]] = None,
    panels: Sequence[str] = (HEADLINE_PANEL,),
    calibrate: str = "both",
    calibrated: Optional[np.ndarray] = None,
    shrinks: Sequence[float] = (1.0,),
    anchors: Sequence[str] = ANCHOR_TAGS,
    min_panel_perts: int = 25,
    small_panel_warn: int = 30,
    celleval_kwargs: Optional[Mapping] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Score every arm on the ``all`` panel, then slice the sub-panels.

    Returns ``{panel: {arm: {metric: value}}}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    celleval_kwargs = dict(celleval_kwargs or {})

    wanted = _resolve_panels(panels, labels, train_pert_names,
                             min_panel_perts, small_panel_warn)

    arms = build_arms(pred, labels, ctrl_used, panel, calibrate=calibrate,
                      shrinks=shrinks, anchors=anchors, calibrated=calibrated)
    anchor_tags = [t for t, _, _ in arms if t in ANCHOR_TAGS]
    print(f"[panel] {len(arms)} arm(s): {', '.join(t for t, _, _ in arms)}")
    print(f"[panel] anchors scored first: {', '.join(anchor_tags)}")

    results: Dict[str, Dict[str, Dict[str, float]]] = {p: {} for p in wanted}

    for tag, matrix, arm_labels in arms:
        arm_dir = os.path.join(out_dir, f"celleval_{tag}")
        pred_path, target_path = write_celleval_v1(matrix, arm_labels, panel, out_dir, tag)

        # Real-side DE is the dominant cost and is identical across arms within a panel:
        # the target file is a pure function of the panel, with no arm dependence. Keyed on
        # the TARGET FILE's own sha256 rather than on "whatever real_de.csv the first arm
        # left behind", so a rebuilt panel misses the cache by construction instead of
        # being scored against a stale ground truth.
        from occam.decoder_train.metrics import real_de_cache_path

        kwargs = dict(celleval_kwargs)
        cache_dir = kwargs.pop("de_real_cache_dir", None) or os.path.join(
            out_dir, "_de_real_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path, digest = real_de_cache_path(cache_dir, target_path)
        if os.path.isfile(cache_path):
            kwargs["de_real"] = cache_path
            print(f"[panel] {tag}: reusing real-side DE for target {digest[:16]}")

        metrics = run_celleval_v1(pred_path, target_path, arm_dir, **kwargs)
        if not metrics:
            print(f"[panel] {tag}: cell-eval returned nothing; skipping")
            continue

        if not os.path.isfile(cache_path):
            produced = os.path.join(arm_dir, "real_de.csv")
            if os.path.isfile(produced):
                import shutil
                shutil.copyfile(produced, cache_path)
                print(f"[panel] cached real-side DE -> {os.path.basename(cache_path)}")

        results[HEADLINE_PANEL][tag] = metrics

        # Sub-panels by re-aggregation, no second cell-eval run.
        per_pert = _read_results_csv(os.path.join(arm_dir, "results.csv"))
        for name in wanted:
            if name == HEADLINE_PANEL:
                continue
            subset = wanted[name]
            if per_pert is None:
                print(f"[panel] {tag}: no results.csv, cannot slice '{name}'")
                continue
            sliced = _aggregate_subset(per_pert, subset)
            if sliced:
                results[name][tag] = sliced

    _write_outputs(results, out_dir, wanted)
    return results


def _resolve_panels(panels, labels, train_pert_names, min_perts, warn_at):
    """``{panel_name: set(perturbations)}`` for the panels actually scorable."""
    present = {str(p) for p in labels if str(p) != "non-targeting"}
    requested = [p.strip() for p in panels if str(p).strip()]
    if HEADLINE_PANEL not in requested:
        requested.insert(0, HEADLINE_PANEL)

    needs_train = {"seen", "unseen"} & set(requested)
    if needs_train and train_pert_names is None:
        raise ValueError(
            f"panels {sorted(needs_train)} need the training perturbation set, but none "
            f"was supplied. Read it from the checkpoint's split_info['train'].\n"
            f"Passing no training set used to yield seen=[] silently, which reads as "
            f"'no seen perturbations' rather than 'unknown' -- so it is an error here."
        )
    train = {str(p) for p in (train_pert_names or ())}

    candidates = {
        HEADLINE_PANEL: present,
        "seen": present & train,
        "unseen": present - train,
    }
    out = {}
    for name in requested:
        if name not in candidates:
            raise ValueError(f"unknown panel {name!r}; expected all|seen|unseen")
        perts = candidates[name]
        if name != HEADLINE_PANEL and len(perts) < min_perts:
            print(f"[panel] skipping '{name}': only {len(perts)} perturbations "
                  f"(minimum {min_perts})")
            continue
        if len(perts) < warn_at:
            print(f"[panel] WARNING: '{name}' has only {len(perts)} perturbations; "
                  f"metric means will be noisy")
        out[name] = perts
        print(f"[panel] {name}: {len(perts)} perturbations")
    return out


def _read_results_csv(path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """``{perturbation: {metric: value}}`` from cell-eval's per-perturbation output."""
    import csv

    if not os.path.isfile(path):
        return None
    out: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = row.get("target_gene") or row.get("perturbation") or row.get("pert")
                if not key:
                    return None
                vals = {}
                for k, v in row.items():
                    if k == key or k in ("target_gene", "perturbation", "pert"):
                        continue
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(f):
                        vals[k] = f
                out[key] = vals
    except Exception as exc:  # noqa: BLE001
        print(f"[panel] could not parse {path}: {exc}")
        return None
    return out


def _aggregate_subset(per_pert, subset) -> Dict[str, float]:
    """Mean each metric over ``subset``, skipping the panel-level ones."""
    rows = [v for k, v in per_pert.items() if k in subset]
    if not rows:
        return {}
    metrics = set()
    for r in rows:
        metrics.update(r)
    out = {}
    for m in sorted(metrics):
        if m in PANEL_LEVEL_METRICS:
            continue          # not per-perturbation; not meaningful to re-average
        vals = [r[m] for r in rows if m in r]
        if vals:
            out[m] = float(np.mean(vals))
    out["n_perturbations"] = float(len(rows))
    return out


def _write_outputs(results, out_dir, wanted) -> None:
    flat = {}
    for panel, arms in results.items():
        for arm, metrics in arms.items():
            for metric, value in metrics.items():
                flat[f"panel/{panel}/{arm}/{metric}"] = value
    for panel, perts in wanted.items():
        flat[f"panel/{panel}/n_perturbations"] = float(len(perts))
        flat[f"panel/{panel}/anchors_scored"] = float(
            any(a in ANCHOR_TAGS for a in results.get(panel, {}))
        )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(flat, fh, indent=2, sort_keys=True)

    import csv

    with open(os.path.join(out_dir, "metrics.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "arm", "metric", "value", "lower_is_better",
                    "ctrlmean", "delta_vs_ctrlmean", "ctrlcell", "delta_vs_ctrlcell"])
        for panel, arms in sorted(results.items()):
            base = {a: arms.get(a, {}) for a in ANCHOR_TAGS}
            for arm, metrics in sorted(arms.items()):
                for metric, value in sorted(metrics.items()):
                    cm = base["ctrlmean"].get(metric)
                    cc = base["ctrlcell"].get(metric)
                    w.writerow([
                        panel, arm, metric, f"{value:.6f}",
                        metric in LOWER_IS_BETTER,
                        "" if cm is None else f"{cm:.6f}",
                        "" if cm is None else f"{value - cm:+.6f}",
                        "" if cc is None else f"{cc:.6f}",
                        "" if cc is None else f"{value - cc:+.6f}",
                    ])

    write_report(results, os.path.join(out_dir, "report.md"))
    print(f"[panel] wrote metrics.json, metrics.csv and report.md to {out_dir}")


def write_report(results, path: str) -> None:
    """Write a human-readable report. Refuses if a panel has no anchor.

    Not a formatting nicety: without the anchor the numbers cannot be read at
    all on this evaluation, because the zero-signal arm beats the model on the
    field's usual headline metric.
    """
    lines = ["# Panel evaluation", ""]

    for panel, arms in sorted(results.items()):
        if not arms:
            continue
        present_anchors = [a for a in ANCHOR_TAGS if a in arms]
        if not present_anchors:
            raise AnchorMissingError(
                f"panel '{panel}' has no zero-signal anchor among its scored arms "
                f"({sorted(arms)}). Refusing to write a report: on this evaluation the "
                f"ctrlmean anchor outscores every model arm on pearson_delta, so a table "
                f"without it invites exactly the wrong conclusion."
            )

        model_arms = [a for a in sorted(arms) if a not in ANCHOR_TAGS]
        lines += [f"## Panel: {panel}", ""]

        # State which metrics the anchors won on THIS run, rather than
        # inheriting a claim from another panel.
        beaten = []
        for metric in REPORTED_METRICS:
            anchor_vals = [arms[a][metric] for a in present_anchors if metric in arms[a]]
            model_vals = [arms[a][metric] for a in model_arms if metric in arms[a]]
            if not anchor_vals or not model_vals:
                continue
            lower_better = metric in LOWER_IS_BETTER
            best_anchor = min(anchor_vals) if lower_better else max(anchor_vals)
            best_model = min(model_vals) if lower_better else max(model_vals)
            model_wins = best_model < best_anchor if lower_better else best_model > best_anchor
            if not model_wins:
                beaten.append(metric)
        if beaten:
            lines += [
                f"> ⚠️ The zero-signal anchor(s) `{', '.join(present_anchors)}` are not "
                f"beaten by any model arm on: **{', '.join(beaten)}**.",
                ">",
                "> Read every number below against its anchor column. Which metric the "
                "anchor wins is a property of this panel, not a general fact.",
                "",
            ]

        header = ["metric", "direction"] + present_anchors + model_arms
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * len(header)) + "|"]
        for metric in list(REPORTED_METRICS) + [
            m for m in PANEL_LEVEL_METRICS if panel == HEADLINE_PANEL
        ]:
            row = [f"`{metric}`",
                   "lower better" if metric in LOWER_IS_BETTER else "higher better"]
            any_value = False
            for arm in present_anchors + model_arms:
                v = arms.get(arm, {}).get(metric)
                row.append("—" if v is None else f"{v:.4f}")
                any_value = any_value or v is not None
            if any_value:
                lines.append("| " + " | ".join(row) + " |")
        n = arms.get(model_arms[0] if model_arms else present_anchors[0], {}).get(
            "n_perturbations")
        if n:
            lines += ["", f"Perturbations scored: {int(n)}"]
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "* Every arm is scored on the **same** panel with cell-eval v1, `--profile full`.",
        "* `ctrlmean` emits the model-visible control mean for every perturbation; "
        "`ctrlcell` emits the drawn control cell unchanged. Neither carries any biology, "
        "so whatever they score is what the metric reads from the emission alone.",
        "* Sub-panels (`seen`, `unseen`) are re-aggregations of the `all` run's "
        "per-perturbation results, not separate cell-eval runs. Panel-level metrics "
        f"({', '.join('`' + m + '`' for m in PANEL_LEVEL_METRICS)}) are reported for "
        "`all` only.",
        "* `mse_delta` and `mae_delta` are ERRORS: lower is better.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
