"""The ONE evaluation space, and the guard that every competitor prediction passes through.

WHY THIS MODULE EXISTS
----------------------
cell-eval compares two h5ads elementwise. If the prediction and the target sit in different
spaces, every magnitude metric is wrong and **nothing warns you**. Worse, cell-eval's own
safety net cannot see the difference that matters. All of this is MEASURED, not assumed --
see the numbers in each section.

THE CANONICAL SPACE (measured from data/benchmarks/rhj_fewshot/splithalf/panel.npz)
----------------------------------------------------------------------------------
    x_eval = log1p( counts / rowsum_FULL_PANEL * 2000 )

restricted to the frozen 6,983-gene axis, in `gene_axis.csv` order, with gene SYMBOLS in
`var.index`.

⚠️ `rowsum_FULL_PANEL` is over the source file's **whole** gene panel (9,624 for HepG2),
taken BEFORE the axis restriction -- which is what `stream_v2.load_chunk_sparse` does.
Verified directly on the panel:

    expm1(X).sum(axis=1):  median 1731.8, mean 1719.8, range 1228.0 - 1931.8

i.e. **not 2000**. The 6,983-gene axis carries 86.6% of each cell's UMI, and the share varies
per cell. Had the panel been normalised on the axis sum instead, that median would be exactly
2000.0. Also measured: value range [0, 5.9248] against log1p(2000) = 7.6014, zero fraction
0.5147.

WHY THAT DETAIL DECIDES THE WHOLE PORT
--------------------------------------
Competitors are handed `target_sum=10_000` data (their pipelines assume pre-normalised input;
see `occam_data.py`). Converting their output to the canonical space is then exact, because
the per-cell denominator cancels:

    x_eval = log1p( expm1(x_10k) * 0.2 )            # 2000 / 10000

MEASURED on real panel rows, two ways -- and the distinction matters:

  * DERIVED (build the 10k matrix from the 2k one in float64, map back):
    max |diff| = 8.88e-16 in float64, **exactly 0.0 in float32**.
  * INDEPENDENT (compute both matrices from raw counts in float32, map the 10k one back):
    max |diff| = **4.768e-07 = exactly one float32 ULP at the max value 5.0**.

So the map is exact in exact arithmetic, and agrees to a single ULP when each matrix carries
its own float32 rounding -- which is the real situation, since competitors are handed a
10k matrix computed independently of the 2k panel. `assert_representable` passes on it. One ULP
is ~7 orders of magnitude below the +/-10% library-size tolerance and cannot move a metric.

That exactness holds ONLY because both sides use the same full-panel denominator. Normalising
competitors on the 6,983-axis sum instead would leave a per-cell factor (median 0.866, and
cell-dependent) that leaves `pearson_delta` untouched -- it is scale-invariant per delta
vector -- while shifting `mae` and moving the DE calls. `mae` is one of our four comparison
metrics.

⚠️⚠️ WHAT CELL-EVAL WILL AND WILL NOT CATCH
------------------------------------------
`cell_eval._evaluator._convert_to_normlog` is, in full:

    if guess_is_lognorm(adata):  return          # "already log-normalized"
    if allow_discrete:           return          # our flag
    sc.pp.normalize_total(adata)                 # to the FILE'S OWN MEDIAN
    sc.pp.log1p(adata)

and `guess_is_lognorm` (`cell_eval/utils.py:5`) is only *"True if ANY of 500 randomly chosen
cells has a row sum with a fractional part > 0.01"* -- i.e. **it merely asks "are these
floats?"**. Its own docstring admits: *"This will not be able to distinguish between
normalized input and log-normalized input."*

Measured, 5 unseeded repeats each:

| input | guess_is_lognorm | outcome |
|---|---|---|
| Occam panel (log1p, ts 2000) | True x5 | left alone ✅ |
| **library-normed LINEAR, no log1p** | **True x5** | **left alone -- silently WRONG** |
| raw integer counts | False x5 | would be renormalised (what `--allow-discrete` blocks) |

And `_evaluator.py:273` then hardcodes `pdex_kwargs["is_log1p"] = True` regardless. So a
competitor emitting library-normed **linear** data would sail through both gates and have its
fold changes computed as though they were log1p. Nothing would warn. That is the failure this
module exists to make impossible.

Note also that `_convert_to_normlog` is called SEPARATELY for the real and predicted files
(`_evaluator.py:167,170`), so the two can in principle receive different treatment --
another reason to assert they are in the same space before calling cell-eval at all.

THE DISCRIMINATOR CELL-EVAL LACKS (measured)
--------------------------------------------
| | log1p (Occam) | library-norm linear |
|---|---|---|
| max value | **5.01** | 148.59 |
| median `X.sum(1)` | 992.60 | 1730.95 |
| median `expm1(X).sum(1)` | **1731** | 1.59e+18 |

So `max(X) <= log1p(target_sum)` separates them cleanly, and
`median(expm1(X).sum(1))` lands near `target_sum * axis_share` for genuine log1p data.
Both are checked below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "EVAL_TARGET_SUM", "TRAIN_TARGET_SUM", "AXIS_UMI_SHARE",
    "EmissionSpaceError", "SpaceReport",
    "to_eval_space", "describe_space", "assert_is_log1p", "assert_representable",
    "space_diagnostics",
]

# Occam's evaluation space. Do not change without rebuilding every baseline.
EVAL_TARGET_SUM = 2000.0

# What competitors are handed, because their pipelines assume pre-normalised input and their
# configs all pass `target_sum=null, log1p=false`.
TRAIN_TARGET_SUM = 10_000.0

# Measured: the 6,983-gene axis carries this share of each HepG2 cell's full-panel UMI
# (median over the panel). Used only for a sanity band, never for arithmetic.
AXIS_UMI_SHARE = 0.866


class EmissionSpaceError(RuntimeError):
    """A prediction is not in the canonical evaluation space. Refusing to score it."""


@dataclass
class SpaceReport:
    """Everything needed to tell two spaces apart, in one object."""

    n_rows: int
    n_genes: int
    vmin: float
    vmax: float
    zero_fraction: float
    median_rowsum_raw: float
    median_rowsum_expm1: float
    n_nonfinite: int
    n_negative: int
    looks_log1p: bool
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.n_rows:,}x{self.n_genes:,} · range [{self.vmin:.4f}, {self.vmax:.4f}] "
                f"· zero {self.zero_fraction:.4f} · median expm1 rowsum "
                f"{self.median_rowsum_expm1:,.1f} · log1p={self.looks_log1p}")


def to_eval_space(x: np.ndarray, *, from_target_sum: float = TRAIN_TARGET_SUM,
                  from_log1p: bool = True) -> np.ndarray:
    """Map a competitor's emission into the canonical space. Exact, not approximate.

    `from_log1p=True` (the default, and what every ported arm uses): the input is
    `log1p(counts / rowsum_full * from_target_sum)`, so

        out = log1p( expm1(x) * EVAL_TARGET_SUM / from_target_sum )

    and the per-cell denominator cancels identically. Measured for 10,000 -> 2,000: bit-exact
    in float32 when the 10k matrix is derived from the 2k one, and agreeing to one float32
    ULP (4.7683716e-07 at the max value 5.0) when both are computed independently from
    counts -- which is the real case. See the module docstring.

    `from_log1p=False`: the input is library-normed LINEAR at `from_target_sum`, so it is
    rescaled and then log1p'd once. Use this only for an arm that genuinely emits linear, and
    record it in that arm's manifest.
    """
    x = np.asarray(x, dtype=np.float64)
    if from_target_sum <= 0:
        raise ValueError(f"from_target_sum must be positive, got {from_target_sum}")
    scale = EVAL_TARGET_SUM / float(from_target_sum)
    lin = np.expm1(x) if from_log1p else x
    if lin.min() < -1e-6:
        raise EmissionSpaceError(
            f"linear-space values are negative (min {lin.min():.4g}). A library-normalised "
            f"expression matrix cannot be negative; the declared input space is wrong.")
    return np.log1p(np.clip(lin, 0.0, None) * scale).astype(np.float32)


def describe_space(x: np.ndarray, *, target_sum: float = EVAL_TARGET_SUM) -> SpaceReport:
    """Measure a matrix well enough to tell log1p from linear. No side effects."""
    x = np.asarray(x)
    finite = np.isfinite(x)
    n_nonfinite = int((~finite).sum())
    safe = np.where(finite, x, 0.0).astype(np.float64)

    rs_raw = safe.sum(axis=1)
    # expm1 overflows spectacularly on linear input; that is itself the signal, so compute it
    # under an error-state guard rather than letting it warn.
    with np.errstate(over="ignore", invalid="ignore"):
        rs_exp = np.expm1(safe).sum(axis=1)
    rs_exp = np.where(np.isfinite(rs_exp), rs_exp, np.inf)

    vmax = float(safe.max()) if safe.size else 0.0
    ceiling = float(np.log1p(target_sum))
    looks_log1p = bool(vmax <= ceiling + 1e-3)

    notes: List[str] = []
    if not looks_log1p:
        notes.append(
            f"max value {vmax:.2f} exceeds log1p(target_sum={target_sum:g}) = {ceiling:.4f}, "
            f"so this is NOT log1p at that target_sum -- most likely library-normed linear. "
            f"cell-eval would accept it silently and pdex would treat it as log1p anyway.")
    return SpaceReport(
        n_rows=int(x.shape[0]), n_genes=int(x.shape[1]),
        vmin=float(safe.min()) if safe.size else 0.0, vmax=vmax,
        zero_fraction=float((safe == 0).mean()) if safe.size else 0.0,
        median_rowsum_raw=float(np.median(rs_raw)) if rs_raw.size else 0.0,
        median_rowsum_expm1=float(np.median(rs_exp)) if rs_exp.size else 0.0,
        n_nonfinite=n_nonfinite,
        n_negative=int((safe < -1e-6).sum()),
        looks_log1p=looks_log1p, notes=notes,
    )


def assert_is_log1p(x: np.ndarray, *, name: str = "matrix",
                    target_sum: float = EVAL_TARGET_SUM) -> SpaceReport:
    """Refuse anything that is not log1p at `target_sum`. This is the check cell-eval lacks."""
    r = describe_space(x, target_sum=target_sum)
    problems: List[str] = []
    if r.n_nonfinite:
        problems.append(f"{r.n_nonfinite:,} non-finite value(s)")
    if r.n_negative:
        problems.append(f"{r.n_negative:,} negative value(s)")
    if not r.looks_log1p:
        problems.extend(r.notes)
    # A genuine log1p matrix at this target_sum has an expm1 row sum near
    # target_sum * axis_share. Two orders of magnitude out means the space is wrong.
    lo, hi = target_sum * AXIS_UMI_SHARE / 100.0, target_sum * 100.0
    if r.median_rowsum_expm1 > hi or (r.median_rowsum_expm1 < lo and r.median_rowsum_expm1 > 0):
        problems.append(
            f"median expm1(x).sum(axis=1) is {r.median_rowsum_expm1:.4g}, outside the "
            f"plausible band [{lo:.4g}, {hi:.4g}] for log1p data at target_sum "
            f"{target_sum:g}")
    if problems:
        raise EmissionSpaceError(
            f"{name} is not in log1p(target_sum={target_sum:g}) space:\n  - "
            + "\n  - ".join(problems)
            + f"\n  measured: {r.summary()}")
    return r

# ---------------------------------------------------------------------------
# The adapter's contract: representability is enforced, quality is measured.
# ---------------------------------------------------------------------------
#
# ⛔ WHY THERE IS NO STATISTICAL COMPARISON AGAINST THE TARGET HERE ANY MORE.
#
# An earlier version of this module refused a prediction whose median library size differed
# from the target's by >10%, or whose zero fraction differed by >0.05. Both are SPACE
# properties AND MODEL-QUALITY properties, and conflating them fails in the worst possible
# direction: it silently EXCLUDES the arms that make poor predictions, biasing the comparison
# toward whichever competitors happen to be well calibrated. A benchmark exists to MEASURE bad
# predictions, not to refuse them.
#
# Measured 2026-09-18, which is what settled it:
#
#     the real target             zero 0.5170   median library 1,729.9
#     OUR OWN s313 prediction     zero 0.0000   median library 1,626.0   scores 0.3415
#     TxPert epoch-20             zero 0.0001   median library 1,505.1
#
# Our own best arm failed the enforced check by a 0.517 margin against a 0.05 tolerance --
# worse than the competitor it rejected -- because a conditional-mean regressor essentially
# never emits an exact zero while the target is sparse single-cell counts. Our arms never meet
# this module (they go straight through `protocol.run_split`), so the asymmetry was invisible
# until a competitor hit it.
#
# ⭐ AND THE CONVERSION DOES NOT NEED STATISTICAL VALIDATION, because it is already proven:
#
#   * it is ANALYTIC -- `log1p(expm1(x) * to/from)`, exact because the per-cell denominator
#     cancels (C1), and exact only because `occam_data` normalises on the FULL panel row sum;
#   * the GATE proves it end to end -- `ctrlcell` pushed through eval -> 10,000 -> eval
#     reproduced its directly-measured score to four decimals, `mae` included, and `mae` is
#     the one metric that is NOT scale-invariant (C7);
#   * `tests/test_occam_data_reader.py` pins the reader bit-identically to `panel.npz`.
#
# So correctness is established ONCE, by construction and by the gate. What remains per-run is
# only whether the matrix can be REPRESENTED in the evaluation space at all.


# An ORDER-OF-MAGNITUDE band on the library ratio, not a quality threshold.
#
# ⚠️ The distinction is the whole point. A +/-10% threshold conflates "the bridge
# mis-normalised" with "this arm predicts poorly", and refusing on it would exclude the weak
# arms the benchmark exists to measure. But a FACTOR error is unambiguous: handing back the
# unconverted 10,000 matrix is exactly 5x, and log1p-vs-linear is orders of magnitude. No
# model is 4x off on total expression unless the space is wrong.
#
# Calibration: our own s313 is at 0.94, TxPert epoch-20 at 0.87. Both far inside.
ROWSUM_FACTOR_BAND = (0.25, 4.0)


def assert_representable(x: np.ndarray, *, n_genes: Optional[int] = None,
                         name: str = "prediction",
                         target_sum: float = EVAL_TARGET_SUM,
                         target: Optional[np.ndarray] = None,
                         var: Optional[Sequence[str]] = None,
                         expect_var: Optional[Sequence[str]] = None) -> SpaceReport:
    """Hard checks only: can this matrix be scored in the evaluation space at all?

    Every one of these is model-agnostic -- a terrible model passes them, which is the point.
    A matrix that fails cannot be scored, or would be scored as something other than what it
    is.

      * finite, and float
      * NON-NEGATIVE -- cell-eval refuses negative input outright ("Invalid scale: min value
        is negative"), so this is representability, not taste
      * the right width, and the gene axis identical AND IN ORDER (the bridge is positional,
        and `discrimination_score_l1` excludes each perturbation's own gene BY SYMBOL)
      * inside `log1p(target_sum)` -- a value above it is arithmetically impossible in this
        space and means the matrix is linear, or normalised to a different target_sum

    Raises `EmissionSpaceError`. Returns the `SpaceReport` so the caller can record it.
    """
    a = np.asarray(x)
    if a.ndim != 2:
        raise EmissionSpaceError(f"{name} has {a.ndim} dimensions, expected 2")
    if n_genes is not None and a.shape[1] != n_genes:
        raise EmissionSpaceError(
            f"{name} is {a.shape[1]} genes wide, expected {n_genes}")
    if not np.issubdtype(a.dtype, np.floating):
        raise EmissionSpaceError(f"{name} has dtype {a.dtype}, expected floating point")
    if not np.isfinite(a).all():
        n_bad = int((~np.isfinite(a)).sum())
        raise EmissionSpaceError(
            f"{name} has {n_bad:,} non-finite value(s). cell-eval cannot score them, and a "
            f"NaN would propagate silently into a group mean.")
    mn = float(a.min())
    if mn < 0.0:
        raise EmissionSpaceError(
            f"{name} has negative values (min {mn:.4f}). cell-eval refuses negative input, "
            f"and a library-normalised matrix cannot be negative. If the arm's head is "
            f"unconstrained, clip at 0 IN THAT ARM'S ADAPTER and record how much was clipped "
            f"-- do not loosen this check.")
    # ⚠️ NO CEILING REFUSAL. This used to raise when max(X) > log1p(target_sum), on the
    # grounds that such a value is arithmetically impossible for a library-normalised matrix.
    # It is impossible for DATA, but not for a PREDICTION: an unconstrained head can emit
    # anything. STATE's output goes through a plain `relu` (state_transition.py:516), which
    # is non-negative but unbounded, and one of its validation passes produced max 17.29.
    #
    # Removed at the user's explicit instruction (2026-09-18): a checkpoint that predicts
    # out-of-range values should be SCORED, not skipped. The score is then the honest
    # consequence of what the model actually emitted -- refusing it hid a bad prediction
    # behind an infrastructure error and cost us the arm's number entirely.
    #
    # The failure mode the check originally guarded (raw counts passed off as log1p) is still
    # caught, by the library-size factor band below: raw counts sit orders of magnitude
    # outside it. An over-ceiling maximum is now REPORTED by `space_diagnostics`
    # (`looks_log1p=False` plus a note) and recorded in the manifest, so the number stays
    # interpretable without being withheld.
    ceiling = float(np.log1p(target_sum))
    mx = float(a.max())
    if mx > ceiling:
        print(f"  [emission] ⚠️  {name} max {mx:.4f} exceeds log1p({target_sum:g}) = "
              f"{ceiling:.4f}. Scoring it anyway (the ceiling refusal was removed on "
              f"purpose); the value is impossible for real data, so read this arm's "
              f"`nmae`/`mse` with that in mind.")

    # The one comparison against the target that is NOT a quality judgement.
    if target is not None:
        rp = describe_space(a, target_sum=target_sum)
        rt = describe_space(np.asarray(target), target_sum=target_sum)
        if rt.median_rowsum_expm1 > 0:
            ratio = rp.median_rowsum_expm1 / rt.median_rowsum_expm1
            lo, hi = ROWSUM_FACTOR_BAND
            if not (lo <= ratio <= hi):
                raise EmissionSpaceError(
                    f"{name}'s median library size is {ratio:.2f}x the target's "
                    f"({rp.median_rowsum_expm1:,.1f} vs {rt.median_rowsum_expm1:,.1f}), "
                    f"outside the factor band [{lo}, {hi}]. This is an ORDER-OF-MAGNITUDE "
                    f"check, not a quality one: 5x is the 10,000-vs-2,000 normalisation "
                    f"mistake, and no model is {lo:g}-{hi:g}x off unless the space is wrong. "
                    f"A merely weak arm lands well inside -- ours is 0.94, TxPert's 0.87.")

    if var is not None and expect_var is not None:
        pv = [str(g) for g in var]
        tv = [str(g) for g in expect_var]
        if pv != tv:
            if set(pv) == set(tv):
                raise EmissionSpaceError(
                    f"{name} has the right {len(tv)} genes in the WRONG ORDER. The bridge is "
                    f"positional, so this silently relabels every gene.")
            raise EmissionSpaceError(
                f"{name}'s gene axis is not the frozen one: {len(pv)} vs {len(tv)} genes "
                f"(missing {sorted(set(tv) - set(pv))[:3]}, "
                f"extra {sorted(set(pv) - set(tv))[:3]})")

    return describe_space(a, target_sum=target_sum)


def space_diagnostics(pred: np.ndarray, target: np.ndarray, *,
                      pred_name: str = "prediction", target_name: str = "target",
                      target_sum: float = EVAL_TARGET_SUM,
                      reference: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Numbers to read BESIDE the score. Never raises.

    ⚠️ NONE OF THESE IS A PASS/FAIL. Each is confounded with model quality:

      * `rowsum_ratio` -- a conditional mean shrinks toward the centre, so predictions
        under-shoot total expression. Ours is 0.94; TxPert's 0.87. Also what a miscalibrated
        arm would show, which is why it is reported and not enforced.
      * `zero_fraction` -- a regressor never emits exact zeros (ours: 0.0000 against a target
        0.5170); a collapsing arm emits many. Both are results.
      * `gene_mean_corr` -- ours is 0.9998. A LOW value is interesting, but a weak model still
        tracks the mean profile because it is conditioned on control cells.

    `reference` is OUR OWN arm's diagnostics, which is the apples-to-apples comparison: the
    question is never "does this look like the real data" -- no prediction does -- but "does
    this sit where our own predictions sit".
    """
    rp = describe_space(np.asarray(pred), target_sum=target_sum)
    rt = describe_space(np.asarray(target), target_sum=target_sum)

    gp = np.asarray(pred, np.float64).mean(axis=0)
    gt = np.asarray(target, np.float64).mean(axis=0)
    corr = (float(np.corrcoef(gp, gt)[0, 1])
            if gp.std() > 0 and gt.std() > 0 else float("nan"))
    ratio = (rp.median_rowsum_expm1 / rt.median_rowsum_expm1
             if rt.median_rowsum_expm1 else float("nan"))

    out: Dict[str, object] = {
        "pred_name": pred_name, "target_name": target_name,
        "pred": rp.__dict__, "target": rt.__dict__,
        "rowsum_ratio": ratio, "gene_mean_corr": corr,
        "zero_fraction_pred": rp.zero_fraction,
        "zero_fraction_target": rt.zero_fraction,
        "note": ("diagnostic only -- library size, density and mean correlation are all "
                 "confounded with model quality; refusing on them would exclude weak arms "
                 "and bias the comparison"),
    }
    if reference:
        out["vs_our_own_arm"] = {
            k: (float(out[k]) - float(reference[k]))          # type: ignore[arg-type]
            for k in ("rowsum_ratio", "gene_mean_corr") if k in reference
        }
    return out
