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
10k matrix computed independently of the 2k panel. `assert_same_space` passes on it. One ULP
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
    "to_eval_space", "describe_space", "assert_is_log1p", "assert_same_space",
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


# How far BELOW the target's library size a prediction may sit. Measured: our own arm -6.0%,
# TxPert -13.0%. A conditional mean shrinks toward the centre, so under-shooting is expected;
# 25% leaves room for a weaker arm without admitting an actual normalisation error.
UNDERSHOOT_TOL = 0.25


def assert_same_space(pred: np.ndarray, target: np.ndarray, *,
                      pred_name: str = "prediction", target_name: str = "target",
                      target_sum: float = EVAL_TARGET_SUM,
                      rowsum_tol: float = 0.10,
                      gene_mean_corr_min: float = 0.95,
                      zero_fraction_tol: float = 0.05,
                      pred_var: Optional[Sequence[str]] = None,
                      target_var: Optional[Sequence[str]] = None,
                      ) -> Dict[str, object]:
    """Refuse to score a prediction that is not in the target's space.

    Run this before EVERY cell-eval call, for competitors and for Occam alike. The tolerances
    are the diagnostics that caught this repo's own emission problems: a `rectified_sigmoid`
    head that left 83.6% of genes constant across cells, and integer re-emission that drove
    the zero fraction to 0.907 against a target's 0.606.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape[1] != target.shape[1]:
        raise EmissionSpaceError(
            f"gene-axis width differs: {pred_name} {pred.shape[1]} vs {target_name} "
            f"{target.shape[1]}. Both must be the frozen 6,983-gene axis.")

    if pred_var is not None and target_var is not None:
        pv, tv = list(map(str, pred_var)), list(map(str, target_var))
        if pv != tv:
            first = next((i for i, (a, b) in enumerate(zip(pv, tv)) if a != b), None)
            raise EmissionSpaceError(
                f"var index differs between {pred_name} and {target_name} "
                f"(first mismatch at position {first}: {pv[first]!r} vs {tv[first]!r}). "
                f"Gene ORDER is part of the contract, and cell-eval's "
                f"discrimination_score_l1 excludes each perturbation's own gene BY SYMBOL.")

    rp = assert_is_log1p(pred, name=pred_name, target_sum=target_sum)
    rt = assert_is_log1p(target, name=target_name, target_sum=target_sum)

    problems: List[str] = []

    # ⚠️ THE LIBRARY-SIZE TOLERANCE IS ASYMMETRIC AND DELIBERATELY LOOSE BELOW 1.0.
    # Predictions systematically under-shoot total expression -- our own arm by -6.0%, TxPert
    # by -13.0% -- because a conditional mean shrinks toward the centre. That is a model
    # property `mae` should MEASURE, not a space error to refuse. An over-shoot, by contrast,
    # has no benign explanation at this magnitude and stays tight.
    ratio = (rp.median_rowsum_expm1 / rt.median_rowsum_expm1
             if rt.median_rowsum_expm1 else float("inf"))
    lo = 1.0 - max(rowsum_tol, UNDERSHOOT_TOL)
    hi = 1.0 + rowsum_tol
    if not (lo <= ratio <= hi):
        problems.append(
            f"median library size differs by {(ratio - 1) * 100:+.1f}% "
            f"({rp.median_rowsum_expm1:,.1f} vs {rt.median_rowsum_expm1:,.1f}); "
            f"allowed band is {(lo - 1) * 100:+.0f}%..{(hi - 1) * 100:+.0f}%. mae and the DE calls are NOT "
            f"scale-invariant, so this would corrupt them while leaving pearson_delta "
            f"looking fine.")

    gp = np.asarray(pred, np.float64).mean(axis=0)
    gt = np.asarray(target, np.float64).mean(axis=0)
    corr = float("nan")
    if gp.std() > 0 and gt.std() > 0:
        corr = float(np.corrcoef(gp, gt)[0, 1])
        if corr < gene_mean_corr_min:
            problems.append(
                f"per-gene mean correlation is {corr:.4f} < {gene_mean_corr_min}; the two "
                f"matrices do not describe the same genes in the same units")
    else:
        problems.append(f"a per-gene mean vector is constant "
                        f"(pred std {gp.std():.3g}, target std {gt.std():.3g})")

    # ⚠️ ZERO FRACTION IS REPORTED, NOT ENFORCED, WHEN COMPARING A PREDICTION TO A TARGET.
    #
    # MEASURED 2026-09-18 and it overturned the original design. A conditional-mean regressor
    # essentially never emits an exact zero, while the target is sparse single-cell counts:
    #
    #     the real target            zero 0.5170   median library 1,729.9
    #     OUR OWN s313 prediction    zero 0.0000   median library 1,626.0  (-6.0%)
    #     TxPert epoch-20            zero 0.0001   median library 1,505.1  (-13.0%)
    #
    # So the enforced version would have REFUSED our own best arm -- the one scoring 0.3415 --
    # by a 0.517 margin against a 0.05 tolerance. Our arms never meet this function (they go
    # straight through `protocol.run_split`), so enforcing it on competitors alone is an
    # asymmetry that rejects them for a property our own model has more extremely.
    #
    # The check it was built for -- integer re-emission driving the zero fraction to 0.907
    # against a target 0.606 -- is still visible in the report, and it is the WRONG DIRECTION
    # from this: too many zeros, not too few. That asymmetry is what `zero_fraction_note`
    # records.
    dz = abs(rp.zero_fraction - rt.zero_fraction)
    zero_note = "within tolerance"
    if dz > zero_fraction_tol:
        if rp.zero_fraction > rt.zero_fraction:
            # MORE zeros than the target: the integer-re-emission signature. Still fatal.
            problems.append(
                f"zero fraction is {rp.zero_fraction:.4f} against a target "
                f"{rt.zero_fraction:.4f} -- the prediction is SPARSER than the real data, "
                f"which is the signature of integer re-emission or a thresholded head. "
                f"(A dense prediction is expected and is not an error; an artificially "
                f"sparse one is.)")
        else:
            zero_note = (
                f"prediction is DENSER than the target ({rp.zero_fraction:.4f} vs "
                f"{rt.zero_fraction:.4f}, delta {dz:.4f}); expected for a conditional-mean "
                f"regressor and NOT enforced -- our own arm is at 0.0000")

    if problems:
        raise EmissionSpaceError(
            f"{pred_name} and {target_name} are not in the same space:\n  - "
            + "\n  - ".join(problems)
            + f"\n  {pred_name}: {rp.summary()}\n  {target_name}: {rt.summary()}")

    return {
        "pred": rp.__dict__, "target": rt.__dict__,
        "rowsum_ratio": ratio, "gene_mean_corr": corr,
        "zero_fraction_delta": dz, "zero_fraction_note": zero_note,
        "target_sum": target_sum,
        "verdict": "same space",
    }
