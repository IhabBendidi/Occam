"""THE reader. Every competitor imports this and sees exactly our data, splits and axis.

WHY THIS MODULE EXISTS
----------------------
The four competitor arms were built against a "hooke-tx zarr" dump on a 2,000-gene panel over
380 test perturbations and 4 cell lines. We benchmark on `rhj_fewshot`: 6,983 genes, 611 test
perturbations, 3 cell lines. Rather than convert our data into their format -- which would
mean a second 18 GB copy on disk and a format to keep in sync -- their dataloaders are
pointed at this module, which reads **our** files in place:

    data/eval/pert/{nadig_2024_hepg2,nadig_2024_jurkat,replogle_2022_rpe1_e}.h5ad
    data/benchmarks/rhj_fewshot/{gene_axis.csv,splits.json,exclusions.json,files/*.json}

⚠️ THE NORMALISATION, AND WHY IT IS NOT NEGOTIABLE
--------------------------------------------------
    x = log1p( counts / rowsum_FULL_PANEL * target_sum )      # restricted to the 6,983 axis

`rowsum_FULL_PANEL` is the row sum over the source file's **whole** gene panel (9,624 for
HepG2), taken **before** the axis restriction. That is what `stream_v2.load_chunk_sparse`
does, and therefore what `splithalf/panel.npz` -- the thing competitors are scored against --
actually contains. Measured on that panel: `expm1(X).sum(axis=1)` has median **1731.8**, not
2000, because the axis carries 86.6% of each cell's UMI and the share varies per cell.

Two consequences:

1. **`target_sum` defaults to 10,000, not our 2,000.** Their pipelines assume pre-normalised
   input -- every arm passes `target_sum=null, log1p=false`, because the hooke-tx data
   arrived already `log1p(1e4)`. Our h5ads are raw integer counts, so we do it for them, at
   the value their recipes expect.
2. **The map back to our evaluation space is then EXACT**, because the per-cell denominator
   cancels: `log1p(expm1(x_10k) * 0.2)`. Measured bit-exact in float32. Had this module
   normalised on the 6,983-axis sum instead, a per-cell factor would remain that leaves
   `pearson_delta` untouched while shifting `mae` and moving the DE calls. See
   `emission_space.py` and `docs/rhj_competitors.md` C1.

The arithmetic below reproduces `scripts/build_splithalf_panel.py::_load_axis_restricted`
**dtype for dtype** -- float32 values, a float64 row-sum accumulation cast back to float32
before the divide -- so that `load_occam_adata(target_sum=2000.0)` is *bit-identical* to
`panel.npz`. `tests/test_occam_data_reader.py` asserts exactly that. Changing the dtype
sequence will silently break the equivalence while still "working".

HOW OUR SPLITS MAP ONTO THEIR SPLIT MODEL
-----------------------------------------
Their loaders treat **a dataset with no split entry as wholly training data**, and TxPert
raises if one perturbation straddles two split sides *within* a dataset. Our benchmark is
exactly that shape, so no new machinery is needed:

| dataset | finetune | validation | test |
|---|---|---|---|
| HepG2 | 1,657 perturbations + **all 4,976 controls** | 103 | 621 |
| Jurkat | everything | — | — |
| RPE1 | everything | — | — |

The 724 held-out HepG2 perturbations legitimately train from Jurkat and RPE1 -- that is the
benchmark's design, and it is why `splits.json` has `train ∩ val ≠ {}` at the *name* level.
Within HepG2 the three sets are disjoint, which is what their loaders require.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "CONTROL_LABEL", "DEFAULT_TARGET_SUM", "OCCAM_CELL_LINES",
    "OccamDataError", "OccamBenchmark",
    "load_occam_benchmark", "load_occam_adata", "occam_split_indices",
    "occam_split_perturbations",
]

# Our control label, from occam.vcell.panel.panel_io.CONTROL.
CONTROL_LABEL = "non-targeting"

# What competitors are handed. NOT our evaluation 2,000 -- see the module docstring.
DEFAULT_TARGET_SUM = 10_000.0

# Exact casing matters: it is what lands in obs["cell_type"] and what their configs match on.
OCCAM_CELL_LINES = ("HEPG2", "jurkat", "RPE1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_BENCH = os.path.join(_REPO, "data", "benchmarks", "rhj_fewshot")


class OccamDataError(RuntimeError):
    """The benchmark artifact or a source file is not what it claims to be."""


@dataclass
class OccamBenchmark:
    """The frozen definition, already validated."""

    dir: str
    benchmark_sha256: str
    axis_ensembl: List[str]
    axis_symbols: List[str]
    axis_sha256: str
    split: Dict[str, object]
    excluded_by_file: Dict[str, Set[str]]
    files: Dict[str, Dict[str, object]]          # cell_line -> per-file manifest
    raw: Dict[str, object] = field(default_factory=dict)

    @property
    def n_genes(self) -> int:
        return len(self.axis_ensembl)

    @property
    def val_perts(self) -> Set[str]:
        return set(self.split["val"])

    @property
    def test_perts(self) -> Set[str]:
        return set(self.split["test_unseen_pert"])

    def summary(self) -> str:
        return (f"rhj_fewshot: {self.n_genes:,} genes · val {len(self.val_perts)} · "
                f"test {len(self.test_perts)} · "
                f"{sum(len(v) for v in self.excluded_by_file.values()):,} (file, pert) "
                f"exclusions · lines {sorted(self.files)}")


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_occam_benchmark(bench_dir: str = _DEFAULT_BENCH, *,
                         verify_sources: bool = True) -> OccamBenchmark:
    """Read and validate the frozen artifact. Raises rather than warns, on everything.

    `verify_sources=True` hashes each source h5ad (1.3-2.0 GB each, ~20 s) and refuses on
    drift: the splits are defined over those exact files, so a changed file means the
    competitor would train on different data than the baselines were measured on.
    """
    need = ("benchmark.json", "gene_axis.csv", "splits.json", "exclusions.json")
    for n in need:
        p = os.path.join(bench_dir, n)
        if not os.path.isfile(p):
            raise OccamDataError(
                f"{bench_dir} is not an rhj_fewshot artifact: missing {n}. Build it with "
                f"scripts/build_rhj_fewshot.py.")

    bench_path = os.path.join(bench_dir, "benchmark.json")
    with open(bench_path) as fh:
        raw = json.load(fh)
    got = _sha256(bench_path)

    for rel, want in (raw.get("artifact_sha256") or {}).items():
        have = _sha256(os.path.join(bench_dir, rel))
        if have != want:
            raise OccamDataError(
                f"{rel} has been edited: {have[:16]} != recorded {want[:16]}. The artifact "
                f"is frozen; regenerate it rather than editing it.")

    rows = list(csv.DictReader(open(os.path.join(bench_dir, "gene_axis.csv"))))
    axis_ens = [r["ensembl_gene_id"] for r in rows]
    axis_sym = [r["gene_name"] for r in rows]
    axis_sha = hashlib.sha256("\n".join(axis_ens).encode()).hexdigest()
    if axis_sha != raw["gene_axis"]["sha256"]:
        raise OccamDataError("gene_axis.csv does not match its recorded hash")
    if len(axis_ens) != int(raw["gene_axis"]["n"]):
        raise OccamDataError(
            f"gene_axis.csv has {len(axis_ens)} rows, benchmark.json declares "
            f"{raw['gene_axis']['n']}")
    if len(set(axis_sym)) != len(axis_sym):
        dupes = sorted({s for s in axis_sym if axis_sym.count(s) > 1})
        raise OccamDataError(
            f"duplicate gene SYMBOLS on the axis: {dupes[:5]}. cell-eval's "
            f"discrimination_score_l1 excludes each perturbation's own gene by symbol, so a "
            f"duplicate makes that exclusion ambiguous.")

    with open(os.path.join(bench_dir, "splits.json")) as fh:
        split = json.load(fh)
    with open(os.path.join(bench_dir, "exclusions.json")) as fh:
        ex = json.load(fh)
    excluded = {k: set(v) for k, v in (ex.get("by_file") or {}).items()}

    files: Dict[str, Dict[str, object]] = {}
    fdir = os.path.join(bench_dir, "files")
    for name in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        if not name.endswith(".json"):
            continue
        with open(os.path.join(fdir, name)) as fh:
            rec = json.load(fh)
        line = str(rec["cell_line"])
        if not os.path.isfile(rec["path"]):
            raise OccamDataError(
                f"{rec['name']} is recorded at {rec['path']}, which does not exist. The "
                f"eval bundle is gitignored; populate it with scripts/export_eval_data.py.")
        if verify_sources:
            have = _sha256(rec["path"])
            if have != rec["sha256"]:
                raise OccamDataError(
                    f"{rec['name']} has changed since the benchmark was built "
                    f"({have[:16]} != {rec['sha256'][:16]}). The frozen splits refer to rows "
                    f"in the old file; training on the new one would silently benchmark "
                    f"something else.")
        files[line] = rec

    missing = [c for c in OCCAM_CELL_LINES if c not in files]
    if missing:
        raise OccamDataError(f"no per-file manifest for cell line(s) {missing}; "
                             f"found {sorted(files)}")

    return OccamBenchmark(
        dir=bench_dir, benchmark_sha256=got,
        axis_ensembl=axis_ens, axis_symbols=axis_sym, axis_sha256=axis_sha,
        split=split, excluded_by_file=excluded, files=files, raw=raw,
    )


def _read_one(path: str, axis_ensembl: Sequence[str], target_sum: float, log1p: bool,
              pert_col: str, batch_col: str, chunk: int = 8192,
              max_cells: Optional[int] = None,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One h5ad -> (X on the axis, perturbation labels, batch labels).

    ⚠️ The dtype sequence here is load-bearing and reproduces
    `build_splithalf_panel.py::_load_axis_restricted` exactly: values read as float32, the
    row sum accumulated in float64 over the FULL panel, then cast back to float32 before the
    divide. That is what makes `target_sum=2000.0` bit-identical to `panel.npz`.
    """
    import h5py

    from occam.h5ad_columns import read_column

    with h5py.File(path, "r") as f:
        ens = read_column(f["var"], "ensembl_gene_id")
        if ens is None:
            raise OccamDataError(f"{path}: var has no ensembl_gene_id column")
        pos = {g: i for i, g in enumerate(ens)}
        absent = [g for g in axis_ensembl if g not in pos]
        if absent:
            raise OccamDataError(
                f"{len(absent)} axis gene(s) are absent from {os.path.basename(path)} "
                f"(e.g. {absent[:5]}). The axis is the 3-way intersection, so this means the "
                f"file changed under the benchmark.")
        keep = np.array([pos[g] for g in axis_ensembl], dtype=np.int64)

        labels = np.asarray(read_column(f["obs"], pert_col), dtype=str)
        batches = np.asarray(read_column(f["obs"], batch_col), dtype=str)

        X = f["X"]
        if not (isinstance(X, h5py.Group) and "indptr" in X):
            raise OccamDataError(f"{path}: X is not CSR; this reader reads sparse rows only")
        n_obs = len(labels) if max_cells is None else min(int(max_cells), len(labels))
        n_vars = len(ens)
        indptr = np.asarray(X["indptr"][:], dtype=np.int64)
        data, indices = X["data"], X["indices"]

        col_of = np.full(n_vars, -1, dtype=np.int64)
        col_of[keep] = np.arange(len(keep))

        out = np.zeros((n_obs, len(keep)), dtype=np.float32)
        for s in range(0, n_obs, chunk):
            e = min(s + chunk, n_obs)
            lo, hi = int(indptr[s]), int(indptr[e])
            v = np.asarray(data[lo:hi], dtype=np.float32)
            c = np.asarray(indices[lo:hi], dtype=np.int64)
            lens = np.diff(indptr[s:e + 1])
            row = np.repeat(np.arange(e - s, dtype=np.int64), lens)
            # the row sum over the FULL panel, before the axis restriction
            rs = np.bincount(row, weights=v.astype(np.float64), minlength=e - s)
            rs = np.where(rs > 0, rs, 1.0)
            tgt = col_of[c]
            m = tgt >= 0
            vals = v[m] / rs[row[m]].astype(np.float32) * float(target_sum)
            out[s + row[m], tgt[m]] = np.log1p(vals) if log1p else vals
    return out, labels[:n_obs], batches[:n_obs]


def load_occam_adata(bench: OccamBenchmark,
                     cell_lines: Sequence[str] = OCCAM_CELL_LINES, *,
                     target_sum: float = DEFAULT_TARGET_SUM,
                     log1p: bool = True,
                     control_label: str = "empty",
                     pert_col_out: str = "gene_pert",
                     max_cells_per_line: Optional[int] = None):
    """Our three h5ads as one AnnData, on the frozen axis, in axis order.

    `target_sum` defaults to 10,000 (what their pipelines expect), NOT our evaluation 2,000.
    Pass 2,000 to reproduce `panel.npz` bit-identically.

    `control_label` is what controls are called in the emitted obs. Their stack uses
    `"empty"`; pass `CONTROL_LABEL` for our `"non-targeting"`.

    obs columns emitted, so each arm finds the names it expects:
      `<pert_col_out>` (default `gene_pert`), `pert`, `target_gene`, `gene`,
      `negative_control` (bool), `is_control` (bool), `cell_type`, `cell_line`,
      `batch_center`, `dataset_batch_num`, `dataset_name`.
    var carries BOTH `ensembl_gene_id` and `gene_symbol`, index = **symbols**.
    """
    import anndata as ad
    import pandas as pd

    unknown = [c for c in cell_lines if c not in bench.files]
    if unknown:
        raise OccamDataError(f"unknown cell line(s) {unknown}; have {sorted(bench.files)}")
    if target_sum <= 0:
        raise OccamDataError(f"target_sum must be positive, got {target_sum}")

    blocks, obs_rows = [], []
    for line in cell_lines:
        rec = bench.files[line]
        X, perts, batches = _read_one(
            rec["path"], bench.axis_ensembl, target_sum, log1p,
            pert_col=str(rec.get("pert_col", "gene_name")),
            batch_col=str(rec.get("batch_col", "dataset_batch_num")),
            max_cells=max_cells_per_line)
        is_ctrl = perts == CONTROL_LABEL
        out_perts = np.where(is_ctrl, control_label, perts)
        blocks.append(X)
        obs_rows.append(pd.DataFrame({
            pert_col_out: out_perts,
            "pert": out_perts,
            "target_gene": out_perts,
            "gene": out_perts,
            "negative_control": is_ctrl,
            "is_control": is_ctrl,
            "cell_type": np.full(len(perts), line),
            "cell_line": np.full(len(perts), line),
            # prefixed for global uniqueness across files, as _load_file_metadata does
            "batch_center": np.char.add(f"{line}__", batches),
            "dataset_batch_num": batches,
            "dataset_name": np.full(len(perts), str(rec["name"]).replace(".h5ad", "")),
        }))
        print(f"  [occam-data] {rec['name']}: {X.shape[0]:,} x {X.shape[1]:,} "
              f"({int(is_ctrl.sum()):,} controls)", flush=True)

    obs = pd.concat(obs_rows, ignore_index=True)
    obs.index = obs.index.astype(str)
    var = pd.DataFrame(
        {"ensembl_gene_id": bench.axis_ensembl, "gene_symbol": bench.axis_symbols},
        index=pd.Index(bench.axis_symbols, name=None))

    adata = ad.AnnData(X=np.concatenate(blocks, axis=0), obs=obs, var=var)
    adata.uns["occam"] = {
        "benchmark_id": "rhj_fewshot",
        "benchmark_sha256": bench.benchmark_sha256,
        "gene_axis_sha256": bench.axis_sha256,
        "n_axis_genes": bench.n_genes,
        "target_sum": float(target_sum),
        "log1p": bool(log1p),
        "normalised_on": "FULL source gene panel row sum, before axis restriction",
        "control_label": control_label,
        "cell_lines": list(cell_lines),
    }
    return adata


def occam_split_indices(bench: OccamBenchmark, adata) -> Dict[str, np.ndarray]:
    """`{"finetune","validation","test"}` as row indices into `adata.obs`.

    HepG2 carries the whole split; Jurkat and RPE1 are wholly `finetune`, which is how our
    per-(file, perturbation) exclusion is expressed in their vocabulary. Every invariant is
    asserted, because a silent split error is the one bug that would invalidate the entire
    comparison while every number still looked plausible.
    """
    line = np.asarray(adata.obs["cell_line"], dtype=str)
    perts = np.asarray(adata.obs["target_gene"], dtype=str)
    is_ctrl = np.asarray(adata.obs["is_control"], dtype=bool)

    val, test = bench.val_perts, bench.test_perts
    hep = line == "HEPG2"

    idx = {
        "validation": np.flatnonzero(hep & ~is_ctrl & np.isin(perts, list(val))),
        "test": np.flatnonzero(hep & ~is_ctrl & np.isin(perts, list(test))),
    }
    held = val | test
    idx["finetune"] = np.flatnonzero(~(hep & ~is_ctrl & np.isin(perts, list(held))))

    n = adata.n_obs
    seen = np.concatenate([idx["finetune"], idx["validation"], idx["test"]])
    if len(np.unique(seen)) != len(seen):
        raise OccamDataError("split index sets overlap; they must be pairwise disjoint")
    if len(seen) != n:
        raise OccamDataError(
            f"the three splits cover {len(seen):,} of {n:,} rows; every row must be in "
            f"exactly one")
    if not np.all(is_ctrl[idx["finetune"]] | True):     # structural: controls are in finetune
        raise OccamDataError("unreachable")
    if is_ctrl[idx["validation"]].any() or is_ctrl[idx["test"]].any():
        raise OccamDataError(
            "a control cell landed in validation or test. Controls are the paired basal pool "
            "and must all stay in finetune.")
    n_ctrl_ft = int(is_ctrl[idx["finetune"]].sum())
    if n_ctrl_ft != int(is_ctrl.sum()):
        raise OccamDataError(
            f"only {n_ctrl_ft:,} of {int(is_ctrl.sum()):,} controls are in finetune")

    for name, want in (("validation", val), ("test", test)):
        got = set(perts[idx[name]].tolist())
        if got != want:
            raise OccamDataError(
                f"{name} covers {len(got)} perturbations, expected exactly the "
                f"{len(want)} in splits.json (missing "
                f"{sorted(want - got)[:5]}, extra {sorted(got - want)[:5]})")

    # Jurkat and RPE1 contribute nothing but finetune -- the positive half of the exclusion
    for other in ("jurkat", "RPE1"):
        rows = np.flatnonzero(line == other)
        if rows.size and not np.isin(rows, idx["finetune"]).all():
            raise OccamDataError(f"{other} rows leaked out of finetune; it must be wholly "
                                 f"training data")
    print(f"  [occam-data] splits: finetune {len(idx['finetune']):,} "
          f"(incl. {n_ctrl_ft:,} controls) · validation {len(idx['validation']):,} "
          f"· test {len(idx['test']):,}")
    return idx


def occam_split_perturbations(bench: OccamBenchmark, adata) -> Dict[str, List[str]]:
    """The same split as perturbation-NAME sets, which TxPert's loader needs.

    ⚠️ `train` deliberately includes the held-out symbols, because they legitimately train
    from Jurkat and RPE1. Within HepG2 the three sets are disjoint, which is what TxPert's
    straddle check actually requires.
    """
    perts = np.asarray(adata.obs["target_gene"], dtype=str)
    is_ctrl = np.asarray(adata.obs["is_control"], dtype=bool)
    present = set(perts[~is_ctrl].tolist())
    val = sorted(bench.val_perts & present)
    test = sorted(bench.test_perts & present)
    train = sorted(present - set())          # every non-control name trains somewhere
    return {"train": train, "validation": val, "test": test}
