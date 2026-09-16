"""Load a frozen benchmark definition and hand the trainer what it needs.

The artifact (`data/benchmarks/<id>/`) is the benchmark; this module is the only thing that
reads it. It returns the gene axis, the per-(file, perturbation) training exclusions and the
split, and it **verifies the sources have not moved under the definition** before training
starts -- data drifting beneath a frozen split is the one failure that produces plausible,
wrong numbers.

Built by `scripts/build_rhj_fewshot.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

__all__ = ["Benchmark", "load_benchmark"]


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Benchmark:
    """A frozen benchmark definition, already validated."""

    benchmark_id: str
    dir: str
    sha256: str                      # of benchmark.json
    axis_ensembl: List[str]          # the scored/shown gene axis, in order
    axis_sha256: str
    axis_symbols: List[str]
    split: Dict[str, object]         # splits.json, verbatim
    exclude_file_perts: Dict[str, Set[str]]
    eval_cell_lines: List[str]
    raw: Dict[str, object] = field(default_factory=dict)

    @property
    def axis_set(self) -> Set[str]:
        return set(self.axis_ensembl)

    def summary(self) -> str:
        n_ex = sum(len(v) for v in self.exclude_file_perts.values())
        return (f"{self.benchmark_id}: {len(self.axis_ensembl):,} genes · "
                f"train {len(self.split.get('train', ())):,} · "
                f"val {len(self.split.get('val', ())):,} · "
                f"test {len(self.split.get('test_unseen_pert', ())):,} · "
                f"{n_ex:,} perturbation(s) excluded from training · "
                f"eval lines {self.eval_cell_lines}")


def load_benchmark(
    bench_dir: str,
    *,
    expect_id: str = "",
    expect_sha256: str = "",
    data_paths: Optional[List[str]] = None,
    encoder_gene_names: Optional[List[str]] = None,
    verify_sources: bool = True,
) -> Benchmark:
    """Read and validate a benchmark definition.

    Every check here is a refusal rather than a warning, because each failure mode it covers
    produces a plausible number rather than an error:

    * a different `benchmark.json` hash than the config declares -- the definition moved;
    * a source file whose sha256 no longer matches -- the data moved under the split;
    * an axis gene outside the encoder vocabulary -- a column that can never be written;
    * an exclusion naming a file that is not being trained on (checked downstream, in
      `prepare_virtual_cell_data`, where the loaded file list is known).
    """
    bench_dir = str(bench_dir)
    need = ("benchmark.json", "gene_axis.csv", "splits.json", "exclusions.json")
    missing = [n for n in need if not os.path.isfile(os.path.join(bench_dir, n))]
    if missing:
        raise FileNotFoundError(
            f"{bench_dir} is not a benchmark definition (missing {missing}). "
            f"Build it with scripts/build_rhj_fewshot.py.")

    bench_path = os.path.join(bench_dir, "benchmark.json")
    with open(bench_path) as fh:
        raw = json.load(fh)
    got_sha = _sha256(bench_path)

    if expect_id and raw.get("benchmark_id") != expect_id:
        raise ValueError(f"benchmark id mismatch: config says {expect_id!r}, "
                         f"artifact says {raw.get('benchmark_id')!r}")
    if expect_sha256 and got_sha != expect_sha256:
        raise ValueError(
            f"benchmark.json hash mismatch.\n  config declares {expect_sha256}\n"
            f"  artifact is     {got_sha}\n"
            f"The definition changed. That may be intended, but it redefines the benchmark, "
            f"so update the config deliberately rather than letting it drift.")

    # the artifact's own recorded hashes
    for rel, want in (raw.get("artifact_sha256") or {}).items():
        have = _sha256(os.path.join(bench_dir, rel))
        if have != want:
            raise ValueError(f"{rel} has been edited since the benchmark was built "
                             f"({have[:16]} != {want[:16]})")

    rows = open(os.path.join(bench_dir, "gene_axis.csv")).read().splitlines()[1:]
    axis = [r.split(",")[0] for r in rows if r]
    symbols = [r.split(",")[1] for r in rows if r]
    axis_sha = hashlib.sha256("\n".join(axis).encode()).hexdigest()
    if axis_sha != raw["gene_axis"]["sha256"]:
        raise ValueError("gene_axis.csv does not match its recorded hash")

    if encoder_gene_names is not None:
        outside = sorted(set(axis) - set(encoder_gene_names))
        if outside:
            raise ValueError(
                f"{len(outside)} axis gene(s) are not in the encoder vocabulary "
                f"(e.g. {outside[:5]}). Those columns could never be written, so the axis "
                f"and this checkpoint do not belong together.")

    with open(os.path.join(bench_dir, "splits.json")) as fh:
        split = json.load(fh)
    with open(os.path.join(bench_dir, "exclusions.json")) as fh:
        ex = json.load(fh)

    exclude = {k: set(v) for k, v in (ex.get("by_file") or {}).items()}
    eval_lines = sorted({cl for v in (ex.get("eval_cell_lines") or {}).values() for cl in v})

    if verify_sources and data_paths:
        recorded = raw.get("sources") or {}
        for p in data_paths:
            rec = recorded.get(Path(p).name)
            if rec is None:
                # Training on a file the benchmark does not know about is allowed -- extra
                # corpora are a legitimate experiment -- but it must be visible.
                print(f"  [bench] NOTE {Path(p).name} is not part of "
                      f"{raw.get('benchmark_id')}; its perturbations are unconstrained")
                continue
            have = _sha256(p)
            if have != rec["sha256"]:
                raise ValueError(
                    f"{Path(p).name} has changed since the benchmark was built "
                    f"({have[:16]} != {rec['sha256'][:16]}). The frozen split refers to rows "
                    f"in the old file; training on the new one would silently score a "
                    f"different benchmark.")

    return Benchmark(
        benchmark_id=str(raw.get("benchmark_id", "")),
        dir=bench_dir, sha256=got_sha,
        axis_ensembl=axis, axis_sha256=axis_sha, axis_symbols=symbols,
        split=split, exclude_file_perts=exclude, eval_cell_lines=eval_lines, raw=raw,
    )
