"""Streaming v2 — sparse, multi-worker, GPU-densified single-cell streaming.

Phase 2 of `VCCv2/EXECUTION_PLAN_2026-09-09.md`. Same semantics as
`VirtualCellStreamingDataset` (one control per perturbed cell, exact/paired identity rows,
per-epoch caps, source interleave, H1 identity extras) with three changes that remove the
measured bottlenecks of the 2026-09-09 audit:

1. **No dense control pool.** The frozen encoder is applied ONCE to every control cell during
   a chunk-wise scan, and only the 768-d embeddings are kept (859k x 768 x fp32 = 2.6 GB,
   resident on the GPU). The old path held 859k x 44,728 fp32 = 154 GB in RAM and built it
   with 2.6 M per-row HDF5 reads (5.4 h). Cached to disk, keyed on file mtime + encoder hash.
2. **Workers yield CSR pieces, the GPU densifies.** Rows leave the loader as (vals, cols in
   encoder space, rowptr, rowsum); the training process scatters them into a dense tensor ON
   THE GPU, normalises, log1p's. No 44,728-wide tensor is ever built on the CPU, and the three
   pageable H2D copies per batch (~0.8 GB) become one pinned copy of ~50 MB.
3. **Identity rows need no full-width control input.** Their control IS the target, so the
   trainer uses `encoder(target)` (which the latent loss already computes) as the control
   embedding. This removes the only reason `control_identity_mode=exact` needed raw controls.

Equivalence to the old path is float-tolerance, not bit-exact: row sums are accumulated in a
different order (see `scripts/stream_v2_equivalence.py`).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import get_worker_info

from occam.vcell.dataset import (FileMetadata, VirtualCellStreamingDataset, _stable_ctrl_index,
                                  _unwrap_indptr, resolve_pert_indices)


def rss_gb(include_children: bool = False) -> float:
    """Resident set size of this process (and, optionally, its live children) in GB."""
    def _rss(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1])
        except OSError:
            return 0
        return 0
    def _pss(pid: int) -> int:
        """Proportional set size: shared (copy-on-write) pages are divided among sharers, so
        summing PSS over parent+children gives the true footprint. RSS would double-count."""
        try:
            with open(f"/proc/{pid}/smaps_rollup") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        return int(line.split()[1])
        except OSError:
            return _rss(pid)
        return _rss(pid)
    if not include_children:
        return _rss(os.getpid()) / 1e6
    total = _pss(os.getpid())
    try:
        import psutil  # optional
        for c in psutil.Process(os.getpid()).children(recursive=True):
            total += _pss(c.pid)
    except Exception:  # noqa: BLE001
        pass
    return total / 1e6


# ----------------------------------------------------------------------------- CSR chunk I/O
def make_col_map(n_vars: int, src_indices: np.ndarray, dst_indices: np.ndarray) -> np.ndarray:
    """file gene position -> encoder column, -1 where the gene is not in the encoder vocab."""
    m = np.full(n_vars, -1, dtype=np.int32)
    m[np.asarray(src_indices)] = np.asarray(dst_indices, dtype=np.int32)
    return m


def load_chunk_sparse(path: str, start: int, end: int, n_vars: int, keep_local: np.ndarray,
                      col_map: np.ndarray, dense_buf: Optional[np.ndarray] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows [start:end) of an h5ad `X`, restricted to `keep_local`, as CSR in ENCODER space.

    Returns (vals float32, cols int32 encoder columns, rowptr int64 (n_kept+1), rowsum float32).
    `rowsum` is the per-row total over the FILE's full gene panel BEFORE alignment, which is
    what the old dense path normalised by (`_load_expression_chunk`: `raw.sum(axis=1)` over
    `n_vars`). Genes absent from the encoder vocabulary are dropped AFTER the row sum.
    """
    keep_local = np.asarray(keep_local, dtype=np.int64)
    n_rows = end - start
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group) and "indptr" in X:
            ip = _unwrap_indptr(np.asarray(X["indptr"][start:end + 1], dtype=np.int64))
            if len(ip) != n_rows + 1:
                raise RuntimeError(f"CSR indptr slice returned {len(ip)} entries for rows "
                                   f"[{start}:{end}] in {path}")
            s0, e0 = int(ip[0]), int(ip[-1])
            span = e0 - s0
            if span > n_rows * n_vars:
                raise RuntimeError(f"CSR span {span:,} exceeds {n_rows * n_vars:,} for rows "
                                   f"[{start}:{end}] of {path}: indptr corrupt/misaligned")
            ip = ip - s0
            if span > 0:
                cols_all = np.asarray(X["indices"][s0:e0])
                vals_all = np.asarray(X["data"][s0:e0], dtype=np.float32)
                if len(cols_all) != span or len(vals_all) != span:
                    raise RuntimeError(f"CSR short read in {path} rows [{start}:{end}]")
            else:
                cols_all = np.zeros(0, dtype=np.int64)
                vals_all = np.zeros(0, dtype=np.float32)
            # row sums over the full panel (float64 accumulation, cast to float32)
            cs = np.concatenate([[0.0], np.cumsum(vals_all, dtype=np.float64)])
            rowsum_all = (cs[ip[1:]] - cs[ip[:-1]]).astype(np.float32)
            lens = np.diff(ip)
            lens_k = lens[keep_local]
            starts_k = ip[:-1][keep_local]
            total = int(lens_k.sum())
            if total > 0:
                offs = np.concatenate([[0], np.cumsum(lens_k)[:-1]])
                idx = np.repeat(starts_k - offs, lens_k) + np.arange(total, dtype=np.int64)
                cols = cols_all[idx]
                vals = vals_all[idx]
            else:
                cols = np.zeros(0, dtype=np.int64)
                vals = np.zeros(0, dtype=np.float32)
            rowptr = np.concatenate([[0], np.cumsum(lens_k)]).astype(np.int64)
            rowsum = rowsum_all[keep_local]
        else:
            if dense_buf is not None and dense_buf.shape[0] >= n_rows and dense_buf.shape[1] == n_vars:
                block = dense_buf[:n_rows]
                X.read_direct(block, np.s_[start:end, :])
            else:
                block = np.asarray(X[start:end]).astype(np.float32, copy=False)
            sub = np.ascontiguousarray(block[keep_local], dtype=np.float32)
            rowsum = sub.sum(axis=1, dtype=np.float32)
            r, c = np.nonzero(sub)
            vals = sub[r, c].astype(np.float32)
            cols = c.astype(np.int64)
            rowptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=len(keep_local)))]).astype(np.int64)
    # remap to encoder columns, dropping genes outside the vocabulary
    mc = col_map[cols]
    keep = mc >= 0
    if not bool(keep.all()):
        nk = len(keep_local)
        row_of = np.repeat(np.arange(nk, dtype=np.int64), np.diff(rowptr))[keep]
        rowptr = np.concatenate([[0], np.cumsum(np.bincount(row_of, minlength=nk))]).astype(np.int64)
        vals = vals[keep]
        mc = mc[keep]
    return vals.astype(np.float32, copy=False), mc.astype(np.int32, copy=False), rowptr, rowsum.astype(np.float32, copy=False)


def densify_csr(vals: torch.Tensor, cols: torch.Tensor, rowptr: torch.Tensor, rowsum: torch.Tensor,
                num_genes: int, target_sum: float, skip_normalize: bool, skip_log1p: bool,
                ) -> torch.Tensor:
    """CSR pieces (already on the target device) -> dense (B, num_genes) log1p-normalised."""
    B = rowptr.numel() - 1
    lens = rowptr[1:] - rowptr[:-1]
    row_of = torch.repeat_interleave(torch.arange(B, device=vals.device), lens)
    v = vals.float()
    # rows with rowsum < 0 are ALREADY transformed (log1p-normalised dense rows converted to
    # CSR, e.g. the H1 identity extras) and must pass through untouched
    pre = (rowsum < 0)[row_of]
    raw = v
    if not skip_normalize:
        rs = rowsum.float()
        rs = torch.where(rs > 0, rs, torch.ones_like(rs))
        v = v / rs[row_of] * float(target_sum)
    if not skip_log1p:
        v = torch.log1p(v)
    v = torch.where(pre, raw, v)
    out = torch.zeros(B, num_genes, device=vals.device, dtype=torch.float32)
    out.index_put_((row_of, cols.long()), v)
    return out


# ----------------------------------------------------------------------------- batches
@dataclass
class SparseBatch:
    """One training batch, target rows as CSR in encoder space, controls as pool indices."""
    vals: torch.Tensor          # (nnz,) float32 raw values
    cols: torch.Tensor          # (nnz,) int32 encoder columns
    rowptr: torch.Tensor        # (B+1,) int64
    rowsum: torch.Tensor        # (B,) float32 full-panel row totals
    ctrl_idx: torch.Tensor      # (B,) or (B, k) int64 into the global control-embedding pool; -1 = identity row
    mask_ids: torch.Tensor      # (B,) int64 into the dataset's mask table
    group_id: torch.Tensor      # (B,) int64 into the group-mean table; -1 = control/identity row
    pert_indices: List[List[int]]
    pert_names: List[str]
    cell_types: List[str]

    @property
    def n_rows(self) -> int:
        return int(self.rowptr.numel() - 1)

    def pin_memory(self):
        return SparseBatch(self.vals.pin_memory(), self.cols.pin_memory(), self.rowptr.pin_memory(),
                           self.rowsum.pin_memory(), self.ctrl_idx.pin_memory(), self.mask_ids.pin_memory(),
                           self.group_id.pin_memory(), self.pert_indices, self.pert_names, self.cell_types)


def _csr_rows_subset(vals, cols, rowptr, order: np.ndarray):
    """Reorder/select CSR rows by `order`, returning new (vals, cols, rowptr)."""
    lens = np.diff(rowptr)[order]
    starts = rowptr[:-1][order]
    total = int(lens.sum())
    if total:
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        idx = np.repeat(starts - offs, lens) + np.arange(total, dtype=np.int64)
        v, c = vals[idx], cols[idx]
    else:
        v, c = vals[:0], cols[:0]
    rp = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return v, c, rp


# ----------------------------------------------------------------------------- dataset
class StreamingDatasetV2(VirtualCellStreamingDataset):
    """Drop-in for `VirtualCellStreamingDataset` that yields `SparseBatch`.

    Extra args
    ----------
    pool_index : {cell_line: (offset, n)} into the global control-embedding pool. Every cell
        line that will be streamed must resolve here (fallback groups are resolved by the
        caller into this dict).
    mask_table : (n_masks, num_genes) bool; `mask_id_of_meta[mi]` maps a file to its row.
    extras_csr : optional H1 identity rows as (vals, cols, rowptr, rowsum, ctrl_idx, mask_id)
        — exact rows have ctrl_idx -1, paired rows carry a pool index.
    interleave : "roundrobin" (old behaviour — the epoch tail is single-source, F162) or
        "proportional" (draw the next chunk from each file with probability proportional to
        its remaining chunks, so batches stay source-mixed to the end).
    """

    def __init__(self, *args, pool_index: Dict[str, Tuple[int, int]], mask_table: torch.Tensor,
                 mask_id_of_meta: Dict[int, int], extras_csr=None, interleave: str = "roundrobin",
                 pert_snr: Optional[Dict[Tuple[str, str, str], float]] = None,
                 min_pert_snr: Optional[float] = None, keep_perts: Optional[set] = None,
                 max_pert_snr: Optional[float] = None,
                 group_index: Optional[Dict[Tuple[str, str, str], int]] = None,
                 ctrl_avg_k: int = 1,
                 **kwargs):
        kwargs.setdefault("extra_samples", None)
        super().__init__(*args, **kwargs)
        self._read_buf = None                     # v2 never builds a dense chunk in RAM
        self.pool_index = pool_index
        self.mask_table = mask_table
        self.mask_id_of_meta = mask_id_of_meta
        self.extras_csr = extras_csr
        self._n_extra = 0 if extras_csr is None else int(extras_csr[2].shape[0] - 1)
        self.interleave = interleave
        # group-mean lookup (denoised targets / delta-magnitude loss): row -> global group row
        self.group_index = group_index
        self.ctrl_avg_k = max(1, int(ctrl_avg_k))
        self._group_of: Dict[int, np.ndarray] = {}
        if self.batch_match:
            raise ValueError("stream_v2 does not support stream_batch_match (withdrawn, F99)")
        # per-file static predicates: which perturbed rows are ever eligible, and their pert id
        self._static_ok: Dict[int, np.ndarray] = {}
        self._pid: Dict[int, np.ndarray] = {}
        self._pid_multi: Dict[int, Dict[int, List[int]]] = {}
        self._col_map: Dict[int, np.ndarray] = {}
        train_cl = set(self.train_cell_lines) if self.train_cell_lines else None
        split = set(self.split_pert_names)
        for mi, fm in enumerate(self.active_metas):
            self._col_map[mi] = make_col_map(fm.n_vars, fm.src_indices, fm.dst_indices)
            is_c = np.asarray(fm.is_control, dtype=bool)
            names = np.asarray(fm.pert_gene_names, dtype=str)
            cls = np.asarray(fm.obs_cell_lines, dtype=str)
            ok = np.ones(fm.n_obs, dtype=bool)
            if train_cl is not None:
                ok &= np.isin(cls, list(train_cl))
            pert_ok = ok & ~is_c
            if fm.controls_only:
                pert_ok[:] = False
            if fm.excluded_pert_lines:
                pert_ok &= ~np.isin(cls, list(fm.excluded_pert_lines))
            if fm.excluded_perts:
                # Per-(file, perturbation) TRAINING exclusion. `_static_ok` is the ONLY
                # per-row eligibility gate `__iter__` consults, so this single line is what
                # enforces the benchmark on the path we actually train on.
                pert_ok &= ~np.isin(names, list(fm.excluded_perts))
            if self.min_cells_per_pert > 0:
                pert_ok &= self._row_ok[mi]
            uniq, inv = np.unique(names, return_inverse=True)
            in_split = np.isin(uniq, list(split))
            pid_u = np.full(len(uniq), -2, dtype=np.int64)
            multi: Dict[int, List[int]] = {}
            for ui, nm in enumerate(uniq):
                if not in_split[ui]:
                    continue
                ids = resolve_pert_indices(nm, self.pert2id)
                if ids is None:
                    continue
                pid_u[ui] = ids[0]
                if len(ids) > 1:
                    multi[ui] = list(ids)
            pid = pid_u[inv]
            pert_ok &= pid >= 0
            self._static_ok[mi] = pert_ok
            self._pid[mi] = np.where(is_c, -1, pid)
            if self.group_index is not None:
                src = source_name_of(fm.path)
                gof = np.full(fm.n_obs, -1, dtype=np.int64)
                ukeys, uinv = np.unique(np.char.add(np.char.add(cls, "|"), names), return_inverse=True)
                lut = np.array([self.group_index.get((src, *k.split("|", 1)), -1) for k in ukeys], dtype=np.int64)
                gof = np.where(is_c, -1, lut[uinv])
                self._group_of[mi] = gof
                miss = int(((gof < 0) & pert_ok).sum())
                if miss:
                    print(f"  [group-mean] {Path(fm.path).name}: {miss:,} eligible rows have no group mean "
                          f"— they keep the single-cell target", flush=True)
            if (min_pert_snr is not None or max_pert_snr is not None) and pert_snr is not None:
                self._total_pert_cells = getattr(self, "_snr_kept", 0)   # tqdm total after the filter
            if multi:
                self._pid_multi[mi] = {int(r): multi[int(u)] for r, u in zip(np.flatnonzero(np.isin(inv, list(multi))), inv[np.isin(inv, list(multi))])}
            # controls eligible as identity rows
            self._ctrl_ok = getattr(self, "_ctrl_ok", {})
            self._ctrl_ok[mi] = is_c & ok if self.include_controls_as_training else np.zeros(fm.n_obs, dtype=bool)
            # ---- effect-strength filter (weak-perturbation hypothesis, 2026-09-09) ----
            # drop perturbed rows whose (source, cell line, pert) group has snr < min_pert_snr in
            # `pert_snr` (from scripts/pert_effect_snr.py); groups absent from the table are
            # dropped too (they had < 1 cell or no controls). `keep_perts` are exempt (the H1
            # panel: dropping them would turn seen perturbations into unseen ones, F73).
            if (min_pert_snr is not None or max_pert_snr is not None) and pert_snr is not None:
                src = source_name_of(fm.path)
                before = int(pert_ok.sum())
                ukeys, uinv = np.unique(np.char.add(np.char.add(cls, "|"), names), return_inverse=True)
                ok_u = np.zeros(len(ukeys), dtype=bool)
                for ui, k in enumerate(ukeys):
                    ln, pt = k.split("|", 1)
                    v = pert_snr.get((src, ln, pt))
                    in_range = (v is not None and (min_pert_snr is None or v >= min_pert_snr)
                                and (max_pert_snr is None or v < max_pert_snr))
                    ok_u[ui] = in_range or (keep_perts is not None and pt in keep_perts)
                pert_ok &= ok_u[uinv]
                self._snr_dropped = getattr(self, "_snr_dropped", 0) + (before - int(pert_ok.sum()))
                self._snr_kept = getattr(self, "_snr_kept", 0) + int(pert_ok.sum())
                print(f"  [snr-filter] {Path(fm.path).name} ({src}): kept {int(pert_ok.sum()):,} of {before:,} "
                      f"perturbed rows at snr in [{min_pert_snr}, {max_pert_snr})"
                      + (f" (+ panel perts exempt)" if keep_perts else ""), flush=True)
            # every streamed line must have a pool entry
            for cl in np.unique(cls[pert_ok]):
                if cl not in self.pool_index:
                    raise ValueError(f"stream_v2: cell line {cl!r} (file {Path(fm.path).name}) has no "
                                     f"control-embedding pool entry")

    # ---- chunk order -----------------------------------------------------------------
    def audit_eligible_perts(self) -> Dict[str, Dict[str, int]]:
        """{file basename: {perturbation: n eligible rows}} from the REAL gate.

        `_static_ok` is the only per-row eligibility mask `__iter__` consults, so this is
        what actually trains, not a reconstruction of it. Used by
        `scripts/verify_rhj_exclusion.py` to prove both halves of the benchmark invariant:
        that no held-out HepG2 row is eligible, AND that the same symbols are still eligible
        from the other two contexts -- the second of which a global name-level exclusion
        would fail.

        Row caps (`max_cells_per_pert`) and chunking can only REMOVE rows downstream of
        this, never add an excluded one back, so a zero here is conclusive.
        """
        out: Dict[str, Dict[str, int]] = {}
        for mi, fm in enumerate(self.active_metas):
            ok = self._static_ok[mi]
            names = np.asarray(fm.pert_gene_names, dtype=str)
            per: Dict[str, int] = {}
            for nm, cnt in zip(*np.unique(names[ok], return_counts=True)):
                per[str(nm)] = int(cnt)
            out[Path(fm.path).name] = per
        return out

    def _ordered_chunks(self, rng) -> List[int]:
        if not self.balance_sources or self.interleave != "proportional":
            return super()._ordered_chunks(rng)
        by_file: Dict[int, List[int]] = {}
        for ci, (mi, _, _) in enumerate(self._chunks):
            by_file.setdefault(mi, []).append(ci)
        lists = []
        for mi in sorted(by_file):
            lst = list(by_file[mi])
            if self.shuffle:
                rng.shuffle(lst)
            lists.append(lst)
        remaining = np.array([len(l) for l in lists], dtype=np.float64)
        pos = [0] * len(lists)
        order: List[int] = []
        while remaining.sum() > 0:
            p = remaining / remaining.sum()
            k = int(rng.choice(len(lists), p=p))
            order.append(lists[k][pos[k]])
            pos[k] += 1
            remaining[k] -= 1
        return order

    # ---- iteration --------------------------------------------------------------------
    def __iter__(self):
        wi = get_worker_info()
        n_workers = wi.num_workers if wi is not None else 1
        worker_id = wi.id if wi is not None else 0
        rng = np.random.default_rng(self.seed + self._epoch)            # identical in every worker
        chunk_order = self._ordered_chunks(rng)
        # per-epoch cap mask per file (bool over n_obs), decided identically in every worker
        cap_mask: Dict[int, Optional[np.ndarray]] = {}
        if self.max_cells_per_pert > 0:
            for mi, fm in enumerate(self.active_metas):
                # SAME row array as V1 (every perturbed row): the cap permutes positions of the
                # array it is given, so a filtered array would select different cells.
                rows = np.where(~np.asarray(fm.is_control, dtype=bool))[0]
                m = np.zeros(fm.n_obs, dtype=bool)
                if len(rows):
                    m[rows[self._pert_cap_mask(fm, rows, self._epoch)]] = True
                cap_mask[mi] = m
        n_groups = max(1, (len(chunk_order) + self.buffer_chunks - 1) // self.buffer_chunks)
        extra_order = rng.permutation(self._n_extra) if self._n_extra else np.empty(0, dtype=np.int64)
        per_group = int(np.ceil(self._n_extra / n_groups)) if self._n_extra else 0
        # worker-private rng for control draws / shuffles (distinct per worker, reproducible)
        wrng = np.random.default_rng([self.seed, self._epoch, worker_id])

        for gi, buf_start in enumerate(range(0, len(chunk_order), self.buffer_chunks)):
            if gi % n_workers != worker_id:
                continue
            group = chunk_order[buf_start:buf_start + self.buffer_chunks]
            pieces_v, pieces_c, pieces_rp, pieces_rs = [], [], [], []
            pid_l, cl_l, mask_l, ident_l, uid_f, uid_r, grp_l = [], [], [], [], [], [], []
            multi_rows: Dict[int, List[int]] = {}
            n_acc = 0
            for ci in group:
                mi, c_start, c_end = self._chunks[ci]
                fm = self.active_metas[mi]
                sl = slice(c_start, c_end)
                pert_ok = self._static_ok[mi][sl].copy()
                if cap_mask.get(mi) is not None:
                    pert_ok &= cap_mask[mi][sl]
                ctrl_ok = self._ctrl_ok[mi][sl]
                wanted = np.flatnonzero(pert_ok | ctrl_ok)
                if wanted.size == 0:
                    continue
                v, c, rp, rs = load_chunk_sparse(fm.path, c_start, c_end, fm.n_vars, wanted,
                                                 self._col_map[mi])
                pieces_v.append(v); pieces_c.append(c); pieces_rs.append(rs)
                pieces_rp.append(rp[1:] + (pieces_rp[-1][-1] if pieces_rp else 0) if False else rp)
                g_rows = c_start + wanted
                pid_l.append(self._pid[mi][g_rows])
                cl_l.append(np.asarray(fm.obs_cell_lines, dtype=str)[g_rows])
                mask_l.append(np.full(len(wanted), self.mask_id_of_meta[mi], dtype=np.int64))
                grp_l.append(self._group_of[mi][g_rows] if mi in self._group_of
                             else np.full(len(wanted), -1, dtype=np.int64))
                ident_l.append(ctrl_ok[wanted])
                uid_f.append(np.full(len(wanted), mi, dtype=np.uint64)); uid_r.append(g_rows.astype(np.uint64))
                if mi in self._pid_multi:
                    for k, r in enumerate(g_rows):
                        if int(r) in self._pid_multi[mi]:
                            multi_rows[n_acc + k] = self._pid_multi[mi][int(r)]
                n_acc += len(wanted)
            # splice this group's share of the H1 identity extras
            if per_group and self.extras_csr is not None:
                sel = extra_order[gi * per_group:(gi + 1) * per_group]
                if len(sel):
                    ev, ec, erp, ers, ectrl, emask = self.extras_csr
                    v, c, rp = _csr_rows_subset(ev, ec, erp, sel)
                    pieces_v.append(v); pieces_c.append(c); pieces_rp.append(rp); pieces_rs.append(ers[sel])
                    pid_l.append(np.full(len(sel), -1, dtype=np.int64))
                    cl_l.append(np.array(["H1"] * len(sel), dtype=str))
                    mask_l.append(np.full(len(sel), emask, dtype=np.int64))
                    grp_l.append(np.full(len(sel), -1, dtype=np.int64))
                    ident_l.append(ectrl[sel] < 0)
                    uid_f.append(np.full(len(sel), 2**40, dtype=np.uint64)); uid_r.append(sel.astype(np.uint64))
                    n_acc += len(sel)
            if n_acc == 0:
                continue
            # concatenate CSR pieces
            vals = np.concatenate(pieces_v); cols = np.concatenate(pieces_c)
            rowsum = np.concatenate(pieces_rs)
            rowptr = np.zeros(n_acc + 1, dtype=np.int64); off = 0; r0 = 0
            for rp in pieces_rp:
                n = len(rp) - 1
                rowptr[r0 + 1:r0 + n + 1] = rp[1:] + off
                off += rp[-1]; r0 += n
            pid = np.concatenate(pid_l); cls = np.concatenate(cl_l); mask_ids = np.concatenate(mask_l)
            grp = np.concatenate(grp_l)
            ident = np.concatenate(ident_l); uf = np.concatenate(uid_f); ur = np.concatenate(uid_r)
            # control indices: one per non-identity row, drawn from its line's pool
            K = self.ctrl_avg_k
            ctrl_idx = np.full((n_acc, K) if K > 1 else n_acc, -1, dtype=np.int64)
            need = np.flatnonzero(~ident)
            if need.size:
                for cl in np.unique(cls[need]):
                    rows = need[cls[need] == cl]
                    off_cl, n_cl = self.pool_index[cl]
                    if self.fixed_pairs:
                        local = _stable_ctrl_index(uf[rows], ur[rows], n_cl, self.seed)
                        if K > 1:      # k draws that stay deterministic per row
                            local = (np.asarray(local)[:, None]
                                     + np.arange(K, dtype=np.int64)[None, :] * 7919) % n_cl
                    else:
                        local = wrng.integers(0, n_cl, size=(len(rows), K) if K > 1 else len(rows))
                    ctrl_idx[rows] = off_cl + np.asarray(local, dtype=np.int64)
            # paired extras carry their own pool index (>=0) — they are not identity rows
            if per_group and self.extras_csr is not None and len(sel):
                e_start = n_acc - len(sel)
                ectrl_sel = self.extras_csr[4][sel]
                sl = slice(e_start, n_acc)
                if K > 1:
                    ctrl_idx[sl][ectrl_sel >= 0] = ectrl_sel[ectrl_sel >= 0][:, None]
                else:
                    ctrl_idx[sl][ectrl_sel >= 0] = ectrl_sel[ectrl_sel >= 0]
            # shuffle rows within the group
            perm = wrng.permutation(n_acc) if self.shuffle else np.arange(n_acc)
            vals, cols, rowptr = _csr_rows_subset(vals, cols, rowptr, perm)
            rowsum = rowsum[perm]; pid = pid[perm]; cls = cls[perm]; mask_ids = mask_ids[perm]
            ctrl_idx = ctrl_idx[perm]; grp = grp[perm]
            inv_multi = {int(np.flatnonzero(perm == k)[0]): v for k, v in multi_rows.items()} if multi_rows else {}
            # yield fixed-size batches; the trailing partial batch is kept if it has >= 2 rows
            for b0 in range(0, n_acc, self.batch_size):
                b1 = min(b0 + self.batch_size, n_acc)
                if b1 - b0 < 2:
                    continue
                s, e = int(rowptr[b0]), int(rowptr[b1])
                pidx = [inv_multi.get(i, [int(pid[i])]) for i in range(b0, b1)]
                names = ["non-targeting" if p[0] < 0 else str(p[0]) for p in pidx]
                yield SparseBatch(
                    vals=torch.from_numpy(np.ascontiguousarray(vals[s:e])),
                    cols=torch.from_numpy(np.ascontiguousarray(cols[s:e])),
                    rowptr=torch.from_numpy(rowptr[b0:b1 + 1] - s),
                    rowsum=torch.from_numpy(np.ascontiguousarray(rowsum[b0:b1])),
                    ctrl_idx=torch.from_numpy(np.ascontiguousarray(ctrl_idx[b0:b1])),
                    mask_ids=torch.from_numpy(np.ascontiguousarray(mask_ids[b0:b1])),
                    group_id=torch.from_numpy(np.ascontiguousarray(grp[b0:b1])),
                    pert_indices=pidx, pert_names=names, cell_types=cls[b0:b1].tolist(),
                )


# ----------------------------------------------------------------------------- effect table
_STEM_TO_SOURCE = {
    "replogle_2022_rpe1_e": "rpe1", "nadig_2024_hepg2": "hepg2", "nadig_2024_jurkat": "jurkat",
    "replogle_2022_k562_e": "k562e", "replogle_2022_k562_wg": "k562wg", "joung_2023": "joung",
    "D1_Rest.assigned_guide": "D1_Rest", "D2_Rest.assigned_guide": "D2_Rest",
    "D3_Rest.assigned_guide": "D3_Rest", "D4_Rest.assigned_guide": "D4_Rest",
    "heracles_HCT116": "HCT116", "heracles_HEK293T": "HEK293T",
}


def source_name_of(path: str) -> str:
    """Short source name used by scripts/pert_effect_snr.py for a training file."""
    stem = Path(path).name.replace(".h5ad", "")
    return _STEM_TO_SOURCE.get(stem, stem)


def load_pert_snr_table(effect_dir: str) -> Dict[Tuple[str, str, str], float]:
    """{(source, cell_line, pert): snr} from cache/pert_effects/all_sources.csv (or per-file CSVs)."""
    import pandas as pd
    d = Path(effect_dir)
    f = d / "all_sources.csv"
    frames = [pd.read_csv(f)] if f.exists() else [pd.read_csv(x) for x in sorted(d.glob("*.csv")) if x.name not in ("panel_perts.csv",)]
    if not frames:
        raise FileNotFoundError(f"no effect tables in {d}")
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for src, ln, pt, v in zip(df.source.astype(str), df.cell_line.astype(str), df.pert.astype(str), df.snr.astype(float)):
        out[(src, ln, pt)] = v
    print(f"  [snr-filter] effect table: {len(out):,} (source, line, pert) groups from {d}")
    return out


# ----------------------------------------------------------------------------- control pools
def encoder_fingerprint(encoder: torch.nn.Module, extra: str = "") -> str:
    h = hashlib.sha1()
    for i, (n, p) in enumerate(encoder.named_parameters()):
        if i >= 3:
            break
        h.update(n.encode()); h.update(np.ascontiguousarray(p.detach().flatten()[:4096].float().cpu().numpy()).tobytes())
    h.update(extra.encode())
    return h.hexdigest()[:16]


CTRL_CACHE_SCHEMA = 2


def control_cache_key(
    *,
    path: str,
    n_obs: int,
    n_genes_encoder: int,
    col_map: np.ndarray,
    control_rows: np.ndarray,
    target_sum: float,
    skip_normalize: bool,
    skip_log1p: bool,
    encoder_key: str,
    axis_sha256: str = "",
    axis_scope: str = "",
) -> Tuple[str, Dict[str, object]]:
    """A canonical, self-describing key for one file's control embeddings.

    Returns ``(sha16, key_dict)``. The dict is written beside the cache as a sidecar so a
    stale entry can explain itself instead of being silently reused.

    Every field here can change an embedding. The one that used to be missing is the gene
    alignment: the old key carried `n_genes` (always 44,728, the encoder width) but not which
    of those columns were actually written, so a cache built before a benchmark-axis
    restriction was reused with embeddings computed from a different input -- wrong control
    embeddings for every pair, no error. `col_map_sha256` fixes that by construction, and
    `n_genes_axis` makes it readable.
    """
    n_axis = int((np.asarray(col_map) >= 0).sum())
    key = {
        "cache_schema": CTRL_CACHE_SCHEMA,
        "file_name": Path(path).name,
        "file_mtime": os.path.getmtime(path),
        "file_size": os.path.getsize(path),
        "n_obs": int(n_obs),
        "n_genes_encoder": int(n_genes_encoder),
        "n_genes_axis": n_axis,
        "col_map_sha256": hashlib.sha256(
            np.asarray(col_map, dtype=np.int64).tobytes()).hexdigest(),
        "control_rows_sha256": hashlib.sha256(
            np.asarray(control_rows, dtype=np.int64).tobytes()).hexdigest(),
        "n_control_rows": int(len(control_rows)),
        "target_sum": float(target_sum),
        "skip_normalize": bool(skip_normalize),
        "skip_log1p": bool(skip_log1p),
        "encoder_fingerprint": str(encoder_key),
        "benchmark_axis_sha256": str(axis_sha256),
        "benchmark_axis_scope": str(axis_scope),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return digest, key


def control_cache_mismatch(want: Dict[str, object], got: Dict[str, object]) -> List[str]:
    """Field names that differ between the recomputed key and a sidecar, most useful first."""
    keys = sorted(set(want) | set(got))
    return [k for k in keys if want.get(k) != got.get(k)]


def build_control_embedding_pool(
    pending: Dict[str, List[Tuple[FileMetadata, np.ndarray]]],
    encode_fn, num_genes: int, target_sum: float, skip_normalize: bool, skip_log1p: bool,
    device, cache_dir: Optional[str], enc_key: str, chunk_size: int = 8192, threads: int = 6,
    extra_dense: Optional[Dict[str, torch.Tensor]] = None,
    axis_sha256: str = "", axis_scope: str = "",
    keep_raw: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray], Optional[Dict[str, torch.Tensor]]]:
    """Encode every control cell once (chunk-wise scan) -> {line: (n, hidden)} + {line: mean log1p row}.

    `pending` = {cell_line: [(file_meta, sorted control row indices), ...]}, i.e. exactly the
    rows the old `_load_expression_rows` would have loaded. `extra_dense` = {line: dense log1p
    rows already in encoder space} for pools that live in no h5ad (the H1 model-visible controls).
    """
    t0 = time.time()
    pools: Dict[str, List[torch.Tensor]] = {}
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    raws: Dict[str, List[torch.Tensor]] = {}

    def _accumulate(cl: str, dense_gpu: torch.Tensor):
        emb = encode_fn(dense_gpu).float().cpu()
        pools.setdefault(cl, []).append(emb)
        if keep_raw:
            # ⚠️ recon_split_mode='raw' needs the per-cell GENE-SPACE control, which this
            # function otherwise discards the moment it is encoded. Kept on CPU: ~5 GB at
            # 44,728 genes for a typical pool; the per-batch gather is small.
            raws.setdefault(cl, []).append(dense_gpu.float().cpu())
        sums[cl] = sums.get(cl, torch.zeros(num_genes, dtype=torch.float64, device=dense_gpu.device)) + dense_gpu.double().sum(0)
        counts[cl] = counts.get(cl, 0) + int(dense_gpu.shape[0])

    # group work by file so each file is scanned once
    by_file: Dict[str, List[Tuple[str, FileMetadata, np.ndarray]]] = {}
    for cl, lst in pending.items():
        for fm, rows in lst:
            by_file.setdefault(fm.path, []).append((cl, fm, np.asarray(rows, dtype=np.int64)))
    for path, entries in by_file.items():
        fm = entries[0][1]
        rows_all = np.concatenate([r for _, _, r in entries])
        cl_of_row = np.concatenate([np.array([cl] * len(r)) for cl, _, r in entries])
        order = np.argsort(rows_all, kind="stable"); rows_all = rows_all[order]; cl_of_row = cl_of_row[order]
        col_map = make_col_map(fm.n_vars, fm.src_indices, fm.dst_indices)
        key, key_dict = control_cache_key(
            path=path, n_obs=fm.n_obs, n_genes_encoder=num_genes, col_map=col_map,
            control_rows=rows_all, target_sum=target_sum,
            skip_normalize=skip_normalize, skip_log1p=skip_log1p, encoder_key=enc_key,
            axis_sha256=axis_sha256, axis_scope=axis_scope,
        )
        cpath = cache / f"ctrl_emb_{Path(path).stem}_{key}.pt" if cache else None
        spath = cpath.with_suffix(".json") if cpath else None
        if cpath and cpath.exists():
            # A cache that cannot explain itself is not used. The digest already encodes
            # every field, so a mismatch here means a hash collision or a hand-edited file --
            # either way, recompute and say which fields disagreed.
            side = {}
            if spath and spath.exists():
                try:
                    with open(spath) as fh:
                        side = json.load(fh)
                except (OSError, ValueError) as exc:
                    print(f"  [ctrl-cache] unreadable sidecar {spath.name}: {exc}")
            diff = control_cache_mismatch(key_dict, side) if side else ["<no sidecar>"]
            if diff:
                print(f"  [ctrl-cache] IGNORING {cpath.name}: key fields differ "
                      f"({', '.join(diff[:6])}) — recomputing")
                cpath = None
        if cpath and cpath.exists():
            blob = torch.load(cpath, map_location="cpu", weights_only=False)
            for cl in np.unique(cl_of_row):
                m = torch.from_numpy(cl_of_row == cl)
                pools.setdefault(cl, []).append(blob["emb"][m])
                sums[cl] = sums.get(cl, torch.zeros(num_genes, dtype=torch.float64)) + blob["sum"][cl].to(torch.float64)
                counts[cl] = counts.get(cl, 0) + int(m.sum())
            if keep_raw:
                raise RuntimeError(
                    f"keep_raw: {Path(path).name} was served from the embedding cache, which "
                    f"stores embeddings only -- the dense gene-space rows do not exist in it. "
                    f"Pass cache_dir=None for this run.")
            print(f"  [ctrl-cache] {Path(path).name}: {len(rows_all):,} controls from cache", flush=True)
            continue
        # chunk spans that contain at least one control
        chunk_ids = np.unique(rows_all // chunk_size)
        spans = [(int(c) * chunk_size, min(fm.n_obs, (int(c) + 1) * chunk_size)) for c in chunk_ids]
        file_emb: List[torch.Tensor] = []
        file_raw: List[torch.Tensor] = []     # keep_raw: dense gene-space rows, same order
        file_rows: List[np.ndarray] = []
        file_sum: Dict[str, torch.Tensor] = {}

        def _job(span):
            s, e = span
            sel = rows_all[(rows_all >= s) & (rows_all < e)]
            return s, sel, load_chunk_sparse(path, s, e, fm.n_vars, sel - s, col_map)

        tf = time.time()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for s, sel, (v, c, rp, rs) in ex.map(_job, spans):
                dense = densify_csr(torch.from_numpy(v).to(device), torch.from_numpy(c).to(device),
                                    torch.from_numpy(rp).to(device), torch.from_numpy(rs).to(device),
                                    num_genes, target_sum, skip_normalize, skip_log1p)
                with torch.no_grad():
                    emb = encode_fn(dense).float()
                file_emb.append(emb.cpu()); file_rows.append(sel)
                if keep_raw:
                    file_raw.append(dense.float().cpu())
                cls_here = cl_of_row[np.searchsorted(rows_all, sel)]
                for cl in np.unique(cls_here):
                    m = torch.from_numpy(cls_here == cl).to(device)
                    file_sum[cl] = file_sum.get(cl, torch.zeros(num_genes, dtype=torch.float64, device=device)) + dense[m].double().sum(0)
                del dense
        emb_all = torch.cat(file_emb, 0); rows_done = np.concatenate(file_rows)
        assert np.array_equal(rows_done, rows_all), "control scan returned rows out of order"
        raw_all = torch.cat(file_raw, 0) if keep_raw else None
        for cl in np.unique(cl_of_row):
            m = torch.from_numpy(cl_of_row == cl)
            pools.setdefault(cl, []).append(emb_all[m])
            if keep_raw:
                # SAME mask, SAME order as the embedding slice above.
                raws.setdefault(cl, []).append(raw_all[m])
            sums[cl] = sums.get(cl, torch.zeros(num_genes, dtype=torch.float64)) + file_sum[cl].cpu()
            counts[cl] = counts.get(cl, 0) + int(m.sum())
        if cpath:
            # atomic: concurrent jobs may build the same cache; readers see a whole file or none
            tmp = cpath.with_suffix(f".tmp{os.getpid()}")
            # The sidecar first: a payload without its key would be ignored on the next run
            # anyway, whereas a key without a payload is simply a miss.
            if spath:
                stmp = spath.with_suffix(f".jtmp{os.getpid()}")
                with open(stmp, "w") as fh:
                    json.dump(key_dict, fh, indent=2, sort_keys=True)
                os.replace(stmp, spath)
            torch.save({"emb": emb_all, "rows": rows_all, "cell_lines": cl_of_row,
                        "sum": {cl: t.cpu().float() for cl, t in file_sum.items()}, "key": key}, tmp)
            os.replace(tmp, cpath)
        print(f"  [ctrl-scan] {Path(path).name}: {len(rows_all):,} controls in {len(spans)} chunks, "
              f"{time.time() - tf:.0f}s, RSS {rss_gb():.1f} GB", flush=True)
    if extra_dense:
        for cl, dense in extra_dense.items():
            for i in range(0, dense.shape[0], 4096):
                d = dense[i:i + 4096].to(device)
                with torch.no_grad():
                    _accumulate(cl, d)
    out_pools = {cl: torch.cat(v, 0) for cl, v in pools.items()}
    means = {cl: (sums[cl].cpu().double() / max(counts[cl], 1)).float().numpy() for cl in out_pools}
    tot = sum(p.shape[0] for p in out_pools.values())
    print(f"  Control-embedding pools: {len(out_pools)} lines, {tot:,} cells, "
          f"{sum(p.numel() * 4 for p in out_pools.values()) / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
    out_raw = None
    if keep_raw:
        out_raw = {cl: torch.cat(v, 0) for cl, v in raws.items()}
        _bad = [cl for cl in out_pools
                if cl not in out_raw or out_raw[cl].shape[0] != out_pools[cl].shape[0]]
        if _bad:
            # A DISK-CACHED embedding skips _accumulate entirely, so that file's raw rows were
            # never materialised. Returning a short pool would misalign every ctrl_idx and
            # produce a plausible wrong number -- refuse, and say how to fix it.
            raise RuntimeError(
                f"keep_raw: raw control rows missing/short for {_bad} "
                f"(raw={{k: v.shape[0] for k, v in out_raw.items()}}, "
                f"emb={{k: v.shape[0] for k, v in out_pools.items()}}). Some code path filled "
                f"`pools` without filling `raws`; every such path must do both or ctrl_idx "
                f"misaligns.")
        print(f"  [stream_v2] RAW control pool kept: "
              f"{sum(v.shape[0] for v in out_raw.values()):,} cells, "
              f"{sum(v.numel() * 4 for v in out_raw.values()) / 2**30:.2f} GB (CPU)", flush=True)
    return out_pools, means, out_raw


def load_group_means(mean_dir: str, device, dtype=torch.float32, sources: Optional[set] = None):
    """Load cache/pert_means/*.npz into one GPU table.

    Returns (index, means, ctrl_means, group_line):
      index[(source, cell_line, pert)] -> row of `means`
      means (G, n_enc)        the group's mean profile, log1p model space
      ctrl_means (L, n_enc)   the control mean of each (source, cell_line)
      group_line (G,)         row of `ctrl_means` for each group
    """
    d = Path(mean_dir)
    files = sorted(d.glob("*.npz"))
    if sources is not None:
        files = [f for f in files if f.stem in sources]
    if not files:
        raise FileNotFoundError(f"no group-mean tables in {d} (run scripts/pert_group_means.py)")
    index, M, C, GL = {}, [], [], []
    for f in files:
        src = f.stem
        z = np.load(f, allow_pickle=True)
        c_off = sum(x.shape[0] for x in C)
        g_off = sum(x.shape[0] for x in M)
        for i, k in enumerate(z["keys"]):
            ln, pt = str(k).split("|", 1)
            index[(src, ln, pt)] = g_off + i
        M.append(z["means"]); C.append(z["ctrl_means"])
        GL.append(np.asarray(z["group_line"], dtype=np.int64) + c_off)
    means = torch.from_numpy(np.concatenate(M, 0)).to(device=device, dtype=dtype)
    ctrl_means = torch.from_numpy(np.concatenate(C, 0)).to(device=device, dtype=dtype)
    group_line = torch.from_numpy(np.concatenate(GL, 0)).to(device)
    print(f"  [group-mean] {len(index):,} groups from {len(files)} file(s), "
          f"means {tuple(means.shape)} ({means.numel() * means.element_size() / 1e9:.2f} GB), "
          f"{ctrl_means.shape[0]} control means", flush=True)
    return index, means, ctrl_means, group_line


def dense_to_csr(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense (n, G) float tensor -> (vals, cols int32, rowptr) with rowsum returned separately by caller."""
    xn = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    r, c = np.nonzero(xn)
    vals = xn[r, c].astype(np.float32)
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=xn.shape[0]))]).astype(np.int64)
    return vals, c.astype(np.int32), rowptr
