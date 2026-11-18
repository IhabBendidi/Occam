"""OptimizedChunkedDataset: batch-level yielding with cross-dataset shuffling.
# NOTE: added __future__ annotations for Python 3.9 compatibility (X | None syntax).

Eliminates the per-row bottleneck in ChunkedDataset by:
1. Loading multiple chunks from different file regions into a shuffle buffer
2. Reading directly into pre-allocated numpy buffers via h5py.read_direct (zero allocation)
3. Shuffling via torch.randperm + tensor indexing across the full buffer
4. Splitting into batch-sized views via torch.split (~0ms)
5. Yielding pre-batched tensors (DataLoader uses batch_size=None)
6. Two-stage pipeline: I/O thread reads while processing thread shuffles in parallel

The buffer_chunks parameter controls cross-dataset mixing. With buffer_chunks=4,
each shuffle buffer combines 4 chunks from randomly selected file positions,
ensuring batches contain cells from multiple datasets even when datasets are
stored contiguously on disk.

Optional cross-buffer carryover (carryover_fraction > 0) allows rows from
adjacent buffers to co-appear in the same batch, improving shuffle correctness
at a small speed cost. Off by default.

Usage:
    dataset = OptimizedChunkedDataset("path/to/file.h5ad", batch_size=512, buffer_chunks=4)
    loader = DataLoader(dataset, batch_size=None)  # IMPORTANT: disables auto-batching
    for batch in loader:
        # batch: torch.FloatTensor of shape (batch_size, n_vars)
        pass
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path
from threading import Thread, Event
from queue import Queue
from typing import Iterator, Optional


class OptimizedChunkedDataset(IterableDataset):
    """Iterable dataset yielding pre-batched tensors with cross-dataset shuffling.

    Loads groups of globally-shuffled chunks into a combined buffer, shuffles
    across the full buffer, then yields batch-sized tensor views.

    Uses a two-stage pipeline when prefetch > 0:
    - I/O thread reads chunks into pre-allocated numpy buffers via read_direct
    - Processing thread converts to tensor, shuffles, and splits into batches
    - Consumer yields batches while both threads work on the next buffer

    Must be used with DataLoader(batch_size=None) to disable auto-batching.

    Args:
        h5ad_path: Path to the h5ad file.
        batch_size: Number of samples per yielded batch.
        chunk_size: Number of contiguous rows per HDF5 read.
            8192 rows = ~1.4 GB per chunk.
        buffer_chunks: Number of chunks to combine into a shuffle buffer.
            Controls cross-dataset mixing quality vs memory tradeoff.
            1 = within-chunk shuffle only.
            4 (default) = 32K rows from ~3-4 different datasets per buffer.
        shuffle: Whether to shuffle chunk order and buffer rows.
        prefetch: Number of buffers to prefetch in background.
        seed: Random seed for reproducible shuffling.
        drop_last: Whether to drop the last incomplete batch per buffer.
        carryover_fraction: Fraction of rows to carry over between buffers
            (0.0-0.5). Allows rows from adjacent buffers to co-appear in
            the same batch. 0.0 (default) = no carryover, zero overhead.
            Only active when shuffle=True. Adds ~5-10ms/batch overhead.
    """

    loader_name = "OptimizedChunkedDataset"

    def __init__(
        self,
        h5ad_path: Path | None = None,
        batch_size: int = 512,
        chunk_size: int = 8192,
        buffer_chunks: int = 4,
        shuffle: bool = True,
        prefetch: int = 2,
        seed: int = 42,
        drop_last: bool = False,
        carryover_fraction: float = 0.0,
        file_infos=None,
        n_target_genes: int | None = None,
        track_presence: bool = False,
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.buffer_chunks = buffer_chunks
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.seed = seed
        self.drop_last = drop_last
        self.carryover_fraction = carryover_fraction if shuffle else 0.0
        self._track_presence = track_presence

        if file_infos is not None:
            # Multi-file mode: chunks from multiple h5ad files with gene alignment
            self._multi_file = True
            self._file_infos = file_infos
            self.h5ad_path = None
            self._n_vars = n_target_genes
            self._n_obs = sum(fi.n_obs for fi in file_infos)
            # Global chunk list: (file_index, local_start, local_end)
            self._chunks = [
                (fi_idx, start, end)
                for fi_idx, fi in enumerate(file_infos)
                for (start, end) in fi.chunks
            ]
            # Pre-allocate temp buffer for non-identity gene alignment reads
            max_file_vars = max(fi.n_vars for fi in file_infos)
            self._read_tmp = np.empty(
                (chunk_size, max_file_vars), dtype=np.float32
            )
        else:
            # Single-file mode (original behaviour)
            self._multi_file = False
            self._file_infos = None
            self._read_tmp = None
            self.h5ad_path = Path(h5ad_path)
            with h5py.File(self.h5ad_path, "r") as f:
                self._n_obs, self._n_vars = f["X"].shape
            self._chunks = []
            for start in range(0, self._n_obs, self.chunk_size):
                end = min(start + self.chunk_size, self._n_obs)
                self._chunks.append((start, end))

        # Compact buffer: +1 column for file index when tracking presence in multi-file mode
        self._buf_width = (self._n_vars + 1) if (track_presence and self._multi_file) else self._n_vars

        self._epoch = 0

    def __len__(self) -> int:
        return self._n_obs

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible per-epoch shuffling (standard DDP pattern)."""
        self._epoch = epoch

    def _prepare_buffer(
        self, chunk_np: np.ndarray, carryover: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        """Convert a numpy buffer into batch tensors with optional carryover.

        Args:
            chunk_np: Numpy array of shape (n_rows, n_vars).
            carryover: Optional tensor of rows carried from the previous buffer.

        Returns:
            (batches, new_carryover): batch tensors and carryover for next buffer.
        """
        chunk_tensor = torch.from_numpy(chunk_np)  # zero-copy

        if carryover is not None and carryover.shape[0] > 0:
            chunk_tensor = torch.cat([carryover, chunk_tensor])

        if self.shuffle:
            perm = torch.randperm(chunk_tensor.shape[0])
            chunk_tensor = chunk_tensor[perm]  # new tensor, owns its memory

        # Split off carryover from the end
        new_carryover = None
        if self.carryover_fraction > 0 and self.shuffle:
            n_carryover = int(chunk_tensor.shape[0] * self.carryover_fraction)
            if n_carryover > 0:
                new_carryover = chunk_tensor[-n_carryover:].clone()
                chunk_tensor = chunk_tensor[:-n_carryover]

        batches = torch.split(chunk_tensor, self.batch_size)

        if self.drop_last and batches and batches[-1].shape[0] < self.batch_size:
            batches = batches[:-1]

        return batches, new_carryover

    def __iter__(self) -> Iterator[torch.Tensor]:
        rng = np.random.default_rng(self.seed + self._epoch)

        chunk_indices = list(range(len(self._chunks)))
        if self.shuffle:
            rng.shuffle(chunk_indices)

        if self._multi_file:
            # Only open files that contain chunks assigned to this worker/rank
            needed_files = {self._chunks[ci][0] for ci in chunk_indices}
            file_handles = {
                i: h5py.File(fi.path, "r")
                for i, fi in enumerate(self._file_infos)
                if i in needed_files
            }
            read_fn = lambda group, buf: self._read_buffer_multi(
                file_handles, group, buf
            )
            try:
                if self.prefetch > 0:
                    yield from self._iter_with_prefetch(read_fn, chunk_indices)
                else:
                    yield from self._iter_simple(read_fn, chunk_indices)
            finally:
                for fh in file_handles.values():
                    fh.close()
        else:
            fh = h5py.File(self.h5ad_path, "r")
            read_fn = lambda group, buf: self._read_buffer(fh, group, buf)
            try:
                if self.prefetch > 0:
                    yield from self._iter_with_prefetch(read_fn, chunk_indices)
                else:
                    yield from self._iter_simple(read_fn, chunk_indices)
            finally:
                fh.close()

    def _read_buffer(self, fh, chunk_indices_group, buf):
        """Read a group of chunks directly into a pre-allocated numpy buffer.

        Uses h5py.Dataset.read_direct to avoid intermediate allocations
        and the np.concatenate copy. Returns the number of rows filled.
        """
        X = fh["X"]
        offset = 0
        for ci in chunk_indices_group:
            start, end = self._chunks[ci]
            n = end - start
            X.read_direct(buf, source_sel=np.s_[start:end],
                          dest_sel=np.s_[offset:offset + n])
            offset += n
        return offset

    def _read_buffer_multi(self, file_handles, chunk_indices_group, buf):
        """Read chunks from multiple files into a pre-allocated buffer.

        Handles per-chunk gene alignment via numpy scatter.  Files whose
        genes match the target vocabulary exactly (``is_identity``) use
        a fast ``read_direct`` path with no temp buffer.

        Returns the number of rows filled.
        """
        offset = 0
        for ci in chunk_indices_group:
            fi_idx, start, end = self._chunks[ci]
            fi = self._file_infos[fi_idx]
            X = file_handles[fi_idx]["X"]
            n = end - start

            if fi.is_identity:
                # Fast path: same genes, same order as target vocab
                if self._track_presence:
                    X.read_direct(buf, source_sel=np.s_[start:end],
                                  dest_sel=np.s_[offset:offset + n, :self._n_vars])
                    buf[offset:offset + n, self._n_vars] = fi_idx
                else:
                    X.read_direct(buf, source_sel=np.s_[start:end],
                                  dest_sel=np.s_[offset:offset + n])
            else:
                # Read into temp buffer, zero-fill destination, scatter columns
                X.read_direct(self._read_tmp, source_sel=np.s_[start:end],
                              dest_sel=np.s_[:n, :fi.n_vars])
                buf[offset:offset + n, :] = 0
                buf[offset:offset + n, fi.dst_indices] = (
                    self._read_tmp[:n, fi.src_indices]
                )
                if self._track_presence:
                    buf[offset:offset + n, self._n_vars] = fi_idx

            offset += n
        return offset

    def _iter_simple(self, read_fn, chunk_indices):
        """No prefetching: load buffer group, convert, yield batches, repeat."""
        buf = np.empty((self.buffer_chunks * self.chunk_size, self._buf_width),
                       dtype=np.float32)
        carryover = None
        for group_start in range(0, len(chunk_indices), self.buffer_chunks):
            group = chunk_indices[group_start : group_start + self.buffer_chunks]
            n_rows = read_fn(group, buf)
            batches, carryover = self._prepare_buffer(buf[:n_rows], carryover)
            yield from batches
        # Flush remaining carryover
        if carryover is not None and carryover.shape[0] >= self.batch_size:
            batches = torch.split(carryover, self.batch_size)
            if self.drop_last and batches[-1].shape[0] < self.batch_size:
                batches = batches[:-1]
            yield from batches

    def _iter_with_prefetch(self, read_fn, chunk_indices):
        """Two-stage pipeline: I/O thread + processing thread + consumer.

        Stage 1 (I/O thread): reads chunks into pre-allocated numpy buffers
        via the supplied *read_fn*. No allocation per buffer -- buffers are
        reused from a pool.

        Stage 2 (processing thread): converts numpy buffer to tensor, shuffles
        via randperm, splits into batch views, manages carryover state. Runs
        in parallel with I/O (h5py/numpy/torch all release GIL).

        Consumer: yields batch tensors, then returns the numpy buffer to the
        pool for reuse.

        Buffer pool sizing: prefetch + 1 buffers. One being read into by I/O,
        one being processed, and up to (prefetch - 1) waiting in the batch queue.
        """
        io_queue: Queue = Queue(maxsize=1)
        batch_queue: Queue = Queue(maxsize=self.prefetch)
        stop_event = Event()

        # Pre-allocate buffer pool
        buf_rows = self.buffer_chunks * self.chunk_size
        free_buffers: Queue = Queue()
        for _ in range(self.prefetch + 1):
            free_buffers.put(
                np.empty((buf_rows, self._buf_width), dtype=np.float32)
            )

        def io_producer():
            """Read chunks from disk into pre-allocated buffers."""
            try:
                for group_start in range(0, len(chunk_indices), self.buffer_chunks):
                    if stop_event.is_set():
                        break
                    group = chunk_indices[group_start : group_start + self.buffer_chunks]
                    # Wait for a free buffer — no short timeout; the consumer
                    # may take tens of seconds per buffer with large models.
                    buf = None
                    while buf is None:
                        if stop_event.is_set():
                            break
                        try:
                            buf = free_buffers.get(timeout=30)
                        except Exception:
                            continue
                    if buf is None:
                        break
                    if stop_event.is_set():
                        break
                    offset = read_fn(group, buf)
                    if stop_event.is_set():
                        break
                    io_queue.put((buf, offset))
            except Exception as e:
                io_queue.put(e)
            finally:
                io_queue.put(None)  # sentinel

        def process_worker():
            """Convert numpy buffers to shuffled batch tensors."""
            carryover = None
            try:
                while True:
                    if stop_event.is_set():
                        break
                    item = io_queue.get()
                    if item is None:
                        # Flush carryover as final batches
                        if carryover is not None and carryover.shape[0] >= self.batch_size:
                            final = torch.split(carryover, self.batch_size)
                            if self.drop_last and final[-1].shape[0] < self.batch_size:
                                final = final[:-1]
                            batch_queue.put((final, None))
                        break
                    if isinstance(item, Exception):
                        batch_queue.put(item)
                        break
                    buf, n_rows = item
                    batches, carryover = self._prepare_buffer(buf[:n_rows], carryover)
                    batch_queue.put((batches, buf))
            except Exception as e:
                batch_queue.put(e)
            finally:
                batch_queue.put(None)  # sentinel

        io_thread = Thread(target=io_producer, daemon=True)
        proc_thread = Thread(target=process_worker, daemon=True)
        io_thread.start()
        proc_thread.start()

        try:
            while True:
                item = batch_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                batches, buf = item
                yield from batches
                # Return buffer to pool (None = carryover flush, no buffer)
                if buf is not None:
                    free_buffers.put(buf)
        finally:
            stop_event.set()
            # Drain queues so threads can unblock from put()
            for q in [io_queue, batch_queue, free_buffers]:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break
            io_thread.join(timeout=5)
            proc_thread.join(timeout=5)

    def close(self):
        pass  # file handle opened/closed per iteration

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
