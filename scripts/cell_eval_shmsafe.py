#!/usr/bin/env python
"""Run the cell-eval CLI with pdex's /dev/shm segment replaced by a segment nobody can unlink.

WHY. pdex 0.1.24 (`parallel_differential_expression`) copies the expression matrix into a
POSIX shared-memory segment (`SharedMemory(create=True)` -> /dev/shm/psm_<hex>) and every
mp.Pool task re-attaches to it BY NAME (`_process_target_batch_shm` -> `SharedMemory(name=)`).
On our GPU nodes systemd-logind runs with RemoveIPC=yes (live value, `busctl get-property
org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager RemoveIPC` ->
true; /etc/systemd/logind.conf has no RemoveIPC line, so the compiled-in default applies).
With RemoveIPC, when a user's LAST logind session on a node ends, logind deletes every
/dev/shm object that user owns. Slurm jobs are not logind sessions (UsePam=no), but any
`ssh <node>` is (pam_systemd via common-session). So a monitoring `ssh hopNN ...` wipes the
live segment of every cell-eval the user has running on that node, ~10 s after it exits
(UserStopDelaySec), and the next pool task dies with
    FileNotFoundError: [Errno 2] No such file or directory: '/psm_3de1974e'
The same wipe explains the `/torch_*` "could not unlink" and `SemLock._rebuild`
FileNotFoundError failures. More jobs on a node = more live segments exposed per wipe.
Reproduced 2026-09-23 on hop01: a /dev/shm/psm_* file vanished after `ssh hop01 true`.

FIX. Before cell-eval imports run, replace `pdex._single_cell.SharedMemory` with a drop-in
that has no name in /dev/shm at all:
  * `anon` (default when the start method is fork, which is Linux's default and what
    cell-eval uses): an anonymous MAP_SHARED mmap created in the parent and INHERITED by the
    forked pool workers. There is no filesystem entry, so nothing can unlink it.
  * `file` (used for spawn/forkserver, or forced via OCCAM_CELLEVAL_SHM_BACKEND=file): a
    file-backed mmap in a private directory under $TMPDIR (ext4 on our nodes, not /dev/shm).
The bytes written and read are identical to stock pdex, so the DE output is identical.

Usage (under the jupyter env): same argv as the `cell-eval` executable, e.g.
    PYENV_VERSION=jupyter python scripts/cell_eval_shmsafe.py run -ap pred.h5ad -ar real.h5ad ...
"""
import atexit
import mmap
import multiprocessing as mp
import os
import shutil
import sys
import tempfile

_REGISTRY = {}          # name -> mmap, populated in the parent BEFORE the pool forks
_PRIVATE_DIR = None


def _private_dir():
    global _PRIVATE_DIR
    if _PRIVATE_DIR is None:
        _PRIVATE_DIR = tempfile.mkdtemp(prefix="celleval_shm_")
        atexit.register(shutil.rmtree, _PRIVATE_DIR, True)
    return _PRIVATE_DIR


def _backend():
    forced = os.environ.get("OCCAM_CELLEVAL_SHM_BACKEND", "").strip().lower()
    if forced in ("anon", "file"):
        return forced
    return "anon" if mp.get_start_method(allow_none=True) in (None, "fork") else "file"


class SafeSharedMemory:
    """The subset of multiprocessing.shared_memory.SharedMemory that pdex uses:
    create=True/size, name=, .name, .buf, .size, .close(), .unlink()."""

    def __init__(self, name=None, create=False, size=0):
        self._owner = bool(create)
        if create:
            if size <= 0:
                raise ValueError("'size' must be a positive number different from zero")
            if _backend() == "anon":
                self._mmap = mmap.mmap(-1, size, flags=mmap.MAP_SHARED,
                                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
                self._name = f"anon:{os.getpid()}:{id(self._mmap):x}"
                _REGISTRY[self._name] = self._mmap
            else:
                fd, path = tempfile.mkstemp(prefix="psm_", dir=_private_dir())
                try:
                    os.ftruncate(fd, size)
                    self._mmap = mmap.mmap(fd, size)
                finally:
                    os.close(fd)
                self._name = "file:" + path
        else:
            if name is None:
                raise ValueError("name is required when create=False")
            self._name = name
            if name.startswith("anon:"):
                if name not in _REGISTRY:
                    raise FileNotFoundError(
                        f"{name}: anonymous segment not inherited -- pool workers must be forked "
                        f"(set OCCAM_CELLEVAL_SHM_BACKEND=file for spawn/forkserver)")
                self._mmap = _REGISTRY[name]
            elif name.startswith("file:"):
                with open(name[5:], "r+b") as f:
                    self._mmap = mmap.mmap(f.fileno(), 0)
            else:
                raise FileNotFoundError(f"unknown segment name {name!r}")
        self._size = len(self._mmap)
        self._buf = memoryview(self._mmap)

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def buf(self):
        return self._buf

    def close(self):
        # Anonymous segments are shared by every task in a worker (and by the parent), so a
        # worker-side close() must not unmap them; the mapping dies with the process.
        if self._name.startswith("anon:"):
            return
        self._release()

    def _release(self):
        try:
            if self._buf is not None:
                self._buf.release()
            self._mmap.close()
        except BufferError:
            pass          # a numpy view is still alive; GC unmaps it when that goes
        self._buf = None

    def unlink(self):
        if self._name.startswith("anon:"):
            m = _REGISTRY.pop(self._name, None)
            if m is not None and self._owner:
                self._release()
        elif self._name.startswith("file:"):
            try:
                os.unlink(self._name[5:])
            except FileNotFoundError:
                pass


def install():
    import pdex._single_cell as sc
    sc.SharedMemory = SafeSharedMemory


# Patch at import time, not only in main(): under spawn/forkserver the pool workers re-import
# this file as __mp_main__ and must get the same SharedMemory class as the parent.
install()


def main():
    from cell_eval.__main__ import main as cell_eval_main
    sys.argv[0] = "cell-eval"
    return cell_eval_main()


if __name__ == "__main__":
    sys.exit(main())
