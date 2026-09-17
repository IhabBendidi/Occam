#!/usr/bin/env bash
# Shared plumbing for every competitor launcher. Sourced, never executed.
#
# WHY A COMMON FILE
# -----------------
# Four arms, three legs each (train / export / score), and the parts that must be identical
# across all twelve are exactly the parts that are easy to get subtly wrong: which repo we are
# in, which interpreter belongs to which arm, and where the run products land.
#
# ⚠️ NEVER put a `#` comment inside a backslash-continued command in any launcher that sources
# this. A comment terminates the continuation: upstream it silently dropped 29 of 32 Hydra
# overrides on STATE, which then fell back to `name=debug`, auto-resumed an unrelated
# checkpoint, and exited 0. That is the most expensive bug recorded in this port.

set -euo pipefail

# Slurm COPIES the batch script into a node spool dir, so ${BASH_SOURCE[0]} does not point into
# the repo. Prefer the submit dir, and verify whichever candidate wins.
_find_repo() {
    local c
    for c in "${OCCAM_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "$c" ] && [ -f "$c/pyproject.toml" ] && [ -d "$c/src/occam" ]; then
            echo "$c"; return 0
        fi
    done
    echo "cannot locate the Occam repo; sbatch from the repo root or set OCCAM_REPO" >&2
    return 1
}

# Each arm's own interpreter. The whole point of the two-phase bridge is that these are NOT
# our interpreter -- an arm's checkpoint can only be loaded by the environment that pinned it.
competitor_venv_python() {
    local arm="$1" repo="$2" p
    case "$arm" in
        txpert)      p="/mnt/ps/home/CORP/ihab.bendidi/.pyenv/versions/gspp/bin/python" ;;
        state)       p="$repo/competitors/STATE/state/.venv/bin/python" ;;
        cellflow)    p="$repo/competitors/CellFlow/cellflow/.venv/bin/python" ;;
        perturbdiff) p="$repo/competitors/PerturbDiff/.venv/bin/python" ;;
        *) echo "unknown arm: $arm (txpert|state|cellflow|perturbdiff)" >&2; return 1 ;;
    esac
    if [ ! -x "$p" ]; then
        echo "no interpreter for $arm at $p" >&2
        echo "  build it:  cd $(dirname "$(dirname "$p")") && uv sync" >&2
        return 1
    fi
    echo "$p"
}

# Our own interpreter, which scores everything. Phase B only.
OCCAM_PY=/mnt/ps/home/CORP/ihab.bendidi/.pyenv/versions/scmae/bin/python

# ⚠️ uv's default cache is on NFS while the venvs are on GPFS; unpacking torch across that
# boundary takes hours instead of a minute. Upstream README_HOOKE_TX.md records this.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"

competitor_banner() {
    local arm="$1" leg="$2" seed="$3" repo="$4"
    echo "=================================================================="
    echo "  competitor: $arm   leg: $leg   seed: $seed"
    echo "  repo   : $repo"
    echo "  node   : $(hostname)   job: ${SLURM_JOB_ID:-none}/${SLURM_ARRAY_TASK_ID:-0}"
    echo "  data   : OUR h5ads, OUR 6,983-gene axis, OUR splits (rhj_fewshot)"
    echo "  scored : OUR cell-eval 0.6.0 path, uncalibrated (model_nocal)"
    echo "  anchors: ctrlcell -0.0018 · meanpert 0.1324 · transfer 0.2323 ·"
    echo "           experimental 0.3722   (pearson_delta, all 5 splits)"
    echo "=================================================================="
}
