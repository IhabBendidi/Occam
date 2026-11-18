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

# ─── wandb: OFFLINE for every competitor leg ────────────────────────────────────────────────
# A PerturbDiff run sat RUNNING for 10 HOURS with a 0% GPU, 27 threads parked in
# futex_wait_queue_me, its 4 dataloader workers idle in do_poll, and its wandb debug log
# 20 lines long and stopped 2 seconds after init while `wandb-core` stayed alive futex-waiting.
# Shared memory (1% of 1 TB) and HDF5 fork-unsafety (zero .h5ad fds) were ruled out by
# measurement; the identical twin run completed, so it is a race, not a config error. The stack
# could not be captured (ptrace_scope=1, no sudo), so wandb is the LEADING but UNPROVEN cause.
#
# Offline removes the network-dependent blocking call outright. Nothing is lost: every metric
# we read comes from Lightning's metrics.csv and the job logs, and a run can still be uploaded
# afterwards with `wandb sync`. Set here, in the one file all twelve legs source, so no arm can
# be missed -- WANDB_MODE overrides the per-arm entity/project/name flags without touching them.
export WANDB_MODE="${WANDB_MODE:-offline}"

# ─── pdex SERIAL during in-training validation ──────────────────────────────────────────────
# pdex's parallel pool deadlocked inside two training processes on 2026-09-22 (TxPert s46,
# PerturbDiff s42, norman_combo): each went SILENT for 7+ h at "Processing batches", 0%
# progress, one H100 held idle apiece. A pool forked from a process that already holds
# CUDA and dataloader workers can deadlock; a serial run has no pool to deadlock. Read by
# TxPert's gspp/celleval_eval.py and PerturbDiff's sampling_eval_celleval.py. The
# validation set is small (norman 20 pairs), so serial costs seconds. The standalone
# SCORE leg (competitor_score.sbatch) is a fresh process with no CUDA context and keeps
# its own threads.
export CELLEVAL_NUM_THREADS="${CELLEVAL_NUM_THREADS:-1}"
# DataLoader tensor hand-off through the filesystem, not /dev/shm fds: the fd path died on
# "could not unlink the shared memory file /torch_*" and hung two PerturbDiff runs
# (2026-09-23). Read by competitors/PerturbDiff/src/apps/run/rawdata_diffusion_training.py.
# ⚠️ REVERTED to opt-in (2026-09-23): `file_system` still uses NAMED /dev/shm files, which
# systemd RemoveIPC deletes on any ssh logout to the node (the actual root cause, F-infra1), and
# the s45 relaunch died on "Pin memory thread exited unexpectedly" with it. Set it explicitly if
# wanted; the robust option for PerturbDiff is PD_WORKERS=0 (no shared memory at all).
export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-}"

# ─── stall watchdog: a hang must free its GPU, not hold it for hours ─────────────────────────
# `competitor_watchdog <pid> <logfile>` kills <pid> if <logfile> has not been written for
# STALL_SECS (default 45 min). Every legitimate phase writes far more often than that: an
# epoch is minutes, a validation cell-eval logs per batch. Both hangs above were caught
# only by a human reading mtimes 7 h later.
competitor_watchdog() {
    # competitor_watchdog <pid> <log> [<log> ...] -- the NEWEST mtime across the logs is
    # what counts: tqdm writes .err constantly during training while loguru writes .out
    # during validation, so either one alone goes quiet during a healthy phase.
    local pid="$1"; shift
    local limit="${STALL_SECS:-2700}"
    ( while kill -0 "$pid" 2>/dev/null; do
          sleep 300
          local newest=0 f m
          for f in "$@"; do
              [ -f "$f" ] || continue
              m=$(stat -c %Y "$f"); [ "$m" -gt "$newest" ] && newest=$m
          done
          [ "$newest" -gt 0 ] || continue
          local age=$(( $(date +%s) - newest ))
          if [ "$age" -gt "$limit" ]; then
              echo "[watchdog] no log written for ${age}s (> ${limit}s): killing $pid so" \
                   "the GPU is released. This is a HANG, not a slow run." >&2
              kill -TERM "$pid" 2>/dev/null; sleep 30; kill -KILL "$pid" 2>/dev/null
              exit 0
          fi
      done ) &
}

# ─── nodes to avoid ─────────────────────────────────────────────────────────────────────────
# hop39's GPU is dead -- `nvidia-smi` there returns "Unable to determine the device handle for
# GPU0 ... Unknown Error" / "No devices were found" -- while sinfo still reports the node `mix`,
# so Slurm keeps scheduling onto it. It failed every split of a STATE re-export with
# "CUDA unknown error". Launchers pass this as --exclude; clear it once the node is drained.
export COMPETITOR_EXCLUDE_NODES="${COMPETITOR_EXCLUDE_NODES:-hop39}"

# ─── which BENCHMARK, and the axis that comes with it ───────────────────────────────────────
# ⛔ THE ONE SOURCE OF TRUTH FOR "SAME GENES, SAME SPLITS". Every leg of every arm resolves
# the benchmark here, and reads the AXIS WIDTH and both content digests out of the artifact
# rather than carrying a literal. Three model configs used to hardcode 6983
# (PerturbDiff's model.input_dim and model.hidden_num, TxPert's hvg_number), so pointing an
# arm at norman_combo's 8,000-gene axis would have trained it on a 6,983-wide input against
# 8,000-wide targets -- which does not crash, it just scores badly and looks like the method
# being bad. Derive, never restate.
#
#   BENCH=rhj_fewshot   (default)  6,983 genes · 611 held-out HepG2 perturbations
#   BENCH=norman_combo             8,000 genes · 111 held-out double perturbations
competitor_resolve_bench() {
    local repo="$1"
    BENCH="${BENCH:-rhj_fewshot}"
    BENCH_DIR="$repo/data/benchmarks/$BENCH"
    PANEL_DIR="$BENCH_DIR/splithalf"
    if [ ! -f "$BENCH_DIR/benchmark.json" ]; then
        echo "no benchmark at $BENCH_DIR (BENCH=$BENCH)." >&2
        echo "Download benchmarks/$BENCH from the Hub, or build it with" >&2
        echo "scripts/repro/build_benchmark_*.sh -- see docs/hf_release.md." >&2
        exit 1
    fi
    if [ ! -f "$PANEL_DIR/panel.npz" ]; then
        echo "no split-half panel at $PANEL_DIR/panel.npz (BENCH=$BENCH)." >&2
        exit 1
    fi
    # The axis width and the two digests come out of the artifact. `n_axis_genes` is what
    # every model dimension must be set from; the digests are what makes "same genes, same
    # splits" checkable rather than asserted in a comment.
    local py_out
    py_out="$(python - "$BENCH_DIR" <<'PYEOF'
import csv, json, os, sys
d = sys.argv[1]
b = json.load(open(os.path.join(d, "benchmark.json")))
axis = b.get("gene_axis") or {}
n = (axis.get("n_genes") or axis.get("n_axis_genes") or b.get("n_axis_genes"))
if n is None:
    with open(os.path.join(d, "gene_axis.csv")) as fh:
        n = sum(1 for _ in csv.reader(fh)) - 1
sha = axis.get("sha256") or b.get("gene_axis_sha256") or ""
m = json.load(open(os.path.join(d, "splithalf", "splits_manifest.json")))
# The EVALUATED line -- the one whose perturbations are held out and scored. Arms
# branch on it (PerturbDiff's split, TxPert's test_cell_type), and every one of them
# had it hardcoded to HepG2.
line = b.get("cell_line") or ""
if not line:
    for rec in (b.get("sources") or {}).values():
        if rec.get("cell_line"):
            line = rec["cell_line"]
            break
if not line:
    ecl = b.get("eval_cell_lines") or []
    line = ecl[0] if ecl else ""
print(int(n))
print(sha)
print(m.get("benchmark_sha256", ""))
print(m.get("cohort_sha256", ""))
print(int(m.get("n_splits", 5)))
print(line)
PYEOF
)" || { echo "could not read $BENCH_DIR" >&2; exit 1; }
    N_AXIS_GENES="$(echo "$py_out" | sed -n 1p)"
    AXIS_SHA256="$(echo "$py_out"  | sed -n 2p)"
    BENCH_SHA256="$(echo "$py_out" | sed -n 3p)"
    COHORT_SHA256="$(echo "$py_out" | sed -n 4p)"
    BENCH_N_SPLITS="$(echo "$py_out" | sed -n 5p)"
    EVALUATED_LINE="$(echo "$py_out" | sed -n 6p)"
    if [ -z "${N_AXIS_GENES:-}" ] || [ "$N_AXIS_GENES" -lt 1 ] 2>/dev/null; then
        echo "could not determine the axis width for BENCH=$BENCH" >&2
        exit 1
    fi
    # Every arm already names its run `occam_s${SEED}${TAG_SUFFIX}` (TxPert, CellFlow,
    # State) or folds TAG_SUFFIX into PD_RUN_NAME (PerturbDiff), so defaulting it here
    # scopes all four training runs by benchmark with no per-arm edit. EMPTY on
    # rhj_fewshot, so every existing checkpoint directory keeps its name.
    # ⚠️ APPEND the benchmark tag to an explicit variant suffix, never replace it: a
    # `TAG_SUFFIX=_single` on norman_combo used to resolve to rhj's `occam_s42_single` dirs
    # (audit 2026-09-23). Unchanged for every existing run: rhj's tag is empty, and an empty
    # TAG_SUFFIX on another benchmark still gives `__<bench>`.
    _bt="$(competitor_bench_tag)"
    case "${TAG_SUFFIX:-}" in
        *"$_bt") TAG_SUFFIX="${TAG_SUFFIX:-}" ;;
        *)       TAG_SUFFIX="${TAG_SUFFIX:-}${_bt}" ;;
    esac
    export BENCH BENCH_DIR PANEL_DIR N_AXIS_GENES AXIS_SHA256 BENCH_SHA256 \
           COHORT_SHA256 BENCH_N_SPLITS TAG_SUFFIX EVALUATED_LINE
}

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
# The interpreter name is the pre-rename one and stays that way: renaming a live pyenv
# would break every other launcher in the repo.
OCCAM_PY=/mnt/ps/home/CORP/ihab.bendidi/.pyenv/versions/scmae/bin/python  # noqa: legacy-name

# ⚠️ uv's default cache is on NFS while the venvs are on GPFS; unpacking torch across that
# boundary takes hours instead of a minute. Upstream the upstream release notes records this.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/rxrx/scratch/sandbox/$USER/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/rxrx/scratch/sandbox/$USER/.uv-python}"

# ─── run paths must carry the BENCHMARK, or two benchmarks collide on disk ───────────────────
# `runs/competitors/<arm>/s42/split_0` and `runs/rhj_splithalf/compet_<arm>_s42` carry the arm
# and the seed but not the benchmark, so a norman_combo export would overwrite the
# rhj_fewshot one silently -- the same class of bug the ARM_SUFFIX comment below records.
# The tag is EMPTY on rhj_fewshot, so every existing directory keeps its name and every
# completed run stays resolvable; anything else appends `__<bench>`.
competitor_bench_tag() {
    if [ "${BENCH:-rhj_fewshot}" = "rhj_fewshot" ]; then echo ""; else echo "__${BENCH}"; fi
}

# Where the SCORED output goes. ⚠️ rhj_fewshot's scores live under `runs/rhj_splithalf/`,
# not `runs/rhj_fewshot/` -- a historical name that `scripts/export_split_metrics.py` maps
# to the benchmark and that 296 existing run directories use. Renaming it would orphan
# every one of them, so it is preserved and only new benchmarks use their own name.
competitor_score_root() {
    if [ "${BENCH:-rhj_fewshot}" = "rhj_fewshot" ]; then echo "runs/rhj_splithalf"
    else echo "runs/${BENCH}"; fi
}

competitor_banner() {
    local arm="$1" leg="$2" seed="$3" repo="$4"
    echo "=================================================================="
    echo "  competitor: $arm   leg: $leg   seed: $seed"
    echo "  repo   : $repo"
    echo "  node   : $(hostname)   job: ${SLURM_JOB_ID:-none}/${SLURM_ARRAY_TASK_ID:-0}"
    echo "  bench  : ${BENCH:-rhj_fewshot}  (${N_AXIS_GENES:-?}-gene axis)"
    echo "  axis   : ${AXIS_SHA256:0:16}   <- must match Occam's for the score to compare"
    echo "  cohort : ${COHORT_SHA256:0:16}   bench ${BENCH_SHA256:0:16}"
    echo "  data   : OUR h5ads, OUR axis, OUR splits -- the digests above are the proof"
    echo "  scored : OUR cell-eval 0.6.0 path, uncalibrated (model_nocal)"
    echo "  anchors: ctrlcell -0.0018 · meanpert 0.1324 · transfer 0.2323 ·"
    echo "           experimental 0.3722   (pearson_delta, all 5 splits)"
    echo "=================================================================="
}
