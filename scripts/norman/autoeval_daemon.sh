#!/usr/bin/env bash
#SBATCH --job-name=norman_autoeval
#SBATCH --partition=cpu
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --output=runs/slurm/norman_autoeval_%j.out
#SBATCH --error=runs/slurm/norman_autoeval_%j.err
# =====================================================================
# AUTO-EVAL: score every norman_combo arm the moment its training finishes
#
# WHY THIS EXISTS
#   `rhj_loss_sweep.sbatch` is training-only -- it contains no `sbatch` call -- so the 30
#   sweep arms would finish and sit as checkpoints with no scores. On this benchmark the
#   eval has always been a SEPARATE submission (build a grid from the finished
#   checkpoints, then norman_models.sbatch), which needs someone present to fire it. This
#   job is that someone, for the ~10 h nobody is watching.
#
# ⚠️ WHY IT CANNOT JUST BE AN `afterok` DEPENDENCY. The eval grid is built from the
#   FINISHED run directory: `build_mode_grid.py` resolves `<arm>__<tag>` by globbing
#   `runs/norman_combo_<arm>__*` and requires the checkpoint to exist. There is nothing to
#   depend on at submit time, because the tag is not known until the run dir is created.
#
# ⚠️ `final_model.pth` IS THE COMPLETION MARKER, not `best_model.pth`. best_model appears
#   as soon as validation improves -- i.e. within the first few epochs -- so keying on it
#   would score a half-trained model and report it as the arm. final_model is written ONLY
#   at the very end (user, 2026-09-21: "final_model is only saved at the end").
#   The eval itself then scores `best_model.pth`, which is the selected checkpoint; the
#   marker and the scored file are deliberately different.
#
# WHAT IT DOES, every POLL_SECS until the arms are all in or the walltime ends
#   1. glob runs/norman_combo_*/vcell/checkpoints/final_model.pth
#   2. skip any arm already submitted (a LEDGER line) or already carrying results
#   3. write a one-arm grid and sbatch norman_models.sbatch --array=0-4%6 for it
#   Submitting PER ARM rather than one big array is deliberate: an arm that finishes at
#   hour 9 still gets scored, and one failed arm cannot take the others' tasks with it.
#
# ⚠️ %6 on the score array is the pdex /dev/shm race limit -- at %12 roughly half the
#   marginal tasks failed. It is per submission, so several arms in flight can still
#   exceed it; POLL_SECS is 600 and an eval task is ~5-7 min, which keeps it loose.
#
# COST  one CPU slot on the `cpu` partition, no GPU. It only submits.
# ⚠️ It MUST be the `cpu` partition: hopper/ampere/def refuse a job that requests no GPU
# ("A GPU must be requested for jobs in the 'hopper', 'ampere', or 'def' partitions"), and
# requesting a GPU for a process that only calls sbatch would hold an H100 idle for 13 h.
#
# USAGE
#   sbatch scripts/norman/autoeval_daemon.sh
#   ARMS_RE='^N[ABCDE][0-9]+$' POLL_SECS=600 MODES=decode_seq_mean sbatch ...
# =====================================================================
set -uo pipefail

HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "$HERE" || exit 1
mkdir -p runs/slurm runs/norman_combo

POLL_SECS="${POLL_SECS:-600}"
MODES="${MODES:-decode_seq_mean}"
# Which arms this daemon owns. The sweep arms are NA../NB../NC../ND../NE..; the published
# bases (NM03/NM06/NQ02*) are already scored and must not be resubmitted.
ARMS_RE="${ARMS_RE:-^N[ABCDE][0-9]+$}"
LEDGER="${LEDGER:-runs/norman_combo/autoeval_submitted.txt}"
DEADLINE=$(( $(date +%s) + ${MAX_SECS:-46800} ))   # 13 h, inside the 14 h walltime
touch "$LEDGER"

echo "=== norman auto-eval daemon ==="
echo "started : $(date -Is)"
echo "host    : $(hostname)   job ${SLURM_JOB_ID:-none}"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "poll    : every ${POLL_SECS}s until $(date -Is -d @$DEADLINE 2>/dev/null || echo '+13h')"
echo "arms    : regex $ARMS_RE   modes: $MODES"
echo "ledger  : $LEDGER ($(wc -l < "$LEDGER") already submitted)"
echo

round=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    round=$(( round + 1 ))
    found=0; new=0
    for marker in runs/norman_combo_*/vcell/checkpoints/final_model.pth; do
        [ -f "$marker" ] || continue
        rundir="${marker%/vcell/checkpoints/final_model.pth}"
        base="$(basename "$rundir")"                 # norman_combo_<ARM>__<TAG>
        rest="${base#norman_combo_}"
        arm="${rest%%__*}"
        tag="${rest#*__}"
        echo "$arm" | grep -qE "$ARMS_RE" || continue
        found=$(( found + 1 ))
        grep -qxF "$arm" "$LEDGER" && continue
        # `best_model.pth` is what gets SCORED; refuse rather than score a training that
        # finished without ever selecting a checkpoint.
        if [ ! -f "$rundir/vcell/checkpoints/best_model.pth" ]; then
            echo "[$(date +%H:%M:%S)] $arm: final_model present but NO best_model -- not" \
                 "scoring. Training finished without selecting a checkpoint." >&2
            echo "$arm" >> "$LEDGER"    # do not retry every 10 min
            continue
        fi
        grid="configs/autoeval_${arm}.tsv"
        {
            echo "# auto-generated by scripts/norman/autoeval_daemon.sh $(date -Is)"
            printf 'arm\ttag\tcompose\n'
            for m in ${MODES//,/ }; do printf '%s\t%s\t%s\n' "$arm" "$tag" "$m"; done
        } > "$grid"
        n_rows=$(( $(echo "${MODES//,/ }" | wc -w) ))
        n_tasks=$(( n_rows * 5 ))
        jid=$(sbatch --parsable --qos=hooke-predict --wckey=hooke-predict \
                     --array=0-$(( n_tasks - 1 ))%6 \
                     --export=ALL,GRID="$grid",N_SPLITS=5 \
                     scripts/slurm/norman_models.sbatch 2>&1 | tail -1)
        if echo "$jid" | grep -qE '^[0-9]+$'; then
            echo "$arm" >> "$LEDGER"
            new=$(( new + 1 ))
            echo "[$(date +%H:%M:%S)] SUBMITTED $arm ($tag) -> job $jid, $n_tasks task(s)"
        else
            echo "[$(date +%H:%M:%S)] $arm: sbatch FAILED, will retry next poll: $jid" >&2
        fi
    done
    echo "[$(date +%H:%M:%S)] round $round: $found finished arm(s) matching, $new newly" \
         "submitted, $(wc -l < "$LEDGER") total in ledger"
    sleep "$POLL_SECS"
done

echo
echo "=== daemon walltime reached: $(date -Is) ==="
echo "submitted in total: $(wc -l < "$LEDGER") arm(s)"
echo "⚠️ Arms that finish AFTER this point are NOT scored. Re-run this daemon, or:"
echo "    python scripts/norman/build_mode_grid.py --arms <arm> --modes $MODES \\"
echo "        --out configs/leftover.tsv"
echo "    GRID=configs/leftover.tsv sbatch --array=0-4%6 scripts/slurm/norman_models.sbatch"
