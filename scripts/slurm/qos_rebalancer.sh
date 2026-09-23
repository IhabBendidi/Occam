#!/usr/bin/env bash
#SBATCH --job-name=qos_rebalance
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/slurm/qos_rebalance_%j.out
#SBATCH --error=runs/slurm/qos_rebalance_%j.err
# =====================================================================
# Move MY jobs that are stuck on a full hooke-predict to a QOS with room.
#
# Rule (user, 2026-09-23):
#   1. hooke-predict first (where everything is submitted);
#   2. if it is full (job pending on QOSGrpGRES) -> vco, when vco has free GPUs;
#   3. if vco is full too -> normal, but ONLY for short jobs (time limit <= SHORT_MAX),
#      because normal is preemptible (PreemptMode=REQUEUE) by hooke-predict / vco jobs.
# wckey is set to match the QOS (vco needs wckey vco).
#
# Only jobs whose pending reason is QOSGrpGRES / QOSGrpCpuLimit / QOSGrpMemLimit are
# touched: a job waiting on a Dependency is left alone until it becomes eligible.
# CPU-only; it only calls squeue/scontrol. Every move is logged.
#
# USAGE  sbatch scripts/slurm/qos_rebalancer.sh
#        POLL_SECS=300 SHORT_MAX_MIN=240 VCO_CAP=32 sbatch scripts/slurm/qos_rebalancer.sh
# =====================================================================
set -uo pipefail
POLL_SECS="${POLL_SECS:-300}"
SHORT_MAX_MIN="${SHORT_MAX_MIN:-240}"
VCO_CAP="${VCO_CAP:-32}"

gpus_running() {   # total GPUs allocated to RUNNING jobs of a QOS (all users)
    squeue -q "$1" -h -t R -O "tres-alloc:120" | grep -oE "gres/gpu=[0-9]+" \
        | cut -d= -f2 | awk '{s+=$1} END{print s+0}'
}
job_gpus() {       # GPUs one task of this job asks for
    local g; g=$(squeue -j "$1" -h -O "tres-per-node:60" | grep -oE "gpu(:[a-z0-9]+)?:[0-9]+" \
        | head -1 | awk -F: '{print $NF}')
    echo "${g:-0}"
}
limit_min() {      # time limit in minutes, from D-HH:MM:SS / HH:MM:SS / MM:SS
    echo "$1" | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3;
                                else if (NF==3) print $1*60+$2; else print $1 }'
}

echo "=== qos rebalancer · started $(date -Is) · job ${SLURM_JOB_ID:-none} ==="
echo "poll ${POLL_SECS}s · vco cap ${VCO_CAP} · normal only if limit <= ${SHORT_MAX_MIN} min"
while true; do
    vco_used=$(gpus_running vco)
    while IFS='|' read -r jid qos lim reason; do
        [ "$qos" = "hooke-predict" ] || continue
        case "$reason" in QOSGrpGRES|QOSGrpCpuLimit|QOSGrpMemLimit) ;; *) continue ;; esac
        g=$(job_gpus "$jid"); [ "$g" -gt 0 ] || continue
        if [ $(( vco_used + g )) -le "$VCO_CAP" ]; then
            if scontrol update JobId="$jid" QOS=vco WCKey=vco 2>/dev/null; then
                vco_used=$(( vco_used + g ))
                echo "[$(date +%H:%M:%S)] $jid -> vco   (limit $lim, $g GPU; vco now ~$vco_used/$VCO_CAP)"
            fi
        elif [ "$(limit_min "$lim")" -le "$SHORT_MAX_MIN" ]; then
            if scontrol update JobId="$jid" QOS=normal 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] $jid -> normal (limit $lim <= ${SHORT_MAX_MIN} min; preemptible)"
            fi
        fi
    done < <(squeue -u "$USER" -h -t PD -o "%i|%q|%l|%r")
    sleep "$POLL_SECS"
done
