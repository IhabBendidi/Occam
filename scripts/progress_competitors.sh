#!/usr/bin/env bash
# Compact per-arm training progress. ONE LINE per job.
#
# ⚠️ WHY THIS EXISTS: PerturbDiff's tqdm line carries ~800 `grad_2.0_norm/<param>` fields, so
# `tail -3` on its log is ~40 KB of noise that hides the three numbers that matter. Never
# tail a competitor log directly; use this.
cd "$(dirname "$0")/.." || exit 1
# Only jobs the watchdog is tracking, unless --all. Superseded attempts are noise: eight
# dead PerturbDiff jobs all sit at the same within-epoch position (the first validation)
# and reading them as live is how a fixed run gets mistaken for a broken one.
TRACKED=$(python -c "
import json,sys
try: d=json.load(open('runs/competitors/watch.json'))
except Exception: sys.exit()
print(' '.join(j for v in d.values() for j in v))" 2>/dev/null)
for f in runs/slurm/cmp_*_*.out runs/slurm/cmp_*_*.err; do
    [ -s "$f" ] || continue
    if [ "$1" != "--all" ] && [ -n "$TRACKED" ]; then
        keep=no
        for j in $TRACKED; do case "$f" in *"$j"*) keep=yes;; esac; done
        [ "$keep" = yes ] || continue
    fi
    case "$f" in *_de.csv) continue;; esac
    line=$(tr '\r' '\n' < "$f" | grep -aE "Epoch [0-9]+:| it/s" | tail -1)
    [ -n "$line" ] || continue
    ep=$(printf '%s' "$line" | grep -oE "Epoch [0-9]+" | tail -1)
    step=$(printf '%s' "$line" | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:?]+" | tail -1)
    loss=$(printf '%s' "$line" | grep -oE "training_loss_step=[0-9.e+-]+|loss=[0-9.e+-]+|v_num[^,]*" | head -1)
    gnorm=$(printf '%s' "$line" | grep -oE "grad_2.0_norm_total=[0-9.e+-]+|grad_norm=[0-9.e+-]+" | head -1)
    rate=$(printf '%s' "$line" | grep -oE "sec/step=[0-9.]+|[0-9.]+it/s" | tail -1)
    nval=$(grep -ac "selection_score=[0-9]" "$f" 2>/dev/null)
    printf "%-30s %-8s %-24s %-26s %-15s val=%s\n" \
        "$(basename "$f")" "${ep:--}" "${step:--}" "${gnorm:--}" "${rate:--}" "${nval:-0}"
done
