#!/usr/bin/env bash
#SBATCH --job-name=score_ledger
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/slurm/score_ledger_%j.out
#SBATCH --error=runs/slurm/score_ledger_%j.err
# The score ledger daemon (scripts/score_ledger.py --daemon): every POLL seconds, resubmit every
# registered split (configs/score_registry.tsv) that is missing and has no live job, up to
# LEDGER_MAX_ATTEMPTS per split; waits for training (final_model.pth) and exports (pred.npy).
# Replaces the per-benchmark autoeval daemons. Needs the Occam env for the eval jobs it submits.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
python -u scripts/score_ledger.py --daemon --poll "${POLL:-600}"
