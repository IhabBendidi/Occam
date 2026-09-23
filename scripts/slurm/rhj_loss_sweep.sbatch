#!/usr/bin/env bash
# The LOSS-COMPOSITION sweep: optimisation knobs FROZEN at our best arm, loss terms varied.
#
#   GRID=scripts/grids/rhj_loss_sweep.tsv \
#   sbatch --qos=hooke-predict --wckey=hooke-predict --array=0-14%8 \
#          scripts/slurm/rhj_loss_sweep.sbatch
#   sbatch --qos=vco --wckey=vco --array=0-14%5 scripts/slurm/rhj_loss_sweep.sbatch
# qos and wckey must MATCH.
#
# WHAT THIS ANSWERS
# -----------------
# Overall goal: predict a perturbation's effect in a context where that perturbation was never
# measured, from a single control cell. Headline metric: cell-eval `pearson_delta` on the
# 611-perturbation HepG2 test cohort. Anchors: meanpert 0.1324 · transfer 0.2323 ·
# experimental 0.3722 · our best 0.3415.
#
# THIS SWEEP'S QUESTION: does the RECONSTRUCTION LIKELIHOOD or the EFFECT-DIRECTION term buy
# anything, once the optimisation knobs are held at the best known setting?
#
# Two things motivate it, both recorded:
#   * F6 concluded `smooth_l1` > `nb` 12/12 -- but F6 ITSELF records the confound: NB and
#     SmoothL1 differ by a measured ~6.5x in scale, so at cosine_loss_weight 0.5 the smooth_l1
#     arms were genuinely half-cosine while the NB arms were effectively PURE NB. F6 measured a
#     loss *configuration*, not a likelihood. It was also GNN-era (`pert_model_type=gnn`) and
#     its arms were NEVER SCORED -- the eval array died on a calibrate='quantile' launch bug,
#     1 of 27 arms produced a result. So NB is untested on adjacency_map. See F-emit1.
#   * Occam's existing `cosine_loss_weight` is cos(prediction, target) on the PROFILE, which is
#     dominated by the baseline expression every perturbation shares. DCOS scores
#     1 - cos(pred - ctrl_mean, group_mean - ctrl_mean) instead: the direction of the EFFECT.
#
# HELD FIXED (our best arm, s313): pert_model_type=adjacency_map, batch_size=2048, lr=1e-3,
# decoder_lr=1e-3, cosine_loss_weight=0.5, data_perts_only=false, film_mode=multiplicative_only,
# film_hidden_dim=256, film_mean_free=true, epochs=30, seed=42, and the frozen benchmark.
#
# VARIED: loss_type, nb_weight, nb_scale, delta_cos_weight (one row per arm in the TSV).
#
# HOW TO READ THE RESULT
# ----------------------
# ⚠️ JUDGE ON THE VALIDATION METRIC FIRST -- that is what this sweep is for. Do NOT score the
# test panel until the validation ordering is decided, or the panel stops being held out.
# ⚠️ AND READ THE PER-EPOCH CURVE BEFORE RANKING. F7: the validation metric oscillates by ±0.20
# between consecutive epochs, against a between-arm span of ~0.012 in the F6 grid. A single
# best-epoch number is NOT a ranking signal on its own.
#
# WHAT EACH OUTCOME MEANS
#   * a scale-matched NB arm (L07-L09) beats smooth_l1 -> F6's verdict was the confound, not
#     the likelihood, and the loss choice should be revisited.
#   * it does not -> F6's conclusion survives on adjmap too, now without the confound.
#   * DCOS (L10-L13) helps -> the profile cosine was the wrong cosine, and the effect direction
#     is worth supervising directly.
#   * DCOS does nothing -> the FiLM/adjacency path already captures direction, and the
#     remaining gap is magnitude or context, not direction.

#SBATCH --job-name=rhj_loss
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=hop39
#SBATCH --output=runs/slurm/rhj_loss_%A_%a.out
#SBATCH --error=runs/slurm/rhj_loss_%A_%a.err

set -euo pipefail

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
HERE="$(_find_repo)" || exit 1
cd "$HERE"
mkdir -p runs/slurm

export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

GRID="${GRID:-scripts/grids/rhj_loss_sweep.tsv}"
CONFIG="${CONFIG:-configs/rhj_fewshot.yaml}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

ROW="$(grep -v '^#' "$GRID" | tail -n +2 | sed -n "$((IDX + 1))p")"
if [ -z "$ROW" ]; then
    echo "no row at index $IDX in $GRID" >&2
    exit 1
fi
# 6th column `delta_mag_weight` is OPTIONAL: rows from the original 5-column grid leave MAGW
# empty and it defaults to 0, which is the value all 540 runs to date used. Without the
# default, adding the column would silently pass an empty string to --set.
# ⚠️ BASE is read even though it is documentation only. With `read -r`, the LAST variable
# absorbs every remaining field -- so omitting it put "1e-7<TAB>M03" into RETA, which corrupted
# both the run suffix and the --set value. Caught by the tab in a run directory name.
# ⚠️ FIELD-COUNT GUARD. `read -r` gives the LAST variable every remaining field, so a grid
# with one unexpected column does not fail -- it silently concatenates into the last one. That
# is exactly how "1e-7<TAB>M03" ended up inside RETA, corrupting both the run suffix and the
# --set value with no error. Refuse instead.
_NF="$(printf '%s\n' "$ROW" | awk -F'\t' '{print NF}')"
if [ "${_NF:-0}" -gt 28 ]; then
    echo "grid row $IDX has $_NF tab-separated fields; this launcher reads at most 28. A new" >&2
    echo "column must be added to the read below, or it lands inside GVW unnoticed." >&2
    exit 1
fi
# The three F-disp1 columns (DCC RSW GVW) are appended AFTER `BASE`, so every 22-column grid
# still parses unchanged and they default to 0.0 -- the no-op value all 570 runs before
# 2026-09-21 used.
read -r ARM LOSS NBW NBS DCW MAGW LLT LLW LCW LCT RSP RCW RDW RSM RDT VVW VCW VGA VTG REP RETA BASE DCC RSW GVW BSZ LR RSEED <<<"$ROW"
# Column 26, OPTIONAL like 20-25, so every 25-column grid parses unchanged and gets the
# 2048 that all 600+ runs before 2026-09-22 used.
# ⚠️ WHY IT IS A COLUMN AND NOT A CONSTANT. batch_size was hardcoded at 2048 in the --set
# list below, tuned on rhj_fewshot's 612,122 training cells = 298 optimiser steps/epoch.
# norman_combo has 56,971 = 27 steps/epoch, so the SAME recipe traverses its cosine
# schedule in 4,050 updates instead of 44,700 -- an 11x difference in the amount of
# optimisation, invisible in the config. Varying it is the point of grid B.
BSZ="${BSZ:-2048}"
# Column 27, OPTIONAL, default 1e-3. ⚠️ `lr` was a CONSTANT in the --set list -- 1e-3 came
# from rhj_fewshot's W grid, where it was raised together with batch_size. It has never
# been varied on norman_combo, and with 11x fewer optimiser steps per run the right lr is
# not obviously the same one. `decoder_lr` tracks it, as it did in the W grid.
LR="${LR:-1e-3}"
# Column 28, OPTIONAL. Lets SEED REPLICATES live in a grid instead of needing one sbatch
# submission per seed: the seed is what re-draws the trained model, so it is the unit of
# replication for any statistical test (scripts/seed_split_test.py), and a 5-seed design
# is a 5-row grid rather than 5 launches. Empty = the SEED env var, then 42.
RSEED="${RSEED:-}"
DCC="${DCC:-0.0}"   # dcos_conc_weight      -- calibrate per-cell effect-direction alignment
RSW="${RSW:-0.0}"   # radial_spread_weight  -- match the spread of ||pred - ctrl_mean||
GVW="${GVW:-0.0}"   # gene_var_weight       -- match per-gene within-group variance
BASE="${BASE:-}"   # consumed so it cannot leak into RETA
VVW="${VVW:-0.0}"      # vicreg_var_weight
VCW="${VCW:-0.0}"      # vicreg_cov_weight
VGA="${VGA:-0.0}"      # vicreg_gamma -- a var weight without it REFUSES, by design
VTG="${VTG:-delta_z}"  # vicreg_target: delta_z | z_hat
RSM="${RSM:-cmean}"   # recon_split_mode:   raw | cmean | delta_only
RDT="${RDT:-cell}"    # recon_delta_target: cell | group_mean  (group_mean = PHASE 2)
# Columns 11-13, OPTIONAL like 7-10, so every earlier grid still parses unchanged.
RSP="${RSP:-false}"   # recon_split
RCW="${RCW:-1.0}"     # recon_ctrl_weight
RDW="${RDW:-1.0}"     # recon_delta_weight
MAGW="${MAGW:-0.0}"
# Latent columns 7-10, all OPTIONAL so every earlier grid still parses. Defaults are the
# no-op values, which is what all 540 runs before 2026-09-20 used.
LLT="${LLT:-none}"      # latent_loss_type: none | delta_z | projected
LLW="${LLW:-0.0}"       # latent_loss_weight (scales the MSE latent loss)
LCW="${LCW:-0.0}"       # latent_cos_weight
LCT="${LCT:-delta_z}"   # latent_cos_target: delta_z | z_hat -- NOT equivalent

# PER-ROW seed (column 28) takes precedence over the env var, matching how REP/RETA
# already work: a grid that VARIES the seed across rows cannot express that with an env
# var, which is global to the job.
SEED="${RSEED:-${SEED:-42}}"
# Optional schedule overrides. Unset = the config's own values (30 epochs, eta_min 1e-7), so
# every earlier grid reproduces exactly.
# Schedule overrides. PER-ROW (grid columns 20-21) takes precedence over the env vars, because
# a sweep that VARIES epochs or eta_min across arms cannot express that with an env var -- those
# are global to the job. Env vars remain for the case where one value applies to every arm.
# Unset in both -> the config's own values, so every earlier grid reproduces exactly.
EPOCHS="${REP:-${EPOCHS:-}}"
ETA_MIN="${RETA:-${ETA_MIN:-}}"
SAVE_EVERY="${SAVE_EVERY:-}"   # >0 also writes epoch_NNN.pth every N epochs (2.13 GB each)
# ⚠️ The suffix must carry EVERY varied axis. It is the run's identity on disk, and two arms
# differing only in `delta_mag_weight` would otherwise write to the same directory and the
# second would silently resume/overwrite the first. `_mag` is appended only when the term is
# on, so every pre-existing run directory name is unchanged and still resolvable.
SUFFIX="${ARM}__${LOSS}_nbw${NBW}_nbs${NBS}_dcos${DCW}"
if [ "${MAGW}" != "0.0" ] && [ "${MAGW}" != "0" ]; then
    SUFFIX="${SUFFIX}_mag${MAGW}"
fi
# Latent axes join the identity only when ON, so every pre-existing run directory name is
# unchanged and still resolvable. Two arms differing only in a latent knob MUST differ here or
# the second silently overwrites the first.
if [ "${LLW}" != "0.0" ] && [ "${LLW}" != "0" ]; then
    SUFFIX="${SUFFIX}_lmse${LLW}"
fi
if [ "${LCW}" != "0.0" ] && [ "${LCW}" != "0" ]; then
    SUFFIX="${SUFFIX}_lcos${LCW}-${LCT}"
fi
# VICReg axes join the identity only when ON, so existing run dirs keep their names.
if [ "${VVW}" != "0.0" ] && [ "${VVW}" != "0" ]; then SUFFIX="${SUFFIX}_vvar${VVW}g${VGA}"; fi
if [ "${VCW}" != "0.0" ] && [ "${VCW}" != "0" ]; then SUFFIX="${SUFFIX}_vcov${VCW}"; fi
if [ "${VVW}" != "0.0" ] || [ "${VCW}" != "0.0" ]; then SUFFIX="${SUFFIX}-${VTG}"; fi
if [ "${RSP}" = "true" ]; then
    # mode AND delta target must be in the identity: two arms differing only in which target
    # the delta regresses against are different experiments, not the same run.
    SUFFIX="${SUFFIX}_rsplit${RSM}-${RDT}-${RCW}-${RDW}"
fi
# ⚠️ The schedule MUST be in the identity. Without it a 60-epoch arm would write to the same
# directory as its 30-epoch namesake and silently overwrite it.
# ⚠️ The BATCH SIZE must be in the identity: two arms differing only in it would write to
# the same directory and the second would silently resume the first. Appended only when it
# is not the historical 2048, so every pre-existing run directory name is unchanged.
if [ "${BSZ}" != "2048" ]; then SUFFIX="${SUFFIX}_bs${BSZ}"; fi
if [ "${LR}" != "1e-3" ]; then SUFFIX="${SUFFIX}_lr${LR}"; fi
if [ "${DCC}" != "0.0" ] && [ "${DCC}" != "0" ]; then SUFFIX="${SUFFIX}_dconc${DCC}"; fi
if [ "${RSW}" != "0.0" ] && [ "${RSW}" != "0" ]; then SUFFIX="${SUFFIX}_rspread${RSW}"; fi
if [ "${GVW}" != "0.0" ] && [ "${GVW}" != "0" ]; then SUFFIX="${SUFFIX}_gvar${GVW}"; fi
if [ -n "${EPOCHS}" ]; then SUFFIX="${SUFFIX}_e${EPOCHS}"; fi
if [ -n "${ETA_MIN}" ]; then SUFFIX="${SUFFIX}_em${ETA_MIN}"; fi
SUFFIX="${SUFFIX}_s${SEED}"

echo "=== rhj loss-composition sweep arm ${ARM} ==="
echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slurm   : job ${SLURM_ARRAY_JOB_ID:-?} task ${IDX}  qos ${SLURM_JOB_QOS:-?}"
echo "varied  : loss=${LOSS} nbw=${NBW} nbs=${NBS} dcos=${DCW} dmag=${MAGW}"
echo "latent  : type=${LLT} mse_w=${LLW} cos_w=${LCW} cos_target=${LCT}"
echo "sched   : epochs=${EPOCHS:-<config>} eta_min=${ETA_MIN:-<config>} save_every=${SAVE_EVERY:-0}"
echo "vicreg  : var=${VVW} cov=${VCW} gamma=${VGA} target=${VTG}"
echo "disp    : dcos_conc=${DCC} radial_spread=${RSW} gene_var=${GVW}"
echo "recon   : split=${RSP} mode=${RSM} delta_target=${RDT} ctrl_w=${RCW} delta_w=${RDW}"
echo "frozen  : adjacency_map batch=2048 lr=1e-3 dlr=1e-3 cos=0.5 dpo=false"
echo "          film=multiplicative_only fh=256 fmf=true epochs=30 seed=${SEED}  (= arm s313)"
# ⚠️ The real run dir is runs/rhj_fewshot_<suffix>/: the orchestrator derives it from the
# config's `name`, NOT from this launcher. Writing the marker to runs/rhj_loss_<suffix>/ (as
# this did) created empty directories that look like runs, are not, and were mistaken for the
# real ones once already.
RUN_DIR="runs/rhj_fewshot_${SUFFIX}"
echo "run dir : ${RUN_DIR}/vcell"
echo

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/slurm_job.json" <<JSON
{"arm_id": "${ARM}", "kind": "loss", "loss_type": "${LOSS}",
 "nb_weight": "${NBW}", "nb_scale": "${NBS}", "delta_cos_weight": "${DCW}",
 "delta_mag_weight": "${MAGW}",
 "latent_loss_type": "${LLT}", "latent_loss_weight": "${LLW}",
 "latent_cos_weight": "${LCW}", "latent_cos_target": "${LCT}",
 "recon_split": "${RSP}", "recon_split_mode": "${RSM}", "recon_delta_target": "${RDT}",
 "recon_ctrl_weight": "${RCW}", "recon_delta_weight": "${RDW}",
 "dcos_conc_weight": "${DCC}", "radial_spread_weight": "${RSW}",
 "gene_var_weight": "${GVW}",
 "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}", "slurm_task_id": "${IDX}",
 "qos": "${SLURM_JOB_QOS:-}", "host": "$(hostname)", "started": "$(date -Is)"}
JSON

python -m occam.train --config "$CONFIG" --stages vcell \
    --run-suffix "$SUFFIX" \
    --set "stages.vcell.graph.pert_model_type=adjacency_map" \
    --set "stages.vcell.optim.batch_size=${BSZ}" \
    --set "stages.vcell.optim.lr=${LR}" \
    --set "stages.vcell.optim.decoder_lr=${LR}" \
    --set "stages.vcell.loss.loss_type=${LOSS}" \
    --set "stages.vcell.loss.cosine_loss_weight=0.5" \
    --set "stages.vcell.loss.nb_weight=${NBW}" \
    --set "stages.vcell.loss.nb_scale=${NBS}" \
    --set "stages.vcell.loss.delta_cos_weight=${DCW}" \
    --set "stages.vcell.loss.delta_mag_weight=${MAGW}" \
    --set "stages.vcell.loss.latent_loss_type=${LLT}" \
    --set "stages.vcell.loss.latent_loss_weight=${LLW}" \
    --set "stages.vcell.loss.latent_cos_weight=${LCW}" \
    --set "stages.vcell.loss.latent_cos_target=${LCT}" \
    --set "stages.vcell.loss.recon_split=${RSP}" \
    --set "stages.vcell.loss.recon_ctrl_weight=${RCW}" \
    --set "stages.vcell.loss.recon_delta_weight=${RDW}" \
    --set "stages.vcell.loss.recon_split_mode=${RSM}" \
    --set "stages.vcell.loss.recon_delta_target=${RDT}" \
    --set "stages.vcell.loss.vicreg_var_weight=${VVW}" \
    --set "stages.vcell.loss.vicreg_cov_weight=${VCW}" \
    --set "stages.vcell.loss.vicreg_gamma=${VGA}" \
    --set "stages.vcell.loss.vicreg_target=${VTG}" \
    --set "stages.vcell.loss.dcos_conc_weight=${DCC}" \
    --set "stages.vcell.loss.radial_spread_weight=${RSW}" \
    --set "stages.vcell.loss.gene_var_weight=${GVW}" \
    --set "stages.vcell.graph.data_perts_only=false" \
    --set "stages.vcell.interaction.interaction_type=film" \
    --set "stages.vcell.interaction.film_mode=multiplicative_only" \
    --set "stages.vcell.interaction.film_hidden_dim=256" \
    --set "stages.vcell.interaction.film_mean_free=true" \
    --set "seed=${SEED}" \
    ${EPOCHS:+--set "stages.vcell.optim.epochs=${EPOCHS}"} \
    ${ETA_MIN:+--set "stages.vcell.optim.eta_min=${ETA_MIN}"} \
    ${SAVE_EVERY:+--set "stages.vcell.optim.save_every_epochs=${SAVE_EVERY}"}

echo
echo "finished: $(date -Is)"
echo
echo "⚠️ JUDGE ON VALIDATION FIRST. Check the per-epoch curve BEFORE ranking (F7, ±0.20 swing):"
echo "    grep 'pearson_delta: mean=' runs/slurm/rhj_loss_${SLURM_ARRAY_JOB_ID:-JOB}_${IDX}.out"
