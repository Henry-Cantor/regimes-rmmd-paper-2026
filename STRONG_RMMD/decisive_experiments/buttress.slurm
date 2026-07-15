#!/bin/bash
# EXP-6 -- four buttressing analyses in ONE per-shot pass: A (u* prediction), B (decision-theoretic alpha +
# the per-shot ROUTER-SOTA grid over every horizon x activity-quartile), C (nonlocality). No training; just forward
# passes of the committed models over the test + holdout shots. Now evaluates all 12 horizons, so budget ~4-6h.
#
#   export CEIL=<full_ld192 checkpoint_best.pt>          # required (the full RMMD model)
#   export ABL_DRES=<abl_dres checkpoint_best.pt>        # NEEDED for Exp A observed crossover AND the Tier-2 nonloc
#                                                        # threshold (err_abl vs err_full). Cantor Equation needs no extra.
#   export DGKNET=<base_dgknet checkpoint_best.pt>       # the DGKNet router arm (Exp A/B) -- optional but recommended
#   # EXTRAP = the router-SOTA BASELINES, pulled from the extrap JSONs (NOT re-evaluated). MACHINE:path pairs; the
#   # ~20 models' per-quartile/per-horizon errors come from holdout_activity_stratified:
#   export EXTRAP="EAST:STRONG_RMMD/theory_validation/results/extrap_strong_report_east.json AUGD:STRONG_RMMD/theory_validation/results/extrap_strong_report_augd.json"
#   export DATA=<phase0NEW> EAST=<...east...> AUGD=<...augd...> OUT=<decisive_out>
#   export NTRAIN=<training-window count>                # optional; N cancels in cross-machine ORDERING
#   sbatch buttress.slurm
#
#SBATCH --job-name=buttress
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=buttress_%j.out
set -uo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-/scratch/gpfs/$USER/decisive_out}"
DATA="${DATA:-/scratch/gpfs/$USER/phase0NEW}"
: "${CEIL:?set (and export) CEIL=<full_ld192 checkpoint_best.pt>}"
: "${EAST:?set (and export) EAST=<phase0NEW_east>/dataset_test_compact.pt}"
: "${AUGD:?set (and export) AUGD=<phase0NEW_augd>/dataset_test_compact.pt}"
cd "$REPO"
VENV="${VIRTUAL_ENV:-$HOME/.torchenv}"; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO/STRONG_RMMD:$REPO:${PYTHONPATH:-}"
python -c "import strong_rmmd" || { echo "FATAL: strong_rmmd not importable"; exit 1; }

DEC=STRONG_RMMD/decisive_experiments
# datasets: pooled test set (all 5 training machines, per-shot machine labels) + the two holdouts
POOL="$DATA/dataset_test_compact.pt"
[ -s "$POOL" ] || { echo "FATAL: pooled test set not found at $POOL"; exit 1; }

ABL_ARG=(); [ -n "${ABL_DRES:-}" ] && ABL_ARG=(--abl-dres-ckpt "$ABL_DRES") || echo "NOTE: ABL_DRES unset -> Exp A observed crossover falls back to flex-overtake only"
DGK_ARG=(); [ -n "${DGKNET:-}" ]   && DGK_ARG=(--dgknet-ckpt "$DGKNET")
# FLEX: space-separated candidates for B's high-drive rung; the router picks the best on the tail. e.g.
#   export FLEX="dgknet_lr1e-4:/path/best/checkpoint_best.pt node_lr1e-4:/path/... node_lr2e-4:/path/..."
FLEX_ARG=(); [ -n "${FLEX:-}" ] && FLEX_ARG=(--flex-ckpts $FLEX)
[ -z "${DGKNET:-}" ] && [ -z "${FLEX:-}" ] && echo "NOTE: no DGKNET/FLEX -> Exp B 3-way router + flex-overtake skipped"
# EXTRAP baselines for the router-SOTA grid (pulled from the extrap JSONs, not re-evaluated)
EXTRAP_ARG=(); [ -n "${EXTRAP:-}" ] && EXTRAP_ARG=(--extrap-json $EXTRAP) || echo "NOTE: EXTRAP unset -> router-SOTA competitors = the router's own arms only (no baselines)"

python $DEC/buttress.py --full-ckpt "$CEIL" "${ABL_ARG[@]}" "${DGK_ARG[@]}" "${FLEX_ARG[@]}" "${EXTRAP_ARG[@]}" \
    --datasets pool:"$POOL" east:"$EAST" augd:"$AUGD" \
    --n-train "${NTRAIN:-1}" --device cuda || { echo "=== FAILED (see traceback above) ==="; exit 1; }

echo "=== DONE. results/buttress.json written. Read: nonloc_equation (Cantor Nonlocality Equation: cv_r2_pooled +"
echo "    VERDICT + per_machine_cv_r2), nonloc_threshold (Tier-2: theta_CV_across_machines + VERDICT), B_router_sota"
echo "    (SOTA/beats-persistence cell counts), B alpha, C window_sensitivity (w50 anchor). ==="
