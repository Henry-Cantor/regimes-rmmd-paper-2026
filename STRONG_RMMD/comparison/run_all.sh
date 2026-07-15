#!/usr/bin/env bash
# Phase 4 — train the full comparison suite (full RMMD + 5 ablations + 4 baselines) under the
# IDENTICAL fast protocol, then build the comparison table.
#
# RESUME: each model writes a .done marker on success. Re-running this script SKIPS finished models
# and retries the rest, so if it breaks partway you just run it again. (A model interrupted
# mid-training has no .done and is retrained from scratch — there is no mid-run checkpoint resume.)
# On a model failure the suite CONTINUES to the next model and reports failures at the end.
#
# Usage:
#   bash STRONG_RMMD/comparison/run_all.sh [CKPT_ROOT] [TEST_DATA]
#   PYBIN=/path/to/python bash STRONG_RMMD/comparison/run_all.sh        # custom interpreter
set -uo pipefail

CKPT_ROOT="${1:-ckpt}"
TEST_DATA="${2:-/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt}"
PYBIN="${PYBIN:-python}"
TRAIN_PY="STRONG_RMMD/training/rmmd_train_eval.py"
COMPARE_PY="STRONG_RMMD/comparison/run_comparison.py"
# CUDA-12 lazy module loading can spuriously throw cudaErrorNoKernelImageForDevice on older GPUs
# (e.g. V100/sm_70); EAGER loads all kernels at context init and fixes it with NO perf cost.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
mkdir -p "$CKPT_ROOT"

FAILED=()
run_one () {
  local label="$1"; shift
  local dir="$CKPT_ROOT/$label"
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo ">> [skip] $label (already done)"; return 0; fi
  echo ">> [train] $label  flags: $* ${TRAIN_EXTRA:-}"
  if "$PYBIN" "$TRAIN_PY" train --fast-protocol ${TRAIN_EXTRA:-} "$@" --checkpoint-dir "$dir"; then
    touch "$dir/.done"; echo ">> [done] $label"
  else
    echo ">> [FAIL] $label (continuing; rerun the script to retry)"; FAILED+=("$label")
  fi
}

# =====================================================================================
# STEP 1 — RMMD config selection (SENSITIVITY-FIRST), over TWO axes: learning rate AND latent dim.
# Pick the best BY VALIDATION; `full`, ALL ablations, AND the dimension-matched dgknet then inherit
# the SAME (latent, lr) — so each ablation isolates a COMPONENT (not an HP difference), the deltas
# are against the BEST full, and dgknet stays dimension-matched to whatever latent we actually use.
# One-at-a-time search around the reference point (latent 256): pick LR @ ld256, then latent @ best
# LR (so latent is compared FAIRLY, not handicapped by the slow default LR). Override the grids with
# env RMMD_LR_GRID / RMMD_LD_GRID; set RMMD_LD_GRID="" to skip latent selection (keep 256).
read -r -a RMMD_LRS <<< "${RMMD_LR_GRID:-5e-5 1e-4 2e-4}"
read -r -a RMMD_LDS <<< "${RMMD_LD_GRID-128 192}"   # below the 256 cap; "" disables latent search
REF_LR="${RMMD_LRS[0]}"
_pick() { "$PYBIN" STRONG_RMMD/comparison/select_lr_winners.py --print-winner "x=$1"; }

# 1a. LR sweep at the reference latent (256).
echo ">> STEP 1a: RMMD LR sweep {${RMMD_LRS[*]}} @ ld256"
for lr in "${RMMD_LRS[@]}"; do run_one "full_lr${lr}" --model rmmd --lr "$lr"; done
_lrc=""; for lr in "${RMMD_LRS[@]}"; do _lrc="${_lrc:+$_lrc,}$CKPT_ROOT/full_lr${lr}"; done
BEST_LR_DIR=$(_pick "$_lrc"); BEST_LR="${BEST_LR_DIR##*_lr}"
case "$BEST_LR" in *e-*) ;; *) BEST_LR="$REF_LR";; esac
echo ">> best LR (by val @ ld256) = $BEST_LR"

# 1b. latent sweep, ALL at the best LR (so the latent comparison isn't confounded by LR). ld256 @
# best LR is the LR winner itself; ld128/192 @ best LR are trained here (the ld*@ref-LR robustness
# points are reused only when best LR == ref). FULL_DIR = the winning (latent,lr) run.
FULL_DIR="$CKPT_ROOT/full_lr${BEST_LR}"; _ldc="$FULL_DIR"
echo ">> STEP 1b: RMMD latent sweep {256 ${RMMD_LDS[*]:-}} @ lr=$BEST_LR"
for ld in "${RMMD_LDS[@]:-}"; do
  [ -z "$ld" ] && continue
  if [ "$BEST_LR" = "$REF_LR" ]; then d="$CKPT_ROOT/sens_ld${ld}"; lab="sens_ld${ld}"
  else d="$CKPT_ROOT/full_ld${ld}_lr${BEST_LR}"; lab="full_ld${ld}_lr${BEST_LR}"; fi
  run_one "$lab" --model rmmd --latent-dim "$ld" --lr "$BEST_LR"
  _ldc="$_ldc,$d"
done
# EXTRA_DIM=1: also probe the (small-latent, 2e-4) corner the one-at-a-time sweep can miss — build
# ld128 & ld192 @ lr=2e-4 as ADDITIONAL candidates for the chosen config (skipped if the best LR
# already IS 2e-4, since the sweep above covered them).
if [ "${EXTRA_DIM:-0}" = "1" ] && [ "$BEST_LR" != "2e-4" ]; then
  echo ">> STEP 1b+: EXTRA_DIM — also building latent {128 192} @ lr=2e-4"
  for ld in 128 192; do
    run_one "full_ld${ld}_lr2e-4" --model rmmd --latent-dim "$ld" --lr 2e-4
    _ldc="$_ldc,$CKPT_ROOT/full_ld${ld}_lr2e-4"
  done
fi
FULL_DIR=$(_pick "$_ldc")
case "$FULL_DIR" in *ld128*) BEST_LD=128;; *ld192*) BEST_LD=192;; *) BEST_LD=256;; esac
# re-derive BEST_LR from the WINNING dir — with EXTRA_DIM the winner's LR can differ from STEP 1a's
# (e.g. ld128 @ 2e-4 won while the ld256 LR sweep picked 1e-4); the ablations must inherit the
# CHOSEN config's LR, not the ld256 winner's.
case "$FULL_DIR" in *_lr*) BEST_LR="${FULL_DIR##*_lr}";; *) BEST_LR="$REF_LR";; esac
case "$BEST_LR" in *e-*) ;; *) BEST_LR="$REF_LR";; esac
echo ">> CHOSEN RMMD config: latent=$BEST_LD  lr=$BEST_LR  (-> $FULL_DIR)"
if [ -e "$CKPT_ROOT/full" ] && [ ! -L "$CKPT_ROOT/full" ]; then rm -rf "$CKPT_ROOT/full"; fi
ln -sfn "$(basename "$FULL_DIR")" "$CKPT_ROOT/full"

# =====================================================================================
# STEP 2 — ablations, trained at the WINNING (latent, lr) so each isolates a COMPONENT only.
# SKIP_ABL=1 skips them entirely -> RMMD-sensitivity + baselines only (the full-vs-baselines run).
if [ "${SKIP_ABL:-0}" != "1" ]; then
  echo ">> STEP 2: ablations @ latent=$BEST_LD lr=$BEST_LR"
  ABL_HP=(--lr "$BEST_LR" --latent-dim "$BEST_LD")
  run_one abl_drivers    "${ABL_HP[@]}" --ablate-drivers
  run_one abl_geometry   "${ABL_HP[@]}" --ablate-geometry
  run_one abl_dres       "${ABL_HP[@]}" --ablate-dres
  run_one abl_transport  "${ABL_HP[@]}" --ablate-transport
  run_one abl_sut        "${ABL_HP[@]}" --loss-sut-weight-base 0
else
  echo ">> STEP 2: ablations SKIPPED (SKIP_ABL=1)"
fi

# =====================================================================================
# STEP 3 — baselines (LR-tuned separately in the grid below).
echo ">> STEP 3: baselines"
run_one base_mlp       --model mlp
run_one base_lstm      --model lstm
run_one base_node      --model node
# dgknet at koopman-dim == RMMD's CHOSEN latent (BEST_LD): the DIMENSION-MATCHED operator control
# (koopman_dim and RMMD latent_dim are the same object — the operator's state dimension; matching
# it isolates operator STRUCTURE from state size). Tracks the latent we actually picked in STEP 1.
export DGKNET_LATENT="$BEST_LD"   # the LR grid reads this for its dgknet candidates too
run_one base_dgknet    --model dgknet --baseline-latent-dim "$BEST_LD"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo ">> FAILED models: ${FAILED[*]} — rerun this script to retry them (finished models are skipped)"
fi

# Plain comparison table = default-LR numbers over all trained models. REDUNDANT with the LR-fair
# table below (which covers full + ablations + LR-tuned baselines), so
# SKIP IT and go straight to the LR grid to save eval time. Run it only as a FALLBACK when the LR
# grid is skipped (so there's always >=1 table), or force with RUN_PLAIN_COMPARISON=1.
if [ "${SKIP_LR_GRID:-0}" = "1" ] || [ "${RUN_PLAIN_COMPARISON:-0}" = "1" ]; then
  echo ">> building plain comparison table over completed models in $CKPT_ROOT"
  "$PYBIN" "$COMPARE_PY" --test-data "$TEST_DATA" --ckpt-root "$CKPT_ROOT" \
      --out STRONG_RMMD/comparison/results/comparison_table.json
else
  echo ">> skipping plain comparison table (redundant with the LR-fair table; RUN_PLAIN_COMPARISON=1 to force)"
fi

# RMMD sensitivity TABLE (robustness plot) — built from the STEP-1 points, already trained (the LR
# sweep + the latent variants). No retraining here; LR selection already happened in STEP 1. Then
# the baseline LR-fairness grid (-> comparison_table_fair.json, THE paper table). Both resumable.
# SKIP_SENSITIVITY=1 / SKIP_LR_GRID=1 to skip.
if [ "${SKIP_SENSITIVITY:-0}" != "1" ]; then
  echo ">> RMMD sensitivity table (-> sensitivity_table.json); chosen latent=$BEST_LD lr=$BEST_LR"
  _sens_models=()
  for lr in "${RMMD_LRS[@]}"; do _sens_models+=("lr${lr}=$CKPT_ROOT/full_lr${lr}"); done   # LR axis @ ld256
  for ld in "${RMMD_LDS[@]:-}"; do                                                         # latent axis @ best LR
    [ -z "$ld" ] && continue
    if [ "$BEST_LR" = "$REF_LR" ]; then _d="$CKPT_ROOT/sens_ld${ld}"; else _d="$CKPT_ROOT/full_ld${ld}_lr${BEST_LR}"; fi
    _sens_models+=("ld${ld}=$_d")
  done
  "$PYBIN" "$COMPARE_PY" --test-data "$TEST_DATA" --models "${_sens_models[@]}" \
      --out STRONG_RMMD/comparison/results/sensitivity_table.json || FAILED+=("sensitivity")
fi
# Per-model CONVERGENCE check: flags any model whose validation NI-NRMSE was still dropping at the
# end of training (undertrained) vs plateaued (converged) — so "is the mlp/dgknet actually trained
# out, or did we just stop early?" is answerable from the val curves, not assumed.
if [ "${SKIP_CONVERGENCE:-0}" != "1" ]; then
  echo ">> convergence check (val-NRMSE plateau per model)"
  "$PYBIN" STRONG_RMMD/comparison/convergence_check.py --ckpt-root "$CKPT_ROOT" \
      --out STRONG_RMMD/comparison/results/convergence_check.json || true
fi
if [ "${SKIP_LR_GRID:-0}" != "1" ]; then
  echo ">> LR-fairness grid (-> comparison_table_fair.json, THE paper table)"
  bash "$(dirname "$0")/run_baseline_lr_grid.sh" "$CKPT_ROOT" "$TEST_DATA" || FAILED+=("lr_grid")
fi

# ZERO-SHOT EXTRAPOLATION report (the headline: in-dist + holdout, activity-stratified + STRONG fit
# + per-machine). Set HOLDOUT_DATA=<holdout dataset_test_compact.pt> (EAST or AUGD) to emit it;
# skipped if unset. Uses the LR-fair selected models (comparison_models.json) when present.
if [ -n "${HOLDOUT_DATA:-}" ]; then
  echo ">> zero-shot extrapolation report (in-dist + holdout=$HOLDOUT_DATA) -> results/extrap.json"
  if [ -f "$CKPT_ROOT/comparison_models.json" ]; then EXM=(--models-json "$CKPT_ROOT/comparison_models.json")
  else EXM=(--ckpt-root "$CKPT_ROOT"); fi
  "$PYBIN" STRONG_RMMD/theory_validation/extrap_strong.py "${EXM[@]}" \
      --indist-data "$TEST_DATA" --holdout-data "$HOLDOUT_DATA" --reference full \
      --out STRONG_RMMD/comparison/results/extrap.json || FAILED+=("extrap")
else
  echo ">> (set HOLDOUT_DATA=<holdout .pt> to also emit the zero-shot extrapolation report)"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo ">> FAILED steps: ${FAILED[*]} — rerun this script to retry (finished work is skipped)"
fi
echo ">> Phase 4 COMPLETE. Tables: comparison_table_fair.json (paper) + sensitivity_table.json${HOLDOUT_DATA:+ + extrap.json (zero-shot)} (in STRONG_RMMD/comparison/results/)"
[ "${#FAILED[@]}" -eq 0 ] || exit 1
