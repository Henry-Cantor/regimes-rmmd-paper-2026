#!/usr/bin/env bash
# NOTE (2026-06): run_all.sh now does RMMD LR selection SENSITIVITY-FIRST inline (trains full_lr*,
# picks the best by validation, points `full` at it, trains the ablations at that LR) and builds
# sensitivity_table.json itself from those points. This standalone script is kept only for ad-hoc
# sweeps; it is NO LONGER called by run_all.sh and its sens_lr* dirs are independent of full_lr*.
#
# Phase 4 — #2 RMMD hyperparameter SENSITIVITY (supplement; run AFTER the ablations).
#
# Vary ONE hyperparameter at a time around the chosen config (latent_dim=384, lr=5e-5) under the
# SAME fast protocol, to show the result is stable (not a lucky HP). The default point reuses
# run_all.sh's ckpt/full. Emits a sensitivity table the notebook plots. Resumable via .done.
# This is a robustness CONFIRMATION around the already-chosen config; it does NOT re-select HPs and
# does NOT affect the ablation table.
#
# Usage: bash STRONG_RMMD/comparison/run_rmmd_sensitivity.sh [CKPT_ROOT] [TEST_DATA]
set -uo pipefail

CKPT_ROOT="${1:-ckpt}"
TEST_DATA="${2:-/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt}"
PYBIN="${PYBIN:-python}"
TRAIN_PY="STRONG_RMMD/training/rmmd_train_eval.py"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"   # avoid CUDA-12 lazy-load no-kernel-image on sm_70

train_one () {  # <dir> <extra flags...>
  local dir="$1"; shift
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo ">> [skip] $dir"; return 0; fi
  echo ">> [train] $dir  flags: $* ${TRAIN_EXTRA:-}"
  if "$PYBIN" "$TRAIN_PY" train --fast-protocol --model rmmd ${TRAIN_EXTRA:-} "$@" --checkpoint-dir "$dir"; then
    touch "$dir/.done"
  else
    echo ">> [FAIL] $dir (rerun to retry)"
  fi
}

# one-at-a-time around the chosen config; the default point is run_all.sh's ckpt/full.
# NOTE: train_command CAPS latent_dim at 256 (the chosen config IS 256), so the latent sweep
# must go BELOW the cap — values above 256 would silently train the IDENTICAL model.
train_one "$CKPT_ROOT/sens_ld128"  --latent-dim 128
train_one "$CKPT_ROOT/sens_ld192"  --latent-dim 192
train_one "$CKPT_ROOT/sens_lr2e-4" --lr 2e-4
train_one "$CKPT_ROOT/sens_lr1e-4" --lr 1e-4

echo ">> building sensitivity table (default = ckpt/full, latent 256 / lr 5e-5)"
"$PYBIN" STRONG_RMMD/comparison/run_comparison.py --test-data "$TEST_DATA" \
    --models "default_ld256_lr5e-5=$CKPT_ROOT/full" \
             "ld128=$CKPT_ROOT/sens_ld128" "ld192=$CKPT_ROOT/sens_ld192" \
             "lr2e-4=$CKPT_ROOT/sens_lr2e-4" "lr1e-4=$CKPT_ROOT/sens_lr1e-4" \
    --out STRONG_RMMD/comparison/results/sensitivity_table.json
echo ">> done. Sensitivity table: STRONG_RMMD/comparison/results/sensitivity_table.json"
