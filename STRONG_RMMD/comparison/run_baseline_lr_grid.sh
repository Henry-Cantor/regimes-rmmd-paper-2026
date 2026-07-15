#!/usr/bin/env bash
# Phase 4 — #1 baseline LR-fairness check (run AFTER run_all.sh; does NOT touch its runs).
#
# run_all.sh trained every model at the shared default LR (5e-5). A reviewer can argue the
# BASELINES were handicapped by RMMD's LR. This trains each baseline at EXTRA LRs into SEPARATE
# dirs, picks the best per baseline BY VALIDATION, and rebuilds a FAIR comparison table. RMMD +
# the 4 ablations are NOT re-tuned (an ablation must inherit RMMD's HPs). Resumable via .done.
#
# Usage: bash STRONG_RMMD/comparison/run_baseline_lr_grid.sh [CKPT_ROOT] [TEST_DATA]
set -uo pipefail

CKPT_ROOT="${1:-ckpt}"
TEST_DATA="${2:-/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt}"
PYBIN="${PYBIN:-python}"
TRAIN_PY="STRONG_RMMD/training/rmmd_train_eval.py"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"   # avoid CUDA-12 lazy-load no-kernel-image on sm_70
DEFAULT_LR="5e-5"                       # what run_all.sh used (dir = ckpt/base_<b>)
EXTRA_LRS=("2e-4" "1e-4")               # additional LRs to try per baseline
BASELINES=("mlp" "lstm" "node" "dgknet")

DGKNET_LATENT="${DGKNET_LATENT:-256}"   # dimension-matched dgknet koopman-dim == RMMD's CHOSEN
                                        # latent (run_all.sh exports this after STEP-1 selection).
train_one () {  # <model> <lr> <dir> [extra_flags]
  local model="$1" lr="$2" dir="$3" extra="${4:-}"
  # canonical dgknet = koopman-dim == RMMD latent_dim (dimension-matched operator control; must
  # match run_all.sh's DGKNET_LATENT).
  if [ -z "$extra" ] && [ "$model" = "dgknet" ]; then extra="--baseline-latent-dim $DGKNET_LATENT"; fi
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo ">> [skip] $dir"; return 0; fi
  echo ">> [train] $model @ lr=$lr -> $dir  $extra ${TRAIN_EXTRA:-}"
  if "$PYBIN" "$TRAIN_PY" train --fast-protocol --model "$model" --lr "$lr" $extra ${TRAIN_EXTRA:-} --checkpoint-dir "$dir"; then
    touch "$dir/.done"
  else
    echo ">> [FAIL] $dir (rerun to retry)"
  fi
}

# Train the EXTRA LR points (the default-LR run from run_all.sh is reused as a candidate).
for b in "${BASELINES[@]}"; do
  for lr in "${EXTRA_LRS[@]}"; do
    train_one "$b" "$lr" "$CKPT_ROOT/base_${b}_lr${lr}"
  done
done

# Build the candidate list per model and pick winners BY VALIDATION.
ENTRIES=( "full=$CKPT_ROOT/full" )
if [ "${SKIP_ABL:-0}" != "1" ]; then                    # ablation rows (omitted when run_all SKIP_ABL=1)
  ENTRIES+=( "abl_drivers=$CKPT_ROOT/abl_drivers" "abl_geometry=$CKPT_ROOT/abl_geometry"
             "abl_dres=$CKPT_ROOT/abl_dres" "abl_transport=$CKPT_ROOT/abl_transport"
             "abl_sut=$CKPT_ROOT/abl_sut" )
fi
for b in "${BASELINES[@]}"; do
  cands="$CKPT_ROOT/base_${b}"                         # default-LR (5e-5) run from run_all.sh
  for lr in "${EXTRA_LRS[@]}"; do cands="$cands,$CKPT_ROOT/base_${b}_lr${lr}"; done
  ENTRIES+=( "base_${b}=$cands" )
done

echo ">> selecting per-baseline LR winners by validation"
"$PYBIN" STRONG_RMMD/comparison/select_lr_winners.py --out "$CKPT_ROOT/comparison_models.json" "${ENTRIES[@]}"

echo ">> building FAIR comparison table"
"$PYBIN" STRONG_RMMD/comparison/run_comparison.py --test-data "$TEST_DATA" \
    --models-json "$CKPT_ROOT/comparison_models.json" \
    --out STRONG_RMMD/comparison/results/comparison_table_fair.json
echo ">> done. Fair table: STRONG_RMMD/comparison/results/comparison_table_fair.json"
