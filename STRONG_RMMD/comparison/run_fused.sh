#!/usr/bin/env bash
# Train the FUSED operator (the SOTA build): RMMD + a DGKNet skip blended by a PER-RADIUS learned gate.
# Rationale (hybrid_analysis/diagnose_complementarity on the real runs): the complementarity is STRUCTURED
# and a fixed 50/50 ensemble already beats BOTH models on the dynamic holdout -> a learned per-radius blend
# can only match-or-beat it. FUSED_RMMD_CKPT loads + freezes the trained RMMD parent so only the gate + the
# DGKNet skip train (cheap, no full retrain; starts from the proven RMMD).
#
#   bash STRONG_RMMD/comparison/run_fused.sh <FULL_CKPT (dir or .pt)> <OUT_DIR> <TRAIN_DATA.pt> [VAL_DATA.pt] [extra args]
#
# TRAIN_DATA is required because the repo's default data path is a scrubbed placeholder
# (/scratch/gpfs/USER/...). VAL_DATA defaults to the train path with dataset_train_compact -> dataset_val_compact.
# Then evaluate it like any other model (it has the MultiMachineRMMD forward interface), e.g. add
#   --models fused=<OUT_DIR>   to theory_validation/extrap_strong.py for its zero-shot NRMSE vs RMMD/DGKNet.
set -uo pipefail
PYBIN="${PYBIN:-python}"
TRAIN_PY="STRONG_RMMD/training/rmmd_train_eval.py"
FULL_CKPT="${1:?need the trained full RMMD checkpoint (dir containing checkpoint_best.pt[.gz], or the .pt itself)}"
OUT="${2:-ckpt/fused}"
TRAIN_DATA="${3:?need the compact TRAIN dataset .pt (e.g. /scratch/gpfs/USER/strong_rmmd/phase0NEW/dataset_train_compact.pt)}"
VAL_DATA="${4:-${TRAIN_DATA/dataset_train_compact/dataset_val_compact}}"   # derive val from train if not given

# resolve the RMMD checkpoint to checkpoint_best.pt OR checkpoint_best.pt.gz (the pipeline saves gzipped)
if [ -d "$FULL_CKPT" ]; then
  for c in checkpoint_best.pt checkpoint_best.pt.gz; do
    [ -f "$FULL_CKPT/$c" ] && { FULL_CKPT="$FULL_CKPT/$c"; break; }
  done
fi
[ -f "$FULL_CKPT" ] || { echo "no checkpoint_best.pt[.gz] at $FULL_CKPT"; exit 1; }
[ -f "$TRAIN_DATA" ] || { echo "TRAIN_DATA not found: $TRAIN_DATA"; exit 1; }

export FUSED_RMMD_CKPT="$FULL_CKPT"     # load the trained RMMD into the frozen parent (gz-aware in FusedRMMD)
export FUSED_FREEZE_RMMD=1              # train only the gate + DGKNet skip (cheap)
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"

# FAST PROTOCOL — exactly what the comparison suite was trained under: --fast-protocol --epochs 35
# --batch-size 32 --max-frontier 50, so the fused-vs-RMMD/DGKNet comparison is apples-to-apples.
# --skip-competence-weight 1.0 trains the DGKNet skip to be a competent predictor; --gate-sup-weight 0
# keeps the per-radius gate a PURE learned blend (no per-shot-switch supervision, which was for the dead switch).
if "$PYBIN" "$TRAIN_PY" train --model fused --fast-protocol --epochs 35 --batch-size 32 --max-frontier 50 --lr 2e-4 \
      --skip-competence-weight 1.0 --gate-sup-weight 0 \
      --compact-train-data "$TRAIN_DATA" --compact-val-data "$VAL_DATA" \
      "${@:5}" --checkpoint-dir "$OUT"; then
  echo ">> fused trained -> $OUT  (gate + skip only; RMMD frozen from $FULL_CKPT)"
  echo ">> evaluate vs RMMD/DGKNet:  add  --models fused=$OUT  to theory_validation/extrap_strong.py"
else
  echo "!! fused training FAILED (see error above)"; exit 1
fi
