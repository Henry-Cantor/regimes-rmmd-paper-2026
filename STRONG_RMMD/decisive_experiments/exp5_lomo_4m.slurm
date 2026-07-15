#!/bin/bash
# EXP-5 LOMO full suite, 4-MACHINE training (the winning setup): each held-out fold trains on ONLY the other 4 pool
# machines (NO EAST/AUGD extra -- train-on-6 was too much for the 35-epoch window and sank D3D/NSTX). Config is the
# reference ld_192 / lr2e-4. Trains the 5 folds (skips any already trained -- e.g. the D3D quick-test model in
# lomo_models_ld192_4m), then runs the full analysis + quartile-skill patch into the FINAL universality_predictive.json.
# AUGD/EAST are NOT retrained -- evaluated with the committed 5-machine model ($CEIL).
#
#   export EAST=... AUGD=... CEIL=... OUT=... DATA=...    # export (sbatch only forwards exported vars)
#   sbatch exp5_lomo_4m.slurm
#
#SBATCH --job-name=lomo_4m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=lomo_4m_%j.out
set -uo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-/scratch/gpfs/$USER/decisive_out}"
DATA="${DATA:-/scratch/gpfs/$USER/phase0NEW}"
: "${EAST:?set (and export) EAST=<phase0NEW_east>/dataset_test_compact.pt}"
: "${AUGD:?set (and export) AUGD=<phase0NEW_augd>/dataset_test_compact.pt}"
: "${CEIL:?set (and export) CEIL=<committed 5-machine full_ld192 checkpoint_best.pt> (for the EAST/AUGD folds)}"
cd "$REPO"
VENV="${VIRTUAL_ENV:-$HOME/.torchenv}"; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO/STRONG_RMMD:$REPO:${PYTHONPATH:-}"
python -c "import strong_rmmd" || { echo "FATAL: strong_rmmd not importable"; exit 1; }

DEC=STRONG_RMMD/decisive_experiments
MACHINES=(CMOD D3D HL2A KSTR NSTX)
DATA4="$OUT/lomo_data_4m"                 # 4-machine folds (no EAST/AUGD extra) -- SAME dir the D3D test used
MODELS="$OUT/lomo_models_ld192_4m"        # SAME dir the D3D test used -> D3D is reused, not retrained

echo "=== [1/4] build the 4-machine folds if missing (train on 4 pool machines, no --extra) ==="
NEED_BUILD=0
for M in "${MACHINES[@]}"; do
  [ -s "$DATA4/holdout_$M/dataset_train_compact.pt" ] \
    && [ -s "$DATA4/holdout_$M/eval_$M.pt" ] || NEED_BUILD=1
done
if [ "${FORCE_BUILD:-0}" = 1 ] || [ "$NEED_BUILD" = 1 ]; then
  python $DEC/exp5_make_lomo_data.py --train $DATA/dataset_train_compact.pt --val $DATA/dataset_val_compact.pt \
      --out-dir "$DATA4" --machines "${MACHINES[@]}" || { echo "FATAL: make-data failed"; exit 1; }
else
  echo "  4-machine folds already built for all ${MACHINES[*]} -> SKIP (FORCE_BUILD=1 to rebuild)"
fi
rm -f "$DATA4"/holdout_*/eval_*.pt.norm.json    # always clear stale sidecars before analysis (priority #2)

echo "=== [2/4] train the 5 folds on 4 machines at ld_192 / lr2e-4 (fast-protocol 35 epochs) ==="
for M in "${MACHINES[@]}"; do
  if [ "${FORCE_TRAIN:-0}" != 1 ] && { [ -s "$MODELS/holdout_$M/checkpoint_best.pt" ] \
       || [ -s "$MODELS/holdout_$M/checkpoint_best.pt.gz" ]; }; then
    echo "--- fold $M already trained -> SKIP (reuses e.g. the D3D quick-test model; FORCE_TRAIN=1 to retrain) ---"
    continue
  fi
  echo "--- training fold $M ---"
  python STRONG_RMMD/training/rmmd_train_eval.py train --model rmmd --fast-protocol \
    --epochs 35 --batch-size 32 --max-frontier 50 --lr 2e-4 --latent-dim 192 --seed 0 --device cuda \
    --compact-train-data "$DATA4/holdout_$M/dataset_train_compact.pt" \
    --compact-val-data   "$DATA4/holdout_$M/dataset_val_compact.pt" \
    --checkpoint-dir     "$MODELS/holdout_$M" \
    || echo "*** FOLD $M TRAINING FAILED (analysis will report it as NOT FOUND)"
done

echo "=== [3/4] full analysis -> FINAL results/universality_predictive.json (5 folds + EAST/AUGD via \$CEIL) ==="
python $DEC/exp5_lomo_analysis.py --models-root "$MODELS" --data-root "$DATA4" --device cuda \
    --machines "${MACHINES[@]}" --holdout-ckpt "$CEIL" --holdout EAST:"$EAST" AUGD:"$AUGD"

echo "=== [4/4] fast quartile-skill patch (persistence-only, no rollout) ==="
python $DEC/exp5_quartile_skill.py --data-root "$DATA4" \
    --machines "${MACHINES[@]}" --holdout EAST:"$EAST" AUGD:"$AUGD"

echo "=== DONE. FINAL results/universality_predictive.json written + quartile-skill patched. ==="
