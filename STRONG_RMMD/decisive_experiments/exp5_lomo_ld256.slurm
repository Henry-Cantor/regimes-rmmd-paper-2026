#!/bin/bash
# >>> DIAGNOSTIC / SUPERSEDED -- ld_256/lr1e-4 config sweep. The canonical config is ld_192/lr2e-4
#     (exp5_lomo_4m.slurm). Kept for the reproducibility trail.
# EXP-5 LOMO, one-command re-do at ld_256 / lr1e-4 (a UNIFORM config for all 5 folds -- NOT per-machine tuning,
# report whatever falls out). Trains the 5 held-out folds, then runs the full analysis + the fast quartile-skill
# patch, all in ONE job. AUGD/EAST are NOT retrained -- they are evaluated with the committed 5-machine model
# ($CEIL), same as before.
#
#   export EAST=<phase0NEW_east>/dataset_test_compact.pt AUGD=<phase0NEW_augd>/dataset_test_compact.pt \
#          CEIL=<committed full_ld192 checkpoint_best.pt> OUT=<decisive_out> DATA=<phase0NEW>
#   sbatch exp5_lomo_ld256.slurm       # NOTE: sbatch only passes EXPORTED vars -> use `export`, not plain assignment
#
#SBATCH --job-name=lomo_ld256
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=lomo_ld256_%j.out
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
MODELS="$OUT/lomo_models_ld256"          # NEW dir -> does not clobber the ld_192 / ld_128 models

echo "=== [1/4] build the LOMO folds if missing (train/val + per-machine-normalized eval_m) ==="
NEED_BUILD=0
for M in "${MACHINES[@]}"; do
  [ -s "$OUT/lomo_data/holdout_$M/dataset_train_compact.pt" ] \
    && [ -s "$OUT/lomo_data/holdout_$M/dataset_val_compact.pt" ] \
    && [ -s "$OUT/lomo_data/holdout_$M/eval_$M.pt" ] || NEED_BUILD=1
done
if [ "${FORCE_BUILD:-0}" = 1 ] || [ "$NEED_BUILD" = 1 ]; then
  python $DEC/exp5_make_lomo_data.py --train $DATA/dataset_train_compact.pt --val $DATA/dataset_val_compact.pt \
      --extra $EAST $AUGD --out-dir $OUT/lomo_data || { echo "FATAL: make-data failed"; exit 1; }
else
  echo "  folds already built for all ${MACHINES[*]} -> SKIP (set FORCE_BUILD=1 to rebuild)"
fi
rm -f "$OUT"/lomo_data/holdout_*/eval_*.pt.norm.json    # always clear stale sidecars before analysis (priority #2)

echo "=== [2/4] train the 5 folds at ld_256 / lr1e-4 (uniform; fast-protocol 35 epochs) ==="
for M in "${MACHINES[@]}"; do
  if [ "${FORCE_TRAIN:-0}" != 1 ] && { [ -s "$MODELS/holdout_$M/checkpoint_best.pt" ] \
       || [ -s "$MODELS/holdout_$M/checkpoint_best.pt.gz" ]; }; then
    echo "--- fold $M already trained -> SKIP (e.g. reuse the D3D quick-test model; FORCE_TRAIN=1 to retrain) ---"
    continue
  fi
  echo "--- training fold $M ---"
  python STRONG_RMMD/training/rmmd_train_eval.py train --model rmmd --fast-protocol \
    --epochs 35 --batch-size 32 --max-frontier 50 --lr 1e-4 --latent-dim 256 --seed 0 --device cuda \
    --compact-train-data "$OUT/lomo_data/holdout_$M/dataset_train_compact.pt" \
    --compact-val-data   "$OUT/lomo_data/holdout_$M/dataset_val_compact.pt" \
    --checkpoint-dir     "$MODELS/holdout_$M" \
    || echo "*** FOLD $M TRAINING FAILED (analysis will report it as NOT FOUND)"
done

echo "=== [3/4] full analysis (5 new ld_256 folds + EAST/AUGD via the committed model) ==="
python $DEC/exp5_lomo_analysis.py --models-root "$MODELS" --data-root "$OUT/lomo_data" --device cuda \
    --machines "${MACHINES[@]}" --holdout-ckpt "$CEIL" --holdout EAST:"$EAST" AUGD:"$AUGD"

echo "=== [4/4] fast quartile-skill patch (persistence-only, no rollout) ==="
python $DEC/exp5_quartile_skill.py --data-root "$OUT/lomo_data" \
    --machines "${MACHINES[@]}" --holdout EAST:"$EAST" AUGD:"$AUGD"

echo "=== DONE. results/universality_predictive.json written + quartile-skill patched. ==="
