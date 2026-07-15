#!/bin/bash
# >>> DIAGNOSTIC / SUPERSEDED -- D3D-only quick test of the ld_256/lr1e-4 config. Canonical run is
#     exp5_lomo_4m.slurm (ld_192/lr2e-4 won). Kept for the reproducibility trail.
# EXP-5 LOMO QUICK TEST -- D3D fold ONLY, at ld_256 / lr1e-4. Same pipeline as exp5_lomo_ld256.slurm but trains and
# evaluates just D3D, so you learn in ~5h (one fold) instead of ~15h (five) whether the new config recovers the
# under-trained fold. If D3D's q4 skill goes positive -> the method works, launch the full 5-fold run (which will
# SKIP re-training D3D, since it writes to the same lomo_models_ld256/ dir). If D3D still fails q4 -> the 6-machine
# train-on-6 is likely too much for the 35-epoch window (or the config does not help) -- rethink before the 15h run.
#
# Writes a SEPARATE results/universality_predictive_d3d.json so it does NOT clobber the full 7-machine JSON.
#
#   export EAST=... AUGD=... CEIL=... OUT=... DATA=...   # (export, not plain assignment -- sbatch passes exports)
#   sbatch exp5_lomo_d3d.slurm
#
#SBATCH --job-name=lomo_d3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=lomo_d3d_%j.out
set -uo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-/scratch/gpfs/$USER/decisive_out}"
DATA="${DATA:-/scratch/gpfs/$USER/phase0NEW}"
: "${EAST:?set (and export) EAST=<phase0NEW_east>/dataset_test_compact.pt}"
: "${AUGD:?set (and export) AUGD=<phase0NEW_augd>/dataset_test_compact.pt}"
cd "$REPO"
VENV="${VIRTUAL_ENV:-$HOME/.torchenv}"; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO/STRONG_RMMD:$REPO:${PYTHONPATH:-}"
python -c "import strong_rmmd" || { echo "FATAL: strong_rmmd not importable"; exit 1; }

DEC=STRONG_RMMD/decisive_experiments
MODELS="$OUT/lomo_models_ld256"          # SAME dir as the full run -> full run reuses this D3D model
OUTJSON="$DEC/results/universality_predictive_d3d.json"

echo "=== [1/4] build the D3D fold if missing ==="
if [ "${FORCE_BUILD:-0}" = 1 ] || ! { [ -s "$OUT/lomo_data/holdout_D3D/dataset_train_compact.pt" ] \
     && [ -s "$OUT/lomo_data/holdout_D3D/eval_D3D.pt" ]; }; then
  python $DEC/exp5_make_lomo_data.py --train $DATA/dataset_train_compact.pt --val $DATA/dataset_val_compact.pt \
      --extra $EAST $AUGD --out-dir $OUT/lomo_data --machines D3D || { echo "FATAL: make-data failed"; exit 1; }
else
  echo "  D3D fold already built -> SKIP (set FORCE_BUILD=1 to rebuild)"
fi
rm -f "$OUT"/lomo_data/holdout_D3D/eval_*.pt.norm.json

echo "=== [2/4] train the D3D fold at ld_256 / lr1e-4 (fast-protocol 35 epochs) ==="
if [ "${FORCE_TRAIN:-0}" != 1 ] && { [ -s "$MODELS/holdout_D3D/checkpoint_best.pt" ] \
     || [ -s "$MODELS/holdout_D3D/checkpoint_best.pt.gz" ]; }; then
  echo "  D3D already trained -> SKIP (FORCE_TRAIN=1 to retrain)"
else
  python STRONG_RMMD/training/rmmd_train_eval.py train --model rmmd --fast-protocol \
    --epochs 35 --batch-size 32 --max-frontier 50 --lr 1e-4 --latent-dim 256 --seed 0 --device cuda \
    --compact-train-data "$OUT/lomo_data/holdout_D3D/dataset_train_compact.pt" \
    --compact-val-data   "$OUT/lomo_data/holdout_D3D/dataset_val_compact.pt" \
    --checkpoint-dir     "$MODELS/holdout_D3D" || { echo "FATAL: D3D training failed"; exit 1; }
fi

echo "=== [3/4] analyze D3D only -> $OUTJSON (does NOT touch the full 7-machine JSON) ==="
python $DEC/exp5_lomo_analysis.py --models-root "$MODELS" --data-root "$OUT/lomo_data" --device cuda \
    --machines D3D --out-json "$REPO/$OUTJSON"

echo "=== [4/4] fast quartile-skill patch (D3D only) ==="
python $DEC/exp5_quartile_skill.py --data-root "$OUT/lomo_data" --machines D3D --json "$REPO/$OUTJSON"

echo "=== DONE. Read [D3D] q4 skill above (>0 = the new config works). Full JSON: $OUTJSON ==="
