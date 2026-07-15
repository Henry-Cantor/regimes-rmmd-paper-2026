#!/bin/bash
# >>> FINDING RECORDED (safe to delete): this diagnostic CONFIRMED 4-machine training fixes the D3D fold
#     (q4 skill -0.16 -> +0.05). That fix is now the canonical exp5_lomo_4m.slurm.
# EXP-5 DIAGNOSTIC (throwaway -- delete after) -- D3D fold trained on ONLY the 4 pool machines (NO EAST/AUGD extra),
# at the REFERENCE config ld_192 / lr2e-4 (same as the 5/7 run) so the ONLY variable vs that run is the machine
# count. Isolates the "6-machine train-on-6 is too much for 35 epochs" hypothesis: if D3D recovers on q4 here
# (4 machines) but not with 6, the extras are the problem; if it still fails, the machine count is not it.
# (If it recovers, a follow-up at ld_256/lr1e-4 can check config next.)
# Writes results to a SEPARATE lomo_data_4m / models_4m / universality_predictive_d3d_4m.json -- clobbers nothing.
# Needs only OUT and DATA (no EAST/AUGD/CEIL -- there is no --extra and no holdout eval).
#
#   export OUT=/scratch/gpfs/$USER/decisive_out DATA=/scratch/gpfs/$USER/phase0NEW
#   sbatch exp5_lomo_d3d_4m.slurm
#
#SBATCH --job-name=lomo_d3d_4m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=lomo_d3d_4m_%j.out
set -uo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-/scratch/gpfs/$USER/decisive_out}"
DATA="${DATA:-/scratch/gpfs/$USER/phase0NEW}"
cd "$REPO"
VENV="${VIRTUAL_ENV:-$HOME/.torchenv}"; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO/STRONG_RMMD:$REPO:${PYTHONPATH:-}"
python -c "import strong_rmmd" || { echo "FATAL: strong_rmmd not importable"; exit 1; }

DEC=STRONG_RMMD/decisive_experiments
DATA4="$OUT/lomo_data_4m"                 # separate fold dir: D3D trained on 4 machines (no EAST/AUGD)
MODELS="$OUT/lomo_models_ld192_4m"
OUTJSON="$DEC/results/universality_predictive_d3d_4m.json"

echo "=== [1/4] build the D3D fold on ONLY the 4 pool machines (no --extra) ==="
if [ "${FORCE_BUILD:-0}" = 1 ] || ! { [ -s "$DATA4/holdout_D3D/dataset_train_compact.pt" ] \
     && [ -s "$DATA4/holdout_D3D/eval_D3D.pt" ]; }; then
  python $DEC/exp5_make_lomo_data.py --train $DATA/dataset_train_compact.pt --val $DATA/dataset_val_compact.pt \
      --out-dir "$DATA4" --machines D3D || { echo "FATAL: make-data failed"; exit 1; }
else
  echo "  D3D 4-machine fold already built -> SKIP (FORCE_BUILD=1 to rebuild)"
fi
rm -f "$DATA4"/holdout_D3D/eval_*.pt.norm.json

echo "=== [2/4] train D3D on 4 machines at ld_192 / lr2e-4 (reference config; fast-protocol 35 epochs) ==="
if [ "${FORCE_TRAIN:-0}" != 1 ] && { [ -s "$MODELS/holdout_D3D/checkpoint_best.pt" ] \
     || [ -s "$MODELS/holdout_D3D/checkpoint_best.pt.gz" ]; }; then
  echo "  D3D (4m) already trained -> SKIP (FORCE_TRAIN=1 to retrain)"
else
  python STRONG_RMMD/training/rmmd_train_eval.py train --model rmmd --fast-protocol \
    --epochs 35 --batch-size 32 --max-frontier 50 --lr 2e-4 --latent-dim 192 --seed 0 --device cuda \
    --compact-train-data "$DATA4/holdout_D3D/dataset_train_compact.pt" \
    --compact-val-data   "$DATA4/holdout_D3D/dataset_val_compact.pt" \
    --checkpoint-dir     "$MODELS/holdout_D3D" || { echo "FATAL: D3D (4m) training failed"; exit 1; }
fi

echo "=== [3/4] analyze D3D only -> $OUTJSON ==="
python $DEC/exp5_lomo_analysis.py --models-root "$MODELS" --data-root "$DATA4" --device cuda \
    --machines D3D --out-json "$REPO/$OUTJSON"

echo "=== [4/4] fast quartile-skill patch (D3D only) ==="
python $DEC/exp5_quartile_skill.py --data-root "$DATA4" --machines D3D --json "$REPO/$OUTJSON"

echo "=== DONE. Read [D3D] q4 skill above. 4-machine result: $OUTJSON ==="
