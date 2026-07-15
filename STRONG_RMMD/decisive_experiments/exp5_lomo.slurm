#!/bin/bash
# >>> SUPERSEDED by exp5_lomo_4m.slurm (the canonical winning LOMO run that writes universality_predictive.json).
#     This original variant added EAST/AUGD as --extra (effectively train-on-6), which was too much for the
#     35-epoch window and sank the D3D/NSTX folds. Kept for the reproducibility trail; do not use for the paper.
# EXP-5 LOMO: train the FULL model 5x, each holding out one training machine (train on the other 4).
# Run exp5_make_lomo_data.py FIRST. Then: sbatch this file. Fair across folds (identical protocol).
#
#SBATCH --job-name=lomo
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=lomo_%A_%a.out
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-/scratch/gpfs/$USER/decisive_out}"
cd "$REPO"
VENV="${VIRTUAL_ENV:-$HOME/.torchenv}"; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO/STRONG_RMMD:$REPO:${PYTHONPATH:-}"
python -c "import strong_rmmd" || { echo "FATAL: strong_rmmd not importable"; exit 1; }

MACHINES=(CMOD D3D HL2A KSTR NSTX)     # match exp5_make_lomo_data.py output dir names
M="${MACHINES[$SLURM_ARRAY_TASK_ID]}"
DATA="$OUT/lomo_data/holdout_$M"
echo "LOMO fold: hold out $M ; train on the other 4 ; data=$DATA"

# EXACT ceiling config (the proven full model that transferred well to AUGD): fast-protocol, 35 epochs,
# max-frontier 50, lr 2e-4, LATENT-DIM 192 (the ld_192 config; the CLI default is 384 -> MUST set 192).
python STRONG_RMMD/training/rmmd_train_eval.py train --model rmmd --fast-protocol \
  --epochs 35 --batch-size 32 --max-frontier 50 --lr 2e-4 --latent-dim 192 --seed 0 --device cuda \
  --compact-train-data "$DATA/dataset_train_compact.pt" \
  --compact-val-data   "$DATA/dataset_val_compact.pt" \
  --checkpoint-dir     "$OUT/lomo_models/holdout_$M"
