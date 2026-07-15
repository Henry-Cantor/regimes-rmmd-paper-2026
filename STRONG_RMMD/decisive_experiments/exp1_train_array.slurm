#!/bin/bash
# EXP-1 — 15 independent single-machine trainings (5 machines x 3 seeds) AS A PARALLEL JOB ARRAY.
# Wall-clock = ONE training, not fifteen; zero rigor cost (identical protocol, 3 seeds, 35 epochs kept).
# Run exp1_make_single_machine_data.py FIRST (it writes $OUT/exp1_data), then: sbatch this file.
#
#SBATCH --job-name=exp1_univ
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=exp1_%A_%a.out
set -euo pipefail

# ---- paths: inherited from the submitting shell if exported, else these defaults ----
REPO="${REPO:-$SLURM_SUBMIT_DIR}"                                   # submit from the repo root
DATA="${DATA:-/scratch/gpfs/$USER/strong_rmmd/phase0NEW}"
OUT="${OUT:-/scratch/gpfs/$USER/decisive_out}"
cd "$REPO"

# ---- environment (REQUIRED: the trainer does `from strong_rmmd...`, which needs STRONG_RMMD on PYTHONPATH;
#      the batch shell has neither your venv nor PYTHONPATH -> without this every task fails with
#      ModuleNotFoundError and writes NO checkpoint). VIRTUAL_ENV is your active .torchenv if sbatch inherited it. ----
VENV="${VIRTUAL_ENV:-$HOME/.torchenv}"; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO/STRONG_RMMD:$REPO:${PYTHONPATH:-}"
python -c "import strong_rmmd" 2>/dev/null || { echo "FATAL: strong_rmmd not importable — fix VENV/PYTHONPATH above"; exit 1; }

MACHINES=(CMOD D3D HL2A KSTR NSTX)     # <-- match exactly what exp1_make_single_machine_data.py printed
SEEDS=(0 1 2)
M="${MACHINES[$((SLURM_ARRAY_TASK_ID / 3))]}"
S="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"

# tiny-data machines: if a machine has < 32 shots, drop batch size for it (keeps the protocol comparable).
NTR=$(python - "$OUT/exp1_data/$M/dataset_train_compact.pt" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
d = p.get("data") or p.get("samples") or []
print(len(d))
PY
)
BS=32; [ "$NTR" -lt 32 ] && BS="$NTR"
echo "task $SLURM_ARRAY_TASK_ID -> machine=$M seed=$S n_train=$NTR batch=$BS"

python STRONG_RMMD/training/rmmd_train_eval.py train --model rmmd --fast-protocol \
  --epochs 35 --batch-size "$BS" --max-frontier 50 --lr 2e-4 --seed "$S" --device cuda \
  --compact-train-data "$OUT/exp1_data/$M/dataset_train_compact.pt" \
  --compact-val-data   "$OUT/exp1_data/$M/dataset_val_compact.pt" \
  --checkpoint-dir     "$OUT/exp1_models/$M/seed$S"
