#!/usr/bin/env bash
# Phase 4 — HEADLINE run: train the 5 reported models (RMMD headline + mlp/lstm/node/dgknet) at the
# FULL budget, each at ITS preferred LR + dimensions (read from the fast-protocol comparison table),
# then build an in-distribution accuracy table AND a zero-shot extrap on the holdout.
#
# Budget (all 5 identical): --epochs 160, curriculum frontier capped at T50, advance when val
# NI-NRMSE < 0.05 else hold a frontier for at most 20 epochs. NOT --fast-protocol (that caps epochs
# at 70); the SUT alignment ramp is set explicitly so it reaches full weight inside 160 epochs.
#
# Preferred config per model is READ from comparison_table_fair.json (each model's winning
# checkpoint dir name encodes its LR, e.g. base_mlp_lr1e-4) + the $SRC_CKPT/full symlink (RMMD's
# chosen latent/lr). dgknet's koopman-dim is dimension-matched to RMMD's chosen latent. Missing
# entries fall back to documented defaults (warned). Resumable via .done.
#
# Usage (set the data paths to your {KSTAR,HL2A,D3D,CMOD} -> NSTX build):
#   SRC_CKPT=/scratch/.../ckpt  HCKPT=/scratch/.../ckpt_headline \
#   TRAIN_DATA=/.../dataset_train_compact.pt  VAL_DATA=/.../dataset_val_compact.pt \
#   TEST_DATA=/.../dataset_test_compact.pt    HOLDOUT_DATA=/.../nstx/dataset_test_compact.pt \
#   bash STRONG_RMMD/comparison/run_headline.sh
set -uo pipefail

PYBIN="${PYBIN:-python}"
TRAIN_PY="STRONG_RMMD/training/rmmd_train_eval.py"
COMPARE_PY="STRONG_RMMD/comparison/run_comparison.py"
EXTRAP_PY="STRONG_RMMD/theory_validation/extrap_strong.py"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"

HCKPT="${HCKPT:-ckpt_headline}"
FAIR_TABLE="${FAIR_TABLE:-STRONG_RMMD/comparison/results/comparison_table_fair.json}"
# RMMD chosen config. The fair table records the RMMD row only as 'full' (no latent), so latent is
# recovered from the $SRC_CKPT/full symlink — UNLESS you set RMMD_LATENT/RMMD_LR explicitly here
# (you know it from the extrap: full_ld<latent>_lr<lr>). If you set RMMD_LATENT, SRC_CKPT is unused.
SRC_CKPT="${SRC_CKPT:-ckpt}"                        # only read to resolve the `full` symlink's latent
RMMD_LATENT="${RMMD_LATENT:-}"                      # e.g. 192 ; blank = auto-read from $SRC_CKPT/full
RMMD_LR="${RMMD_LR:-}"                              # e.g. 1e-4 ; blank = auto-read from $SRC_CKPT/full

# --- data (the new {KSTAR,HL2A,D3D,CMOD} train/val/test + NSTX holdout) ---
TRAIN_DATA="${TRAIN_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2/dataset_train_compact.pt}"
VAL_DATA="${VAL_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2/dataset_val_compact.pt}"
TEST_DATA="${TEST_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2/dataset_test_compact.pt}"
HOLDOUT_DATA="${HOLDOUT_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2_nstx/dataset_test_compact.pt}"

# --- budget / curriculum (override via env) ---
EPOCHS="${EPOCHS:-160}"
MAX_FRONTIER="${MAX_FRONTIER:-50}"
MAX_HOLD="${MAX_HOLD:-20}"
ADV_THRESH="${ADV_THRESH:-0.05}"
SUT_RAMP_START="${SUT_RAMP_START:-$((EPOCHS/5))}"   # SUT reaches full weight by ~epoch end/2 ...
SUT_RAMP_END="${SUT_RAMP_END:-$((EPOCHS/2))}"       # ... enforced for the back half of training

for pair in "TRAIN_DATA:$TRAIN_DATA" "VAL_DATA:$VAL_DATA" "TEST_DATA:$TEST_DATA" "HOLDOUT_DATA:$HOLDOUT_DATA"; do
  n="${pair%%:*}"; f="${pair#*:}"
  [ -f "$f" ] || { echo "ERROR: $n not found: $f  (set the env var to your build)" >&2; exit 1; }
done

# ---- resolve preferred (lr, dims) per model from comparison_table_fair.json -----------------
# Each model's winning checkpoint dir name encodes its LR (base_mlp_lr1e-4). RMMD's chosen
# latent/lr comes from that table's 'full' entry, falling back to the $SRC_CKPT/full symlink.
# Emits one "label model lr extra_flags" line per model to stdout (warnings -> stderr).
CONFIGS=$("$PYBIN" - "$FAIR_TABLE" "$SRC_CKPT" "${RMMD_LATENT:-}" "${RMMD_LR:-}" <<'PY'
import json, os, re, sys
from pathlib import Path
ft, src = Path(sys.argv[1]), Path(sys.argv[2])
ov_ld = sys.argv[3] if len(sys.argv) > 3 else ""
ov_lr = sys.argv[4] if len(sys.argv) > 4 else ""
models = {}
if ft.exists():
    try: models = (json.load(open(ft)) or {}).get("models", {})
    except Exception as e: print(f"[headline] could not read {ft}: {e}", file=sys.stderr)
else:
    print(f"[headline] {ft} missing -> default LRs (run the LR grid for the real winners)", file=sys.stderr)

def dirname_of(label):
    cp = (models.get(label) or {}).get("checkpoint", "")
    return Path(cp).parent.name if cp else ""
def lr_of(name, default):
    m = re.search(r'_lr([0-9.eE+-]+)', name or ""); return m.group(1) if m else default
def latent_of(name, default):
    m = re.search(r'ld(\d+)', name or ""); return m.group(1) if m else default

# RMMD chosen config: explicit RMMD_LATENT/RMMD_LR win; else the fair-table 'full' dir name; else
# resolve the $SRC_CKPT/full symlink target (the table records RMMD only as 'full', no latent).
full_name = dirname_of("full")
if "ld" not in full_name and (src / "full").exists():
    full_name = Path(os.path.realpath(src / "full")).name
rmmd_ld = ov_ld or latent_of(full_name, "256")
rmmd_lr = ov_lr or lr_of(full_name, "1e-4")
_srcs = ("override" if ov_ld or ov_lr else (f"'{full_name}'" if full_name else "default"))
print(f"[headline] RMMD chosen: latent={rmmd_ld} lr={rmmd_lr} (from {_srcs})", file=sys.stderr)

# baselines: LR from their winner dir; documented fallbacks (the prior winners) if absent
defaults = {"base_mlp": "1e-4", "base_lstm": "2e-4", "base_node": "2e-4", "base_dgknet": "2e-4"}
rows = [f"headline rmmd {rmmd_lr} --latent-dim {rmmd_ld}"]
for lab, mdl in (("base_mlp", "mlp"), ("base_lstm", "lstm"), ("base_node", "node")):
    rows.append(f"{lab} {mdl} {lr_of(dirname_of(lab), defaults[lab])} -")
# dgknet dimension-matched to RMMD's chosen latent
rows.append(f"base_dgknet dgknet {lr_of(dirname_of('base_dgknet'), defaults['base_dgknet'])} --baseline-latent-dim {rmmd_ld}")
print("\n".join(rows))
PY
)
echo "$CONFIGS"

# ---- train each at the headline budget + its preferred config ------------------------------
mkdir -p "$HCKPT"; FAILED=()
COMMON=(--epochs "$EPOCHS" --max-frontier "$MAX_FRONTIER"
        --curriculum-advance-threshold "$ADV_THRESH" --curriculum-max-hold-epochs "$MAX_HOLD"
        --train-data "$TRAIN_DATA" --val-data "$VAL_DATA"
        --compact-train-data "$TRAIN_DATA" --compact-val-data "$VAL_DATA")
while read -r label model lr extra; do
  [ -z "${label:-}" ] && continue
  [ "$extra" = "-" ] && extra=""
  dir="$HCKPT/$label"; mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo ">> [skip] $label (done)"; continue; fi
  sut=""; [ "$model" = "rmmd" ] && sut="--loss-sut-ramp-start $SUT_RAMP_START --loss-sut-ramp-end $SUT_RAMP_END"
  echo ">> [train] $label: $model lr=$lr $extra $sut (epochs=$EPOCHS frontier<=$MAX_FRONTIER hold<=$MAX_HOLD)"
  if "$PYBIN" "$TRAIN_PY" train --model "$model" --lr "$lr" $extra $sut "${COMMON[@]}" --checkpoint-dir "$dir"; then
    touch "$dir/.done"; echo ">> [done] $label"
  else
    echo ">> [FAIL] $label (rerun to retry)"; FAILED+=("$label")
  fi
done <<< "$CONFIGS"
[ "${#FAILED[@]}" -gt 0 ] && echo ">> FAILED: ${FAILED[*]} (rerun to retry; done models skipped)"

# ---- in-distribution accuracy table over the 5 headline models ----------------------------
echo ">> in-dist accuracy table -> headline_comparison.json"
"$PYBIN" "$COMPARE_PY" --test-data "$TEST_DATA" --ckpt-root "$HCKPT" --reference headline \
    --out STRONG_RMMD/comparison/results/headline_comparison.json || FAILED+=("compare")

# ---- add them to the zero-shot extrap on the holdout (NSTX) --------------------------------
echo ">> zero-shot extrap on holdout -> headline_extrap.json (reference=headline)"
"$PYBIN" "$EXTRAP_PY" --ckpt-root "$HCKPT" --holdout-data "$HOLDOUT_DATA" \
    --reference headline --skip-indist \
    --out STRONG_RMMD/theory_validation/results/headline_extrap.json || FAILED+=("extrap")

echo ">> done. headline_comparison.json (in-dist) + headline_extrap.json (zero-shot)."
[ "${#FAILED[@]}" -eq 0 ] || { echo ">> with failures: ${FAILED[*]}"; exit 1; }
