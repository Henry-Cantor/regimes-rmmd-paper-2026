#!/usr/bin/env bash
# Phase 4 — HYBRID ablation suite (replaces the RMMD-side ablations). Trains HybridRMMD (operator +
# SUT + gated MLP skip) + every ablation at the FAST budget (the transfer sweet spot — long training
# overfits, so NO headline needed), then the in-dist accuracy table + the NSTX zero-shot head-to-head
# vs the existing fast full/mlp. Paths are HARDCODED below — just `bash STRONG_RMMD/comparison/run_hybrid.sh`.
# (Every value is still env-overridable if you ever need to, but you don't have to set anything.)
set -uo pipefail

PYBIN="${PYBIN:-python}"
TRAIN_PY="STRONG_RMMD/training/rmmd_train_eval.py"
COMPARE_PY="STRONG_RMMD/comparison/run_comparison.py"
EXTRAP_PY="STRONG_RMMD/theory_validation/extrap_strong.py"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"

# ---- hardcoded cluster paths (USER) ----
HCKPT="${HCKPT:-/scratch/gpfs/USER/ckpt_hybrid}"                                   # hybrid suite + matched baselines
TRAIN_DATA="${TRAIN_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2/dataset_train_compact.pt}"
VAL_DATA="${VAL_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2/dataset_val_compact.pt}"
TEST_DATA="${TEST_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2/dataset_test_compact.pt}"
HOLDOUT_DATA="${HOLDOUT_DATA:-/scratch/gpfs/USER/strong_rmmd/data_build_v2_nstx/dataset_test_compact.pt}"
LR="${LR:-1e-4}"            # RMMD's chosen LR (consistent across the ablation family)
LATENT="${LATENT:-192}"    # RMMD's CHOSEN latent. WITHOUT this it defaults to 384->cap 256 = a BIGGER
                           # RMMD than the known-good fast full (ld192) -> overfits + kills transfer.
EPOCHS="${EPOCHS:-35}"     # locked-protocol budget: EVERYTHING trains to 35 (not 70)
FRONTIER="${FRONTIER:-50}" # locked protocol = reach T50 only. fast-protocol's default is 100 -> a
                           # DEEPER frontier overfits the training machines' long-horizon -> wrecks
                           # zero-shot (in-dist up, holdout down). Eval still degrades out to T100.
BATCH="${BATCH:-32}"       # MUST match run_all's locked protocol (--batch-size 32). The train default
                           # is 16 = ~2x more gradient steps/epoch = effectively more training =
                           # OVERFITS -> baselines transfer ~3x worse than run_all's. (the hybrid bug)
SMOKE="${SMOKE:-0}"        # SMOKE=1 -> 1-epoch wiring check, no eval
EVAL="${EVAL:-1}"          # EVAL=0 -> train only (skip the compare + extrap)
FAST="${FAST:-0}"         # FAST=1 -> quick holdout-only extrap (--skip-indist --max-shots 300); 0 = full

for pair in "TRAIN_DATA:$TRAIN_DATA" "VAL_DATA:$VAL_DATA" "TEST_DATA:$TEST_DATA" "HOLDOUT_DATA:$HOLDOUT_DATA"; do
  n="${pair%%:*}"; f="${pair#*:}"
  [ -f "$f" ] || { echo "ERROR: $n not found: $f" >&2; exit 1; }
done

mkdir -p "$HCKPT" STRONG_RMMD/comparison/results; FAILED=()
DATA=(--train-data "$TRAIN_DATA" --val-data "$VAL_DATA" --compact-train-data "$TRAIN_DATA" --compact-val-data "$VAL_DATA")
EPFLAG=(--epochs "$EPOCHS"); [ "$SMOKE" = "1" ] && EPFLAG=(--epochs 1)

train () {  # <label> <extra flags...>
  local label="$1"; shift
  local dir="$HCKPT/$label"; mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo ">> [skip] $label"; return 0; fi
  echo ">> [train] $label  (hybrid @ fast-protocol lr=$LR latent=$LATENT frontier<=$FRONTIER)  $*"
  if "$PYBIN" "$TRAIN_PY" train --fast-protocol --model hybrid --lr "$LR" --latent-dim "$LATENT" \
        --max-frontier "$FRONTIER" --batch-size "$BATCH" "${EPFLAG[@]}" "${DATA[@]}" "$@" --checkpoint-dir "$dir"; then
    touch "$dir/.done"; echo ">> [done] $label"
  else
    echo ">> [FAIL] $label (rerun to retry)"; FAILED+=("$label")
  fi
}

train_base () {  # <label> <model> <lr> [extra flags...]  -- baseline at the SAME 30-epoch budget
  local label="$1" model="$2" lr="$3"; shift 3
  local dir="$HCKPT/$label"; mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then echo ">> [skip] $label"; return 0; fi
  echo ">> [train-base] $label ($model lr=$lr epochs=$EPOCHS frontier<=$FRONTIER)  $*"
  if "$PYBIN" "$TRAIN_PY" train --fast-protocol --model "$model" --lr "$lr" \
        --max-frontier "$FRONTIER" --batch-size "$BATCH" "${EPFLAG[@]}" "$@" "${DATA[@]}" --checkpoint-dir "$dir"; then
    touch "$dir/.done"; echo ">> [done] $label"
  else
    echo ">> [FAIL] $label (rerun to retry)"; FAILED+=("$label")
  fi
}

echo "=== HYBRID ablation suite @ fast budget (replaces the RMMD-side ablations) ==="
train hybrid                                              # the full hybrid (new reference)
train hybrid_abl_skip       --ablate-skip                # == the exact RMMD (skip removed)
train hybrid_abl_dres       --ablate-dres
train hybrid_abl_transport  --ablate-transport
train hybrid_abl_sut        --loss-sut-weight-base 0
train hybrid_abl_drivers    --ablate-drivers
train hybrid_abl_geometry   --ablate-geometry

echo "=== baselines @ the SAME 30-epoch budget (matched comparison; mlp is the one to beat) ==="
train_base base_mlp    mlp    1e-4
train_base base_lstm   lstm   2e-4
train_base base_node   node   2e-4
train_base base_dgknet dgknet 2e-4 --baseline-latent-dim "$LATENT"   # dimension-matched to RMMD's latent
[ "${#FAILED[@]}" -gt 0 ] && echo ">> FAILED: ${FAILED[*]} (rerun to retry; done models skipped)"

if [ "$SMOKE" = "1" ]; then echo ">> SMOKE done (1-epoch wiring check; no eval). rm -rf $HCKPT/*_smoke* when happy."; exit 0; fi
if [ "$EVAL" != "1" ]; then echo ">> EVAL=0 -> training only, done."; exit "$([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)"; fi

# (1) full in-dist accuracy table over all 11 models (paired vs hybrid).
echo ">> in-dist accuracy table -> results/hybrid_comparison.json (ref=hybrid)"
"$PYBIN" "$COMPARE_PY" --test-data "$TEST_DATA" --ckpt-root "$HCKPT" --reference hybrid \
    --out STRONG_RMMD/comparison/results/hybrid_comparison.json || FAILED+=("compare")

# (2) FULL extrapolation (in-dist + NSTX zero-shot, all shots/horizons, activity-stratified +
# STRONG fit + per-machine) over all 11 models -> the paper-grade table. FAST=1 = quick probe.
if [ "$FAST" = "1" ]; then
  EXTRAP_ARGS=(--holdout-data "$HOLDOUT_DATA" --skip-indist --max-shots 300)
  echo ">> extrapolation (FAST: holdout-only, 300 shots) -> results/hybrid_extrap.json (ref=hybrid)"
else
  EXTRAP_ARGS=(--indist-data "$TEST_DATA" --holdout-data "$HOLDOUT_DATA")
  echo ">> extrapolation (FULL: in-dist + NSTX, all shots) -> results/hybrid_extrap.json (ref=hybrid)"
fi
"$PYBIN" "$EXTRAP_PY" "${EXTRAP_ARGS[@]}" --ckpt-root "$HCKPT" --reference hybrid \
    --out STRONG_RMMD/comparison/results/hybrid_extrap.json || FAILED+=("extrap")

echo ">> PHASE-4 DONE. Next: sut_confirmation --checkpoint $HCKPT/hybrid --compare-checkpoint $HCKPT/hybrid_abl_sut ;"
echo ">>   theorems_validation --checkpoint $HCKPT/hybrid --abl-dres-checkpoint $HCKPT/hybrid_abl_dres ; then the notebook."
echo ">> artifacts: results/hybrid_comparison.json (in-dist) + results/hybrid_extrap.json (in-dist + NSTX zero-shot)."
[ "${#FAILED[@]}" -eq 0 ] || { echo ">> with failures: ${FAILED[*]}"; exit 1; }
