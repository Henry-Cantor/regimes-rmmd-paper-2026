#!/usr/bin/env python
"""Train a 1-D Fourier Neural Operator baseline at three learning rates on the 5-machine pool, then
zero-shot extrapolate on the EAST and AUGD holdouts, writing all results to results/fno.json.

The FNO is a first-class baseline (strong_rmmd/baselines.py, --model fno) sharing the same rollout,
curriculum, and loss harness as the other baselines. Reuses the train and extrapolation entry points.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

_USER = os.environ.get("USER", "USER")

HERE = Path(__file__).resolve().parent                    # .../STRONG_RMMD/decisive_experiments
REPO = HERE.parents[1]                                     # decisive_experiments -> STRONG_RMMD -> repo root
TRAIN = REPO / "STRONG_RMMD" / "training" / "rmmd_train_eval.py"
EXTRAP = REPO / "STRONG_RMMD" / "theory_validation" / "extrap_strong.py"
LRS = ["5e-5", "2e-4", "1e-4"]                            # the 3 baseline LRs (default + 2 extra)


def run(cmd):
    print("\n>> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=f"/scratch/gpfs/{_USER}/strong_rmmd",
                    help="holds phase0NEW/ (5-machine train/val/test) + phase0NEW_east/ + phase0NEW_augd/")
    ap.add_argument("--out-root", default=f"/scratch/gpfs/{_USER}/decisive_out", help="where FNO checkpoints go")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out-json", default=str(HERE / "results" / "fno.json"))
    a = ap.parse_args()

    dr = Path(a.data_root)
    train = dr / "phase0NEW" / "dataset_train_compact.pt"
    val = dr / "phase0NEW" / "dataset_val_compact.pt"
    indist = dr / "phase0NEW" / "dataset_test_compact.pt"
    east = dr / "phase0NEW_east" / "dataset_test_compact.pt"
    augd = dr / "phase0NEW_augd" / "dataset_test_compact.pt"
    for p in (train, val, indist, east, augd):
        if not p.exists():
            sys.exit(f"FATAL: missing data file {p} (fix --data-root)")
    models_dir = Path(a.out_root) / "fno_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # ---- 1) TRAIN the FNO at the 3 LRs on the 5-machine pool (fast-protocol) ----
    if not a.skip_train:
        for lr in LRS:
            ckpt = models_dir / f"fno_lr{lr}"
            if (ckpt / "checkpoint_best.pt").exists() or (ckpt / "checkpoint_best.pt.gz").exists():
                print(f"  [train] fno lr={lr} already trained -> SKIP ({ckpt})"); continue
            run([py, TRAIN, "train", "--model", "fno", "--fast-protocol", "--epochs", a.epochs,
                 "--lr", lr, "--seed", 0, "--device", a.device, "--max-frontier", 50,
                 "--compact-train-data", train, "--compact-val-data", val, "--checkpoint-dir", ckpt])

    # ---- 2) ZERO-SHOT extrap on EAST and AUGD (each report has indist + that holdout) ----
    ref = f"fno_lr{LRS[0]}"                               # self-reference (a FNO ckpt that exists)
    rep_east = HERE / "results" / "fno_extrap_east.json"
    rep_augd = HERE / "results" / "fno_extrap_augd.json"
    run([py, EXTRAP, "--indist-data", indist, "--holdout-data", east, "--ckpt-root", models_dir,
         "--reference", ref, "--device", a.device, "--out", rep_east])
    run([py, EXTRAP, "--indist-data", indist, "--holdout-data", augd, "--ckpt-root", models_dir,
         "--reference", ref, "--device", a.device, "--out", rep_augd])

    # ---- 3) MERGE -> fno.json (indist + east + augd, per LR) ----
    re, ra = json.loads(rep_east.read_text()), json.loads(rep_augd.read_text())
    out = {"meta": {"horizons": re["meta"]["horizons"], "lrs": LRS, "indist_data": str(indist),
                    "east_data": str(east), "augd_data": str(augd), "n_indist": re["meta"].get("n_indist"),
                    "n_east": re["meta"].get("n_holdout"), "n_augd": ra["meta"].get("n_holdout")},
           "models": {}}
    for name, me in re.get("models", {}).items():
        ma = ra.get("models", {}).get(name, {})
        out["models"][name] = {
            "model_type": me.get("model_type"), "n_params": me.get("n_params"),
            "indist": me.get("indist"), "indist_activity_stratified": me.get("indist_activity_stratified"),
            "east": me.get("holdout"), "east_activity_stratified": me.get("holdout_activity_stratified"),
            "augd": ma.get("holdout"), "augd_activity_stratified": ma.get("holdout_activity_stratified")}
    Path(a.out_json).write_text(json.dumps(out, indent=1, default=float))

    print(f"\nwrote {a.out_json}  (models: {list(out['models'])})")
    def nr(x):
        return round(x["nrmse"], 3) if isinstance(x, dict) and x.get("nrmse") is not None else None
    print("  model            T50 NRMSE:  indist   east    augd")
    for name, m in out["models"].items():
        print(f"  {name:14s}            {nr((m['indist'] or {}).get('50'))}   "
              f"{nr((m['east'] or {}).get('50'))}   {nr((m['augd'] or {}).get('50'))}")


if __name__ == "__main__":
    main()
