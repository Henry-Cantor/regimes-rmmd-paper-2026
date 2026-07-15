#!/usr/bin/env python3
"""Run fast hyperparameter sensitivity sweeps for RMMD and DGKNet.

Generates short 4-epoch runs for each hyperparameter combination and
stores runs under --run-root/runs/hp_<name>.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
RMMD_SCRIPT = ROOT / "STRONG_RMMD" / "training" / "rmmd_train_eval.py"
DGK_SCRIPT = ROOT / "STRONG_RMMD" / "training" / "dgknet_train_eval.py"


def _run(cmd: List[str], cwd: Path, dry_run: bool) -> int:
    print(("DRY:" if dry_run else "RUN:"), " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def build_rmmd_cmd(python: str, ckpt_dir: Path, lr: float, wd: float, epochs: int = 4) -> List[str]:
    return [
        python,
        str(RMMD_SCRIPT),
        "train",
        "--checkpoint-dir",
        str(ckpt_dir),
        "--train-data",
        "/scratch/gpfs/USER/strong_rmmd/phase0/dataset_train.pt",
        "--val-data",
        "/scratch/gpfs/USER/strong_rmmd/phase0/dataset_val.pt",
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--batch-size",
        "8",
        "--max-rollout-train",
        "100",
        "--lr",
        str(lr),
        "--weight-decay",
        str(wd),
    ]


def build_dgk_cmd(python: str, ckpt_dir: Path, lr: float, wd: float, koopman_dim: int, epochs: int = 4) -> List[str]:
    return [
        python,
        str(DGK_SCRIPT),
        "train",
        "--checkpoint-dir",
        str(ckpt_dir),
        "--train-data",
        "/scratch/gpfs/USER/strong_rmmd/phase0/dataset_train.pt",
        "--val-data",
        "/scratch/gpfs/USER/strong_rmmd/phase0/dataset_val.pt",
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--batch-size",
        "8",
        "--max-rollout-train",
        "100",
        "--lr",
        str(lr),
        "--weight-decay",
        str(wd),
        "--koopman-dim",
        str(koopman_dim),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="/scratch/gpfs/USER/strong_rmmd/comparison")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    lrs = [3e-5, 5e-5, 1e-4]
    wds = [1e-6, 1e-5, 5e-5]
    koop_dims = [512, 1024]

    runs_dir = Path(args.run_root) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # RMMD sweep
    for lr, wd in itertools.product(lrs, wds):
        name = f"hp_rmmd_lr{lr:.0e}_wd{wd:.0e}"
        ckpt = runs_dir / name / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        cmd = build_rmmd_cmd(args.python, ckpt, lr, wd, epochs=4)
        rc = _run(cmd, ROOT, args.dry_run)
        if rc != 0:
            print(f"{name} failed with rc={rc}")

    # DGKNet sweep (smaller grid)
    for lr, wd, kd in itertools.product([5e-5, 1e-4], [1e-6, 1e-5], koop_dims):
        name = f"hp_dgk_lr{lr:.0e}_wd{wd:.0e}_kd{kd}"
        ckpt = runs_dir / name / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        cmd = build_dgk_cmd(args.python, ckpt, lr, wd, kd, epochs=4)
        rc = _run(cmd, ROOT, args.dry_run)
        if rc != 0:
            print(f"{name} failed with rc={rc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
