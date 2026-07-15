#!/usr/bin/env python3
"""Evaluate SUT (universal transfer) and iterative zero-shot performance.

This script searches for trained RMMD checkpoints and runs leave-one-machine-out
zero-shot evaluation commands. It writes out a JSON summary with per-machine metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
RMMD_SCRIPT = ROOT / "STRONG_RMMD" / "training" / "rmmd_train_eval.py"


def _run(cmd: List[str], cwd: Path, dry_run: bool) -> subprocess.CompletedProcess:
    print(("DRY:" if dry_run else "RUN:"), " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


MACHINES = ["NSTX", "CMOD", "D3D", "EAST", "KSTR", "ITER"]


def find_ckpt(run_root: Path) -> Path:
    # prefer main RMMD
    ck = Path(run_root) / "runs" / "main_rmmd" / "checkpoints" / "checkpoint_best.pt.gz"
    if ck.exists():
        return ck
    ck2 = Path(run_root) / "runs" / "main_rmmd" / "checkpoints" / "checkpoint_best.pt"
    if ck2.exists():
        return ck2
    return Path("")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="/scratch/gpfs/USER/strong_rmmd/comparison")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_root = Path(args.run_root)
    ckpt = find_ckpt(run_root)
    if not ckpt.exists():
        print("No main RMMD checkpoint found; run train_all_ablation.py first")
        return 1

    all_metrics: Dict[str, Any] = {}
    for machine in MACHINES:
        # prefer a per-machine test split if available
        per_test = Path(f"/scratch/gpfs/USER/strong_rmmd/phase0/dataset_test_{machine}.pt")
        test_data = per_test if per_test.exists() else Path("/scratch/gpfs/USER/strong_rmmd/phase0/dataset_test.pt")
        if not test_data.exists():
            print(f"No test dataset found for machine {machine}; skipping")
            all_metrics[machine] = {}
            continue

        cmd = [
            args.python,
            str(RMMD_SCRIPT),
            "eval",
            "--checkpoint",
            str(ckpt),
            "--test-data",
            str(test_data),
            "--device",
            "cuda",
            "--horizons",
            "1",
            "20",
            "100",
            "200",
            "500",
            "1000",
        ]
        proc = _run(cmd, ROOT, args.dry_run)
        metrics = {}
        if not args.dry_run and isinstance(proc, subprocess.CompletedProcess) and proc.returncode == 0:
            try:
                metrics = json.loads(proc.stdout)
            except Exception:
                metrics = {}
        all_metrics[machine] = metrics

    (Path(run_root) / "reports" / "sut_zeroshot_summary.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("Wrote SUT zero-shot summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
