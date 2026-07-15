#!/usr/bin/env python3
"""Compute per-horizon means and 95% CIs from comparison_summary.json for profiles/geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def ci95(data):
    if not data:
        return None, None
    m = mean(data)
    if len(data) < 2:
        return m, None
    s = stdev(data)
    se = s / math.sqrt(len(data))
    return m, 1.96 * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="/scratch/gpfs/USER/strong_rmmd/comparison/reports/comparison_summary.json")
    p.add_argument("--out", default="/scratch/gpfs/USER/strong_rmmd/comparison/reports/comparison_uncertainty.json")
    args = p.parse_args()

    data = []
    try:
        data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    except Exception:
        print("could not read summary", args.summary)
        raise

    horizons = ["1", "20", "100", "200", "500", "1000"]
    groups = ["profiles", "geometry"]
    out = {g: {h: {} for h in horizons} for g in groups}

    # collect per-horizon lists
    per = {g: {h: [] for h in horizons} for g in groups}
    for item in data:
        metrics = item.get("metrics") or {}
        for g in groups:
            for h in horizons:
                try:
                    v = (metrics.get(g) or {}).get(h, {}).get("nrmse_mean")
                except Exception:
                    v = None
                if v is not None:
                    per[g][h].append(float(v))

    for g in groups:
        for h in horizons:
            m, ci = ci95(per[g][h])
            out[g][h]["nrmse_mean"] = m
            out[g][h]["nrmse_95ci"] = ci
            out[g][h]["n_shots"] = len(per[g][h])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote:", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
