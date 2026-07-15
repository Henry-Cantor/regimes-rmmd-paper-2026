#!/usr/bin/env python
"""Pick the best learning-rate variant per model BY VALIDATION (never test), and write a
{label: winning_dir} JSON for run_comparison.py.

Selection metric: best (lowest) validation NI-NRMSE from each candidate's training_summary.json
(history['best_ni_nrmse']). Selecting on validation and reporting on test avoids test-set leakage
in model selection.

Usage:
  python select_lr_winners.py --out ckpt/comparison_models.json \
      full=ckpt/full abl_drivers=ckpt/abl_drivers ... \
      base_mlp=ckpt/base_mlp,ckpt/base_mlp_lr2e-4  base_lstm=ckpt/base_lstm,ckpt/base_lstm_lr2e-4 ...
Each entry is label=dir or label=dir1,dir2,... (comma-separated candidates).
"""
import argparse
import json
import sys
from pathlib import Path


def best_val(d: Path):
    f = d / "training_summary.json"
    if not f.exists():
        return None
    try:
        h = json.loads(f.read_text())
    except Exception:
        return None
    v = h.get("best_ni_nrmse")
    if v is None:
        vals = h.get("val_ni_nrmse") or []
        v = min(vals) if vals else None
    return float(v) if v is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=False, help="output JSON (required unless --print-winner)")
    ap.add_argument(
        "--print-winner", action="store_true",
        help="Print ONLY the single best candidate dir (lowest val NI-NRMSE across ALL entries' "
             "candidates) to stdout, for shell capture. Diagnostics go to stderr; no JSON written. "
             "Used by run_all.sh to pick RMMD's best-LR run before training the ablations.",
    )
    ap.add_argument("entries", nargs="+", help="label=dir or label=dir1,dir2,...")
    args = ap.parse_args()

    if args.print_winner:
        cands = []
        for e in args.entries:
            _, _, paths = e.partition("=")
            cands += [Path(p) for p in paths.split(",") if p]
        scored = [(best_val(c), c) for c in cands]
        valid = [(v, c) for v, c in scored if v is not None]
        if valid:
            v, c = min(valid, key=lambda x: x[0])
            others = "  ".join(f"{cc.name}={vv:.4f}" for vv, cc in valid)
            print(f"[select] winner {c.name} (val NRMSE {v:.4f})  candidates: {others}", file=sys.stderr)
            print(str(c))  # ONLY the winning dir to stdout
        else:
            existing = [c for c in cands if c.exists()]
            fallback = existing[0] if existing else (cands[0] if cands else Path(""))
            print(f"[select] no val metrics yet; falling back to {fallback}", file=sys.stderr)
            print(str(fallback))
        return

    if not args.out:
        ap.error("--out is required unless --print-winner is set")

    winners = {}
    for e in args.entries:
        label, _, paths = e.partition("=")
        cands = [Path(p) for p in paths.split(",") if p]
        scored = [(best_val(c), c) for c in cands]
        valid = [(v, c) for v, c in scored if v is not None]
        if not valid:
            # fall back to the first existing dir even if no summary (e.g. still training)
            existing = [c for c in cands if c.exists()]
            if existing:
                winners[label] = str(existing[0])
                print(f"  {label}: no val metric, using {existing[0]}")
            continue
        v, c = min(valid, key=lambda x: x[0])
        winners[label] = str(c)
        others = "  ".join(f"{cc.name}={vv:.4f}" for vv, cc in valid)
        print(f"  {label}: best val NRMSE {v:.4f} -> {c.name}   (candidates: {others})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(winners, indent=2))
    print(f"Wrote {args.out} with {len(winners)} models")


if __name__ == "__main__":
    main()
