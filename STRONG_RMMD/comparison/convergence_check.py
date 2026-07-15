#!/usr/bin/env python
"""Per-model convergence check from the validation NI-NRMSE curve each run logs.

For every model under --ckpt-root, reads training_summary.json's validation curve and reports whether
validation was still dropping at the end (undertrained) or had plateaued (converged), so a matched-budget
comparison is not confounded by an undertrained baseline. Run with --help.
"""
import argparse
import json
from pathlib import Path


def _val_curve(summary: dict):
    """Return the validation NI-NRMSE curve (list of floats) from a training_summary.json dict,
    tolerant to a couple of schemas."""
    for getter in (
        lambda h: h.get("val_ni_nrmse"),
        lambda h: (h.get("history") or {}).get("val_ni_nrmse"),
        lambda h: (h.get("history") or {}).get("val_nrmse"),
    ):
        v = getter(summary)
        if isinstance(v, list) and v:
            return [float(x) for x in v if x is not None]
    return []


def assess(curve: list, best_reported=None) -> dict:
    n = len(curve)
    if n == 0:
        return {"verdict": "no_val_curve", "n_epochs": 0}
    best = min(curve)
    argmin = int(min(range(n), key=lambda i: curve[i]))
    best_frac = argmin / (n - 1) if n > 1 else 1.0
    if n >= 4:
        q = max(1, n // 4)
        before = min(curve[: n - q])
        late = min(curve[n - q:])
        last_quarter_gain = (before - late) / before if before > 0 else 0.0
    else:
        last_quarter_gain = 0.0
    if n < 5:
        verdict = "too_short"
    elif best_frac >= 0.90 or last_quarter_gain > 0.02:
        verdict = "UNDERTRAINED"          # best is at/near the end, or still dropping >2%
    elif last_quarter_gain > 0.005:
        verdict = "near-converged"
    else:
        verdict = "converged"
    return {
        "verdict": verdict,
        "n_epochs": n,
        "best_val_nrmse": round(best, 5),
        "final_val_nrmse": round(curve[-1], 5),
        "best_epoch_frac": round(best_frac, 3),
        "last_quarter_gain": round(last_quarter_gain, 4),
        "best_reported": best_reported,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    rows = {}
    for summ in sorted(root.glob("*/training_summary.json")):
        if summ.parent.is_symlink():
            continue  # skip the `full` -> full_lr<best> symlink (its real dir is listed already)
        model = summ.parent.name
        try:
            h = json.loads(summ.read_text())
        except Exception:
            continue
        rows[model] = assess(_val_curve(h), best_reported=h.get("best_ni_nrmse"))

    order = {"UNDERTRAINED": 0, "too_short": 1, "near-converged": 2, "converged": 3, "no_val_curve": 4}
    print(f"{'model':22s} {'verdict':14s} {'epochs':>6s} {'best':>8s} {'best@frac':>9s} {'lastQ_gain':>10s}")
    for model, r in sorted(rows.items(), key=lambda kv: (order.get(kv[1]['verdict'], 9), kv[0])):
        print(f"{model:22s} {r['verdict']:14s} {r.get('n_epochs',0):6d} "
              f"{str(r.get('best_val_nrmse','-')):>8s} {str(r.get('best_epoch_frac','-')):>9s} "
              f"{str(r.get('last_quarter_gain','-')):>10s}")
    under = [m for m, r in rows.items() if r["verdict"] == "UNDERTRAINED"]
    if under:
        print(f"\n*** UNDERTRAINED (val still improving at budget end): {', '.join(sorted(under))}")
        print("    -> matched-budget numbers UNDERSTATE these models; extend their budget before "
              "reading their result as final.")
    else:
        print("\nAll models with a val curve reached a plateau by the budget end (no undertraining flagged).")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
