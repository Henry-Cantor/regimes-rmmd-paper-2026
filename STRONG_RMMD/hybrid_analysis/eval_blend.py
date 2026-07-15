"""Evaluate the RMMD+DGKNet fixed blend with the same NRMSE as the comparison suite. (Superseded by the
driver-keyed router in ../decisive_experiments/exp4b_router_sota.py; kept for reference.) Run with --help.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def _imports():
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    def imp(rel, name):
        spec = importlib.util.spec_from_file_location(name, strong / rel)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    rc = imp("training/rmmd_train_eval_impl.py", "rmmd_train_eval_impl")
    cmp_mod = imp("comparison/run_comparison.py", "comparison_run_comparison")
    ex = imp("theory_validation/extrap_strong.py", "extrap_strong")   # eval_dataset + summarize + activity_stratified
    return rc, cmp_mod, ex


class FixedBlendRMMD(torch.nn.Module):
    """(1-w)*RMMD + w*DGKNet at every rollout step. Frozen sub-models; no parameters of its own."""
    def __init__(self, m_rmmd, m_dgknet, w: float):
        super().__init__()
        self.m_r, self.m_d, self.w = m_rmmd, m_dgknet, float(w)
        for a in ("n_radial", "n_drivers", "machine_to_idx", "machine_embedding"):
            if hasattr(m_rmmd, a):
                try: setattr(self, a, getattr(m_rmmd, a))
                except Exception: pass

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        out_r = self.m_r(*args, **kwargs)
        out_d = self.m_d(*args, **kwargs)
        x = (1.0 - self.w) * out_r.x_next + self.w * out_d.x_next
        x = torch.nan_to_num(torch.clamp(x, -8.0, 8.0), nan=0.0)
        return dataclasses.replace(out_r, x_next=x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset", action="append", required=True, metavar="name:path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ws", default="0.0,0.1,0.2,0.3,0.4,0.5", help="blend weights to sweep")
    ap.add_argument("--horizons", type=int, nargs="*",
                    default=[1, 2, 3, 5, 8, 12, 16, 20, 32, 50, 75, 100])   # the full extrap horizon set
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--report-dir", default=str(REPO / "STRONG_RMMD/theory_validation/results"))
    ap.add_argument("--out", default=str(REPO / "STRONG_RMMD/hybrid_analysis/results/blend_sota.json"))
    args = ap.parse_args()
    rc, cmp_mod, ex = _imports()

    ck = Path(args.ckpt_root)
    mr, norm, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "full"), args.device)
    md, _, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "base_dgknet"), args.device)
    ws = [float(x) for x in args.ws.split(",")]

    def load_ds(path):
        payload = rc._load_phase0_dataset(Path(path))
        n = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        return rc.CompactRolloutDataset(payload, max_time=max(args.horizons), normalization_stats=n), n

    out = {}
    for spec in args.dataset:
        name, path = spec.split(":", 1)
        ds, n = load_ds(path)
        per_w = {}
        for w in ws:
            blend = FixedBlendRMMD(mr, md, w).eval()
            acc = ex.eval_dataset(rc, blend, ds, args.horizons, args.device, n, args.max_shots or None)
            summ = ex.summarize(acc, args.horizons, cmp_mod)
            strat = ex.activity_stratified(acc, args.horizons)
            per_w[w] = {"pooled": {h: (summ.get(h) or {}).get("nrmse") for h in args.horizons},
                        "quartiles": {h: strat.get(str(h), {}) for h in args.horizons}}   # per-horizon quartiles
        # best w by pooled T50
        t50 = {w: (per_w[w]["pooled"].get(50) or 9) for w in ws}
        best_w = min(t50, key=t50.get)
        out[name] = {"per_w_pooled_T50": t50, "best_w": best_w,
                     "best_w_pooled_by_horizon": per_w[best_w]["pooled"],           # all 12 horizons (like extrap)
                     "best_w_quartiles_by_horizon": per_w[best_w]["quartiles"],     # q1..q4 at every horizon (like extrap)
                     "rmmd_only_pooled_by_horizon": per_w[0.0]["pooled"] if 0.0 in per_w else None}
        print(f"[{name}] pooled T50 by w: " + "  ".join(f"{w}:{t50[w]:.3f}" for w in ws) + f"  -> best w={best_w}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", args.out)
    print("\nCompare best_w_pooled_by_horizon + best_w_quartiles_T50 against RMMD / baselines in "
          f"{args.report_dir}/extrap_strong_report_*.json to state the SOTA claim (w=0 row = RMMD alone).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
