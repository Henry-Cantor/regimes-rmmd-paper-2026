"""Diagnosis of RMMD vs DGKNet complementarity. (Superseded by the driver-keyed router in
../decisive_experiments/exp4b_router_sota.py and buttress.py analysis B; kept for reference.)

Tests whether the per-shot RMMD/DGKNet complementarity has spatial or temporal structure that a fused operator
could exploit, before building one. Run with --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
H_GRID = [20, 50, 100]


def _imports():
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    def imp(rel, name):
        spec = importlib.util.spec_from_file_location(name, strong / rel)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    return imp("training/rmmd_train_eval_impl.py", "rmmd_train_eval_impl"), imp("comparison/run_comparison.py", "comparison_run_comparison")


@torch.no_grad()
def per_point_errors(rc, model, dataset, device, norm, max_shots):
    """Return dict h -> per-shot per-RADIUS squared error, target variance, and the raw prediction
    (predictions are kept so main() can also evaluate a FIXED RMMD+DGKNet ensemble = the cheap,
    deployable, no-oracle test of whether plain averaging beats either model)."""
    out = {h: {"se": [], "tvar": [], "shot": [], "pred": [], "tgt": []} for h in H_GRID}
    n = min(len(dataset), max_shots) if max_shots else len(dataset)
    for i in range(n):
        s = dataset[i]; T = int(s["ni_traj"].shape[0])
        if T < min(H_GRID):
            continue
        ni_preds, _ = rc._rollout_compact_shot_to_checkpoints(
            model, s["ni_t0"], s["geom_t0"], s["pre_shot_context"], s["limiter_geometry_tensor"],
            s["ni_traj"], s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}),
            device, norm, max_time_step=min(max(H_GRID), T), drivers_traj=s.get("drivers_traj"),
            report_horizons=H_GRID)
        for h in H_GRID:
            if h > T or h not in ni_preds:
                continue
            tgt = s["ni_traj"][h - 1].numpy().astype(np.float64)
            pred = ni_preds[h].numpy().astype(np.float64)
            out[h]["se"].append((pred - tgt) ** 2)
            out[h]["tvar"].append(float(np.var(tgt)) + 1e-9)
            out[h]["shot"].append(i); out[h]["pred"].append(pred); out[h]["tgt"].append(tgt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset", action="append", required=True, metavar="name:path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "STRONG_RMMD/hybrid_analysis/results/complementarity.json"))
    args = ap.parse_args()

    rc, cmp_mod = _imports()
    ck = Path(args.ckpt_root)
    mr, nr, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "full"), args.device)
    md, nd, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "base_dgknet"), args.device)

    report = {}
    for spec in args.dataset:
        name, path = spec.split(":", 1)
        payload = rc._load_phase0_dataset(Path(path))
        norm = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        ds = rc.CompactRolloutDataset(payload, max_time=max(H_GRID), normalization_stats=norm)
        eR = per_point_errors(rc, mr, ds, args.device, nr, args.max_shots or None)
        eD = per_point_errors(rc, md, ds, args.device, nd, args.max_shots or None)

        rep = {"per_radius": {}, "per_horizon": {}, "win_consistency": {}, "ceilings": {}}
        # align shots present for both at each horizon
        for h in H_GRID:
            shR = {s: k for k, s in enumerate(eR[h]["shot"])}
            shD = {s: k for k, s in enumerate(eD[h]["shot"])}
            common = sorted(set(shR) & set(shD))
            if not common:
                continue
            SR = np.array([eR[h]["se"][shR[s]] for s in common])      # (n, n_radial)
            SD = np.array([eD[h]["se"][shD[s]] for s in common])
            tv = np.array([eR[h]["tvar"][shR[s]] for s in common])[:, None]
            # 1. per-radius mean NRMSE (sqrt of mean SE / target var), each model
            rad_R = np.sqrt((SR / tv).mean(0)); rad_D = np.sqrt((SD / tv).mean(0))
            rep["per_radius"][h] = {"rmmd": rad_R.tolist(), "dgknet": rad_D.tolist(),
                                     "rmmd_wins_frac_by_radius": (rad_R < rad_D).mean().item()}
            # 2. per-horizon pooled NRMSE + FIXED ENSEMBLE (deployable 50/50 average of the two predictions)
            nR = float(np.sqrt((SR.sum(1) / (tv[:, 0] * SR.shape[1])).mean()))
            nD = float(np.sqrt((SD.sum(1) / (tv[:, 0] * SD.shape[1])).mean()))
            PR = np.array([eR[h]["pred"][shR[s]] for s in common])
            PD = np.array([eD[h]["pred"][shD[s]] for s in common])
            TG = np.array([eR[h]["tgt"][shR[s]] for s in common])
            SE_ens = (0.5 * (PR + PD) - TG) ** 2
            nEns = float(np.sqrt((SE_ens.sum(1) / (tv[:, 0] * SE_ens.shape[1])).mean()))
            # RATIO SWEEP: best FIXED blend (1-w)*RMMD + w*DGKNet at this horizon. This is the deployable
            # SOTA route (no training) -- the learned gate collapses to RMMD, but a fixed blend can't.
            ws = np.linspace(0, 1, 21)
            nrmse_w = [float(np.sqrt(((( (1 - w) * PR + w * PD) - TG) ** 2).sum(1) / (tv[:, 0] * PR.shape[1])).mean())
                       for w in ws]
            j = int(np.argmin(nrmse_w))
            rep["per_horizon"][h] = {"rmmd": nR, "dgknet": nD, "ensemble_50_50": nEns,
                                     "best_blend_w": float(ws[j]), "best_blend_nrmse": float(nrmse_w[j]),
                                     "best_blend_beats_both": float(nrmse_w[j]) < min(nR, nD) - 1e-4,
                                     "rmmd_better": nR < nD,
                                     "ensemble_beats_both": nEns < min(nR, nD) - 1e-4}
            # 4. ceilings at this horizon: shot-oracle (pick whole-shot best) vs radius-blend oracle
            shot_err_R = np.sqrt((SR / tv).mean(1)); shot_err_D = np.sqrt((SD / tv).mean(1))
            shot_oracle = float(np.minimum(shot_err_R, shot_err_D).mean())
            radius_oracle = float(np.sqrt((np.minimum(SR, SD) / tv).mean()))   # pick best model PER radius
            rep["ceilings"][h] = {"rmmd": float(shot_err_R.mean()), "dgknet": float(shot_err_D.mean()),
                                   "shot_oracle": shot_oracle, "radius_blend_oracle": radius_oracle,
                                   "radius_gain_over_shot_oracle": shot_oracle - radius_oracle}
        # 3. win-consistency across horizons: does the per-shot winner agree at T20 vs T50 vs T100?
        def shot_winner(h):
            shR = {s: k for k, s in enumerate(eR[h]["shot"])}; shD = {s: k for k, s in enumerate(eD[h]["shot"])}
            common = sorted(set(shR) & set(shD))
            w = {}
            for s in common:
                r = np.sqrt(np.mean(eR[h]["se"][shR[s]] / eR[h]["tvar"][shR[s]]))
                d = np.sqrt(np.mean(eD[h]["se"][shD[s]] / eD[h]["tvar"][shD[s]]))
                w[s] = int(r < d)
            return w
        ws = {h: shot_winner(h) for h in H_GRID}
        for a, b in [(20, 50), (50, 100), (20, 100)]:
            common = sorted(set(ws.get(a, {})) & set(ws.get(b, {})))
            if len(common) > 4:
                va = np.array([ws[a][s] for s in common]); vb = np.array([ws[b][s] for s in common])
                rep["win_consistency"][f"T{a}_vs_T{b}"] = {"agree_frac": float((va == vb).mean()),
                                                            "n": len(common)}
        report[name] = rep

    # ---------- verdict ----------
    def agg(metric):
        vals = []
        for nm, r in report.items():
            for h, d in r.get("ceilings", {}).items():
                vals.append(d.get(metric))
        return float(np.mean([v for v in vals if v is not None])) if vals else None
    consist = [c["agree_frac"] for r in report.values() for c in r.get("win_consistency", {}).values()]
    radius_gain = agg("radius_gain_over_shot_oracle")
    mean_consist = float(np.mean(consist)) if consist else None
    structured = bool((mean_consist is not None and mean_consist > 0.62) or
                      (radius_gain is not None and radius_gain > 0.02))
    ens_flags = [d.get("ensemble_beats_both") for r in report.values() for d in r.get("per_horizon", {}).values()]
    fixed_ensemble_helps = bool(ens_flags) and (sum(bool(x) for x in ens_flags) / len(ens_flags) >= 0.5)
    report["VERDICT"] = {
        "win_consistency_across_horizons": mean_consist,   # >~0.6 => structured (winner is stable, not noise)
        "radius_blend_gain_over_shot_oracle": radius_gain,  # >0 => spatial structure a gate could exploit
        "complementarity_is_STRUCTURED": structured,
        "fixed_ensemble_beats_both": fixed_ensemble_helps,  # deployable 50/50 average already beats both? (cheapest win)
        "interpretation": ("STRUCTURED: the winner is stable across horizons and/or a per-radius blend beats "
                           "the shot oracle -> a FUSED operator with a spatial-temporal gate is worth building."
                           if structured else
                           "looks like NOISE: winner flips across horizons (~0.5 agreement) and a per-radius "
                           "blend adds little over the shot oracle -> combining RMMD+DGKNet cannot reach SOTA; "
                           "pursue a better SINGLE model instead (do NOT build the fused operator)."),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print("VERDICT:", json.dumps(report["VERDICT"], indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
