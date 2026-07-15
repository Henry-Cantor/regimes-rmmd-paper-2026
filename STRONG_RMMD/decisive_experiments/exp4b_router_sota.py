"""EXP-4b -- the deployable driver-keyed router.

Routes per shot on driver_ptp (peak-to-peak of the prescribed heating drive): above a threshold theta use
DGKNet (for dynamic shots), else RMMD. Drivers are known control inputs, so this is available at inference.
Scored with the same extrap harness (per horizon and activity quartile) as the paper table. theta_zeroshot is
fit on --fit-dataset and applied blind to the holdouts; theta_tuned is the within-dataset optimum (an upper
bound). Run with --help.
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
RESULTS = Path(__file__).resolve().parent / "results"


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
    ex = imp("theory_validation/extrap_strong.py", "extrap_strong")
    return rc, cmp_mod, ex


def _driver_feature(s, kind="ptp"):
    """Routing feature from the PRESCRIBED drive (channel 0 = PINJ). 'ptp' is RAW (machine-scale-dependent);
    'rel_ptp'/'cv'/'spec_entropy' are SCALE-INVARIANT (candidates for a threshold that transfers zero-shot)."""
    drv = s.get("drivers_traj")
    if not (isinstance(drv, torch.Tensor) and drv.numel()):
        return np.nan
    p = drv.detach().cpu().numpy().astype(np.float64)[:, 0]
    if kind == "ptp":
        return float(np.ptp(p))
    if kind == "rel_ptp":
        return float(np.ptp(p) / (np.abs(p).mean() + 1e-9))          # relative drive swing (dimensionless)
    if kind == "cv":
        return float(p.std() / (np.abs(p).mean() + 1e-9))            # coefficient of variation
    if kind == "spec_entropy":
        if p.size < 5:
            return 0.0
        ps = np.abs(np.fft.rfft(p - p.mean())) ** 2; ps = ps / (ps.sum() + 1e-12)
        return float(-(ps * np.log(ps + 1e-12)).sum())
    return np.nan


def route_acc(acc_r, acc_d, ptp_by_shot, theta, horizons):
    """Build a routed acc (extrap format) picking per shot from RMMD or DGKNet by driver_ptp vs theta."""
    routed = {h: {"nrmse": [], "shots": [], "machine": [], "pers": [], "gate": [],
                  "geom_nrmse": [], "geom_pers": []} for h in horizons}
    for h in horizons:
        d_nr = dict(zip(acc_d[h]["shots"], acc_d[h]["nrmse"]))
        for k, sh in enumerate(acc_r[h]["shots"]):
            use_d = (ptp_by_shot.get(sh, np.nan) > theta) and (sh in d_nr)
            routed[h]["nrmse"].append(d_nr[sh] if use_d else acc_r[h]["nrmse"][k])
            routed[h]["shots"].append(sh); routed[h]["machine"].append(acc_r[h]["machine"][k])
            routed[h]["pers"].append(acc_r[h]["pers"][k]); routed[h]["gate"].append(float("nan"))
    return routed


def pooled_T(acc, h):
    v = acc[h]["nrmse"]; return float(np.mean(v)) if v else 9.0


def load_baselines(name, horizons):
    """ALL prior methods (RMMD/DGKNet/LSTM/NODE/MLP) at their best-by-T50 config, from the committed reports,
    so the router's SOTA claim is checked against EVERY baseline -- not just the two models it routes between."""
    base = REPO / "STRONG_RMMD"
    if name in ("augd", "east"):
        p = base / "theory_validation" / "results" / f"extrap_strong_report_{name}.json"; key = "holdout"
    elif name == "indist":
        p = base / "comparison" / "results" / "comparison_table_fair.json"; key = "by_horizon"
    else:
        return {}
    if not p.exists():
        return {}
    M = __import__("json").loads(p.read_text())["models"]
    def nr(m, h):
        v = (M.get(m, {}).get(key) or {}).get(str(h)); return v.get("nrmse") if isinstance(v, dict) else v
    out = {}
    for label, pref in [("RMMD", "full"), ("DGKNet", "base_dgknet"), ("LSTM", "base_lstm"),
                        ("NODE", "base_node"), ("MLP", "base_mlp")]:
        cands = [m for m in M if (m == "full" or m.startswith("full_"))] if pref == "full" else [m for m in M if m.startswith(pref)]
        cands = [m for m in cands if nr(m, 50) is not None]
        if cands:
            best = min(cands, key=lambda m: nr(m, 50))
            out[label] = {str(h): nr(best, h) for h in horizons}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset", action="append", required=True, metavar="name:path")
    ap.add_argument("--fit", default="indist", help="dataset name whose shots set theta for the ZERO-SHOT router")
    ap.add_argument("--feature", default="ptp", choices=["ptp", "rel_ptp", "cv", "spec_entropy"],
                    help="routing feature: ptp=raw (machine-scale-dependent); others are scale-invariant (zero-shot candidates)")
    ap.add_argument("--second", default="base_dgknet",
                    help="second model to route with (base_dgknet | base_lstm | base_node | base_mlp). "
                         "LSTM is the stronger complement to RMMD on AUGD -> likely a better router.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fit-horizon", type=int, default=50)
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 2, 3, 5, 8, 12, 16, 20, 32, 50, 75, 100])
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "router_sota.json"))
    args = ap.parse_args()
    rc, cmp_mod, ex = _imports()
    ck = Path(args.ckpt_root)
    mr, _, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "full"), args.device)
    md, _, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / args.second), args.device)

    data = {}
    for spec in args.dataset:
        name, path = spec.split(":", 1)
        payload = rc._load_phase0_dataset(Path(path))
        n = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        ds = rc.CompactRolloutDataset(payload, max_time=max(args.horizons), normalization_stats=n)
        acc_r = ex.eval_dataset(rc, mr, ds, args.horizons, args.device, n, args.max_shots or None)
        acc_d = ex.eval_dataset(rc, md, ds, args.horizons, args.device, n, args.max_shots or None)
        ptp = {i: _driver_feature(ds[i], args.feature) for i in range(len(ds))}
        data[name] = {"acc_r": acc_r, "acc_d": acc_d, "ptp": ptp}

    # ---- theta_zeroshot: fit on --fit dataset (minimize routed pooled@fit-horizon), never touches the holdout ----
    def best_theta(d, h):
        vals = np.array([v for v in d["ptp"].values() if np.isfinite(v)])
        if vals.size < 8:
            return None
        grid = np.quantile(vals, np.linspace(0.05, 0.95, 19))
        best = None
        for th in grid:
            r = route_acc(d["acc_r"], d["acc_d"], d["ptp"], th, [h])
            p = pooled_T(r, h)
            if best is None or p < best[0]:
                best = (p, float(th))
        return best[1] if best else None
    theta_zs = best_theta(data[args.fit], args.fit_horizon) if args.fit in data else None

    report = {"fit_dataset": args.fit, "second_model": args.second, "routing_feature": args.feature,
              "theta_zeroshot": theta_zs, "horizons": args.horizons, "per_dataset": {}}
    for name, d in data.items():
        theta_tuned = best_theta(d, args.fit_horizon)
        baselines = load_baselines(name, args.horizons)   # ALL prior methods, best-by-T50 config
        out = {"theta_tuned_on_this_set": theta_tuned, "rmmd_pooled": {}, "second_pooled": {},
               "baselines_pooled": baselines, "router_zeroshot_pooled": {}, "router_tuned_pooled": {},
               "router_zeroshot_quartiles": {}, "router_tuned_quartiles": {},
               "router_zs_beats_all_baselines_by_horizon": {}, "n_shots": len(d["ptp"])}
        r_zs = route_acc(d["acc_r"], d["acc_d"], d["ptp"], theta_zs, args.horizons) if theta_zs is not None else None
        r_tn = route_acc(d["acc_r"], d["acc_d"], d["ptp"], theta_tuned, args.horizons) if theta_tuned is not None else None
        strat_zs = ex.activity_stratified(r_zs, args.horizons) if r_zs else {}
        strat_tn = ex.activity_stratified(r_tn, args.horizons) if r_tn else {}
        for h in args.horizons:
            out["rmmd_pooled"][str(h)] = pooled_T(d["acc_r"], h)
            out["second_pooled"][str(h)] = pooled_T(d["acc_d"], h)
            bl = [v.get(str(h)) for v in baselines.values() if v.get(str(h)) is not None]
            if r_zs:
                rz_h = pooled_T(r_zs, h)
                out["router_zeroshot_pooled"][str(h)] = rz_h
                out["router_zeroshot_quartiles"][str(h)] = strat_zs.get(str(h), {})
                out["router_zs_beats_all_baselines_by_horizon"][str(h)] = bool(bl and rz_h < min(bl) - 0.003)
            if r_tn:
                out["router_tuned_pooled"][str(h)] = pooled_T(r_tn, h)
                out["router_tuned_quartiles"][str(h)] = strat_tn.get(str(h), {})
        # per-dataset: does the router beat ALL prior methods, and at which horizons?
        wins_all = [h for h, ok in out["router_zs_beats_all_baselines_by_horizon"].items() if ok]
        out["router_zs_SOTA_horizons"] = sorted(wins_all, key=int)
        H = str(args.fit_horizon)
        rz = out["router_zeroshot_pooled"].get(H); rt = out["router_tuned_pooled"].get(H)
        rm = out["rmmd_pooled"].get(H); dg = out["second_pooled"].get(H)
        out["router_zeroshot_beats_both_routed"] = bool(rz is not None and rz < min(rm, dg) - 0.003)
        out["router_tuned_beats_both_routed"] = bool(rt is not None and rt < min(rm, dg) - 0.003)
        report["per_dataset"][name] = out

    holds = {k: v for k, v in report["per_dataset"].items() if k != args.fit}
    zs_win = any(v["router_zeroshot_beats_both_routed"] for v in holds.values())
    tn_win = any(v["router_tuned_beats_both_routed"] for v in holds.values())
    zs_sota_all = any(v["router_zs_SOTA_horizons"] for v in holds.values())   # beats EVERY baseline at some horizon
    sota_by_ds = {k: v["router_zs_SOTA_horizons"] for k, v in holds.items()}
    report["VERDICT_router"] = {
        "second_model": args.second, "routing_feature": args.feature, "theta_zeroshot": theta_zs,
        "zeroshot_router_beats_BOTH_ROUTED_on_a_holdout": zs_win,
        "tuned_router_beats_BOTH_ROUTED_on_a_holdout": tn_win,
        "zeroshot_router_beats_ALL_BASELINES_at_horizons": sota_by_ds,   # THE honest SOTA claim (vs every prior method)
        "reading": ("ZERO-SHOT SOTA vs ALL baselines at horizons %s -> airtight, scoped to those horizons." % sota_by_ds)
                   if zs_sota_all else
                   ("beats the two ROUTED models zero-shot but NOT every baseline at any holdout horizon -> NOT SOTA; "
                    "report as 'beats RMMD and %s' only, and name which baseline wins where." % args.second),
        "VERDICT": ("ZERO-SHOT SOTA (vs all baselines, scoped)" if zs_sota_all
                    else ("beats both routed but not all baselines" if zs_win else "INCONCLUSIVE")),
        "NOTE": "SOTA here = beats EVERY prior method (RMMD/DGKNet/LSTM/NODE/MLP, best config) at that horizon, "
                "not just the two models routed. Check router_zs_SOTA_horizons per dataset.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["VERDICT_router"], indent=1, default=float))
    for name, v in report["per_dataset"].items():
        H = str(args.fit_horizon)
        bl = v.get("baselines_pooled", {})
        best_bl = min([b.get(H) for b in bl.values() if b.get(H) is not None], default=None)
        print(f"[{name}] T{H}: router_zs={v['router_zeroshot_pooled'].get(H)}  tuned={v['router_tuned_pooled'].get(H)}  "
              f"rmmd={v['rmmd_pooled'].get(H)}  {args.second}={v['second_pooled'].get(H)}  best_baseline={best_bl}  "
              f"SOTA_horizons={v['router_zs_SOTA_horizons']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
