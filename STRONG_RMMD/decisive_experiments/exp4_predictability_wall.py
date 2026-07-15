"""EXP-4 -- the predictability wall.

Tests whether any input feature predicts which model (RMMD vs DGKNet) wins a dynamic shot. Adds harder
features (driver spectral entropy, driver max-slope, NI curvature, short-window growth-rate proxy) and
recomputes both the winner-prediction AUC and the predict-absolute-NRMSE R^2. Pre-registered: max AUC stays
below 0.60, i.e. forecast difficulty on dynamic shots is not explained by observed macroscopic inputs.
Run with --help.
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
    return imp("training/rmmd_train_eval_impl.py", "rmmd_train_eval_impl"), imp("comparison/run_comparison.py", "comparison_run_comparison")


def shot_features(s):
    """Macroscopic input features (extended per spec) — NONE use the future NI(h), so no leakage."""
    f = {}
    ni0 = s["ni_t0"].reshape(-1).cpu().numpy().astype(np.float64)
    f["ni_peaking"] = float(ni0.max() / (ni0.mean() + 1e-9))
    f["ni_curvature"] = float(np.abs(np.diff(ni0, 2)).mean()) if ni0.size > 2 else 0.0
    f["ni_mean0"] = float(ni0.mean()); f["ni_std0"] = float(ni0.std())
    drv = s.get("drivers_traj")
    if isinstance(drv, torch.Tensor) and drv.numel():
        d = drv.detach().cpu().numpy().astype(np.float64)   # (T, n_drivers)
        p = d[:, 0]                                         # channel 0 = PINJ
        f["driver_ptp"] = float(np.ptp(p)); f["driver_std"] = float(p.std())
        f["driver_max_slope"] = float(np.abs(np.diff(p)).max()) if p.size > 1 else 0.0
        # short-window growth-rate proxy: max relative change over a 3-step window
        if p.size > 3:
            w = p[3:] - p[:-3]; f["driver_growth3"] = float(np.abs(w).max())
        else:
            f["driver_growth3"] = 0.0
        # spectral entropy of the drive (flat spectrum = high entropy = broadband dynamism)
        if p.size > 4:
            ps = np.abs(np.fft.rfft(p - p.mean())) ** 2; ps = ps / (ps.sum() + 1e-12)
            f["driver_spec_entropy"] = float(-(ps * np.log(ps + 1e-12)).sum())
        else:
            f["driver_spec_entropy"] = 0.0
    else:
        for k in ("driver_ptp", "driver_std", "driver_max_slope", "driver_growth3", "driver_spec_entropy"):
            f[k] = np.nan
    return f


@torch.no_grad()
def shot_nrmse(rc, model, s, device, norm, H):
    T = int(s["ni_traj"].shape[0])
    if T < 2:
        return None
    preds, _ = rc._rollout_compact_shot_to_checkpoints(
        model, s["ni_t0"], s["geom_t0"], s["pre_shot_context"], s["limiter_geometry_tensor"],
        s["ni_traj"], s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}),
        device, norm, max_time_step=min(H, T), drivers_traj=s.get("drivers_traj"), report_horizons=[min(H, T)])
    h = min(H, T)
    if h not in preds:
        return None
    e, _ = rc._normalized_rmse_mae(preds[h].numpy(), s["ni_traj"][h - 1].numpy())
    return float(e)


def auc(x, wins):
    x = np.asarray(x, float); wins = np.asarray(wins, int)
    ok = np.isfinite(x); x, wins = x[ok], wins[ok]
    pos, neg = x[wins == 1], x[wins == 0]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    # Mann-Whitney AUC
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(max(a, 1 - a))          # orientation-free (feature could point either way)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset", action="append", required=True, metavar="name:path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--max-shots", type=int, default=0)
    args = ap.parse_args()
    rc, cmp_mod = _imports()
    ck = Path(args.ckpt_root)
    mr, nr, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "full"), args.device)
    md, nd, _ = cmp_mod._build_model(rc, cmp_mod._find_checkpoint(ck / "base_dgknet"), args.device)

    report = {"horizon": args.horizon, "per_dataset": {}}
    for spec in args.dataset:
        name, path = spec.split(":", 1)
        payload = rc._load_phase0_dataset(Path(path))
        n = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        ds = rc.CompactRolloutDataset(payload, max_time=args.horizon, normalization_stats=n)
        N = min(len(ds), args.max_shots) if args.max_shots else len(ds)
        feats, wins, rmmd_err, dgk_err = [], [], [], []
        for i in range(N):
            s = ds[i]
            er = shot_nrmse(rc, mr, s, args.device, nr, args.horizon)
            ed = shot_nrmse(rc, md, s, args.device, nd, args.horizon)
            if er is None or ed is None:
                continue
            feats.append(shot_features(s)); wins.append(int(ed < er)); rmmd_err.append(er); dgk_err.append(ed)
        if not feats:
            report["per_dataset"][name] = "NOT FOUND (no valid shots)"; continue
        keys = sorted(feats[0])
        X = np.array([[f[k] for k in keys] for f in feats], float)
        wins = np.array(wins); rmmd_err = np.array(rmmd_err); dgk_err = np.array(dgk_err)
        # AUC per feature for "DGKNet wins"  (in-sample: necessary, NOT sufficient)
        aucs = {k: auc(X[:, j], wins) for j, k in enumerate(keys)}
        # R^2 predicting absolute NRMSE from ALL features (ridge, standardized, 5-fold)
        r2 = _cv_r2(X, rmmd_err)
        # THE REAL TEST: does a router (threshold fit on held-out folds) beat BOTH models OUT-OF-SAMPLE?
        router = _router_cv(X, keys, rmmd_err, dgk_err, aucs)
        report["per_dataset"][name] = {
            "n_shots": int(len(wins)), "n_features_tested": len(keys), "dgknet_win_rate": float(wins.mean()),
            "auc_per_feature": aucs, "max_feature_auc": float(np.nanmax(list(aucs.values()))),
            "r2_predict_nrmse_from_inputs": r2,
            "router_out_of_sample": router,   # in-sample AUC can lie; THIS is what decides "router is real"
        }

    # ---- VERDICT ----
    dd = [d for d in report["per_dataset"].values() if isinstance(d, dict)]
    max_auc = max((d.get("max_feature_auc", 0) for d in dd), default=float("nan"))
    max_r2 = max((d.get("r2_predict_nrmse_from_inputs", 0) for d in dd), default=float("nan"))
    # a router is REAL only if it beats BOTH models OUT-OF-SAMPLE (in-sample AUC alone can be a
    # multiple-comparison / overfit artifact across the ~9 features tested).
    router_oos_wins = [d["router_out_of_sample"].get("router_beats_both_oos")
                       for d in dd if isinstance(d.get("router_out_of_sample"), dict)]
    router_real = bool(np.isfinite(max_auc) and max_auc > 0.65 and any(router_oos_wins))
    high_auc_only = bool(np.isfinite(max_auc) and max_auc > 0.65 and not any(router_oos_wins))
    report["VERDICT_4"] = {
        "claim_preregistered": "max AUC < 0.60 and low R^2 -> forecast difficulty not explained by observed inputs.",
        "refuted_if": "a router (threshold fit out-of-sample) BEATS BOTH models -> the SOTA combo is real; build it.",
        "max_feature_auc": max_auc, "max_r2_predict_nrmse": max_r2,
        "router_beats_both_out_of_sample": bool(any(router_oos_wins)),
        "VERDICT": ("REFUTED -> router is REAL, build it" if router_real
                    else ("INCONCLUSIVE: in-sample AUC>0.65 but router does NOT beat both out-of-sample "
                          "(likely multiple-comparison / in-sample artifact; wall effectively stands)" if high_auc_only
                          else ("SUPPORTED (predictability wall)" if np.isfinite(max_auc) and max_auc < 0.60 else "INCONCLUSIVE"))),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "predictability_wall.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["VERDICT_4"], indent=1, default=float))
    print("wrote", RESULTS / "predictability_wall.json")


def _router_cv(X, keys, er, ed, aucs, folds=5, margin=0.003):
    """Cross-validated hard router on the single best feature: fit threshold+orientation on train folds to
    minimize hybrid NRMSE, apply to the held-out fold, and compare the OUT-OF-SAMPLE hybrid NRMSE to
    RMMD-alone and DGKNet-alone. This is the honest 'is a router real' test (in-sample AUC can lie)."""
    fj = max(range(len(keys)), key=lambda j: abs((aucs[keys[j]] or 0.5) - 0.5))
    x = X[:, fj].astype(float); ok = np.isfinite(x)
    x, er, ed = x[ok], np.asarray(er)[ok], np.asarray(ed)[ok]
    n = len(x)
    if n < 2 * folds:
        return {"feature": keys[fj], "note": "too few shots for CV"}
    idx = np.random.default_rng(0).permutation(n); parts = np.array_split(idx, folds)
    routed = np.empty(n)
    for k in range(folds):
        te = parts[k]; trn = np.concatenate([parts[j] for j in range(folds) if j != k])
        best = None
        for th in np.quantile(x[trn], np.linspace(0.1, 0.9, 17)):
            for orient in (1, -1):
                hyb = np.where(orient * x[trn] > orient * th, ed[trn], er[trn]).mean()
                if best is None or hyb < best[0]:
                    best = (hyb, th, orient)
        _, th, orient = best
        routed[te] = np.where(orient * x[te] > orient * th, ed[te], er[te])
    hyb, rmmd, dgk = float(routed.mean()), float(er.mean()), float(ed.mean())
    return {"feature": keys[fj], "auc": aucs[keys[fj]], "hybrid_nrmse_oos": hyb,
            "rmmd_nrmse": rmmd, "dgknet_nrmse": dgk,
            "router_beats_both_oos": bool(hyb < min(rmmd, dgk) - margin)}


def _cv_r2(X, y, folds=5, lam=1.0):
    X = np.nan_to_num(X); y = np.asarray(y, float)
    if len(y) < 2 * folds:
        return float("nan")
    mu, sd = X.mean(0), X.std(0) + 1e-9
    idx = np.random.default_rng(0).permutation(len(y)); parts = np.array_split(idx, folds)
    preds = np.zeros_like(y)
    for k in range(folds):
        te = parts[k]; trn = np.concatenate([parts[j] for j in range(folds) if j != k])
        Xt = (X[trn] - mu) / sd; Xv = (X[te] - mu) / sd
        A = Xt.T @ Xt + lam * np.eye(Xt.shape[1]); w = np.linalg.solve(A, Xt.T @ (y[trn] - y[trn].mean()))
        preds[te] = Xv @ w + y[trn].mean()
    ss_res = float(((y - preds) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1 - ss_res / (ss_tot + 1e-12))


if __name__ == "__main__":
    main()
