"""Find the RMMD->DGKNet switch threshold from per_shot.json. (Superseded by the driver-keyed router in
../decisive_experiments/exp4b_router_sota.py; kept for reference.)

Tests whether a hard per-shot switch (use DGKNet when a feature exceeds a threshold, else RMMD) beats either
model alone, reporting both an oracle switch (on true activity) and a deployable switch (on inference
features). Run with --help.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FEATURES = ["activity", "driver_ptp_norm", "driver_std_mean", "driver_max_slope",
            "ni_peaking", "ni_mean0", "geom_mean0", "geom_std0", "geom_ptp0"]
FIT_SETS = {"train", "val", "test"}   # in-distribution: theta is chosen here. HELD-OUT = the rest (machines).
AUC_MIN = 0.55                        # a feature with AUC<=~0.5 carries NO signal; "beating both" on it is
                                      # a multiple-comparisons artifact, NOT a real switch. Gate on real signal.


def auc(x, wins):
    """Rank-AUC that higher x predicts wins=1. 0.5 = no signal."""
    pos, neg = x[wins], x[~wins]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hybrid_nrmse(x, theta, rmmd, dgknet):
    """RMMD where x<theta, DGKNet where x>=theta."""
    return float(np.where(x < theta, rmmd, dgknet).mean())


def best_theta(x, rmmd, dgknet):
    cands = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 49)))
    vals = [hybrid_nrmse(x, t, rmmd, dgknet) for t in cands]
    j = int(np.argmin(vals))
    return float(cands[j]), float(vals[j])


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "STRONG_RMMD/hybrid_analysis/results/per_shot.json"
    rows = json.load(open(path))["shots"]
    if not rows:
        print("no shots"); return 1
    ds = np.array([r["dataset"] for r in rows])
    rmmd = np.array([r["nrmse_rmmd"] for r in rows], float)
    dgk = np.array([r["nrmse_dgknet"] for r in rows], float)
    wins = np.array([r["dgknet_wins"] for r in rows], bool)
    fit = np.array([d in FIT_SETS for d in ds])
    holdouts = sorted(set(ds[~fit]))

    def pooled(mask, arr):
        return float(arr[mask].mean()) if mask.any() else float("nan")
    oracle = np.minimum(rmmd, dgk)
    print(f"n={len(rows)}  fit(val+test)={fit.sum()}  holdout={(~fit).sum()}  "
          f"DGKNet-wins overall={wins.mean():.0%}\n")
    print(f"{'set':10s} {'RMMD':>7s} {'DGKNet':>7s} {'oracle':>7s}")
    for grp, m in [("FIT", fit)] + [(h, ds == h) for h in holdouts] + [("ALL", np.ones(len(rows), bool))]:
        print(f"{grp:10s} {pooled(m, rmmd):7.3f} {pooled(m, dgk):7.3f} {pooled(m, oracle):7.3f}")

    print("\nfeature                AUC   theta   hybrid(FIT)  " + "  ".join(f"hyb({h})" for h in holdouts))
    results = {}
    for f in FEATURES:
        x = np.array([r.get(f, 0.0) for r in rows], float)
        a = auc(x, wins)
        # orient so "higher -> dgknet wins"; if AUC<0.5 the feature is inverted -> flip sign
        xs = x if (np.isnan(a) or a >= 0.5) else -x
        theta, hfit = best_theta(xs[fit], rmmd[fit], dgk[fit])
        hold = [hybrid_nrmse(xs[ds == h], theta, rmmd[ds == h], dgk[ds == h]) for h in holdouts]
        results[f] = {"auc": a, "theta": theta, "hybrid_fit": hfit,
                      "hybrid_holdout": dict(zip(holdouts, hold))}
        print(f"{f:20s} {a:5.2f} {theta:7.3f} {hfit:11.3f}  " + "  ".join(f"{v:7.3f}" for v in hold))

    # ---- conservative verdict (deployable only; activity is the oracle, excluded) ----
    deploy = {f: r for f, r in results.items() if f != "activity"}
    def beats_both_on_holdouts(r):
        return all(r["hybrid_holdout"][h] < min(pooled(ds == h, rmmd), pooled(ds == h, dgk)) - 1e-4
                   for h in holdouts) if holdouts else False
    # a winner must BOTH beat both alone on every holdout AND carry real signal (AUC >= AUC_MIN) — the
    # AUC gate kills the multiple-comparisons false positive (a no-signal feature that clears by chance).
    winners = [f for f, r in deploy.items()
               if beats_both_on_holdouts(r) and not np.isnan(r["auc"]) and r["auc"] >= AUC_MIN]
    oracle_thr = results["activity"]
    verdict = {
        "deployable_switch_works": bool(winners),
        "winning_features": winners,
        "max_feature_auc": float(max((r["auc"] for r in deploy.values() if not np.isnan(r["auc"])), default=float("nan"))),
        "complementarity_exists": bool(pooled(np.ones(len(rows), bool), oracle) < 0.97 * min(
            pooled(np.ones(len(rows), bool), rmmd), pooled(np.ones(len(rows), bool), dgk))),
        "best_deployable_feature": (min(deploy, key=lambda f: np.mean(list(deploy[f]["hybrid_holdout"].values())))
                                    if holdouts else None),
        "oracle_threshold_helps": bool(holdouts and all(
            oracle_thr["hybrid_holdout"][h] < min(pooled(ds == h, rmmd), pooled(ds == h, dgk)) - 1e-4
            for h in holdouts)),
        "note": ("DEPLOYABLE switch beats BOTH alone on every holdout AND keys on a real-signal feature "
                 "(AUC>=%.2f) -> build the hard-switch hybrid." % AUC_MIN
                 if winners else
                 "NO deployable feature with real signal beats both alone (max feature AUC=%.2f ~ noise). "
                 "If complementarity_exists is true the two models ARE complementary per-shot but that "
                 "complementarity is NOT predictable from any input feature -> a feature-keyed switch OR "
                 "blend cannot capture it; the only path to SOTA-everywhere is an end-to-end FUSED operator "
                 "(or a fixed ensemble), NOT a learned per-shot selector." % (max(
                     (r["auc"] for r in deploy.values() if not np.isnan(r["auc"])), default=float("nan")))),
    }
    print("\nVERDICT:", json.dumps(verdict, indent=2))
    out = str(Path(path).with_name("threshold_report.json"))   # never overwrite the input
    json.dump({"baselines": {"rmmd": pooled(np.ones(len(rows), bool), rmmd),
                             "dgknet": pooled(np.ones(len(rows), bool), dgk),
                             "oracle": pooled(np.ones(len(rows), bool), oracle)},
               "features": results, "verdict": verdict}, open(out, "w"), indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
