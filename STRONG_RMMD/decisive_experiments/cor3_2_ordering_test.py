"""Cor 3.2 ordering test (analysis only; reads buttress.json A_ustar).

Theorem 3 crossover: beta_hat_m(u*) = sigma^2_m * p/(N*T), with beta_hat_m(u) = a_m + b_m*u. Folding all
constants into one global C gives x*_pred,m = (C*sigma^2_m - a_m)/b_m; the test asks whether one C
reproduces the ordering of the observed per-machine crossovers (leave-one-machine-out). Pre-registered
verdict: PASS iff LOO Spearman >= 0.7 and all predictions lie in their machine's omega_d range.
"""
import argparse, json
from pathlib import Path
import numpy as np

# observed crossovers (omega_d axis, ablation D_res sign-change) -- AUGD/KSTR are nan (excluded from scoring)
OBS = {"HL2A": 0.205647, "CMOD": 0.248026, "NSTX": 0.700668, "EAST": 0.818367, "D3D": 1.148103}
OMEGA_D_RANGE = (0.0, 3.0)   # the axis the crossovers live on (0.2-1.15); exact per-machine ranges only tighten FAIL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buttress", default=str(Path(__file__).parent / "results" / "buttress.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "ustar_ordering_test.json"))
    a = ap.parse_args()
    A = json.loads(Path(a.buttress).read_text())["A_ustar"]["per_machine"]
    M = [m for m in OBS if m in A and (A[m].get("beta_fit") or {}).get("b")]
    am = {m: A[m]["beta_fit"]["a"] for m in M}
    bm = {m: A[m]["beta_fit"]["b"] for m in M}
    s2 = {m: A[m]["sigma2_T1"] for m in M}

    def xpred(m, C):
        return (C * s2[m] - am[m]) / bm[m]

    grid = np.concatenate([np.zeros(1), np.logspace(-1, 7, 5000)])
    loo_pred, loo_C = {}, {}
    for held in M:                                   # leave-one-machine-out: fit C on the other 4, predict held
        tr = [m for m in M if m != held]
        C = float(min(grid, key=lambda c: np.mean([abs(xpred(m, c) - OBS[m]) for m in tr])))
        loo_C[held] = C; loo_pred[held] = float(xpred(held, C))

    xo = np.array([OBS[m] for m in M]); xp = np.array([loo_pred[m] for m in M])
    rx, ry = np.argsort(np.argsort(xo)), np.argsort(np.argsort(xp))
    rho = float(np.corrcoef(rx, ry)[0, 1]); mae = float(np.mean(np.abs(xp - xo)))

    # DEEPER DIAGNOSTIC: is the failure just the 5-order sigma^2 spread (a normalization issue), or does the
    # crossover ORDERING simply not track sigma^2 / b at all? If Spearman(sigma^2, x_obs) ~ 0, then NO normalization
    # of sigma^2 can rescue the formula -- the crossover is not a function of sigma^2 through it.
    def _sp(u, v):
        return float(np.corrcoef(np.argsort(np.argsort(np.array(u))), np.argsort(np.argsort(np.array(v))))[0, 1])
    diag = {"spearman_sigma2_vs_xobs": _sp([s2[m] for m in M], xo),
            "spearman_b_vs_xobs": _sp([bm[m] for m in M], xo),
            "spearman_a_vs_xobs": _sp([am[m] for m in M], xo),
            "sigma2_orders_of_magnitude": float(np.log10(max(s2.values()) / min(s2.values())))}
    lo, hi = OMEGA_D_RANGE
    in_range = {m: bool(lo <= loo_pred[m] <= hi) for m in M}
    verdict = "PASS" if (rho >= 0.7 and all(in_range.values())) else "FAIL"

    table = [{"machine": m, "a_m": am[m], "b_m": bm[m], "R2": A[m]["beta_fit"].get("R2"),
              "sigma2_T1": s2[m], "x_obs": OBS[m], "x_pred_LOO": loo_pred[m],
              "fitted_C": loo_C[m], "in_range": in_range[m],
              "C_needed_for_inrange(a/sigma2)": am[m] / s2[m]} for m in M]
    out = {"VERDICT": verdict, "LOO_spearman": rho, "LOO_MAE": mae, "diagnostics": diag,
           "criteria": "PASS iff LOO Spearman >= 0.7 AND all predictions in omega_d range",
           "omega_d_range_used": list(OMEGA_D_RANGE), "n_scored": len(M), "table": table,
           "why": ("sigma^2_m spans ~5 orders (CMOD 2.5e-6 .. NSTX 0.25) while a_m ~= 0.94 for all machines, so "
                   "C*sigma^2_m ~= a_m (the in-range condition) needs a per-machine C spanning 5 orders -- no single "
                   "global C works. DEEPER: spearman(sigma^2, x_obs) is also ~0, so this is NOT just a normalization "
                   "bug -- the crossover ordering is not a monotone function of sigma^2 (or b), so no rescaling of "
                   "sigma^2 rescues the formula. The theorem's crossover formula does not predict the observed "
                   "crossovers; keep the mechanism regression + observed ordering as descriptive (Extended Data).")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"VERDICT: {verdict} -- LOO Spearman = {rho:.3f}, MAE = {mae:.3g}")
    print(f"{'machine':6s} {'sigma2':>10s} {'x_obs':>7s} {'x_pred_LOO':>12s} {'in_range':>9s}")
    for r in table:
        print(f"{r['machine']:6s} {r['sigma2_T1']:10.3e} {r['x_obs']:7.3f} {r['x_pred_LOO']:12.3e} {str(r['in_range']):>9s}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
