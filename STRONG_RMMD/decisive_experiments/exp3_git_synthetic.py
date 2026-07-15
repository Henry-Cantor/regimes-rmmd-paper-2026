"""EXP-3 -- GIT validation on systems with known off-diagonal coupling.

Validates the GIT machinery where the coupling is analytically known and tunable, so the theorem can be
refuted. Self-contained (numpy + scipy), deterministic. EXP-3A: a linear Ito SDE dz = A z dt + Sigma dW with
A = S - D_diag - D_res, comparing the closed-form GIT KL, a Monte-Carlo Girsanov estimate, and
diagonal-vs-full surrogate NRMSE. EXP-3B: Lorenz-96 (nonlinear stress test), diagonal-vs-full surrogate NRMSE
vs coupling. Writes results/git_synthetic.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

RESULTS = Path(__file__).resolve().parent / "results"
K = 16                       # latent dim (matches the model)
EPS_GRID = [0.0, 0.02, 0.05, 0.1, 0.2]
SIGMA = 0.5                  # isotropic diffusion  Sigma = SIGMA * I  -> (Sigma Sigma^T)^-1 = SIGMA**-2 * I


# ----------------------------------------------------------------------------- operator construction
def build_operators(seed: int = 0):
    """S skew-symmetric (known), D_diag > 0 diagonal (known), D_res_unit symmetric OFF-DIAGONAL (known).
    D_res(eps) = eps * D_res_unit. A(eps) = S - D_diag - eps*D_res_unit is kept Hurwitz by a strong D_diag."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((K, K))
    S = M - M.T                                   # skew-symmetric
    D_diag = np.diag(1.0 + 0.5 * rng.random(K))   # SPD diagonal, entries in [1,1.5]
    R = rng.standard_normal((K, K)); R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)                       # symmetric, zero-diagonal
    R = R / np.linalg.norm(R)                      # unit Frobenius norm -> eps is the coupling magnitude
    return S, D_diag, R


def is_hurwitz(A):
    return bool(np.all(np.linalg.eigvals(A).real < 0))


# ----------------------------------------------------------------------------- EXP-3A: linear SDE + GIT KL
def exp3a():
    S, D_diag, R = build_operators(seed=0)
    Sinv2 = SIGMA ** -2                            # scalar, since Sigma Sigma^T = SIGMA^2 * I
    Q = (SIGMA ** 2) * np.eye(K)                    # Sigma Sigma^T
    T_grid = [10.0, 25.0, 50.0, 100.0]             # PHYSICAL TIME (not step counts) — the bug in v1 conflated them
    dt = 0.01
    n_paths = 1500

    out = {"eps_grid": EPS_GRID, "T_grid": T_grid, "dt": dt, "n_paths": n_paths, "K": K, "sigma": SIGMA,
           "discretization": "EXACT (M=expm(A dt), process-noise cov Q_dt = P - M P M^T) -> no Euler bias, "
                             "sampled AT stationarity so P_emp->P (sampling noise only)",
           "per_eps": {}}
    kl_rate_closed_by_eps = {}
    for eps in EPS_GRID:
        D_res = eps * R
        A = S - D_diag - D_res
        hur = is_hurwitz(A)
        # --- closed form: stationary cov P via Lyapunov, then dKL/dt = 0.5 tr(D_res^T Q^-1 D_res P) ---
        P = solve_continuous_lyapunov(A, -Q)       # A P + P A^T + Q = 0
        kl_rate_closed = 0.5 * Sinv2 * float(np.trace(D_res.T @ D_res @ P))
        kl_rate_closed_by_eps[eps] = float(kl_rate_closed)
        kl_closed_T = {T: float(kl_rate_closed * T) for T in T_grid}

        # --- EXACT discretization of the linear SDE (no Euler bias): z_{t+dt} = M z_t + noise ---
        # M = expm(A dt); exact process-noise cov Q_dt = P - M P M^T (so the chain's stationary cov IS P).
        M = expm(A * dt)
        Q_dt = P - M @ P @ M.T; Q_dt = 0.5 * (Q_dt + Q_dt.T)
        w, V = np.linalg.eigh(Q_dt); w = np.clip(w, 0.0, None); L_noise = V @ np.diag(np.sqrt(w))
        L_P = np.linalg.cholesky(P + 1e-10 * np.eye(K))
        rng = np.random.default_rng(1234)
        z = rng.standard_normal((n_paths, K)) @ L_P.T                  # start AT stationarity
        n_steps = int(round(max(T_grid) / dt))
        record_steps = {int(round(T / dt)): T for T in T_grid}
        cum = np.zeros(n_paths); kl_mc_T = {}
        P_emp_accum = np.zeros((K, K)); n_acc = 0
        for step in range(1, n_steps + 1):
            integrand = ((z @ D_res.T) ** 2).sum(axis=1) * Sinv2       # (D_res z)^T Q^-1 (D_res z)
            cum = cum + 0.5 * integrand * dt
            z = z @ M.T + rng.standard_normal((n_paths, K)) @ L_noise.T
            P_emp_accum += (z.T @ z) / n_paths; n_acc += 1
            if step in record_steps:
                kl_mc_T[record_steps[step]] = float(cum.mean())
        P_emp = P_emp_accum / n_acc
        # --- linearity fit KL_mc vs T, slope, ratio to closed-form rate ---
        Ts = np.array(T_grid, float); kls = np.array([kl_mc_T[T] for T in T_grid], float)
        if kls.max() > 1e-30:
            A_fit = np.vstack([Ts, np.ones_like(Ts)]).T
            coef, *_ = np.linalg.lstsq(A_fit, kls, rcond=None)
            slope_mc = float(coef[0]); pred = A_fit @ coef
            ss_res = float(((kls - pred) ** 2).sum()); ss_tot = float(((kls - kls.mean()) ** 2).sum())
            r2 = float(1 - ss_res / (ss_tot + 1e-30))
        else:
            slope_mc, r2 = 0.0, float("nan")
        ratio = float(slope_mc / kl_rate_closed) if abs(kl_rate_closed) > 1e-12 else (1.0 if slope_mc < 1e-9 else float("inf"))

        # --- surrogate: predicted-mean divergence from OMITTING D_res (isolates D_res, matches GIT baseline) ---
        # full op = expm(A t); D_res-removed op = expm((A+D_res) t) = expm((S-D_diag) t). Short horizons where the
        # contracting mean is non-trivial. gap = NRMSE(no-D_res predictor vs true mean) grows with eps.
        surro = _surrogate_dres_divergence(A, D_res, P, seed=7)
        out["per_eps"][str(eps)] = {
            "hurwitz": hur,
            "kl_rate_closed_form": float(kl_rate_closed),
            "kl_closed_by_T": kl_closed_T,
            "kl_mc_by_T": kl_mc_T,
            "mc_linear_fit_R2": r2,
            "mc_slope": slope_mc,
            "slope_ratio_mc_over_closed": ratio,
            "P_lyap_vs_emp_relerr": float(np.linalg.norm(P - P_emp) / (np.linalg.norm(P) + 1e-12)),
            "dres_omission_nrmse_by_t": surro,          # {t: NRMSE of omitting D_res}
            "dres_omission_nrmse_t5": float(surro.get("5.0", np.nan)),
        }

    # --- eps^2 scaling of the closed-form rate (KL ∝ ||D_res||^2 = eps^2) ---
    nz = [(e, kl_rate_closed_by_eps[e]) for e in EPS_GRID if e > 0]
    eps_arr = np.array([e for e, _ in nz]); rate_arr = np.array([r for _, r in nz])
    # fit log(rate) = a + b log(eps); b should be ~2
    lx = np.log(eps_arr); ly = np.log(rate_arr + 1e-30)
    b = float(np.polyfit(lx, ly, 1)[0])
    out["eps2_scaling_loglog_slope"] = b   # ~2.0 if KL ∝ eps^2

    # --- VERDICT (spec EXP-3A) ---
    r2s = [out["per_eps"][str(e)]["mc_linear_fit_R2"] for e in EPS_GRID if e > 0]
    ratios = [out["per_eps"][str(e)]["slope_ratio_mc_over_closed"] for e in EPS_GRID if e > 0]
    omit = [out["per_eps"][str(e)]["dres_omission_nrmse_t5"] for e in EPS_GRID]
    lin_ok = all(r >= 0.98 for r in r2s if np.isfinite(r))
    slope_ok = all(0.9 <= r <= 1.1 for r in ratios if np.isfinite(r))
    eps2_ok = 1.7 <= b <= 2.3
    omit_grows = omit[-1] > omit[0] + 1e-6            # omitting D_res costs more prediction error as eps grows
    supported = bool(lin_ok and slope_ok and eps2_ok and omit_grows)
    out["VERDICT_3A"] = {
        "prediction_if_true": "MC KL T-linear (R2>0.98), slope matches closed form (ratio 0.9-1.1), rate ~eps^2 (loglog slope ~2), omitting-D_res prediction error grows with eps.",
        "refuted_if": "KL not T-linear, OR slope mismatch (ratio outside 0.9-1.1), OR D_res omission is costless at large eps.",
        "mc_linear_R2_all>=0.98": lin_ok, "slope_ratio_all_0.9_1.1": slope_ok,
        "eps2_loglog_slope": b, "eps2_ok_1.7_2.3": eps2_ok, "dres_omission_grows_with_eps": omit_grows,
        "dres_omission_nrmse_t5_by_eps": {str(e): g for e, g in zip(EPS_GRID, omit)},
        "VERDICT": "SUPPORTED" if supported else "REFUTED",
    }
    return out


def _surrogate_dres_divergence(A, D_res, P, seed):
    """Predicted-MEAN divergence from OMITTING D_res (isolates D_res exactly; matches the GIT baseline that keeps
    S and D_diag but drops D_res). full op = expm(A t); no-D_res op = expm((A+D_res) t). Over z0 ~ N(0,P),
    NRMSE(t) = mean||expm(A t)z0 - expm((A+D_res)t)z0|| / mean||expm(A t)z0||. Short t (mean not yet decayed)."""
    rng = np.random.default_rng(seed)
    L_P = np.linalg.cholesky(P + 1e-10 * np.eye(K))
    z0 = rng.standard_normal((2000, K)) @ L_P.T
    A_nod = A + D_res                                    # S - D_diag (D_res removed)
    errs = {}
    for t in (0.5, 1.0, 2.0, 5.0):
        full = z0 @ expm(A * t).T
        nod = z0 @ expm(A_nod * t).T
        num = np.linalg.norm(full - nod, axis=1).mean()
        den = np.linalg.norm(full, axis=1).mean() + 1e-12
        errs[str(t)] = float(num / den)
    return errs


# ----------------------------------------------------------------------------- EXP-3B: Lorenz-96
def _l96_step(x, F, dt):
    def deriv(x):
        return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F
    k1 = deriv(x); k2 = deriv(x + 0.5 * dt * k1); k3 = deriv(x + 0.5 * dt * k2); k4 = deriv(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _l96_features(traj):
    """Per-coordinate LOCAL-STENCIL features for a spatially-homogeneous rule (fit once, applied to all i).
    Returns (X_diag, X_coupled, Y) pooled over (coordinate, time). xm2..xp1 = x_{i-2..i+1}."""
    xm2 = np.roll(traj, 2, axis=1); xm1 = np.roll(traj, 1, axis=1)
    x0 = traj; xp1 = np.roll(traj, -1, axis=1)
    y = np.roll(traj, -1, axis=0)                         # next-time state (same coordinate)
    def flat(a): return a[:-1].reshape(-1)                # drop last row (no next), pool (t,i)
    ones = np.ones_like(x0)
    # DIAGONAL: coordinate i's next value from ONLY its own value (no cross-coordinate coupling)
    Xd = np.stack([flat(ones), flat(x0), flat(x0 ** 2)], axis=1)
    # COUPLED: local stencil incl. the cross products that span the Lorenz-96 quadratic coupling
    Xc = np.stack([flat(ones), flat(xm2), flat(xm1), flat(x0), flat(xp1),
                   flat(xm1 * xp1), flat(xm2 * xm1), flat(xm1 * x0), flat(x0 * xp1)], axis=1)
    return Xd, Xc, flat(y)


def exp3b():
    N, dt = 40, 0.05
    F_grid = [4.0, 8.0, 16.0]
    horizons = [1, 5, 10, 20, 50]
    out = {"N": N, "dt": dt, "F_grid": F_grid, "horizons": horizons,
           "surrogate": "per-coordinate homogeneous rule; DIAGONAL=[1,x_i,x_i^2] (no coupling) vs "
                        "COUPLED=local stencil incl. cross-products (can represent off-diagonal coupling). "
                        "Linear Koopman was dropped: it cannot represent Lorenz-96 at any F, confounding the test.",
           "per_F": {}}
    adv_h1, offdiags = [], []
    for F in F_grid:
        rng = np.random.default_rng(int(F))
        x = F * np.ones(N) + 0.01 * rng.standard_normal(N)
        for _ in range(2000):
            x = _l96_step(x, F, dt)
        n_train, n_roll = 6000, 60
        traj = [x.copy()]
        for _ in range(n_train + n_roll + 10):
            x = _l96_step(x, F, dt); traj.append(x.copy())
        traj = np.array(traj); sc = traj[:n_train].std() + 1e-12
        trn = (traj[:n_train] - traj[:n_train].mean()) / sc
        Xd, Xc, y = _l96_features(trn)
        wd = np.linalg.lstsq(Xd, y, rcond=None)[0]        # diagonal coeffs (3,)
        wc = np.linalg.lstsq(Xc, y, rcond=None)[0]        # coupled coeffs (9,)
        # coupling load: fraction of coupled-model prediction variance from the cross terms
        cross_cols = Xc[:, 5:9] @ wc[5:9]
        coupling_load = float(cross_cols.std() / ((Xc @ wc).std() + 1e-12))

        def step_model(state, w, coupled):
            xm2 = np.roll(state, 2); xm1 = np.roll(state, 1); xp1 = np.roll(state, -1); x0 = state
            if coupled:
                feat = np.stack([np.ones_like(x0), xm2, xm1, x0, xp1,
                                 xm1 * xp1, xm2 * xm1, xm1 * x0, x0 * xp1], axis=1)
            else:
                feat = np.stack([np.ones_like(x0), x0, x0 ** 2], axis=1)
            return feat @ w
        start = n_train + 2
        true = (traj[start:start + max(horizons) + 1] - traj[:n_train].mean()) / sc
        def rollout(w, coupled):
            z = true[0].copy(); errs = {}; var = true.var() + 1e-12
            for t in range(1, max(horizons) + 1):
                z = step_model(z, w, coupled)
                z = np.clip(np.nan_to_num(z, nan=1e3, posinf=1e3, neginf=-1e3), -1e3, 1e3)   # guard chaotic rollout blowup
                if t in horizons:
                    e = float(np.sqrt(((z - true[t]) ** 2).mean() / var))
                    errs[str(t)] = min(e, 1e3) if np.isfinite(e) else 1e3
            return errs
        ed, ec = rollout(wd, False), rollout(wc, True)
        adv = {str(h): float(ed[str(h)] - ec[str(h)]) for h in horizons}   # diag - coupled (positive = coupling wins)
        out["per_F"][str(F)] = {"nrmse_diagonal": ed, "nrmse_coupled": ec,
                                "advantage_diag_minus_coupled": adv, "coupling_load_frac": coupling_load,
                                "note": "T>=10 rollouts diverge (chaos amplifies any fitted-map error) — judge at T=1/T=5."}
        adv_h1.append(adv["1"]); offdiags.append(coupling_load)
    # decisive test: coupling-capable beats diagonal on a KNOWN-coupled system, at the STABLE horizon (T=1), every F
    coupled_wins_all = all(out["per_F"][str(f)]["advantage_diag_minus_coupled"]["1"] > 0.05 for f in F_grid)
    corr_adv_coupling = float(np.corrcoef(offdiags, adv_h1)[0, 1]) if len(F_grid) > 2 and np.std(offdiags) > 1e-9 else float("nan")
    out["VERDICT_3B"] = {
        "prediction_if_true": "off-diagonal-capable (coupled) surrogate beats the diagonal one on Lorenz-96 (a system with KNOWN coupling), at every F (judged at the stable horizon T=1).",
        "refuted_if": "coupled gives no advantage over diagonal at any F -> off-diagonal capacity buys nothing even where coupling provably exists.",
        "advantage_diag_minus_coupled_T1_by_F": {str(f): a for f, a in zip(F_grid, adv_h1)},
        "coupling_load_frac_by_F": {str(f): o for f, o in zip(F_grid, offdiags)},
        "coupled_beats_diagonal_all_F": coupled_wins_all,
        "note_F_monotonicity": "F sets forcing amplitude/chaoticity, not coupling FRACTION; do not expect the "
                               "advantage to grow monotonically with F (see coupling_load_frac_by_F).",
        "VERDICT": "SUPPORTED" if coupled_wins_all else "REFUTED",
    }
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"EXP3A_linear_SDE": exp3a(), "EXP3B_lorenz96": exp3b()}
    (RESULTS / "git_synthetic.json").write_text(json.dumps(report, indent=1, default=float))
    print("=" * 78)
    print("EXP-3A (linear SDE, GIT KL):")
    print(json.dumps(report["EXP3A_linear_SDE"]["VERDICT_3A"], indent=1, default=float))
    print("-" * 78)
    print("EXP-3B (Lorenz-96):")
    print(json.dumps(report["EXP3B_lorenz96"]["VERDICT_3B"], indent=1, default=float))
    print("=" * 78)
    print("wrote", RESULTS / "git_synthetic.json")


if __name__ == "__main__":
    main()
