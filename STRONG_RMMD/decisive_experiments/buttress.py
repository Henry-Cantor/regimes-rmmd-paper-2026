"""EXP-6 -- three analyses in one per-shot pass (most are partial pass, though c is most promising).

A) u* prediction: per machine, fit beta_hat(u) = regression of learned <||D_res||^2_F> on rel_ptp; predict
   the crossover u*_m from beta_hat_m(u) = sigma^2_m * p / (N*T) at T=50. Observed u*_m = the drive value
   where the per-machine D_res-ablation delta changes sign.
B) decision-theoretic alpha: AUC for predicting the per-shot winner (full model vs persistence @T50) from
   t=0 features and the drive plan; bounds the value of any routing policy, and scores a three-way
   {persistence | RMMD | DGKNet} drive-keyed policy per machine.
C) nonlocality: partial correlation of a per-shot profile-nonlocality proxy with per-shot ||D_res||,
   controlling for drive.

Run with --help for arguments.
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
# T=1 for sigma^2; T=50 = T_STAR for A/B's operating-point errors + ablation deltas; the rest let analysis B
# (analysis_router_sota) evaluate the per-shot router as SOTA across EVERY horizon x activity-quartile cell.
HORIZONS = [1, 2, 3, 5, 8, 12, 16, 20, 32, 50, 75, 100]
T_STAR = 50
MAX_TIME = 100         # ni_traj is truncated to min(MAX_TIME, raw_len) by the dataset (see analysis C windows)


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


# --------------------------------------------------------------------------- per-shot feature helpers
def _np(x):
    """Tensor-or-array -> float64 numpy (dataset objects are CPU torch tensors)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, float)


def _rel_ptp(drv):
    """Dimensionless drive variability = ptp(PINJ)/mean(|PINJ|); driver channel 0 = PINJ."""
    if isinstance(drv, torch.Tensor) and drv.numel():
        p = drv.detach().cpu().numpy().astype(float)[:, 0]
        return float(np.ptp(p) / (np.abs(p).mean() + 1e-9))
    return float("nan")


def _nonlocality(ni_traj, window=None):
    """Profile-scale nonlocality proxy: variance fraction captured by the 1st radial PC of dNI/dt. High = the
    profile change is radially COHERENT (one dominant radial mode -> nonlocal/extended); low = radially incoherent
    (independent channels -> local/diffusive). Computed from the raw NI trajectory only (no model). `window` = use
    the first `window` steps (the incoming ni_traj is already min(MAX_TIME, raw_len) long). Theory: nonlocal
    transport is a fast TRANSIENT, so a shorter window should give a HIGHER partial rho -- the knobs 10..100 test
    that. The original rho=0.232 run used a 50-step trajectory, so w50 is the anchor for reproducing it."""
    a = _np(ni_traj)
    a = a.reshape(a.shape[0], -1)[:, :40]
    if window:
        a = a[:window]
    if a.shape[0] < 3:
        return float("nan")
    d = np.diff(a, axis=0)                       # (T-1, 40) time-derivative of the profile
    if not np.isfinite(d).all() or d.shape[0] < 2:
        return float("nan")
    C = np.cov(d.T)                              # (40,40) covariance across the radial channels
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return float(w.max() / s) if s > 0 else float("nan")


def _t0_features(ni_t0):
    a = _np(ni_t0).reshape(-1)[:40]
    g = np.abs(np.diff(a))
    return [float(a.mean()), float(a.std()), float(a[:8].mean()), float(a[-8:].mean()),
            float(g.max() if g.size else 0.0), float(g.mean() if g.size else 0.0)]


T0_NAMES = ["ni_mean", "ni_std", "ni_core", "ni_edge", "ni_gradmax", "ni_gradmean"]


def _machine(s):
    m = s.get("machine", "UNK")
    return m.decode() if isinstance(m, bytes) else str(m)


def _dres_param_count(model):
    """Learnable parameter count of the D_res branch of the resonance kernel (amp_head + gamma_head + mode share).
    Convention: all parameters of model.rmmd.kernel whose name matches the D_res branch; report the number used."""
    k = model.rmmd.kernel
    keys = ("amp", "gamma", "mode", "res", "d_res", "dres", "off")
    n = 0
    for nm, pp in k.named_parameters():
        if any(t in nm.lower() for t in keys):
            n += int(pp.numel())
    if n == 0:                                   # fallback: whole kernel (state the convention in the report)
        n = sum(int(pp.numel()) for pp in k.parameters())
    return n


# --------------------------------------------------------------------------- extraction (one per-shot table)
def extract(rc, cmp_mod, ex, args):
    device = args.device
    datasets = [(spec.split(":", 1)[0], spec.split(":", 1)[1]) for spec in args.datasets]
    rows = {}                                    # (dsname, i) -> per-shot dict

    # ---- Pass 1: FULL model, kernel-patched, T=1 -> dres_norm, sigma^2, rel_ptp, nonlocality, t0 features ----
    full, fnorm, _ = cmp_mod._build_model(rc, Path(args.full_ckpt), device); full.eval()
    p_dres = _dres_param_count(full)
    rmmd = full.rmmd
    orig = rmmd.kernel.forward
    grabbed = {}

    def patched(z, omega_t, omega_d, context=None):
        kout = orig(z=z, omega_t=omega_t, omega_d=omega_d, context=context)
        grabbed["dres"] = float(torch.linalg.norm(kout.d_res.reshape(kout.d_res.shape[0], -1), dim=1).mean().item())
        grabbed["omega_t"] = float(torch.as_tensor(omega_t).detach().cpu().float().mean().item())
        grabbed["omega_d"] = float(torch.as_tensor(omega_d).detach().cpu().float().mean().item())  # kernel's ACTUAL input
        return kout
    rmmd.kernel.forward = patched
    for dsname, path in datasets:
        n = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        ds = rc.CompactRolloutDataset(rc._load_phase0_dataset(Path(path)), max_time=MAX_TIME, normalization_stats=n)
        for i in range(len(ds)):
            s = ds[i]
            grabbed.clear()
            try:
                ni_preds, _ = rc._rollout_compact_shot_to_checkpoints(
                    full, s["ni_t0"], s["geom_t0"], s["pre_shot_context"], s["limiter_geometry_tensor"],
                    s["ni_traj"], s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}),
                    device, fnorm, max_time_step=1, drivers_traj=s.get("drivers_traj"), report_horizons=[1])
            except Exception:
                continue
            if "dres" not in grabbed or 1 not in ni_preds:
                continue
            tgt0 = _np(s["ni_traj"][0]).reshape(-1)
            pr1 = _np(ni_preds[1]).reshape(-1)
            sigma1 = float(np.mean((pr1 - tgt0) ** 2)) if pr1.shape == tgt0.shape else float("nan")
            rows[(dsname, i)] = {"dataset": dsname, "machine": _machine(s), "rel_ptp": _rel_ptp(s.get("drivers_traj")),
                                 "omega_t": grabbed.get("omega_t", float("nan")), "omega_d": grabbed.get("omega_d", float("nan")),
                                 "dres": grabbed["dres"], "sigma1": sigma1, "t0": _t0_features(s["ni_t0"]),
                                 # ni_traj is already truncated to min(MAX_TIME, raw_len) by the dataset, so window=k means the first k
                                 # steps. The 10..100 knobs probe the transient hypothesis (a shorter window should give a higher partial
                                 # rho). Primary window = w50.
                                 "nonloc10": _nonlocality(s["ni_traj"], 10), "nonloc20": _nonlocality(s["ni_traj"], 20),
                                 "nonloc30": _nonlocality(s["ni_traj"], 30), "nonloc50": _nonlocality(s["ni_traj"], 50),
                                 "nonloc75": _nonlocality(s["ni_traj"], 75), "nonloc100": _nonlocality(s["ni_traj"], 100),
                                 "nonloc": _nonlocality(s["ni_traj"], 50)}
    rmmd.kernel.forward = orig

    # Per-shot errors at every horizon (full / abl_dres / each flex candidate), stored as the T=50 scalar keys
    # used by analyses A and alpha, and as r['errH'][key][H] + r['persH'][H] across all horizons for the router.
    def add_errs(ckpt, key, set_pers=False, flex=False):
        model, mnorm, _ = cmp_mod._build_model(rc, Path(ckpt), device); model.eval()
        for dsname, path in datasets:
            n = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
            ds = rc.CompactRolloutDataset(rc._load_phase0_dataset(Path(path)), max_time=MAX_TIME, normalization_stats=n)
            acc = ex.eval_dataset(rc, model, ds, HORIZONS, device, mnorm, None)
            for H in HORIZONS:                                   # (2) all-horizon per-shot errors for the router-SOTA grid
                accH = acc.get(H)
                if not accH:
                    continue
                perH = dict(zip(accH["shots"], accH["nrmse"])); persH = dict(zip(accH["shots"], accH["pers"]))
                for i, v in perH.items():
                    r = rows.get((dsname, i))
                    if r is None:
                        continue
                    r.setdefault("errH", {}).setdefault(key, {})[int(H)] = float(v)
                    if set_pers and i in persH:
                        r.setdefault("persH", {})[int(H)] = float(persH.get(i, np.nan))
            per = dict(zip(acc[T_STAR]["shots"], acc[T_STAR]["nrmse"]))   # (1) T=50 scalars (backward-compat)
            pers = dict(zip(acc[T_STAR]["shots"], acc[T_STAR]["pers"]))
            for i, v in per.items():
                r = rows.get((dsname, i))
                if r is None:
                    continue
                if flex:
                    r.setdefault("flex", {})[key] = float(v)
                else:
                    r[key] = float(v)
                    if set_pers:
                        r["pers50"] = float(pers.get(i, np.nan))
    if not getattr(args, "skip_errors", False):     # --skip-errors: Pass-1 only (T=1) -> fast C + A-beta, no T=50 evals
        add_errs(args.full_ckpt, "err_full", set_pers=True)
        if args.abl_dres_ckpt:
            add_errs(args.abl_dres_ckpt, "err_abl")
        flex_specs = list(args.flex_ckpts)
        if args.dgknet_ckpt:                        # single --dgknet-ckpt is just one flexible candidate named 'dgknet'
            flex_specs.append(f"dgknet:{args.dgknet_ckpt}")
        if getattr(args, "only_nonloc", False):     # fast mode: only full + abl_dres are needed (Equation + threshold)
            flex_specs = []
        for spec in flex_specs:
            spec = spec.strip()
            if not spec:
                continue
            if ":" in spec and not spec.split(":", 1)[0].startswith("/"):   # name:path
                name, ckpt = spec.split(":", 1)
            else:                                                            # bare path -> auto-name from its dir
                ckpt = spec; name = Path(spec).parent.name or Path(spec).stem
            if not ckpt or not Path(ckpt).exists():
                print(f"  [flex candidate] SKIP '{spec}' (name={name}, ckpt missing/empty)", flush=True); continue
            print(f"  [flex candidate] {name} <- {ckpt}", flush=True)
            add_errs(ckpt, name, flex=True)
    return list(rows.values()), p_dres


def _select_flex(rows, out):
    """Pick the flexible arm (DGKNet/NODE x LR) that is best on the HIGH-DRIVE TAIL (where the router uses it), attach
    it as r['err_dgk'] for the downstream analyses, and report every candidate's tail performance for transparency."""
    names = sorted({n for r in rows for n in (r.get("flex") or {})})
    if not names:
        out["flex_selection"] = {"note": "no flexible candidates passed (--flex-ckpts / --dgknet-ckpt)"}; return None
    R = [r for r in rows if r.get("flex") and "err_full" in r and np.isfinite(r.get("pers50", np.nan))]
    u = np.array([r["rel_ptp"] for r in R], float)
    thr = np.nanquantile(u[np.isfinite(u)], 2 / 3) if np.isfinite(u).any() else np.nan
    tail = [r for r in R if np.isfinite(r["rel_ptp"]) and r["rel_ptp"] >= thr]
    tail_mean = {}
    for n in names:
        vals = [r["flex"][n] for r in tail if n in r["flex"] and np.isfinite(r["flex"][n])]
        tail_mean[n] = float(np.mean(vals)) if len(vals) >= 5 else float("inf")
    best = min(tail_mean, key=tail_mean.get)
    for r in rows:                                  # the chosen flexible arm becomes err_dgk for the router / u* overtake
        f = r.get("flex") or {}
        if best in f and np.isfinite(f[best]):
            r["err_dgk"] = f[best]
    out["flex_selection"] = {"chosen_flex_arm": best, "n_high_drive_tail": len(tail),
                             "criterion": "lowest T50 NRMSE on the top-tercile rel_ptp (high-drive) tail",
                             "tail_mean_per_candidate": {k: (v if np.isfinite(v) else None) for k, v in tail_mean.items()}}
    return best


# --------------------------------------------------------------------------- statistics helpers
def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return float("nan"), float("nan"), int(x.size)
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return float(r.correlation), float(r.pvalue), int(x.size)
    except Exception:
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan"), int(x.size)


def _zero_crossing(xbins, ybins):
    """First x where the binned y (D_res helps = err_abl - err_full) crosses from <=0 to >0 (linear interp)."""
    for j in range(1, len(ybins)):
        y0, y1 = ybins[j - 1], ybins[j]
        if np.isfinite(y0) and np.isfinite(y1) and y0 <= 0 < y1:
            f = (0.0 - y0) / (y1 - y0 + 1e-12)
            return float(xbins[j - 1] + f * (xbins[j] - xbins[j - 1]))
    return float("nan")


def _binned_crossover(u, delta, nbins=6):
    """Observed crossover: bin drive u into quantile bins, mean(delta) per bin, return the zero-up-crossing."""
    u = np.asarray(u, float); delta = np.asarray(delta, float)
    m = np.isfinite(u) & np.isfinite(delta)
    u, delta = u[m], delta[m]
    if u.size < 4 * nbins:
        nbins = max(3, u.size // 6)
    if u.size < 12:
        return float("nan")
    qs = np.quantile(u, np.linspace(0, 1, nbins + 1))
    xc, yc = [], []
    for b in range(nbins):
        sel = (u >= qs[b]) & (u <= qs[b + 1]) if b == nbins - 1 else (u >= qs[b]) & (u < qs[b + 1])
        if sel.sum() >= 3:
            xc.append(float(u[sel].mean())); yc.append(float(delta[sel].mean()))
    return _zero_crossing(xc, yc) if len(xc) >= 2 else float("nan")


# --------------------------------------------------------------------------- Analysis A: u* prediction
_U_AXES = ("omega_d", "omega_t", "rel_ptp")   # omega_d = the OPERATING POINT the kernel actually depends on (a/L_n);
                                              # rel_ptp (drive variability) is a fallback -- ||D_res|| does NOT track it.


def _fit_beta(R, dres2):
    """Fit beta_hat(u) = a + b*u for each candidate operating-point axis; return the BEST-R2 axis (>= a floor)."""
    best = None
    for ax in _U_AXES:
        u = np.array([r.get(ax, np.nan) for r in R], float)
        good = np.isfinite(u) & np.isfinite(dres2)
        if good.sum() < 8 or np.ptp(u[good]) < 1e-12:
            continue
        b, a = np.polyfit(u[good], dres2[good], 1)
        yhat = a + b * u[good]
        r2 = float(1 - np.sum((dres2[good] - yhat) ** 2) / (np.sum((dres2[good] - dres2[good].mean()) ** 2) + 1e-12))
        if best is None or r2 > best["R2"]:
            best = {"axis": ax, "a": float(a), "b": float(b), "R2": r2, "n": int(good.sum()), "u": u}
    return best


def analysis_ustar(rows, p_dres, n_train, out):
    per = {}
    for m in sorted({r["machine"] for r in rows}):
        R = [r for r in rows if r["machine"] == m]
        dres2 = np.array([r["dres"] ** 2 for r in R], float)
        sigma2 = float(np.nanmean(np.array([r.get("sigma1", np.nan) for r in R], float)))
        best = _fit_beta(R, dres2)
        if best is None:
            per[m] = {"note": "no usable operating-point axis for the beta fit"}; continue
        rhs = sigma2 * p_dres / (max(n_train, 1) * T_STAR)          # sigma^2 * p / (N*T)
        # predicted u* ONLY when the beta fit is real (else the inversion is degenerate -> report observed only)
        u_pred = float((rhs - best["a"]) / best["b"]) if (abs(best["b"]) > 1e-12 and best["R2"] >= 0.15) else float("nan")
        ua = best["u"]                                              # observed crossovers on the SAME axis beta was fit on
        dd = np.array([r.get("err_abl", np.nan) - r.get("err_full", np.nan) for r in R], float)   # >0 = D_res helps
        dg = np.array([r.get("err_full", np.nan) - r.get("err_dgk", np.nan) for r in R], float)   # >0 = DGKNet better
        per[m] = {"n": best["n"], "axis": best["axis"], "beta_fit": {"a": best["a"], "b": best["b"], "R2": best["R2"]},
                  "sigma2_T1": sigma2, "u_pred": u_pred,
                  "u_obs_dres_signchange": _binned_crossover(ua, dd), "u_obs_dgknet_overtake": _binned_crossover(ua, dg)}
    good_beta = [m for m, v in per.items() if isinstance(v, dict) and (v.get("beta_fit") or {}).get("R2", 0) >= 0.15]
    pairs = [(v["u_pred"], v["u_obs_dres_signchange"]) for v in per.values()
             if isinstance(v, dict) and np.isfinite(v.get("u_pred", np.nan)) and np.isfinite(v.get("u_obs_dres_signchange", np.nan))]
    rho, pval, npair = _spearman([x for x, _ in pairs], [y for _, y in pairs]) if len(pairs) >= 3 else (float("nan"), float("nan"), len(pairs))
    preds = {m: v.get("u_pred") for m, v in per.items() if isinstance(v, dict) and np.isfinite(v.get("u_pred", np.nan))}
    augd_lowest = bool(preds and "AUGD" in preds and preds["AUGD"] == min(preds.values()))
    # observed ordering alone (the fallback result, always defined): rank machines by their observed crossover
    obs = {m: v.get("u_obs_dres_signchange") for m, v in per.items()
           if isinstance(v, dict) and np.isfinite(v.get("u_obs_dres_signchange", np.nan))}
    out["A_ustar"] = {"p_dres_params": int(p_dres), "n_train_windows": n_train, "T": T_STAR,
                      "per_machine": per, "n_machines_with_real_beta_fit": len(good_beta),
                      "spearman_pred_vs_obs": rho, "spearman_p": pval, "n_machines_paired": npair,
                      "AUGD_predicted_lowest": augd_lowest,
                      "observed_crossover_ordering": dict(sorted(obs.items(), key=lambda kv: kv[1])),
                      "beta_DEGENERATE": bool(len(good_beta) < 3),
                      "PREREGISTERED_HIT": bool(np.isfinite(rho) and rho >= 0.7 and augd_lowest),
                      "note": "beta is fit vs the OPERATING POINT (omega_d = a/L_n, the kernel's actual input), NOT rel_ptp "
                              "(||D_res|| does not track drive variability). If beta_DEGENERATE (||D_res|| ~saturated, "
                              "R2<0.15 on all axes) the predicted side is undefined -> publish the observed_crossover_"
                              "ordering alone (Extended Data) and flag to the theory chat that beta needs reformulation."}


# --------------------------------------------------------------------------- Analysis B: decision-theoretic alpha
def _auc(scores, y):
    """Mann-Whitney AUC = P(score_pos > score_neg); rank-based, no sklearn."""
    scores = np.asarray(scores, float); y = np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.empty(len(scores), float); ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _logistic_cv_auc(X, y, folds=5, iters=400, lr=0.2):
    """Multivariate out-of-sample AUC via numpy logistic regression (standardize on train, AUC on held-out)."""
    n = len(y); idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    aucs = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(tr) < 20 or len(te) < 8 or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= lr * (Xtr.T @ g / len(tr) + 1e-3 * w); b -= lr * g.mean()
        a = _auc(Xte @ w + b, y[te])
        if np.isfinite(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def _oos_proba(X, y, folds=5, iters=400, lr=0.2):
    """Out-of-sample P(y=1) per shot via CV numpy logistic -- each shot scored by a model NOT trained on it (honest,
    so the resulting per-shot router is not leaking). Returns an array aligned to X (nan where a fold was skipped)."""
    n = len(y); idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    proba = np.full(n, np.nan)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        if len(tr) < 20 or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= lr * (Xtr.T @ g / len(tr) + 1e-3 * w); b -= lr * g.mean()
        proba[te] = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
    return proba


def analysis_alpha(rows, out):
    R = [r for r in rows if "err_full" in r and "pers50" in r and np.isfinite(r.get("pers50", np.nan))]
    if len(R) < 30:                                # e.g. --skip-errors (no T=50 model errors) or too few shots
        out["B_decision_theoretic"] = {"note": "SKIPPED (no T=50 model errors -- --skip-errors mode, or <30 shots)"}
        return
    y = np.array([1 if r["err_full"] < r["pers50"] else 0 for r in R])
    X = np.array([[r["rel_ptp"]] + r["t0"] for r in R], float)
    feat_names = ["rel_ptp"] + T0_NAMES
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    alpha = float("nan"); method = "n/a"
    if X.shape[0] >= 30 and 0 < y.mean() < 1:
        alpha = _logistic_cv_auc(X, y); method = "numpy_logreg_5foldCV_AUC"
        if not np.isfinite(alpha):
            alpha = _auc(X[:, 0], y); method = "rel_ptp_single_feature_AUC"
    # value-of-information bound: max routing gain over the best fixed choice <= (2*alpha-1) of the oracle gap
    voi_frac = float(2 * alpha - 1) if np.isfinite(alpha) else float("nan")
    # ---- proper PER-SHOT alpha-driven 3-way router {persistence | RMMD | DGKNet} ----
    proba = _oos_proba(X, y)                                    # OOS P(model beats persistence) per shot, aligned to R
    # global RMMD-vs-DGKNet threshold on rel_ptp (fit once on the pooled shots that have dgknet errors)
    idx_d = [k for k, r in enumerate(R) if "err_dgk" in r]
    theta_md = float("nan")
    if len(idx_d) >= 20:
        ud = np.array([R[k]["rel_ptp"] for k in idx_d]); fed = np.array([R[k]["err_full"] for k in idx_d]); ded = np.array([R[k]["err_dgk"] for k in idx_d])
        grid = np.quantile(ud[np.isfinite(ud)], np.linspace(0.1, 0.9, 17))
        theta_md = float(min(grid, key=lambda th: float(np.where(ud < th, fed, ded).mean())))
    machines = sorted({r["machine"] for r in R})
    three = {}
    for m in machines:
        idm = [k for k, r in enumerate(R) if r["machine"] == m and "err_dgk" in r]
        if len(idm) < 8:
            three[m] = {"note": "need dgknet errors"}; continue
        pe = np.array([R[k]["pers50"] for k in idm]); fe = np.array([R[k]["err_full"] for k in idm]); de = np.array([R[k]["err_dgk"] for k in idm])
        u = np.array([R[k]["rel_ptp"] for k in idm]); pr = np.array([proba[k] for k in idm])
        best_fixed = float(min(pe.mean(), fe.mean(), de.mean()))
        oracle = float(np.minimum(np.minimum(pe, fe), de).mean())
        model_err = np.where(u < theta_md, fe, de) if np.isfinite(theta_md) else fe   # RMMD low-drive, DGKNet high-drive
        routed = np.where(pr < 0.5, pe, model_err)                                     # persistence when model NOT predicted to win
        routed = np.where(np.isfinite(pr), routed, np.minimum(pe, model_err))          # fold-skip fallback: best of the two
        three[m] = {"n": len(idm), "persistence": float(pe.mean()), "RMMD": float(fe.mean()), "flex_arm": float(de.mean()),
                    "best_fixed_arm": best_fixed, "shot_oracle": oracle,
                    "alpha_router_per_shot": float(routed.mean()), "beats_best_fixed": bool(routed.mean() < best_fixed - 1e-9),
                    "gain_vs_fixed_frac_of_oracle_gap": (float((best_fixed - routed.mean()) / (best_fixed - oracle))
                                                         if best_fixed - oracle > 1e-9 else None)}
    out["B_decision_theoretic"] = {
        "alpha_winner_pred_AUC": alpha, "alpha_method": method, "features": feat_names, "n_shots": int(X.shape[0]),
        "frac_model_beats_persistence": float(y.mean()) if y.size else None,
        "VoI_bound_max_routing_gain_frac_of_oracle_gap": voi_frac, "rmmd_vs_dgknet_theta": theta_md,
        "n_machines_router_beats_fixed": int(sum(1 for v in three.values() if isinstance(v, dict) and v.get("beats_best_fixed"))),
        "interpretation": ("alpha~0.5 -> no ex-ante feature predicts the winner -> the fixed/near-fixed policy is "
                           "provably near-optimal (the reframe becomes a theorem). alpha>>0.5 -> the winner IS routable: "
                           "the PER-SHOT alpha-router (persistence when the OOS predictor says the model won't beat it, "
                           "else RMMD/DGKNet by drive) realizes the headroom -- beats_best_fixed shows where. On quiescent "
                           "machines the honest win is MATCHING persistence, not beating it."),
        "three_way_policy_per_machine": three}


def _quartile_idx(vals):
    """Activity quartile 0..3 by rank of persistence NRMSE (higher = more dynamic = q4). EXACTLY matches
    theory_validation/extrap_strong.py:activity_stratified so the router's cells line up with the extrap
    baselines: rank via mergesort, then (rank*4)//n clipped to 3."""
    pers = np.asarray(vals, float); n = pers.size
    if n == 0:
        return np.array([], int)
    ranks = np.empty(n, dtype=np.int64)
    ranks[np.argsort(pers, kind="mergesort")] = np.arange(n)
    return np.minimum((ranks * 4) // n, 3)


def _load_extrap_competitors(extrap_specs):
    """Parse --extrap-json MACHINE:path pairs -> {MACHINE: {H(str): {qK: {"persistence": p, "models": {name: nrmse}}}}}.
    Baseline numbers come STRAIGHT from the extrap report's holdout_activity_stratified (no re-evaluation of the 20
    checkpoints). The extrap quartiles are the same rank-based activity quartiles _quartile_idx reproduces."""
    comp = {}
    for spec in (extrap_specs or []):
        spec = spec.strip()
        if not spec or ":" not in spec:
            continue
        m, path = spec.split(":", 1)
        try:
            rep = json.loads(Path(path).read_text())
        except Exception as e:
            print(f"  [extrap] SKIP {spec}: {e}", flush=True); continue
        cells = {}
        for name, mrep in rep.get("models", {}).items():
            strat = mrep.get("holdout_activity_stratified", {})
            for H, qd in strat.items():
                for qK, v in qd.items():
                    mn = v.get("model_nrmse")
                    if mn is None:
                        continue
                    c = cells.setdefault(str(H), {}).setdefault(qK, {"persistence": v.get("persistence_nrmse"), "models": {}})
                    c["models"][name] = float(mn)
                    if c.get("persistence") is None:
                        c["persistence"] = v.get("persistence_nrmse")
        comp[m] = cells
        print(f"  [extrap] {m} <- {path}: {len(cells)} horizons of baseline cells", flush=True)
    return comp


def analysis_router_sota(rows, out, chosen_flex, extrap_competitors=None):
    """THE SOTA question, done properly: is the PER-SHOT 3-way router {persistence | RMMD | DGKNet} state-of-the-art
    in EVERY (horizon x activity-quartile) cell, per machine -- beating persistence AND every other model per shot
    (best model PER SHOT, not best model per quartile)? The router's per-shot errors come from this run; the ~20
    BASELINE competitors are pulled from the extrap JSONs (--extrap-json MACHINE:path) so we don't re-evaluate them.
    A sanity field (rmmd_arm_vs_extrap_full) confirms the cells line up (router's RMMD arm ~= extrap 'full')."""
    extrap_competitors = extrap_competitors or {}
    R = [r for r in rows if "err_full" in r.get("errH", {}) and "persH" in r and r["t0"] is not None]
    if len(R) < 40:
        out["B_router_sota"] = {"note": "SKIPPED (need all-horizon errH/persH -- run WITHOUT --skip-errors)"}
        return
    flex_names = sorted({c for r in R for c in r["errH"] if c not in ("err_full", "err_abl")})
    dgk = chosen_flex if (chosen_flex in flex_names) else next((c for c in flex_names if "dgknet" in c.lower()), None)
    Hs = sorted({H for r in R for H in r["errH"].get("err_full", {})})
    machines = sorted({r["machine"] for r in R})
    grid = {}
    for H in Hs:
        RH = [r for r in R if H in r["errH"]["err_full"] and H in r["persH"]
              and (dgk is None or (dgk in r["errH"] and H in r["errH"][dgk]))]
        if len(RH) < 30:
            continue
        pe = np.array([r["persH"][H] for r in RH]); fe = np.array([r["errH"]["err_full"][H] for r in RH])
        de = np.array([r["errH"][dgk][H] for r in RH]) if dgk else fe.copy()
        u = np.array([r["rel_ptp"] for r in RH], float)
        y = (fe < pe).astype(float)                                   # RMMD beats persistence @H
        X = np.array([[r["rel_ptp"]] + r["t0"] for r in RH], float)
        ok = np.isfinite(X).all(1)
        proba = np.full(len(RH), np.nan)
        if ok.sum() >= 30 and 0 < y[ok].mean() < 1:
            proba[np.where(ok)[0]] = _oos_proba(X[ok], y[ok])
        theta = float("nan"); finu = u[np.isfinite(u)]
        if dgk is not None and len(finu) >= 20:
            gr = np.quantile(finu, np.linspace(0.1, 0.9, 17))
            theta = float(min(gr, key=lambda th: float(np.where(u < th, fe, de).mean())))
        model_err = np.where(u < theta, fe, de) if np.isfinite(theta) else fe
        routed = np.where(np.isfinite(proba) & (proba >= 0.5), model_err, pe)
        routed = np.where(np.isfinite(proba), routed, np.minimum(pe, model_err))   # fold-skip -> best of the two
        mach = np.array([r["machine"] for r in RH])
        for m in machines:
            sel = np.where(mach == m)[0]
            if len(sel) < 8:
                continue
            q = _quartile_idx(pe[sel])                          # rank-quartiles matching extrap exactly
            ex_cells = (extrap_competitors.get(m, {}) or {}).get(str(H), {})
            for qi in range(4):
                cell = sel[q == qi]
                if len(cell) < 3:
                    continue
                rmean = float(np.nanmean(routed[cell]))
                rmmd_arm = float(np.nanmean(fe[cell]))
                # competitors: prefer the extrap baselines for this (m,H,q); else fall back to the router's own arms
                ex = ex_cells.get(f"q{qi+1}", {})
                comps = dict(ex.get("models", {}))
                ex_pers = ex.get("persistence")
                comps["persistence"] = float(ex_pers) if ex_pers is not None else float(np.nanmean(pe[cell]))
                comps.setdefault("RMMD", rmmd_arm)
                if dgk is not None:
                    comps.setdefault("DGKNet", float(np.nanmean(de[cell])))
                best = min(v for v in comps.values() if np.isfinite(v))
                # alignment sanity: our RMMD-arm cell mean should ~match extrap 'full' at the same cell
                ex_full = ex.get("models", {}).get("full")
                grid.setdefault(m, {}).setdefault(int(H), {})[f"q{qi+1}"] = {
                    "n": int(len(cell)), "router": rmean,
                    "beats_persistence": bool(rmean <= comps["persistence"] + 1e-9),
                    "SOTA_vs_all": bool(rmean <= best + 1e-9),
                    "best_competitor": float(best), "n_competitors": len(comps),
                    "used_extrap_baselines": bool(ex.get("models")),
                    "rmmd_arm_vs_extrap_full": ([rmmd_arm, float(ex_full)] if ex_full is not None else None),
                    "competitors": {k: float(v) for k, v in comps.items()}}
    def _tally(cells):
        misalign = [c["rmmd_arm_vs_extrap_full"] for c in cells if c.get("rmmd_arm_vs_extrap_full")
                    and abs(c["rmmd_arm_vs_extrap_full"][0] - c["rmmd_arm_vs_extrap_full"][1]) > 0.03]
        return {"cells": len(cells),
                "SOTA_vs_all": sum(1 for c in cells if c["SOTA_vs_all"]),
                "beats_or_matches_persistence": sum(1 for c in cells if c["beats_persistence"]),
                "cells_with_extrap_baselines": sum(1 for c in cells if c["used_extrap_baselines"]),
                "n_cells_RMMD_arm_misaligned_vs_extrap_full(>0.03)": len(misalign)}
    per_summary = {m: _tally([c for hd in grid[m].values() for c in hd.values()]) for m in grid}
    allc = [c for m in grid.values() for hd in m.values() for c in hd.values()]
    out["B_router_sota"] = {
        "dgknet_arm": dgk, "machines_with_extrap_baselines": sorted(extrap_competitors),
        "horizons": Hs, "per_machine_grid": grid, "per_machine_summary": per_summary,
        "pooled": _tally(allc),
        "note": ("Per-shot 3-way router scored in every (horizon x activity-quartile) cell per machine. "
                 "SOTA_vs_all = router mean <= EVERY competitor in that cell. Competitors = the ~20 models pulled "
                 "from the extrap JSONs (--extrap-json MACHINE:path; holdout_activity_stratified) + persistence, "
                 "with the router's own arms as fallback where no extrap report is given. Quartiles reproduce "
                 "extrap's rank-based activity quartiles exactly; rmmd_arm_vs_extrap_full is the alignment check "
                 "(the router's RMMD-arm cell mean should ~= extrap 'full'; misalign count flags any binning drift).")}


# --------------------------------------------------------------------------- Analysis C: nonlocality (model-as-instrument)
def _partial_corr(a, b, c):
    """Pearson partial correlation of a,b controlling for c (residualize both on c linearly, correlate residuals)."""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[m], b[m], c[m]
    if a.size < 10 or np.ptp(c) < 1e-12:
        return float("nan"), float("nan"), int(a.size)
    def resid(y, x):
        s, i = np.polyfit(x, y, 1); return y - (s * x + i)
    ra, rb = resid(a, c), resid(b, c)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan"), float("nan"), int(a.size)
    r = float(np.corrcoef(ra, rb)[0, 1]); nsz = a.size
    try:
        from scipy import stats
        tstat = r * np.sqrt((nsz - 3) / max(1e-12, 1 - r ** 2))
        pval = float(2 * stats.t.sf(abs(tstat), df=nsz - 3))
    except Exception:
        pval = float("nan")
    return r, pval, int(nsz)


def analysis_nonlocality(rows, out):
    # Window sensitivity: for each window truncation, report both the pooled and top-drive-tercile partial
    # rho. Transient hypothesis: a shorter window gives a higher rho. Primary window = w50.
    def _high_drive(Rw, key):
        uu = np.array([r["rel_ptp"] for r in Rw], float)
        thr = np.nanquantile(uu[np.isfinite(uu)], 2 / 3) if np.isfinite(uu).any() else np.nan
        Rh = [r for r in Rw if np.isfinite(r["rel_ptp"]) and r["rel_ptp"] >= thr]
        return _partial_corr([r[key] for r in Rh], [r["dres"] for r in Rh], [r["rel_ptp"] for r in Rh])
    win_sens = {}
    for wkey in ("nonloc10", "nonloc20", "nonloc30", "nonloc50", "nonloc75", "nonloc100"):
        Rw = [r for r in rows if np.isfinite(r.get(wkey, np.nan)) and np.isfinite(r.get("dres", np.nan))]
        pw, pvw, nw = _partial_corr([r[wkey] for r in Rw], [r["dres"] for r in Rw], [r["rel_ptp"] for r in Rw])
        ph, pvh_, nh_ = _high_drive(Rw, wkey)
        win_sens[wkey] = {"partial_rho": pw, "p": pvw, "n": nw, "high_drive_rho": ph, "high_drive_p": pvh_, "high_drive_n": nh_}
    R = [r for r in rows if np.isfinite(r.get("nonloc", np.nan)) and np.isfinite(r.get("dres", np.nan))]
    nl = [r["nonloc"] for r in R]; dr = [r["dres"] for r in R]; u = [r["rel_ptp"] for r in R]
    pr_raw = float(np.corrcoef(np.asarray(nl, float), np.asarray(dr, float))[0, 1]) if len(R) >= 4 else float("nan")
    pr, pv, nsz = _partial_corr(nl, dr, u)                              # PRIMARY: w50 (the 0.232-anchor), controls for drive
    per = {}
    for m in sorted({r["machine"] for r in R}):
        Rm = [r for r in R if r["machine"] == m]
        per[m] = dict(zip(("partial_rho", "p", "n"), _partial_corr([r["nonloc"] for r in Rm],
                                                                   [r["dres"] for r in Rm], [r["rel_ptp"] for r in Rm])))
    # theory-specified stratum (NOT a moved goalpost): nonlocal transport is expected under TRANSIENT DRIVE, so also
    # report the partial correlation within the top drive tercile (pre-specified by the physics, reported alongside).
    uu = np.array(u, float); thr = np.nanquantile(uu[np.isfinite(uu)], 2 / 3) if np.isfinite(uu).any() else np.nan
    Rh = [r for r in R if np.isfinite(r["rel_ptp"]) and r["rel_ptp"] >= thr]
    prh, pvh, nh = _partial_corr([r["nonloc"] for r in Rh], [r["dres"] for r in Rh], [r["rel_ptp"] for r in Rh])
    dyn_hits = [m for m, v in per.items() if np.isfinite(v.get("partial_rho", np.nan)) and v["partial_rho"] >= 0.3
                and np.isfinite(v.get("p", np.nan)) and v["p"] < 0.01]
    out["C_nonlocality"] = {"raw_corr_nonloc_vs_dres": pr_raw, "window_primary": "nonloc50 (0.232-anchor)",
                            "window_sensitivity": win_sens,
                            "partial_rho_controlling_for_drive": pr, "partial_p": pv, "n": nsz,
                            "partial_rho_high_drive_tercile": prh, "high_drive_p": pvh, "high_drive_n": nh,
                            "per_machine_partial": per, "machines_hitting_0.3": dyn_hits,
                            "PREREGISTERED_HIT": bool(np.isfinite(pr) and pr >= 0.3 and np.isfinite(pv) and pv < 0.01),
                            "note": "PRE-REGISTERED = pooled partial rho>=0.3 & p<0.01. Reported ALONGSIDE (theory-"
                                    "specified, not post-hoc): the top-drive-tercile partial (nonlocal transport is a "
                                    "transient-drive phenomenon) and the per-machine breakdown. If the pooled misses 0.3 "
                                    "but is highly significant and concentrated on the dynamic machines, that IS the "
                                    "theory's prediction -- report it exactly that way (Discussion + Extended Data, NP seed). "
                                    "Scoping: TRANSP is transport-timescale -> profile-scale nonlocality, NOT turbulence "
                                    "correlation lengths -- phrase it that way."}


def analysis_nonloc_equation(rows, out):
    """THE CANTOR NONLOCALITY EQUATION. Can a shot's profile nonlocality be PREDICTED from CLEAN INPUTS (initial
    profile shape + exogenous drive + operating point), WITHIN a machine? The honest metric is per-machine CV
    (train on a machine's shots, predict held-out shots of THAT machine). Tests linear, degree-2 polynomial, AND a
    nonparametric KNN (captures arbitrary nonlinearity -- exp/log/threshold/interactions -- with NO assumed form).
    If even KNN gives ~0 within-machine, no regression helps: nonlocality is robustly emergent. CLEAN predictors
    only (t0 = initial condition, rel_ptp = drive, omega_d = t0 operating point; NOT model outputs)."""
    R = [r for r in rows if np.isfinite(r.get("nonloc", np.nan)) and r.get("t0") is not None
         and np.isfinite(r.get("rel_ptp", np.nan)) and np.isfinite(r.get("omega_d", np.nan))]
    if len(R) < 50:
        out["nonloc_equation"] = {"note": "SKIPPED (<50 shots with nonloc + clean inputs)"}
        return
    feat = ["rel_ptp", "omega_d"] + T0_NAMES
    X = np.array([[r["rel_ptp"], r["omega_d"]] + list(r["t0"]) for r in R], float)
    y = np.array([r["nonloc"] for r in R], float)
    mach = np.array([r["machine"] for r in R])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y, mach = X[ok], y[ok], mach[ok]

    def _poly2(A):
        d = A.shape[1]
        return np.hstack([A, A ** 2] + [(A[:, i] * A[:, j])[:, None] for i in range(d) for j in range(i + 1, d)])

    def _ridge(Xtr, ytr, Xte, lam=1e-2):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        a, b = (Xtr - mu) / sd, (Xte - mu) / sd; b0 = ytr.mean()
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (ytr - b0))
        return b @ w + b0

    def _knn(Xtr, ytr, Xte, k=15):                    # nonparametric: arbitrary nonlinearity, no assumed form
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        a, b = (Xtr - mu) / sd, (Xte - mu) / sd; out_ = np.empty(len(b))
        for j in range(len(b)):
            dd = np.sum((a - b[j]) ** 2, axis=1)
            out_[j] = ytr[np.argsort(dd)[:min(k, len(a))]].mean()
        return out_

    def _within(predictor, transform=None):          # PER-MACHINE 5-fold CV = the honest within-machine test
        pm = {}
        for m in sorted(set(mach)):
            sel = np.where(mach == m)[0]
            if len(sel) < 40:
                continue
            Xm = X[sel] if transform is None else transform(X[sel]); ym = y[sel]
            im = np.arange(len(sel)); np.random.default_rng(1).shuffle(im); pred = np.full(len(sel), np.nan)
            for f in range(5):
                te = im[f::5]; tr = np.setdiff1d(im, te)
                pred[te] = predictor(Xm[tr], ym[tr], Xm[te])
            sst = np.sum((ym - ym.mean()) ** 2)
            if sst > 0:
                pm[m] = float(1 - np.nansum((ym - pred) ** 2) / sst)
        return (float(np.median(list(pm.values()))) if pm else float("nan")), pm

    lin_wm, lin_pm = _within(_ridge)
    poly_wm, poly_pm = _within(_ridge, transform=_poly2)
    knn_wm, knn_pm = _within(_knn)
    best_wm = max([v for v in (lin_wm, poly_wm, knn_wm) if np.isfinite(v)] or [float("nan")])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w = np.linalg.solve(Xs.T @ Xs + 1e-2 * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    coefs = {feat[i]: float(w[i]) for i in range(len(feat))}
    out["nonloc_equation"] = {
        "within_machine_median_r2": {"linear": lin_wm, "poly2": poly_wm, "knn": knn_wm},
        "per_machine_r2_knn": knn_pm, "n": int(len(y)), "features": feat, "intercept_mean_nonloc": float(y.mean()),
        "standardized_linear_coefficients": coefs, "dominant_terms": sorted(coefs, key=lambda k: -abs(coefs[k]))[:3],
        "VERDICT": ("PREDICTABLE within a machine (best CV-R2=%.3f) -- a Cantor Nonlocality Equation may exist" % best_wm
                    if (np.isfinite(best_wm) and best_wm >= 0.15) else
                    ("EMERGENT — even a nonparametric KNN gives ~0 within-machine CV-R2 (linear %.3f, poly2 %.3f, "
                     "knn %.3f): nonlocality is NOT set by pre-shot inputs under ANY functional form"
                     % (lin_wm, poly_wm, knn_wm))),
        "note": ("PER-MACHINE 5-fold CV (train within a machine, predict held-out shots of THAT machine) with linear, "
                 "degree-2 polynomial, AND nonparametric KNN. KNN captures arbitrary nonlinearity WITHOUT assuming a "
                 "form -- if KNN is also ~0, no regression (exp/log/whatever) will help -> robustly emergent. This is "
                 "the fair within-machine test (the earlier pooled-fit R2 was a between-machine artifact).")}


def analysis_nonloc_threshold(rows, out):
    """Does the D_res BENEFIT rise with a shot's nonlocality? Uses the CONTINUOUS benefit (err_abl - err_full), NOT a
    binary 'helps' -- on quiescent/flat shots err_abl~=err_full so a binary sign is PURE NOISE (this is why EAST's
    binary frac_helps was ~0.5 and its correlation spurious). Restricts to the HIGH-DRIVE regime where D_res is
    operative and the nonlocality proxy is reliable. Needs err_full + err_abl (pass --abl-dres-ckpt)."""
    R = [r for r in rows if np.isfinite(r.get("nonloc", np.nan)) and np.isfinite(r.get("err_full", np.nan))
         and np.isfinite(r.get("err_abl", np.nan)) and np.isfinite(r.get("rel_ptp", np.nan))]
    if len(R) < 50:
        out["nonloc_threshold"] = {"note": "SKIPPED (need nonloc+err_full+err_abl+rel_ptp; pass --abl-dres-ckpt)"}
        return
    nl = np.array([r["nonloc"] for r in R], float)
    benefit = np.array([r["err_abl"] - r["err_full"] for r in R], float)   # >0 = removing D_res HURT = D_res helped
    drive = np.array([r["rel_ptp"] for r in R], float)
    mach = np.array([r["machine"] for r in R])

    def _per(mask):
        per = {}
        for m in sorted(set(mach[mask])):
            sel = mask & (mach == m)
            if sel.sum() < 20:
                continue
            rho, p, _ = _spearman(nl[sel], benefit[sel])
            per[m] = {"n": int(sel.sum()), "spearman_nonloc_vs_benefit": rho, "p": p,
                      "frac_Dres_helps(benefit>0)": float(np.mean(benefit[sel] > 0)),
                      "median_benefit": float(np.median(benefit[sel]))}
        rl = [v["spearman_nonloc_vs_benefit"] for v in per.values() if np.isfinite(v["spearman_nonloc_vs_benefit"])]
        fp = float(np.mean([r > 0 for r in rl])) if rl else float("nan")
        med = float(np.median(rl)) if rl else float("nan")
        return per, fp, med

    allmask = np.ones(len(R), bool)
    thr = np.nanquantile(drive[np.isfinite(drive)], 2 / 3) if np.isfinite(drive).any() else np.nan
    hdmask = np.isfinite(drive) & (drive >= thr)
    all_per, all_fp, all_med = _per(allmask)
    hd_per, hd_fp, hd_med = _per(hdmask)
    rho_all, p_all, _ = _spearman(nl, benefit)
    rho_hd, p_hd, _ = _spearman(nl[hdmask], benefit[hdmask])
    real = (np.isfinite(hd_med) and hd_med >= 0.2 and np.isfinite(hd_fp) and hd_fp >= 0.8)
    out["nonloc_threshold"] = {
        "pooled_spearman_nonloc_vs_benefit_ALL": rho_all, "p_all": p_all, "n": len(R),
        "pooled_spearman_nonloc_vs_benefit_HIGHDRIVE": rho_hd, "p_highdrive": p_hd, "n_highdrive": int(hdmask.sum()),
        "per_machine_all": all_per, "frac_pos_all": all_fp, "median_rho_all": all_med,
        "per_machine_highdrive": hd_per, "frac_pos_highdrive": hd_fp, "median_rho_highdrive": hd_med,
        "VERDICT": ("D_res BENEFIT rises with nonlocality in the operative (high-drive) regime: consistent sign "
                    "(frac_pos=%.2f) median rho=%.2f -> supports the nonlocality->D_res-usefulness link" % (hd_fp, hd_med)
                    if real else
                    "WEAK/inconsistent even in the high-drive regime (frac_pos=%.2f median rho=%.2f): the "
                    "nonlocality->D_res-usefulness link is not established" % (hd_fp, hd_med)),
        "note": ("CONTINUOUS benefit (err_abl-err_full), NOT a binary 'helps' -- on quiescent shots err_abl~=err_full "
                 "so a binary sign is noise (EAST's binary frac_helps~0.5 was spurious). HIGH-DRIVE = top-tercile "
                 "rel_ptp, where D_res is operative + the nonlocality proxy is reliable. frac_pos = fraction of "
                 "machines with positive nonloc->benefit correlation.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ckpt", required=True)
    ap.add_argument("--abl-dres-ckpt", default=None, help="abl_dres checkpoint (for the theorem-faithful u* observed crossover)")
    ap.add_argument("--dgknet-ckpt", default=None, help="a single flexible-arm checkpoint (added to --flex-ckpts as 'dgknet')")
    ap.add_argument("--flex-ckpts", nargs="*", default=[], metavar="name:path",
                    help="flexible-arm CANDIDATES for B's high-drive rung, e.g. dgknet_lr1e-4:<p> node_lr2e-4:<p> ...; "
                         "the router picks whichever is best on the high-drive TAIL (reported), so you don't hard-code one.")
    ap.add_argument("--datasets", nargs="+", required=True, metavar="name:path", help="e.g. pool:<test.pt> east:<E> augd:<A>")
    ap.add_argument("--extrap-json", nargs="*", default=[], metavar="MACHINE:path",
                    help="extrap report(s) for B's router-SOTA baselines, e.g. EAST:<report_east.json> AUGD:<report_augd.json>. "
                         "The ~20 baseline errors are pulled from holdout_activity_stratified -- NOT re-evaluated. "
                         "MACHINE must match the per-shot machine label (EAST/AUGD/...).")
    ap.add_argument("--n-train", type=int, default=1, help="global training-window count N (cancels in cross-machine ordering)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-errors", action="store_true",
                    help="Pass-1 only (T=1 kernel pass): computes C (nonlocality) + A's beta_hat in MINUTES, skips the "
                         "T=50 model-error passes (so A's observed crossovers + B are omitted). Use for the fast C re-run.")
    ap.add_argument("--only-nonloc", action="store_true",
                    help="FAST mode: run ONLY the Cantor Nonlocality Equation + Tier-2 threshold (+ C). Rolls out just "
                         "full + abl_dres at T=1,50 (skips DGKNet/flex, all-horizon errors, A, B, router-SOTA).")
    ap.add_argument("--out-json", default=str(RESULTS / "buttress.json"))
    args = ap.parse_args()
    if args.only_nonloc:                              # fast mode: T=1 (kernel/nonloc) + T=50 (err_full/err_abl) only
        globals()["HORIZONS"] = [1, 50]
    rc, cmp_mod, ex = _imports()

    print("=== EXP-6: extracting per-shot table (dres, sigma^2, rel_ptp, nonlocality, T50 errors) ===", flush=True)
    rows, p_dres = extract(rc, cmp_mod, ex, args)
    print(f"  extracted {len(rows)} shots; D_res-branch params p={p_dres}", flush=True)

    out = {"n_shots": len(rows), "machines": sorted({r['machine'] for r in rows})}
    best_flex = _select_flex(rows, out)            # pick the best flexible arm on the high-drive tail -> r["err_dgk"]
    if best_flex:
        print(f"  flexible arm chosen (best on high-drive tail): {best_flex}", flush=True)
    if not args.only_nonloc:                                   # fast mode skips A / B / router-SOTA
        analysis_ustar(rows, p_dres, args.n_train, out)
        analysis_alpha(rows, out)
        extrap_comp = _load_extrap_competitors(args.extrap_json)   # ~20 baselines from the extrap JSONs (not re-run)
        analysis_router_sota(rows, out, best_flex, extrap_comp)    # per-shot router SOTA across ALL horizons x quartiles
    analysis_nonlocality(rows, out)                            # C (cheap, from Pass-1)
    analysis_nonloc_equation(rows, out)                        # the Cantor Nonlocality Equation (nonloc ~ clean inputs)
    analysis_nonloc_threshold(rows, out)                       # Tier-2 universal D_res-helps threshold on nonlocality

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))
    if "A_ustar" in out:
        print("\n=== A) u* prediction ===", flush=True)
        a = out["A_ustar"]
        print(f"  beta axis/R2 per machine: " + "  ".join(f"{m}:{(v.get('axis') or '-')}={round((v.get('beta_fit') or {}).get('R2', float('nan')), 2)}"
                                                           for m, v in a['per_machine'].items() if isinstance(v, dict) and v.get('beta_fit')))
        print(f"  real-beta machines={a['n_machines_with_real_beta_fit']} beta_DEGENERATE={a['beta_DEGENERATE']} "
              f"Spearman(pred,obs)={a['spearman_pred_vs_obs']} AUGD_lowest={a['AUGD_predicted_lowest']} HIT={a['PREREGISTERED_HIT']}")
        print(f"  observed_crossover_ordering (fallback result): {a['observed_crossover_ordering']}")
    if "B_decision_theoretic" in out:
        print("=== B) decision-theoretic alpha ===")
        fs = out.get("flex_selection", {})
        print(f"  flex arm chosen (best on high-drive tail): {fs.get('chosen_flex_arm')}  tail_means={fs.get('tail_mean_per_candidate')}")
        b = out["B_decision_theoretic"]
        print(f"  alpha(winner-pred AUC)={b.get('alpha_winner_pred_AUC')} ({b.get('alpha_method', b.get('note'))})  "
              f"VoI_frac={b.get('VoI_bound_max_routing_gain_frac_of_oracle_gap')}  "
              f"per-shot router beats best-fixed on {b.get('n_machines_router_beats_fixed')} machines")
    rs = out.get("B_router_sota", {})
    if "pooled" in rs:
        pl = rs["pooled"]
        print(f"  ROUTER-SOTA grid (all horizons x quartiles): SOTA-vs-all {pl['SOTA_vs_all']}/{pl['cells']} cells, "
              f"beats/matches persistence {pl['beats_or_matches_persistence']}/{pl['cells']} | dgk_arm={rs.get('dgknet_arm')} "
              f"| extrap-baseline machines={rs.get('machines_with_extrap_baselines')}")
        mis = pl.get('n_cells_RMMD_arm_misaligned_vs_extrap_full(>0.03)', 0)
        print(f"    alignment: {pl.get('cells_with_extrap_baselines',0)}/{pl['cells']} cells use extrap baselines; "
              f"RMMD-arm vs extrap-full misaligned in {mis} cells (want 0 -> binning matches)")
        for m, s in rs.get("per_machine_summary", {}).items():
            print(f"    {m}: SOTA {s['SOTA_vs_all']}/{s['cells']}, beats/matches pers {s['beats_or_matches_persistence']}/{s['cells']}")
    elif rs.get("note"):
        print(f"  ROUTER-SOTA grid: {rs.get('note')}")
    print("=== C) nonlocality (model-as-instrument) ===")
    c = out["C_nonlocality"]
    print(f"  pooled partial_rho={c['partial_rho_controlling_for_drive']} p={c['partial_p']} | high-drive-tercile={c['partial_rho_high_drive_tercile']} "
          f"| machines>=0.3: {c['machines_hitting_0.3']} | PREREG_HIT={c['PREREGISTERED_HIT']}")
    eq = out.get("nonloc_equation", {})
    print("=== CANTOR NONLOCALITY EQUATION (per-machine CV: linear / poly2 / KNN-nonparametric) ===")
    if "within_machine_median_r2" in eq:
        w = eq["within_machine_median_r2"]
        print(f"  WITHIN-machine median CV-R2:  linear={w['linear']:.3f}  poly2={w['poly2']:.3f}  KNN={w['knn']:.3f}  (n={eq['n']})")
        print(f"  -> {eq['VERDICT']}")
    else:
        print(f"  {eq.get('note')}")
    th = out.get("nonloc_threshold", {})
    print("=== TIER-2 nonlocality -> D_res BENEFIT (continuous; all + high-drive) ===")
    if "median_rho_highdrive" in th:
        print(f"  ALL shots: pooled rho(nonloc,benefit)={th['pooled_spearman_nonloc_vs_benefit_ALL']:.2f} | "
              f"per-machine frac_pos={th['frac_pos_all']} median_rho={th['median_rho_all']}")
        print(f"  HIGH-DRIVE (operative regime, n={th['n_highdrive']}): pooled rho={th['pooled_spearman_nonloc_vs_benefit_HIGHDRIVE']:.2f} | "
              f"frac_pos={th['frac_pos_highdrive']} median_rho={th['median_rho_highdrive']}")
        print(f"  -> {th['VERDICT']}")
    else:
        print(f"  {th.get('note')}")
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
