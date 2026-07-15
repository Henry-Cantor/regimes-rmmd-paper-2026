#!/usr/bin/env python3
"""SUT (Spectral Universality Theorem) confirmation. Outdated file, initially believed to be a result but
in fact circular, as SUT Loss is a method.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _imports():
    """Load the proven implementation modules (single source of truth for loading/eval)."""
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    rc = _import_module(strong / "training" / "rmmd_train_eval_impl.py", "rmmd_train_eval_impl")
    cmp_mod = _import_module(strong / "comparison" / "run_comparison.py", "comparison_run_comparison")
    return rc, cmp_mod


# ---------------------------------------------------------------------------
# Capture: record the EXACT inputs the RMMD block / kernel see during a model call
# (wrapping bound forwards is torch-version-proof, unlike register_forward_hook kwargs).
# ---------------------------------------------------------------------------
class RMMDCapture:
    def __init__(self, model):
        self.model = model
        self.records = []          # one dict per model() call (i.e. per rollout step)
        self._orig_rmmd_fwd = None

    def __enter__(self):
        rmmd = self.model.rmmd
        self._orig_rmmd_fwd = rmmd.forward

        def wrapped(*a, **k):
            # multi_machine_rmmd.forward calls: self.rmmd(x_t=..., omega_t=..., omega_d=...,
            # context=..., z_t=...) — all kwargs (multi_machine_rmmd.py ~line 531).
            rec = {
                "z_t": k.get("z_t", a[0] if a else None),
                "context": k.get("context"),
                "omega_t": k.get("omega_t"),
                "omega_d": k.get("omega_d"),
            }
            out = self._orig_rmmd_fwd(*a, **k)
            rec["d_res"] = out.d_res
            rec["d_psd"] = out.d_psd
            self.records.append({kk: (vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv)
                                 for kk, vv in rec.items()})
            return out

        rmmd.forward = wrapped
        return self

    def __exit__(self, *exc):
        self.model.rmmd.forward = self._orig_rmmd_fwd
        return False


# ---------------------------------------------------------------------------
# Operator spectra from captured conditioning (math pinned to rmmd_block.forward)
# ---------------------------------------------------------------------------
@torch.no_grad()
def operator_spectra(model, z_t: torch.Tensor, context: torch.Tensor,
                     omega_t: torch.Tensor, omega_d: torch.Tensor, top_k: int) -> dict:
    rmmd = model.rmmd
    dev = next(rmmd.parameters()).device
    z = z_t.to(dev); ctx = context.to(dev)
    w_t = omega_t.to(dev).reshape(-1); w_d = omega_d.to(dev).reshape(-1)

    # --- conservative half-map (rmmd_block.py forward ~185-211) ---
    A_raw = rmmd.A_sym_param
    A_skew = torch.tanh(A_raw - A_raw.T) * 0.8
    K_sym = rmmd._get_K_sym(A_skew)
    sym_gate = rmmd.geom_gate_sym(ctx).view(z.shape[0], rmmd.half_dim, rmmd.half_dim)
    K_sym_mod = K_sym.unsqueeze(0) + rmmd.residual_gate_scale * sym_gate
    K_sym_mod = torch.nan_to_num(K_sym_mod, nan=0.0, posinf=0.0, neginf=0.0)
    sym_fro = torch.norm(K_sym_mod, dim=(1, 2), keepdim=True)
    K_sym_mod = K_sym_mod / torch.clamp(sym_fro / 8.0, min=1.0)
    eye = torch.eye(rmmd.half_dim, device=dev)
    M_q = eye.unsqueeze(0) + float(rmmd.dissipation_step) * (K_sym_mod - eye.unsqueeze(0))
    mu = torch.linalg.eigvals(M_q[0].double())                     # complex (half_dim,)
    lam = torch.log(mu + 1e-30)                                    # per-step generator
    order = torch.argsort(mu.abs(), descending=True)[:top_k]
    cons_freq = lam.imag[order].abs()
    cons_grow = lam.real[order]

    # --- dissipative resonance contraction: I - g*D_psd_hat (rmmd_block ~233-251) ---
    kout = rmmd.kernel(z=z, omega_t=w_t, omega_d=w_d, context=ctx)
    d_psd_sym = 0.5 * (kout.d_psd + kout.d_psd.transpose(-1, -2))
    diag_only = torch.diag_embed(torch.diagonal(d_psd_sym, dim1=-2, dim2=-1))
    off = d_psd_sym - diag_only
    offdiag_frac = float((off.reshape(1, -1).norm() / (d_psd_sym.reshape(1, -1).norm() + 1e-12)).item())
    lam_max = rmmd._operator_norm(d_psd_sym)
    d_hat = d_psd_sym[0] / (lam_max[0] + 1e-3)
    g = float((torch.sigmoid(rmmd.dissipation_gain) * rmmd.max_dissipation_gain).item())
    diss_rates = g * torch.linalg.eigvalsh(d_hat.double())         # real, in [~0, g]
    diss_rates = torch.sort(diss_rates, descending=True).values[:top_k]

    # --- resonance landscape on the GB axis ---
    weights = (kout.amplitudes * kout.lorentz_weights)[0]          # (n_harmonics,)
    return {
        "cons_freq": cons_freq.detach().cpu().float().numpy(),     # |Im log mu|, per-step
        "cons_grow": cons_grow.detach().cpu().float().numpy(),
        "diss_rates": diss_rates.detach().cpu().float().numpy(),
        "offdiag_frac": offdiag_frac,
        "res_weights": weights.detach().cpu().float().numpy(),     # a_m * L_m at (w_t, w_d)
        "res_gammas": kout.gammas[0].detach().cpu().float().numpy(),
        "omega_t": float(w_t[0].item()),
        "omega_d": float(w_d[0].item()),
    }


@torch.no_grad()
def resonance_landscape_sweep(model, z_t, context, omega_d_mean: float, n_grid: int = 48) -> dict:
    """Learned resonance response a_m*L_m vs omega_t on a COMMON grid (units of omega_d):
    if the learned landscape is universal, these curves coincide across machines."""
    rmmd = model.rmmd
    dev = next(rmmd.parameters()).device
    z = z_t.to(dev); ctx = context.to(dev)
    grid = np.linspace(0.05, 3.0, n_grid)                          # omega_t / omega_d
    curves = np.zeros((n_grid, int(rmmd.kernel.n_harmonics)), dtype=np.float64)
    w_d = torch.tensor([omega_d_mean], device=dev)
    for i, r in enumerate(grid):
        w_t = torch.tensor([r * omega_d_mean], device=dev)
        kout = rmmd.kernel(z=z, omega_t=w_t, omega_d=w_d, context=ctx)
        curves[i] = (kout.amplitudes * kout.lorentz_weights)[0].detach().cpu().numpy()
    return {"grid_omega_t_over_omega_d": grid.tolist(), "curves": curves.tolist()}


def jacobian_spectrum(rc, model, sample, device, norm_stats, top_k: int) -> np.ndarray | None:
    """Eigenvalues |mu| of the full one-step NI map at t=0 (40x40), via autograd.
    batch_data construction pinned to rc._rollout_compact_shot_to_checkpoints (~1539-1578)."""
    pre_shot = sample["pre_shot_context"].unsqueeze(0).to(device)
    pre_shot = torch.nan_to_num(pre_shot, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(pre_shot.abs().amax(dim=1, keepdim=True) > 100.0):
        pre_shot = torch.sign(pre_shot) * torch.log1p(pre_shot.abs())
    pre_shot = torch.clamp(pre_shot, min=-12.0, max=12.0)
    limiter = sample["limiter_geometry_tensor"].unsqueeze(0).to(device)
    geom0 = sample["geom_t0"].unsqueeze(0).to(device)
    step_dt = torch.full((1, 1), rc._normalized_step_dt(1.0), device=device)
    omega_t, omega_d = rc._compute_omegas_for_compact_batch(
        sample["ni_t0"].unsqueeze(0), [sample.get("pre_shot_scalars", {})],
        [sample["machine"]], device, norm_stats)
    drv = sample.get("drivers_traj")
    drivers0 = drv[0].unsqueeze(0).to(device) if isinstance(drv, torch.Tensor) and drv.shape[0] > 0 else None

    def f(ni_flat):
        bd = {"compact_mode": True, "pre_shot_context": pre_shot,
              "limiter_geometry_tensor": limiter, "ni_t0": ni_flat.unsqueeze(0),
              "geometry_tensor": geom0, "step_dt": step_dt}
        if drivers0 is not None:
            bd["drivers"] = drivers0
        out = model(x_t=ni_flat.unsqueeze(0), machine_names=[sample["machine"]],
                    omega_t=omega_t, omega_d=omega_d, batch_data=bd)
        return out.x_next.squeeze(0)

    try:
        J = torch.autograd.functional.jacobian(f, sample["ni_t0"].to(device), vectorize=False)
        mu = torch.linalg.eigvals(J.double().cpu())
        return torch.sort(mu.abs(), descending=True).values[:top_k].float().numpy()
    except Exception as exc:  # noqa: BLE001 — jacobian is optional; never kill the run
        print(f"[sut] jacobian failed ({exc}); skipping for this shot", flush=True)
        return None


# ---------------------------------------------------------------------------
# Universality statistics
# ---------------------------------------------------------------------------
def _bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (multiple-comparison control)."""
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return pvals
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    adj[order] = np.minimum.accumulate((pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1))[::-1])[::-1]
    return np.clip(adj, 0.0, 1.0)


def universality_stats(per_machine: dict[str, np.ndarray], n_perm: int = 1000, seed: int = 0) -> dict:
    """per_machine: machine -> (n_shots, n_modes) spectra. Returns per-mode-rank stats."""
    machines = sorted(per_machine)
    if len(machines) < 2:
        return {"error": "need >=2 machines"}
    n_modes = min(v.shape[1] for v in per_machine.values())
    X = {m: np.asarray(v)[:, :n_modes] for m, v in per_machine.items()}
    means = np.stack([X[m].mean(axis=0) for m in machines])               # (M, n_modes)
    within = np.stack([X[m].var(axis=0, ddof=1) if X[m].shape[0] > 1 else np.zeros(n_modes)
                       for m in machines])
    between_std = means.std(axis=0, ddof=1)
    grand_mean = np.abs(means.mean(axis=0)) + 1e-12
    std_over_mean = between_std / grand_mean                              # universality gate < 0.2
    U = (between_std ** 2) / (within.mean(axis=0) + 1e-12)                # between/within variance

    # permutation null: shuffle machine labels across the pooled shots
    pooled = np.concatenate([X[m] for m in machines], axis=0)
    sizes = [X[m].shape[0] for m in machines]
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, n_modes))
    for p in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        ofs, ms = 0, []
        for s in sizes:
            ms.append(pooled[perm[ofs:ofs + s]].mean(axis=0)); ofs += s
        null[p] = np.stack(ms).std(axis=0, ddof=1)
    pvals = ((null >= between_std[None, :]).sum(axis=0) + 1) / (n_perm + 1)
    # Benjamini-Hochberg adjusted p-values (multiple-comparison control across mode ranks)
    adj = _bh_adjust(pvals)
    # Scale-floored dispersion: std_over_mean blows up on near-zero-mean modes; relative to the
    # FAMILY's grand scale it stays interpretable (used by the verdict; both are reported).
    family_scale = max(float(np.mean(np.abs(means))), 1e-12)
    som_scaled = (between_std / family_scale)
    return {
        "perm_pvalues_bh_adjusted": adj.tolist(),
        "std_over_family_scale": som_scaled.tolist(),
        "machines": machines, "n_modes": int(n_modes),
        "machine_means": {m: X[m].mean(axis=0).tolist() for m in machines},
        "machine_stds": {m: X[m].std(axis=0, ddof=1).tolist() if X[m].shape[0] > 1 else None
                          for m in machines},
        "between_std": between_std.tolist(),
        "std_over_mean": std_over_mean.tolist(),
        "variance_ratio_U": U.tolist(),
        "perm_pvalues": pvals.tolist(),
    }


def east_containment(train: dict[str, np.ndarray], east: np.ndarray | None) -> dict | None:
    if east is None or not train:
        return None
    n_modes = min(min(v.shape[1] for v in train.values()), east.shape[1])
    means = np.stack([np.asarray(v)[:, :n_modes].mean(axis=0) for v in train.values()])
    lo, hi = means.min(axis=0), means.max(axis=0)
    spread = (hi - lo) + 1e-12
    em = east[:, :n_modes].mean(axis=0)
    inside = (em >= lo - 0.25 * spread) & (em <= hi + 0.25 * spread)
    dist = np.maximum(lo - em, em - hi) / spread                          # <=0 inside
    return {"east_mean": em.tolist(), "train_lo": lo.tolist(), "train_hi": hi.tolist(),
            "inside_frac": float(inside.mean()),
            "normalized_excess_distance": np.maximum(dist, 0.0).tolist()}


FAMILY_NAMES = ("cons_freq_raw", "cons_freq_over_wd", "cons_freq_over_wt",
                "diss_rates", "res_weights", "res_gammas", "jac_mu_abs")


def _link_horizons(args) -> list[int]:
    """Sorted unique horizons for the SUT->extrap link. Default {8,20,50} capped at --link-horizon;
    the primary --link-horizon is always included."""
    hs = list(args.link_horizons) if getattr(args, "link_horizons", None) else \
        [h for h in (8, 20, 50) if h <= args.link_horizon]
    hs.append(int(args.link_horizon))
    return sorted({int(h) for h in hs if int(h) >= 1})


def collect_spectra(model, datasets, rc, args):
    """Per-shot operator spectra for ONE model over the datasets. Holdout shots are labeled
    '<MACHINE>(zero-shot)' (machine-derived — works for ANY holdout, EAST or D3D).
    HOLDOUT shots additionally roll out to --link-horizon and record per-shot zero-shot
    NRMSE (shot_meta 'nrmse_link') for the SUT->extrapolation link test."""
    spectra = {fam: defaultdict(list) for fam in FAMILY_NAMES}
    shot_meta = defaultdict(list)
    landscape_inputs = {}     # machine -> (z_t, context, mean omega_d) for the common sweep
    counts = defaultdict(int)
    link_hs = _link_horizons(args)
    link_max = max(link_hs)
    cap_n = int(args.max_shots_per_machine)   # 0 = no cap (use all shots)
    for tag, ds, norm in datasets:
        # RIGOR: omega denormalization uses the stats the DATASET was normalized with.
        for i in range(len(ds)):
            s = ds[i]
            is_holdout = tag != "indist"
            mach = s["machine"] if not is_holdout else f"{s['machine']}(zero-shot)"
            if cap_n and counts[mach] >= cap_n:
                continue
            counts[mach] += 1
            T_av = int(s["ni_traj"].shape[0])
            # holdout rolls to the deepest link horizon it can reach; record NRMSE at EVERY link
            # horizon for the per-horizon + horizon-averaged link (in-dist needs only t0 spectra).
            hs_here = [h for h in link_hs if h <= T_av]
            steps = max(1, min(link_max, T_av)) if is_holdout else 1
            rep = hs_here if (is_holdout and hs_here) else [steps]
            with RMMDCapture(model) as cap:
                # rollout = the exact eval conditioning path; records[0] = TRUE-t0 conditioning.
                ni_preds, _ = rc._rollout_compact_shot_to_checkpoints(
                    model, s["ni_t0"], s["geom_t0"], s["pre_shot_context"],
                    s["limiter_geometry_tensor"], s["ni_traj"], s["geom_traj"],
                    s["machine"], s.get("pre_shot_scalars", {}), args.device, norm,
                    max_time_step=steps, drivers_traj=s.get("drivers_traj"),
                    report_horizons=rep)
            if not cap.records:
                continue
            nrmse_by_h = {}
            if is_holdout:
                for h in hs_here:
                    if h > 1 and h in ni_preds:
                        e, _ = rc._normalized_rmse_mae(ni_preds[h].numpy(), s["ni_traj"][h - 1].numpy())
                        nrmse_by_h[h] = float(e)
            nrmse_link = nrmse_by_h.get(int(args.link_horizon))
            rec = cap.records[0]
            sp = operator_spectra(model, rec["z_t"], rec["context"], rec["omega_t"],
                                  rec["omega_d"], args.top_modes)
            wd = max(sp["omega_d"], 1e-6); wt = max(sp["omega_t"], 1e-6)
            spectra["cons_freq_raw"][mach].append(sp["cons_freq"])
            spectra["cons_freq_over_wd"][mach].append(sp["cons_freq"] / wd)
            spectra["cons_freq_over_wt"][mach].append(sp["cons_freq"] / wt)
            spectra["diss_rates"][mach].append(sp["diss_rates"])
            spectra["res_weights"][mach].append(sp["res_weights"])
            spectra["res_gammas"][mach].append(sp["res_gammas"])
            shot_meta[mach].append({"omega_t": sp["omega_t"], "omega_d": sp["omega_d"],
                                    "offdiag_frac": sp["offdiag_frac"],
                                    "nrmse_link": nrmse_link, "nrmse_by_h": nrmse_by_h})
            if mach not in landscape_inputs:
                landscape_inputs[mach] = (rec["z_t"], rec["context"], wd)
            if args.jacobian:
                jm = jacobian_spectrum(rc, model, s, args.device, norm, args.top_modes)
                if jm is not None:
                    spectra["jac_mu_abs"][mach].append(jm)
        print(f"[sut] {tag}: machines so far {dict(counts)}", flush=True)
    return spectra, shot_meta, landscape_inputs, counts


def _spearman_perm(a, b, n_perm: int = 2000, seed: int = 0):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 8 or a.std() < 1e-12 or b.std() < 1e-12:
        return None, None, int(a.size)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    r = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, rb[rng.permutation(rb.size)])[0, 1] for _ in range(n_perm)])
    p = float(((np.abs(null) >= abs(r)).sum() + 1) / (n_perm + 1))
    return r, p, int(a.size)


def sut_extrapolation_link(spectra, shot_meta, holdout_key, n0, families, nr_values=None) -> dict | None:
    """THE mechanistic SUT->extrapolation test: per holdout shot, distance of its operator
    spectrum from the TRAINING-machine envelope vs its zero-shot NRMSE at the link horizon.
    A positive correlation = shots whose learned operator falls inside the universal family
    extrapolate better -> universality is the MECHANISM of zero-shot transfer (predictive
    validity), independent of the abl_sut NRMSE row."""
    if holdout_key is None or holdout_key not in shot_meta:
        return None
    metas = shot_meta[holdout_key]
    if nr_values is None:
        nr = np.array([m["nrmse_link"] if m.get("nrmse_link") is not None else np.nan for m in metas])
    else:
        nr = np.asarray(nr_values, dtype=float)
    if nr.shape[0] != len(metas) or not np.isfinite(nr).any():
        return None
    total_d = np.zeros(len(metas)); used = []
    per_family = {}

    def _link(dvec):
        # correlate ONLY over shots with a finite spectral distance AND a finite accuracy at this
        # horizon (shots too short to reach it are dropped, not NaN-poisoned).
        m = np.isfinite(dvec) & np.isfinite(nr)
        if m.sum() < 5:
            return None, None, int(m.sum())
        return _spearman_perm(dvec[m], nr[m])

    for fam in families:
        hold = spectra[fam].get(holdout_key)
        train = {m: np.stack(v) for m, v in spectra[fam].items()
                 if v and not m.endswith("(zero-shot)")}
        if not hold or len(train) < 2 or len(hold) != len(metas):
            continue
        H = np.stack(hold)
        n_modes = int(min(min(v.shape[1] for v in train.values()), H.shape[1], n0))
        means = np.stack([v[:, :n_modes].mean(axis=0) for v in train.values()])
        lo, hi = means.min(axis=0), means.max(axis=0)
        spread = (hi - lo) + 1e-12
        d = np.maximum(np.maximum(lo - H[:, :n_modes], H[:, :n_modes] - hi), 0.0) / spread
        d = d.mean(axis=1)
        total_d += d; used.append(fam)
        r, p, n = _link(d)
        per_family[fam] = {"spearman": r, "p_perm": p, "n": n}
    if not used:
        return None
    r, p, n = _link(total_d / len(used))
    return {"link_quantity": "per-shot spectral distance from training envelope vs zero-shot NRMSE",
            "families_used": used, "overall_spearman": r, "overall_p_perm": p, "n_shots": n,
            "per_family": per_family,
            "interpretation": ("r > 0 (p<0.05): operator-spectrum proximity to the universal "
                               "family PREDICTS zero-shot accuracy shot-by-shot — universality "
                               "is the mechanism of extrapolation (Theorem 3 -> STRONG bound), "
                               "independent of the abl_sut NRMSE ablation.")}


def machine_identity_ancova(per_machine: dict, meta: dict, n0: int, n_perm: int = 1000,
                            seed: int = 0) -> dict:
    """THE strongest universality statistic (ANCOVA): once the GB operating point
    (omega_t, omega_d, quadratic terms) is controlled, does MACHINE IDENTITY explain ANY
    residual spectral variance? SUT predicts ~none (Theorem 3: spectral differences are a
    function of dimensionless parameters, not identity). Pooled OLS per mode rank over the
    top-n0 modes; Delta-R^2 of machine one-hots over physics covariates; Freedman-Lane
    permutation (permute reduced-model residuals) for the p-value. Training machines only."""
    machines = sorted(m for m in per_machine if not m.endswith("(zero-shot)"))
    if len(machines) < 3:
        return {"error": "<3 training machines"}
    # alignment guard (jacobian failures can desync spectra vs meta for jac_mu_abs)
    for m in machines:
        if np.asarray(per_machine[m]).shape[0] != len(meta[m]):
            return {"error": f"spectra/meta misaligned for {m} (family has per-shot failures)"}
    n_modes = int(min(min(np.asarray(per_machine[m]).shape[1] for m in machines), n0))
    Y, wt, wd, lab = [], [], [], []
    for mi, m in enumerate(machines):
        arr = np.asarray(per_machine[m])[:, :n_modes]
        Y.append(arr)
        wt += [r["omega_t"] for r in meta[m]]
        wd += [r["omega_d"] for r in meta[m]]
        lab += [mi] * arr.shape[0]
    Y = np.concatenate(Y, axis=0)
    wt = np.asarray(wt, dtype=np.float64); wd = np.asarray(wd, dtype=np.float64)
    lab = np.asarray(lab)
    wt = (wt - wt.mean()) / (wt.std() + 1e-12); wd = (wd - wd.mean()) / (wd.std() + 1e-12)
    Xp = np.stack([np.ones_like(wt), wt, wd, wt ** 2, wd ** 2, wt * wd], axis=1)
    D = np.zeros((len(lab), len(machines) - 1))
    for j in range(1, len(machines)):
        D[lab == j, j - 1] = 1.0
    Xf = np.concatenate([Xp, D], axis=1)
    Xm = np.concatenate([np.ones((len(lab), 1)), D], axis=1)

    def fit_r2(X, y):
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ c
        res = y - yhat
        ss = float(((y - y.mean()) ** 2).sum()) + 1e-12
        return 1.0 - float(res @ res) / ss, yhat, res

    rng = np.random.default_rng(seed)
    d_obs, r2_phys, r2_mach = [], [], []
    null = np.zeros(n_perm)
    for k in range(n_modes):
        y = Y[:, k]
        rp, yhat_p, res_p = fit_r2(Xp, y)
        rf, _, _ = fit_r2(Xf, y)
        rm, _, _ = fit_r2(Xm, y)
        d_obs.append(rf - rp); r2_phys.append(rp); r2_mach.append(rm)
        for p in range(n_perm):   # Freedman-Lane
            y_star = yhat_p + res_p[rng.permutation(res_p.size)]
            rfs, _, _ = fit_r2(Xf, y_star)
            rps, _, _ = fit_r2(Xp, y_star)
            null[p] += (rfs - rps) / n_modes
    mean_d = float(np.mean(d_obs))
    return {
        "delta_R2_machine_after_physics": mean_d,
        "p_freedman_lane": float(((null >= mean_d).sum() + 1) / (n_perm + 1)),
        "R2_physics_only_mean": float(np.mean(r2_phys)),
        "R2_machine_only_mean": float(np.mean(r2_mach)),
        "n_modes": n_modes, "n_shots": int(Y.shape[0]), "machines": machines,
        "interpretation": ("UNIVERSAL if delta_R2 ~ 0 / p n.s.: machine identity adds nothing "
                           "once the dimensionless operating point is controlled (Theorem 3's "
                           "C_SUT*||dbeta|| form). R2_machine_only shows the UNCONTROLLED "
                           "machine effect for contrast."),
    }


def landscape_universality(landscapes: dict) -> dict:
    """Scalar universality index for the learned resonance-response curves on the common GB
    grid: pairwise RMS distance between training machines' total-response curves, normalized
    by the mean curve RMS (small = universal); plus holdout-to-train distance."""
    keys = sorted(landscapes)
    curves = {m: np.asarray(landscapes[m]["curves"]).sum(axis=1) for m in keys}
    base = float(np.mean([np.sqrt(np.mean(c ** 2)) for c in curves.values()])) + 1e-12
    train = [m for m in keys if not m.endswith("(zero-shot)")]
    hold = [m for m in keys if m.endswith("(zero-shot)")]
    pair = [float(np.sqrt(np.mean((curves[a] - curves[b]) ** 2)) / base)
            for i, a in enumerate(train) for b in train[i + 1:]]
    out = {"pairwise_rms_distance_mean": float(np.mean(pair)) if pair else None,
           "n_train_pairs": len(pair)}
    if hold and train:
        out["holdout_to_train_rms_distance_mean"] = float(np.mean(
            [np.sqrt(np.mean((curves[hold[0]] - curves[m]) ** 2)) / base for m in train]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="dir or file (headline/full RMMD)")
    ap.add_argument("--indist-data", default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt")
    ap.add_argument("--east-data", default=None, help="holdout compact dataset (zero-shot; any machine)")
    ap.add_argument("--compare-checkpoint", default=None,
                    help="second model (e.g. abl_sut) — quantifies the SUT loss's effect on "
                         "cross-machine spectral alignment (its ACTUAL training objective)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shots-per-machine", type=int, default=128,
                    help="cap shots/machine for spectra + the link (more = less noisy dispersion/"
                         "ANCOVA and larger link n). 0 = use ALL shots (max power; slower).")
    ap.add_argument("--top-modes", type=int, default=16)
    ap.add_argument("--n0", type=int, default=8, help="top-N0 modes for the SUT pass gate")
    ap.add_argument("--jacobian", action="store_true", help="also extract NI one-step Jacobian spectra (slower)")
    ap.add_argument("--link-horizons", type=int, nargs="*", default=None,
                    help="horizons for the SUT->extrap link, reported EACH separately + a per-shot "
                         "horizon-AVERAGE (denoised; n stays = n_shots, no pseudo-replication). "
                         "Default = {8, 20, 50} capped at --link-horizon. Holdout rolls to the max.")
    ap.add_argument("--link-horizon", type=int, default=20,
                    help="holdout shots roll to this T for the SUT->extrapolation link test")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--out", default=str(HERE / "results" / "sut_report.json"))
    args = ap.parse_args()

    rc, cmp_mod = _imports()
    t_start = time.time()
    cp = cmp_mod._find_checkpoint(Path(args.checkpoint))
    if cp is None:
        raise SystemExit(f"No checkpoint under {args.checkpoint}")
    model, norm_ck, mtype = cmp_mod._build_model(rc, cp, args.device)
    if mtype != "rmmd":
        raise SystemExit(f"SUT requires the RMMD model (got model_type={mtype})")

    def load_ds(path, max_time=2):
        payload = rc._load_phase0_dataset(Path(path))
        norm = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        return rc.CompactRolloutDataset(payload, max_time=max_time, normalization_stats=norm), norm

    datasets = [("indist", *load_ds(args.indist_data))]
    if args.east_data:
        # holdout needs trajectory depth for the SUT->extrapolation link rollout (to the DEEPEST
        # link horizon, so per-horizon accuracy can be recorded at each).
        _link_max = max(_link_horizons(args))
        datasets.append(("holdout", *load_ds(args.east_data, max_time=max(2, _link_max))))

    spectra, shot_meta, landscape_inputs, per_machine_count = collect_spectra(model, datasets, rc, args)
    holdout_keys = sorted(k for k in per_machine_count if k.endswith("(zero-shot)"))
    holdout_key = holdout_keys[0] if holdout_keys else None
    report = {
        "meta": {
            "checkpoint": str(cp), "indist_data": args.indist_data, "east_data": args.east_data,
            "n_shots_per_machine": dict(per_machine_count),
            "units_note": ("Operator spectra in PER-STEP units (unit-step compact dataset; physical "
                           "dt/step not stored). 'over_wd' = divided by the shot's GB-normalized "
                           "omega_d (kernel harmonic axis) = the primary dimensionless SUT statistic. "
                           "res_weights/res_gammas live natively on the GB axis."),
            "elapsed_s": None,
        },
        "shared_private": {
            "s_universal_norm": float(model.s_universal.detach().norm().item()),
            "delta_s_norms": {m: float((0.05 * (model.delta_s_machines[i]
                               - model.delta_s_machines[i].T)).detach().norm().item())
                              for m, i in model.machine_to_idx.items()},
        },
        "families": {}, "resonance_landscape": {}, "verdict": {},
        # Operating-point confound check: per-machine omega distributions. If machines sit at very different
        # (omega_t, omega_d), spectral differences may reflect operating point rather than non-universality.
        "machine_operating_points": {
            m: {"omega_t_mean": float(np.mean([r["omega_t"] for r in v])),
                "omega_d_mean": float(np.mean([r["omega_d"] for r in v])),
                "omega_t_std": float(np.std([r["omega_t"] for r in v])),
                "omega_d_std": float(np.std([r["omega_d"] for r in v])),
                "offdiag_frac_mean": float(np.mean([r["offdiag_frac"] for r in v]))}
            for m, v in shot_meta.items() if v},
    }

    for fam, by_mach in spectra.items():
        by_mach = {m: np.stack(v) for m, v in by_mach.items() if v}
        if not by_mach:
            continue
        train = {m: v for m, v in by_mach.items() if not m.endswith("(zero-shot)")}
        stats = universality_stats(train, n_perm=args.n_perm) if len(train) >= 2 else {"error": "<2 machines"}
        entry = {"universality": stats,
                 "holdout": east_containment(train, by_mach.get(holdout_key) if holdout_key else None),
                 # ANCOVA: machine identity vs dimensionless operating point (the strongest test)
                 "ancova": machine_identity_ancova(train, shot_meta, args.n0, n_perm=args.n_perm)}
        report["families"][fam] = entry

    # common-grid resonance landscape per machine (universality of the learned response curve)
    for mach, (z, ctx, wd) in landscape_inputs.items():
        report["resonance_landscape"][mach] = resonance_landscape_sweep(model, z, ctx, wd)
    report["resonance_landscape_universality"] = landscape_universality(report["resonance_landscape"])

    # Primary evidence: the operator-structure families (resonance landscape weights/widths, dissipation
    # rates) plus the raw conservative spectrum. The omega-relative conservative families are diagnostic only,
    # since the learned per-step frequencies have no physical-time conversion. jac_mu_abs (with --jacobian) is
    # fully conditioning-dependent and counts as primary evidence when computed.
    PRIMARY_FAMILIES = ("res_weights", "res_gammas", "diss_rates", "cons_freq_raw", "jac_mu_abs")
    per_family = {}
    for fam in report["families"]:
        u = report["families"][fam].get("universality", {})
        if "std_over_mean" not in u:
            continue
        n0 = min(args.n0, len(u["std_over_mean"]))
        som_sc = np.asarray(u.get("std_over_family_scale", u["std_over_mean"])[:n0])
        pv_bh = np.asarray(u.get("perm_pvalues_bh_adjusted", u["perm_pvalues"])[:n0])
        e = report["families"][fam].get("holdout")
        anc = report["families"][fam].get("ancova") or {}
        per_family[fam] = {
            "role": "primary" if fam in PRIMARY_FAMILIES else "diagnostic(units)",
            "top_modes_std_over_family_scale": som_sc.tolist(),
            "gate_dispersion_lt_0.2": bool(np.all(som_sc < 0.2)),
            "n_modes_bh_significant_nonuniversal": int((pv_bh < 0.05).sum()),
            "holdout_inside_frac": (e or {}).get("inside_frac"),
            "ancova_delta_R2": anc.get("delta_R2_machine_after_physics"),
            "ancova_p": anc.get("p_freedman_lane"),
            # EFFECT-SIZE gate (not significance: with N~300 a trivial effect can be 'significant'):
            # machine identity explains <5% of spectral variance beyond the physics covariates.
            "gate_ancova_identity_small": (anc.get("delta_R2_machine_after_physics") is not None
                                           and anc["delta_R2_machine_after_physics"] < 0.05),
        }
    # Triviality tier: cons_freq_raw is dominated by the shared A_sym operator (universal by construction);
    # the conditioning-dependent families (resonance weights/widths, dissipation rates, Jacobian) are the
    # non-trivial evidence. A 'supported' verdict must rest on at least one non-trivial family.
    NON_TRIVIAL = ("res_weights", "res_gammas", "diss_rates", "jac_mu_abs")
    def _disp(f): return bool(per_family.get(f, {}).get("gate_dispersion_lt_0.2"))
    def _anc(f):  return bool(per_family.get(f, {}).get("gate_ancova_identity_small"))
    primary_pass = [f for f in PRIMARY_FAMILIES if _disp(f) or _anc(f)]       # OR-gate (WEAK)
    # STRICT: clears BOTH gates (low dispersion AND machine-identity effect small). The OR-gate can
    # be dispersion-only while ANCOVA says machine matters, so BOTH is the honest bar.
    primary_pass_both = [f for f in PRIMARY_FAMILIES if _disp(f) and _anc(f)]
    nontrivial_both = [f for f in primary_pass_both if f in NON_TRIVIAL]
    # omega_d operating-range check: is the holdout queried OUTSIDE the trained omega_d range?
    ops = report["machine_operating_points"]
    hold_op = ops.get(holdout_key) if holdout_key else None
    train_wd = [d["omega_d_mean"] for m, d in ops.items() if not m.endswith("(zero-shot)")]
    wd_caveat = None
    if hold_op and train_wd and not (min(train_wd) <= hold_op["omega_d_mean"] <= max(train_wd)):
        wd_caveat = (f"holdout omega_d mean {hold_op['omega_d_mean']:.3f} is OUTSIDE the training "
                     f"range [{min(train_wd):.3f}, {max(train_wd):.3f}]: the kernel is queried "
                     "out-of-range on the holdout. Landscape/dissipation containment despite this "
                     "strengthens the universality claim; conservative-frequency containment is "
                     "weakened by it.")
    report["verdict"] = {
        "N0": args.n0,
        "primary_families": list(PRIMARY_FAMILIES),
        "per_family": per_family,
        "non_trivial_families": list(NON_TRIVIAL),
        "primary_families_passing_OR_gate": primary_pass,
        "primary_families_passing_BOTH_gates": primary_pass_both,
        "non_trivial_families_passing_BOTH_gates": nontrivial_both,
        "n_pass_or_gate": len(primary_pass),
        "n_pass_both_gates": len(primary_pass_both),
        # Headline: at least 2 families clear both gates and at least 1 is conditioning-dependent. The OR-gate
        # count is retained only as a separate, weaker flag.
        "sut_supported": bool(len(primary_pass_both) >= 2 and len(nontrivial_both) >= 1),
        "sut_weakly_supported_or_gate_only": bool(len(primary_pass) >= 2),
        "omega_d_range_caveat": wd_caveat,
        "units_rationale": ("Conservative spectra are PER-STEP (no stored dt/step); omega-relative "
                            "normalizations are reported but NOT gating (over_wd divides a shared "
                            "per-step spectrum by machine-dependent omega_d -> manufactured "
                            "dependence). Physical GB normalization requires per-shot dt: builder "
                            "rebuild TODO."),
        "triviality_check": (
            "Shared kernel/operator parameters make SOME cross-machine agreement built-in; the "
            "non-trivial evidence is (a) landscape/dissipation collapse across machine-specific "
            "(z, context, omega) INCLUDING zero-shot EAST queried out-of-range, (b) the Jacobian "
            "family (--jacobian, fully conditioning-dependent), (c) small geometry-gated "
            "deviations in cons_freq_raw. The paper claim: the LEARNED transport operator "
            "structure is machine-invariant and transfers zero-shot; not a measured plasma-"
            "spectrum universality (that needs physical dt + fluctuation data)."),
    }

    # SUT -> extrapolation link, reported at every link horizon plus a per-shot horizon-average (denoises
    # each shot's accuracy; n stays = n_shots, no pseudo-replication).
    metas = shot_meta.get(holdout_key, [])
    link_hs = _link_horizons(args)
    by_h = {}
    for h in link_hs:
        nr_h = np.array([(m.get("nrmse_by_h") or {}).get(h, np.nan) for m in metas], dtype=float)
        lk = sut_extrapolation_link(spectra, shot_meta, holdout_key, args.n0, PRIMARY_FAMILIES, nr_values=nr_h)
        if lk is not None:
            lk["link_horizon"] = int(h)
            by_h[int(h)] = lk
    if by_h:
        # Benjamini-Hochberg correct the per-horizon p-values (multiple comparisons); the per-horizon tests are
        # exploratory, and the primary link test is the denoised per-shot horizon-average below.
        _hs = sorted(by_h)
        _praw = np.array([by_h[h]["overall_p_perm"] if by_h[h]["overall_p_perm"] is not None else np.nan
                          for h in _hs], dtype=float)
        if np.isfinite(_praw).any():
            _fin = np.isfinite(_praw)
            _adj = np.full_like(_praw, np.nan)
            _adj[_fin] = _bh_adjust(_praw[_fin])
            for h, pa in zip(_hs, _adj):
                by_h[h]["overall_p_perm_bh_adjusted"] = (None if not np.isfinite(pa) else float(pa))
        report["sut_extrapolation_link_by_horizon"] = by_h
        report["sut_extrapolation_link"] = by_h.get(int(args.link_horizon)) or next(iter(by_h.values()))

        def _avg_acc(m):
            vals = [v for v in (m.get("nrmse_by_h") or {}).values() if v is not None and np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan
        nr_avg = np.array([_avg_acc(m) for m in metas], dtype=float)
        lk_avg = sut_extrapolation_link(spectra, shot_meta, holdout_key, args.n0, PRIMARY_FAMILIES, nr_values=nr_avg)
        if lk_avg is not None:
            lk_avg["link_horizon"] = "avg(" + ",".join(str(h) for h in link_hs) + ")"
            lk_avg["role"] = "PRIMARY link test (pre-registered; denoised per-shot average; per-horizon = exploratory)"
            report["sut_extrapolation_link_horizon_avg"] = lk_avg
            report["sut_extrapolation_link_primary"] = lk_avg

    # SUT-loss effect (--compare-checkpoint). The abl_sut NRMSE row is a weak test of the SUT loss (its
    # objective is spectral alignment, not NRMSE); the direct test is the same spectra extraction on a model
    # trained without the SUT loss. The abl_sut sibling is auto-discovered next to the main checkpoint.
    compare_ckpt = args.compare_checkpoint
    if not compare_ckpt:
        sib = Path(args.checkpoint).resolve().parent / "abl_sut"
        if cmp_mod._find_checkpoint(sib) is not None:
            compare_ckpt = str(sib)
            print(f"[sut] auto-using compare-checkpoint {sib} (no-SUT-loss sibling) for the "
                  "SUT-loss enforcement test; pass --compare-checkpoint to override.")
    if compare_ckpt:
        cp2 = cmp_mod._find_checkpoint(Path(compare_ckpt))
        if cp2 is None:
            print(f"[sut] compare checkpoint not found under {compare_ckpt}; skipping")
        else:
            model2, _, mtype2 = cmp_mod._build_model(rc, cp2, args.device)
            spectra2, meta2, land2_inputs, _ = collect_spectra(model2, datasets, rc, args)
            eff = {"compare_checkpoint": str(cp2)}
            for fam in FAMILY_NAMES:
                a = {m: np.stack(v) for m, v in spectra[fam].items() if v}
                b = {m: np.stack(v) for m, v in spectra2[fam].items() if v}
                ta = {m: v for m, v in a.items() if not m.endswith("(zero-shot)")}
                tb = {m: v for m, v in b.items() if not m.endswith("(zero-shot)")}
                if len(ta) < 2 or len(tb) < 2:
                    continue
                ua = universality_stats(ta, n_perm=200)
                ub = universality_stats(tb, n_perm=200)
                anc_b = machine_identity_ancova(tb, meta2, args.n0, n_perm=min(args.n_perm, 500))
                da = float(np.mean(ua["std_over_family_scale"][: args.n0]))
                db = float(np.mean(ub["std_over_family_scale"][: args.n0]))
                ea = east_containment(ta, a.get(holdout_key) if holdout_key else None)
                eb = east_containment(tb, b.get(holdout_key) if holdout_key else None)
                eff[fam] = {
                    "dispersion_primary": da, "dispersion_compare": db,
                    "dispersion_ratio_compare_over_primary": db / max(da, 1e-12),
                    "ancova_delta_R2_compare": anc_b.get("delta_R2_machine_after_physics"),
                    "holdout_inside_primary": (ea or {}).get("inside_frac"),
                    "holdout_inside_compare": (eb or {}).get("inside_frac"),
                }
            land2 = {m: resonance_landscape_sweep(model2, z, ctx, wd)
                     for m, (z, ctx, wd) in land2_inputs.items()}
            eff["landscape_primary"] = report["resonance_landscape_universality"]
            eff["landscape_compare"] = landscape_universality(land2)
            eff["interpretation"] = (
                "ratio > 1 (and/or larger compare ANCOVA delta_R2, lower compare holdout "
                "containment, larger compare landscape distance) = the model WITHOUT the SUT "
                "loss has LOOSER cross-machine spectral alignment -> the SUT loss measurably "
                "enforces operator universality, independent of its (modest) NRMSE ablation.")
            report["sut_loss_effect"] = eff

    report["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))

    print("\n=== SUT verdict (primary = operator-structure families) ===")
    for fam, v in per_family.items():
        print(f"  {fam:18s} [{v['role']:17s}] disp<0.2: {v['gate_dispersion_lt_0.2']}  "
              f"ancova dR2: {v['ancova_delta_R2'] if v['ancova_delta_R2'] is None else round(v['ancova_delta_R2'], 4)}"
              f" (small: {v['gate_ancova_identity_small']})  holdout_inside: {v['holdout_inside_frac']}")
    _v = report['verdict']
    print(f"  -> SUT SUPPORTED (STRICT: >=2 families pass BOTH gates incl. >=1 non-trivial): "
          f"{_v['sut_supported']}")
    print(f"     both-gate {_v['n_pass_both_gates']}/{len(PRIMARY_FAMILIES)}: {_v['primary_families_passing_BOTH_gates']}"
          f"  | of which non-trivial: {_v['non_trivial_families_passing_BOTH_gates']}")
    print(f"     [weaker OR-gate flag only (NOT the headline): {_v['sut_weakly_supported_or_gate_only']} "
          f"({_v['n_pass_or_gate']}/{len(PRIMARY_FAMILIES)})]")
    lu = report.get("resonance_landscape_universality", {})
    print(f"  landscape: train-pair dist={lu.get('pairwise_rms_distance_mean')}, "
          f"holdout dist={lu.get('holdout_to_train_rms_distance_mean')}")
    if "sut_extrapolation_link_by_horizon" in report:
        print("  SUT->extrap link (r>0 = spectral proximity to the universal family predicts "
              "zero-shot accuracy):")
        for h, lk in sorted(report["sut_extrapolation_link_by_horizon"].items()):
            sig = "*" if (lk["overall_p_perm"] is not None and lk["overall_p_perm"] < 0.05) else " "
            print(f"    @T{h:<3} spearman={lk['overall_spearman']:.3f}{sig} "
                  f"(p={lk['overall_p_perm']}, n={lk['n_shots']})")
        av = report.get("sut_extrapolation_link_horizon_avg")
        if av:
            sig = "*" if (av["overall_p_perm"] is not None and av["overall_p_perm"] < 0.05) else " "
            print(f"    {av['link_horizon']:<8} spearman={av['overall_spearman']:.3f}{sig} "
                  f"(p={av['overall_p_perm']}, n={av['n_shots']})  <- denoised per-shot average")
    if "sut_loss_effect" in report:
        print("  SUT-loss effect (compare/primary dispersion ratio; >1 = loss tightens alignment):")
        for fam in FAMILY_NAMES:
            v = report["sut_loss_effect"].get(fam)
            if v:
                print(f"    {fam:18s} {v['dispersion_ratio_compare_over_primary']:.2f}")
    if wd_caveat:
        print(f"  caveat: {wd_caveat}")
    print(f"Wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
