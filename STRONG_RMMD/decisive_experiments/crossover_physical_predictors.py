"""Per-machine Delta t from the raw CDFs (the C transient-timescale gate) and a pre-registered test of
whether any physical per-machine quantity predicts the D_res crossover.

Caveat: the crossover is one number per machine, so only 5 are scorable (HL2A, CMOD, NSTX, EAST, D3D;
AUGD/KSTR have no CLEAN sign change). At n=5, |Spearman| >= 0.9 is p ~ 0.04 and testing many candidates
will produce a false hit, so this is hypothesis-generation, not prediction: report every candidate and
apply Bonferroni.
"""
import argparse, json, glob, os, sys
from pathlib import Path
import numpy as np

# self-bootstrap: put the repo root (the dir holding dgknet_baseline/ and STRONG_RMMD/) on sys.path so
# `from dgknet_baseline...` works without the caller setting PYTHONPATH.
for _p in Path(__file__).resolve().parents:
    if (_p / "dgknet_baseline").is_dir():
        for _d in (str(_p), str(_p / "STRONG_RMMD")):
            if _d not in sys.path:
                sys.path.insert(0, _d)
        break

# observed D_res crossovers (omega_d axis, ablation sign-change) from buttress A_ustar; loaded from json if present
FALLBACK_XSTAR = {"HL2A": 0.2056, "CMOD": 0.2480, "NSTX": 0.7007, "EAST": 0.8184, "D3D": 1.1481}
# tabulated energy-confinement time tau_E (ms), order-of-magnitude, for the Delta t timescale gate (edit if you have better)
TAU_E_MS = {"AUGD": 150, "CMOD": 40, "D3D": 150, "EAST": 100, "HL2A": 40, "KSTR": 120, "NSTX": 50}


def _spearman(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 4:
        return float("nan"), int(m.sum())
    ru, rv = np.argsort(np.argsort(u[m])), np.argsort(np.argsort(v[m]))
    return float(np.corrcoef(ru, rv)[0, 1]), int(m.sum())


def _nonlocality(ni_traj):
    """Model-free profile-scale nonlocality proxy (SAME as buttress C): variance fraction of the 1st radial PC of
    dNI/dt over the full trajectory. High = radially coherent (nonlocal); low = incoherent (local)."""
    a = np.asarray(ni_traj, float)
    a = a.reshape(a.shape[0], -1)[:, :40]
    if a.shape[0] < 3:
        return float("nan")
    d = np.diff(a, axis=0)
    if not np.isfinite(d).all() or d.shape[0] < 2:
        return float("nan")
    w = np.clip(np.linalg.eigvalsh(np.cov(d.T)), 0.0, None)
    s = w.sum()
    return float(w.max() / s) if s > 0 else float("nan")


# Driver provenance (see the locked driver set PINJ+PCUR+gas; everything else is a CONSEQUENCE of NI and would
# leak if used as a model driver). For CROSSOVER prediction, a correlation is only a CLEAN physical claim if the
# predictor is externally set (actuator or machine geometry); consequence-of-state predictors are CIRCULAR.
_EXOGENOUS = {"PINJ", "PCUR", "SESGF", "PECH", "PRFE", "PRFI", "PLH", "GAS", "PNBI", "PBEAM"}
_GEOMETRY = {"RAXIS", "RMAJB", "RMJB", "YAXIS", "ZAXIS"}


def _provenance(feature_key):
    """EXOGENOUS (clean actuator) / GEOMETRY (clean machine property) / CONSEQUENCE (circular for a crossover claim)."""
    if feature_key.startswith("gradlen"):
        return "CONSEQUENCE(profile-derived omega_d)"
    if feature_key.startswith("derived"):
        return "MIXED(derived)"
    key = feature_key.split(":", 1)[-1].split("(")[0].strip().upper()
    if key in _EXOGENOUS:
        return "EXOGENOUS(actuator)"
    if key in _GEOMETRY:
        return "GEOMETRY(machine)"
    return "CONSEQUENCE(state)"


def _omega_d_proxy(ni_t0):
    """Model-free operating-point proxy omega_d ~ a/L_n = |d ln(NI)/d(r/a)| median over radius (normalized x in [0,1])."""
    p = np.asarray(ni_t0, float).ravel()[:40]
    if p.size < 5 or not np.all(p > 0):
        return float("nan")
    return float(np.median(np.abs(np.gradient(np.log(p), 1.0 / p.size))))


def _read_time_dt_ms(cdf_path):
    """Median physical time-step (ms) from the CDF TIME axis (for the C transient-timescale gate)."""
    try:
        import netCDF4
        ds = netCDF4.Dataset(cdf_path)
        var = next((k for k in ("TIME", "TIME3", "TIMELST") if k in ds.variables),
                   next((k for k in ds.variables if "TIME" in k.upper() and ds.variables[k].ndim == 1), None))
        if var is None:
            return float("nan")
        t = np.asarray(ds.variables[var][:], float).ravel()
        return float(np.median(np.diff(t)) * 1e3) if t.size > 2 else float("nan")
    except Exception:
        return float("nan")


def _machine_features(machine, cdf_paths, cap):
    """Per-machine physical candidate features, self-discovered from build_sample. Median over shots x time."""
    from dgknet_baseline.phases.phase0_multicdf import build_sample
    scal = {}   # global_scalars: key -> list of medians (one per shot)
    lgrad = {}  # kinetic profiles: key -> list of median |d ln p / dx| (an omega_d = a/L proxy, normalized radius)
    dts = []
    per_shot = {"nonloc": [], "omega_d": [], "pinj": []}   # WITHIN-machine: model-free per-shot arrays
    used = 0
    for p in cdf_paths[:cap]:
        dts.append(_read_time_dt_ms(p))
        try:
            s = build_sample(str(p))
            if not s or not s.get("state_trajectory"):
                continue
        except Exception:
            continue
        gs_shot, lg_shot = {}, {}
        for st in s["state_trajectory"]:
            for k, v in (st.get("global_scalars") or {}).items():
                try:
                    gs_shot.setdefault(k, []).append(float(v))
                except Exception:
                    pass
            for k, prof in (st.get("kinetic_profiles") or {}).items():
                a = np.asarray(prof, float).ravel()
                if a.size >= 5 and np.all(a > 0):
                    dx = 1.0 / a.size
                    lg_shot.setdefault(k, []).append(float(np.median(np.abs(np.gradient(np.log(a)) / dx))))
        for k, vals in gs_shot.items():
            scal.setdefault(k, []).append(float(np.median(vals)))
        for k, vals in lg_shot.items():
            lgrad.setdefault(k, []).append(float(np.median(vals)))
        # per-shot (nonlocality, omega_d, drive) -- all model-free, for the WITHIN-machine threshold test
        ni = np.asarray(s.get("ni_trajectory"), float)
        if ni.ndim == 2 and ni.shape[0] >= 3:
            per_shot["nonloc"].append(_nonlocality(ni))
            per_shot["omega_d"].append(_omega_d_proxy(ni[0]))
            per_shot["pinj"].append(float(np.median(gs_shot["PINJ"])) if "PINJ" in gs_shot else float("nan"))
        used += 1
    feats = {}
    for k, vals in scal.items():
        feats[f"scalar:{k}"] = float(np.median(vals)) if vals else float("nan")
    for k, vals in lgrad.items():
        feats[f"gradlen:{k}(omega_d~a/L)"] = float(np.median(vals)) if vals else float("nan")
    # PRE-REGISTERED derived candidates (only if constituents were found)
    def g(*names):
        for n in names:
            if f"scalar:{n}" in feats and np.isfinite(feats[f"scalar:{n}"]):
                return feats[f"scalar:{n}"]
        return float("nan")
    pinj, pvol, rmajb, raxis, pcur = g("PINJ"), g("PVOL"), g("RMAJB", "RMJB"), g("RAXIS"), g("PCUR")
    if np.isfinite(pinj) and np.isfinite(pvol) and pvol > 0:
        feats["derived:power_density(PINJ/PVOL)"] = pinj / pvol
    if np.isfinite(rmajb) and np.isfinite(raxis) and (rmajb - raxis) != 0:
        feats["derived:aspect_ratio(RMAJB/(RMAJB-RAXIS))"] = rmajb / (rmajb - raxis)
    # drive-variability (rate): std/mean of PINJ over time, pooled across shots
    if "PINJ" in scal:
        pv = np.asarray(scal["PINJ"], float)
        if np.isfinite(pv).sum() >= 3 and np.nanmean(pv) != 0:
            feats["derived:PINJ_variability(std/mean)"] = float(np.nanstd(pv) / abs(np.nanmean(pv)))
    dt_ms = float(np.nanmedian(dts)) if np.isfinite(dts).any() else float("nan")
    return feats, dt_ms, used, per_shot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdf-root", required=True, help="e.g. /scratch/gpfs/$USER/cdf (7 machine subfolders)")
    ap.add_argument("--buttress", default=str(Path(__file__).parent / "results" / "buttress.json"))
    ap.add_argument("--per-machine-cap", type=int, default=40, help="max CDFs per machine (speed)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "crossover_predictors.json"))
    a = ap.parse_args()

    # crossovers + the model-side b_m slope (the current best ordinal correlate) from buttress
    xstar = dict(FALLBACK_XSTAR); bslope = {}
    try:
        A = json.loads(Path(a.buttress).read_text())["A_ustar"]["per_machine"]
        xstar = {m: v["u_obs_dres_signchange"] for m, v in A.items()
                 if isinstance(v, dict) and isinstance(v.get("u_obs_dres_signchange"), (int, float))
                 and np.isfinite(v["u_obs_dres_signchange"])}
        bslope = {m: (v.get("beta_fit") or {}).get("b") for m, v in A.items() if isinstance(v, dict)}
    except Exception as e:
        print(f"[warn] using fallback crossovers ({e})")

    subdirs = [d for d in sorted(glob.glob(os.path.join(a.cdf_root, "*"))) if os.path.isdir(d)]
    def match(name):
        u = name.upper().replace("-", "").replace("_", "")
        for m in list(xstar) + list(TAU_E_MS):
            if m.upper() in u or u in m.upper() or u.startswith(m.upper()[:3]):
                return m
        return name

    per_machine = {}
    for d in subdirs:
        machine = match(os.path.basename(d))
        cdfs = sorted(glob.glob(os.path.join(d, "*.cdf")) + glob.glob(os.path.join(d, "*.CDF")))
        # drop the auxiliary companion files (e.g. <runid>PH.CDF -- a plasma-state/heating file with no time axis);
        # only the main time-dependent TRANSP CDF (<runid>.CDF) has the trajectory. Keeps the cap on REAL shots.
        cdfs = [c for c in cdfs if not Path(c).stem.upper().endswith("PH")]
        if not cdfs:
            print(f"[skip] {os.path.basename(d)} -> {machine}: no usable CDFs"); continue
        print(f"[read] {os.path.basename(d)} -> {machine}: {len(cdfs)} CDFs (cap {a.per_machine_cap})", flush=True)
        feats, dt_ms, used, per_shot = _machine_features(machine, cdfs, a.per_machine_cap)
        per_machine[machine] = {"n_cdfs_used": used, "dt_ms": dt_ms, "features": feats, "_per_shot": per_shot}
        print(f"        dt={dt_ms:.3g}ms, {len(feats)} features, from {used} shots", flush=True)

    # ---- (1) Delta t timescale gate for C (transient peak ~ w30-50 steps) ----
    dts = {m: v["dt_ms"] for m, v in per_machine.items() if np.isfinite(v.get("dt_ms", np.nan))}
    dt_spread = (max(dts.values()) / min(dts.values())) if len(dts) >= 2 and min(dts.values()) > 0 else float("nan")
    timescale = {m: {"dt_ms": dts[m], "peak30_50_ms": [30 * dts[m], 50 * dts[m]], "tau_E_ms": TAU_E_MS.get(m),
                     "peak_over_tauE": ([30 * dts[m] / TAU_E_MS[m], 50 * dts[m] / TAU_E_MS[m]] if TAU_E_MS.get(m) else None)}
                 for m in dts}

    # ---- (2) crossover-predictor test (n=5, PRE-REGISTERED, honest) ----
    machines = [m for m in xstar if m in per_machine]
    all_feats = sorted({f for m in machines for f in per_machine[m]["features"]})
    xv = [xstar[m] for m in machines]
    results = []
    for f in all_feats:
        fv = [per_machine[m]["features"].get(f, np.nan) for m in machines]
        rho, n = _spearman(fv, xv)
        results.append({"feature": f, "provenance": _provenance(f), "spearman_vs_crossover": rho, "n": n,
                        "values": {m: per_machine[m]["features"].get(f) for m in machines}})
    # include the model-side slope b_m for reference
    if bslope:
        bv = [bslope.get(m, np.nan) for m in machines]
        rho, n = _spearman(bv, xv)
        results.append({"feature": "MODEL:beta_slope_b_m(reference)", "spearman_vs_crossover": rho, "n": n})
    results.sort(key=lambda r: -abs(r["spearman_vs_crossover"]) if np.isfinite(r["spearman_vs_crossover"]) else 0)
    n_tested = sum(1 for r in results if np.isfinite(r["spearman_vs_crossover"]))

    # ---- (3) WITHIN-machine (POWERED, n=hundreds, model-free): does drive drive nonlocality, and is the
    # nonlocality AT each machine's crossover machine-INDEPENDENT (a universal transition threshold)? ----
    within = {}
    for m in per_machine:
        ps = per_machine[m].get("_per_shot", {})
        nl = np.asarray(ps.get("nonloc", []), float); od = np.asarray(ps.get("omega_d", []), float)
        pj = np.asarray(ps.get("pinj", []), float)
        r_od, _ = _spearman(nl, od); r_pj, _ = _spearman(nl, pj)
        e = {"n_shots": int(np.isfinite(nl).sum()), "spearman_nonloc_vs_omega_d": r_od,
             "spearman_nonloc_vs_PINJ": r_pj, "nonloc_median": float(np.nanmedian(nl)) if np.isfinite(nl).any() else None,
             "omega_d_range": ([float(np.nanmin(od)), float(np.nanmax(od))] if np.isfinite(od).any() else None)}
        xc = xstar.get(m)                                   # crossover on the MODEL omega_d axis
        good = np.isfinite(nl) & np.isfinite(od)
        if xc is not None and good.sum() >= 15:
            lo, hi = float(np.nanmin(od[good])), float(np.nanmax(od[good]))
            dist = np.abs(od[good] - xc); k = max(5, int(0.2 * good.sum()))
            e.update({"crossover_omega_d_model": float(xc), "crossover_in_CDF_omega_d_range": bool(lo <= xc <= hi),
                      "nonloc_at_crossover": float(np.median(nl[good][np.argsort(dist)[:k]]))})
        within[m] = e
    # universal-threshold read: spread of nonloc-at-crossover across machines where it is placeable IN-RANGE
    nlx = [within[m]["nonloc_at_crossover"] for m in within if within[m].get("crossover_in_CDF_omega_d_range")
           and np.isfinite(within[m].get("nonloc_at_crossover", np.nan))]
    thr_cv = float(np.std(nlx) / np.mean(nlx)) if len(nlx) >= 3 and np.mean(nlx) != 0 else float("nan")
    n_pos = sum(1 for m in within if np.isfinite(within[m]["spearman_nonloc_vs_omega_d"])
                and within[m]["spearman_nonloc_vs_omega_d"] > 0)
    n_wm = sum(1 for m in within if np.isfinite(within[m]["spearman_nonloc_vs_omega_d"]))
    within_test = {"per_machine": within, "n_machines_nonloc_RISES_with_omega_d": [n_pos, n_wm],
                   "nonloc_at_crossover_CV": thr_cv, "n_machines_crossover_in_range": len(nlx),
                   "READ": ("(a) POWERED backbone: n_machines where per-shot nonlocality rises with omega_d "
                            "(model-free, n=hundreds each) -- if most/all, drive organizes nonlocality within every "
                            "machine (a real result, no n=5 problem). (b) UNIVERSAL-THRESHOLD lead: nonloc_at_crossover_CV "
                            "small (say <0.25) across in-range machines => D_res starts helping at a machine-INDEPENDENT "
                            "nonlocality level = the law tying A+C+crossover. CAVEAT: the crossover is on the MODEL "
                            "omega_d; this places it on the model-free omega_d proxy, so it is only valid where the two "
                            "axes are ~proportional (check crossover_in_CDF_omega_d_range). Out-of-range machines are "
                            "not scored -- if most fall out of range, the proxy axis is mis-scaled and this needs the "
                            "model's per-shot omega_d (a buttress per-shot dump).")}
    # honest significance: per-test p on n=5 for |rho|>=0.9 ~ 0.037; Bonferroni threshold
    bonferroni_rho_needed = 0.9  # |rho|>=0.9 (p~0.04) is the floor even to be a per-test hit at n=5

    out = {"n_scorable_machines": len(machines), "machines": machines, "n_features_tested": n_tested,
           "HARD_CAVEAT": ("n=5 -> hypothesis-generation only. A |rho|>=0.9 is p~0.04 per test; with "
                           f"{n_tested} features tested, expect ~{n_tested*0.05:.0f} false hits at p<0.05. Any hit is "
                           "a LEAD for the theory chat + a test-on-more-machines, NOT a prediction. Do not p-hack."),
           "bonferroni_note": f"Bonferroni p<0.05 over {n_tested} tests needs p<{0.05/max(n_tested,1):.4f}; "
                              "on n=5 the max achievable p (rho=1) is ~0.017 -> essentially NOTHING survives correction.",
           "PROVENANCE_note": ("A correlation is a CLEAN physical prediction ONLY if the predictor is EXOGENOUS "
                               "(actuator: PINJ/PCUR/gas) or GEOMETRY (machine property). CONSEQUENCE features "
                               "(BETA*/ZEFF0/Q0/PVOL and the profile-derived omega_d/gradient-scale-length) are "
                               "downstream of the predicted NI, so correlating them with the crossover is CIRCULAR "
                               "(a proxy of the crossover), not prediction. Each candidate is tagged; trust only the "
                               "CLEAN ones. Same for the within-machine test: rho(nonloc,PINJ) is clean, "
                               "rho(nonloc,omega_d) is consequence-vs-consequence."),
           "crossover_used": {m: xstar[m] for m in machines},
           "ranked_candidates": results,
           "delta_t_gate": {"per_machine": timescale, "dt_spread_ratio(max/min)": dt_spread,
                            "verdict": ("commensurate -> transient-timescale story is testable" if (np.isfinite(dt_spread) and dt_spread < 3)
                                        else "cadences differ >3x -> pooled unit-step peak is NOT interpretable as one physical timescale; drop or redo C per-machine in physical time")},
           "within_machine_test": within_test}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("\n=== Delta t gate (C transient timescale) ===")
    for m, v in timescale.items():
        pk = v["peak_over_tauE"]
        print(f"  {m}: dt={v['dt_ms']:.3g}ms  w30-50={v['peak30_50_ms'][0]:.0f}-{v['peak30_50_ms'][1]:.0f}ms  "
              f"tau_E~{v['tau_E_ms']}ms  peak/tauE={'-' if pk is None else f'{pk[0]:.2f}-{pk[1]:.2f}'}")
    print(f"  dt spread (max/min) = {dt_spread:.2f}  -> {out['delta_t_gate']['verdict']}")
    print("\n=== crossover predictors (n=5, ranked; HYPOTHESIS-GENERATION ONLY) ===")
    print("  [CLEAN = EXOGENOUS/GEOMETRY (a real predictive claim);  CIRCULAR = CONSEQUENCE/derived (proxy of itself)]")
    for r in results[:14]:
        if np.isfinite(r["spearman_vs_crossover"]):
            prov = r.get("provenance", "")
            tag = "CLEAN " if prov.startswith(("EXO", "GEO")) else "circ. "
            print(f"  {tag} rho={r['spearman_vs_crossover']:+.2f}  {r['feature']:42s} [{prov}]")
    clean = [r for r in results if r.get("provenance", "").startswith(("EXO", "GEO")) and np.isfinite(r["spearman_vs_crossover"])]
    if clean:
        best = max(clean, key=lambda r: abs(r["spearman_vs_crossover"]))
        print(f"  --> best CLEAN (non-circular) predictor: {best['feature']} rho={best['spearman_vs_crossover']:+.2f} "
              f"(n=5 -> lead only)")
    print(f"\n  {out['bonferroni_note']}")
    print("\n=== WITHIN-machine (POWERED, model-free) ===")
    print("  CLEAN driver = rho(nonloc, PINJ) [external actuator];  rho(nonloc, omega_d) is CONSEQUENCE-vs-consequence (circular-ish)")
    print(f"  nonlocality RISES with omega_d in {within_test['n_machines_nonloc_RISES_with_omega_d'][0]}"
          f"/{within_test['n_machines_nonloc_RISES_with_omega_d'][1]} machines (per-shot, n=hundreds each):")
    for m, e in within.items():
        nac = e.get("nonloc_at_crossover")
        print(f"    {m}: rho(nonloc,PINJ)={e['spearman_nonloc_vs_PINJ']:+.2f} [CLEAN]  "
              f"rho(nonloc,omega_d)={e['spearman_nonloc_vs_omega_d']:+.2f} [circ]  (n={e['n_shots']})  "
              f"nonloc@crossover={'-' if nac is None else f'{nac:.3f}'} "
              f"{'(in-range)' if e.get('crossover_in_CDF_omega_d_range') else '(OUT, unscored)'}")
    print(f"  UNIVERSAL-THRESHOLD: nonloc@crossover CV = {within_test['nonloc_at_crossover_CV']:.3f} across "
          f"{within_test['n_machines_crossover_in_range']} in-range machines  (small <0.25 => a real lead)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
