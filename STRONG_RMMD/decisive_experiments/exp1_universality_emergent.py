"""EXP-1 -- is the operator universality emergent or imposed?

The committed model shares one operator across machines, so this experiment trains separate models per
machine (no shared parameters, multiple seeds), extracts each operator's basis-invariant eigenvalue spectrum,
and tests whether the spectra diverge raw and collapse only under the correct gyro-Bohm normalization.
Decision statistic F = between-machine variance / within-machine (seed) variance, with bootstrap CIs.
Run with --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RATE_FAMILIES = ("cons_freq_raw", "diss_rates")   # these are RATES (get GB-normalized); offdiag_frac is a scalar


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
    sut = imp("theory_validation/sut_confirmation.py", "sut_confirmation")
    return rc, cmp_mod, sut


def _find_ckpt(d: Path):
    for c in ("checkpoint_best.pt", "checkpoint_best.pt.gz"):
        if (d / c).exists():
            return d / c
    return None


# Known per-machine physical radii (m). The compact datasets do NOT carry a_minor/Te, so
# compute_resonance_frequencies falls back to CONSTANT defaults -> a constant tau_GB across machines ->
# the N1/N3 comparison is degenerate. The minor radius IS the gyro-Bohm length scale; c_s reference is a
# global constant that cancels in the between/within variance RATIO, so tau_GB \propto a_minor per machine.
A_MINOR = {"CMOD": 0.21, "D3D": 0.67, "HL2A": 0.22, "KSTR": 0.5, "NSTX": 0.67, "EAST": 0.45, "AUGD": 0.5}
R_MAJOR = {"CMOD": 0.68, "D3D": 1.67, "HL2A": 0.67, "KSTR": 1.8, "NSTX": 0.85, "EAST": 1.75, "AUGD": 1.65}


def collect_one(rc, cmp_mod, sut, ckpt, val_path, device, machine):
    """Return per-shot spectra + per-machine physical scale for ONE independently trained model."""
    model, norm, _ = cmp_mod._build_model(rc, ckpt, device); model.eval()
    payload = rc._load_phase0_dataset(Path(val_path))
    n = rc._ensure_normalization_stats(Path(val_path), checkpoint_dir=None, require=False)
    ds = rc.CompactRolloutDataset(payload, max_time=2, normalization_stats=n)
    args = SimpleNamespace(device=device, top_modes=8, jacobian=False, max_shots_per_machine=0,
                           link_horizon=1, link_horizons=[1])
    spectra, shot_meta, _, _ = sut.collect_spectra(model, [("indist", ds, n)], rc, args)
    # one machine key inside spectra (in-dist -> raw machine name)
    key = next(iter(spectra["cons_freq_raw"]), None)
    out = {"a_minor": A_MINOR.get(machine, 0.5), "r_major": R_MAJOR.get(machine, 1.0),
           "n_shots": len(ds), "offdiag_frac": float(np.median([m["offdiag_frac"] for m in shot_meta[key]])) if key else None}
    for fam in RATE_FAMILIES + ("res_gammas",):
        arr = np.asarray(spectra[fam][key]) if key and spectra[fam][key] else np.zeros((0, args.top_modes))
        # sorted spectrum per shot (descending magnitude) -> shot-mean spectrum
        srt = np.sort(np.abs(arr), axis=1)[:, ::-1] if arr.size else arr
        out[fam] = {"shot_mean_spectrum": srt.mean(0).tolist() if srt.size else [],
                    "shot_spectra": srt.tolist() if srt.size else []}
    return out


def _hungarian(a, b):
    """OT/Hungarian L2 matching distance between two eigenvalue sets (scipy if available, else sorted-L2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = min(len(a), len(b))
    if m == 0:
        return float("nan")
    a, b = a[:m], b[:m]
    try:
        from scipy.optimize import linear_sum_assignment
        C = np.abs(a[:, None] - b[None, :])
        r, c = linear_sum_assignment(C)
        return float(np.sqrt((C[r, c] ** 2).mean()))
    except Exception:
        return float(np.sqrt(((np.sort(a)[::-1] - np.sort(b)[::-1]) ** 2).mean()))


def variance_ratio(per_ms, fam, tau_map):
    """F = between_machine_var / within_machine(seed)_var on the tau-normalized shot-mean spectrum (summarized
    by its L2 norm). per_ms[machine][seed] = collect_one output. tau_map[machine] = normalization time."""
    machines = sorted(per_ms)
    machine_vals = {}   # machine -> list over seeds of a scalar summary
    for mach in machines:
        vals = []
        for seed, rec in per_ms[mach].items():
            spec = np.asarray(rec[fam]["shot_mean_spectrum"], float)
            if spec.size == 0:
                continue
            vals.append(float(np.linalg.norm(spec * tau_map[mach])))     # normalize the RATE by tau
        if vals:
            machine_vals[mach] = vals
    if len(machine_vals) < 2:
        return {"F": None, "note": "need >=2 machines with spectra"}
    means = np.array([np.mean(v) for v in machine_vals.values()])
    within = np.mean([np.var(v, ddof=1) if len(v) > 1 else 0.0 for v in machine_vals.values()])
    between = float(np.var(means, ddof=1))
    F = float(between / (within + 1e-12)) if within > 0 else float("inf")
    # bootstrap CI over seeds
    rng = np.random.default_rng(0); Fs = []
    keys = list(machine_vals)
    for _ in range(1000):
        bs_means, bs_within = [], []
        for k in keys:
            v = np.array(machine_vals[k]); idx = rng.integers(0, len(v), len(v))
            bs_means.append(v[idx].mean()); bs_within.append(np.var(v[idx], ddof=1) if len(v) > 1 else 0.0)
        wi = np.mean(bs_within); Fs.append(np.var(bs_means, ddof=1) / (wi + 1e-12) if wi > 0 else np.nan)
    Fs = np.array([f for f in Fs if np.isfinite(f)])
    ci = [float(np.percentile(Fs, 2.5)), float(np.percentile(Fs, 97.5))] if Fs.size else [None, None]
    # sorted-spectrum + OT distance between machine-mean spectra (permutation-invariant)
    ms = {k: np.asarray(per_ms[k][list(per_ms[k])[0]][fam]["shot_mean_spectrum"], float) * tau_map[k] for k in keys}
    pair_l2, pair_ot = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ms[keys[i]], ms[keys[j]]
            if a.size and b.size:
                pair_l2.append(float(np.linalg.norm(np.sort(a)[::-1] - np.sort(b)[::-1])))
                pair_ot.append(_hungarian(a, b))
    return {"F": F, "F_CI95": ci, "between_var": between, "within_seed_var": float(within),
            "mean_sorted_L2_between_machines": float(np.mean(pair_l2)) if pair_l2 else None,
            "mean_OT_between_machines": float(np.mean(pair_ot)) if pair_ot else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--machines", nargs="*", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rc, cmp_mod, sut = _imports()

    per_ms, missing = {}, []
    for mach in args.machines:
        per_ms[mach] = {}
        for seed in args.seeds:
            ck = _find_ckpt(Path(args.models_root) / mach / f"seed{seed}")
            val = Path(args.data_root) / mach / "dataset_val_compact.pt"
            if ck is None or not val.exists():
                missing.append(f"{mach}/seed{seed} (ckpt={ck}, val_exists={val.exists()})"); continue
            per_ms[mach][seed] = collect_one(rc, cmp_mod, sut, ck, val, args.device, mach)
        if not per_ms[mach]:
            del per_ms[mach]

    a_minor = {m: per_ms[m][list(per_ms[m])[0]]["a_minor"] for m in per_ms}
    r_major = {m: per_ms[m][list(per_ms[m])[0]]["r_major"] for m in per_ms}
    # four normalization conditions (tau_GB \propto a_minor; c_s reference cancels in the F ratio)
    norms = {
        "N0_raw": {m: 1.0 for m in per_ms},
        "N1_gyrobohm": {m: a_minor[m] for m in per_ms},                    # a/c_s  (correct GB length scale)
        "N2_wrong_R_over_cs": {m: r_major[m] for m in per_ms},             # R/c_s  (WRONG: major radius)
    }
    mach_order = list(per_ms)
    shifted = mach_order[1:] + mach_order[:1]
    norms["N3_scrambled"] = {m: a_minor[shifted[i]] for i, m in enumerate(mach_order)}   # permuted a_minor

    report = {"machines": list(per_ms), "seeds": args.seeds, "missing": missing,
              "a_minor_per_machine": {m: float(a_minor[m]) for m in per_ms},
              "r_major_per_machine": {m: float(r_major[m]) for m in per_ms},
              "shot_counts": {m: {s: per_ms[m][s]["n_shots"] for s in per_ms[m]} for m in per_ms},
              "offdiag_frac_per_model": {m: {s: per_ms[m][s]["offdiag_frac"] for s in per_ms[m]} for m in per_ms},
              "per_family": {}}
    for fam in RATE_FAMILIES:
        report["per_family"][fam] = {nm: variance_ratio(per_ms, fam, tau) for nm, tau in norms.items()}

    # ---- DECISION RULE (spec step 7), on diss_rates (the dissipation spectrum) ----
    def Fof(fam, nm):
        return report["per_family"].get(fam, {}).get(nm, {})
    fam = "diss_rates"
    N0 = Fof(fam, "N0_raw"); N1 = Fof(fam, "N1_gyrobohm"); N2 = Fof(fam, "N2_wrong_R_over_cs"); N3 = Fof(fam, "N3_scrambled")
    def lo(x): return (x.get("F_CI95") or [None, None])[0]
    def hi(x): return (x.get("F_CI95") or [None, None])[1]
    enough = all(x.get("F") is not None for x in (N0, N1, N2, N3))
    supported = bool(enough and (lo(N0) or 0) > 1 and (lo(N1) or 9) <= 1.5 and (lo(N2) or 0) > 1 and (lo(N3) or 0) > 1)
    refuted = bool(enough and ((lo(N1) or 0) > 1.5 or (hi(N2) or 9) <= 1.2 or (hi(N3) or 9) <= 1.2))
    report["VERDICT_1"] = {
        "prediction_if_true": "F(N0) raw > 1 (diverge); F(N1) GB ~1 (collapse to seed noise); F(N2),F(N3) > 1 (wrong norms don't collapse).",
        "refuted_if": "F(N1) still >>1 (no real collapse) OR N2/N3 collapse ~as well as N1 (trivial variance-reduction, not physics).",
        "F_diss_rates": {"N0_raw": N0.get("F"), "N1_gyrobohm": N1.get("F"), "N2_wrong": N2.get("F"), "N3_scrambled": N3.get("F")},
        "F_CI95": {"N0": N0.get("F_CI95"), "N1": N1.get("F_CI95"), "N2": N2.get("F_CI95"), "N3": N3.get("F_CI95")},
        "VERDICT": ("SUPPORTED" if supported else ("REFUTED" if refuted else "INCONCLUSIVE")),
        "note": "INCONCLUSIVE if seed noise swamps between-machine signal (overfitting on small-data machines) -> "
                "report shot_counts + per-model offdiag_frac and STOP (spec step 7). If offdiag_frac now DIFFERS "
                "across independently-trained models (it was 0.9802 identical when shared), that ALONE shows the "
                "prior identity was an artifact of sharing.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "universality_emergent.json").write_text(json.dumps(report, indent=1, default=float))
    print("missing:", missing)
    print("offdiag_frac_per_model (was 0.9802 identical under sharing):")
    print(json.dumps(report["offdiag_frac_per_model"], indent=1, default=float))
    print("VERDICT_1:", json.dumps(report["VERDICT_1"], indent=1, default=float))
    print("wrote", RESULTS / "universality_emergent.json")


if __name__ == "__main__":
    main()
