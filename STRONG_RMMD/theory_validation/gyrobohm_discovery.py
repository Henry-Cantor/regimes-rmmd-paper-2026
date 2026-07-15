"""Quantify the gyro-Bohm universality claim from the SUT report(s).

Different machines sit at different operating frequencies (omega_d varies ~2.4x), yet the learned resonance-
response landscape, expressed on the gyro-Bohm axis, is compared for machine-invariance. Run with --help.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def cv(xs):
    xs = np.asarray([x for x in xs if x is not None], float)
    m = np.mean(np.abs(xs))
    return float(np.std(xs) / m) if m > 1e-9 else float("nan")


def analyze(path):
    s = json.load(open(path))
    mop = s.get("machine_operating_points", {})
    fams = s.get("families", {})
    rlu = s.get("resonance_landscape_universality", {})

    # 1. how much do the machines DIFFER physically (operating point spread)?
    omega_d = [v.get("omega_d_mean") for v in mop.values() if isinstance(v, dict)]
    omega_t = [v.get("omega_t_mean") for v in mop.values() if isinstance(v, dict)]
    op_cv = {"omega_d_cv": cv(omega_d), "omega_t_cv": cv(omega_t),
             "omega_d_range_ratio": (max(omega_d) / min(omega_d)) if omega_d and min(omega_d) > 0 else None,
             "n_machines": len(mop)}

    # 2. how much does the learned RESPONSE LANDSCAPE differ across machines + to the holdout?
    land = {"pairwise_rms_distance_mean": rlu.get("pairwise_rms_distance_mean"),
            "holdout_to_train_rms_distance_mean": rlu.get("holdout_to_train_rms_distance_mean")}

    # 3. which learned quantities are universal (operator) vs machine-specific (frequencies)?
    def med(fam):
        u = (fams.get(fam, {}) or {}).get("universality", {})
        som = u.get("std_over_mean")
        return float(np.median(som)) if som else None
    universal = {f: med(f) for f in ("res_weights", "diss_rates", "jac_mu_abs")}
    machine_specific = {f: med(f) for f in ("cons_freq_raw", "cons_freq_over_wd", "cons_freq_over_wt")}

    # 4. NON-TRIVIALITY: operating points vary a lot, landscape barely -> the collapse is not because the
    #    machines are trivially identical. Ratio = how many times tighter the response is than the operating point.
    pw = land["pairwise_rms_distance_mean"]
    nontriv_ratio = (op_cv["omega_d_cv"] / pw) if (pw and op_cv["omega_d_cv"]) else None

    return {"operating_point_spread": op_cv, "landscape_universality": land,
            "universal_operator_families": universal, "machine_specific_families": machine_specific,
            "nontriviality_ratio_opCV_over_landscapeRMS": nontriv_ratio}


def main() -> int:
    paths = sys.argv[1:] or ["STRONG_RMMD/theory_validation/results/sut_report_east.json"]
    out = {}
    for p in paths:
        try:
            out[p] = analyze(p)
        except FileNotFoundError:
            print("missing:", p); continue
    # combined verdict
    op_cvs = [r["operating_point_spread"]["omega_d_cv"] for r in out.values()]
    pws = [r["landscape_universality"]["pairwise_rms_distance_mean"] for r in out.values() if r["landscape_universality"]["pairwise_rms_distance_mean"]]
    hos = [r["landscape_universality"]["holdout_to_train_rms_distance_mean"] for r in out.values() if r["landscape_universality"]["holdout_to_train_rms_distance_mean"]]
    op_cv = float(np.nanmean(op_cvs)) if op_cvs else None
    pw = float(np.mean(pws)) if pws else None
    ho = float(np.mean(hos)) if hos else None
    # discovery is supported if machines differ materially (op CV > 0.15) AND the landscape is ~identical
    # (pairwise RMS < 0.02) AND the holdout is contained (within ~the training pairwise spread).
    supported = bool(op_cv and op_cv > 0.15 and pw is not None and pw < 0.02 and ho is not None and ho < 3 * pw)
    out["VERDICT"] = {
        "machines_differ_operating_point_cv": op_cv,
        "landscape_pairwise_rms": pw,
        "holdout_landscape_rms": ho,
        "gyrobohm_universality_supported_NONtrivial": supported,
        "interpretation": ("NON-TRIVIAL gyro-Bohm universality: machines differ materially in operating "
                           "frequency yet the normalized resonance response collapses to one curve and "
                           "contains the held-out machine -> the learned operator confirms gyro-Bohm "
                           "similarity and uses it for zero-shot transfer. STILL CHECK (physics, for the "
                           "paper): does the universal response SHAPE match the theoretical gyro-Bohm "
                           "prediction, not just collapse? That is the difference between 'consistent with' "
                           "and 'confirms' -- a domain expert call."
                           if supported else
                           "NOT clearly non-trivial from the report alone -- either machines are too similar "
                           "in operating point or the landscape spread is not small enough; investigate before "
                           "claiming a gyro-Bohm discovery."),
    }
    from pathlib import Path as _P
    _op = _P(paths[0]).parent / "gyrobohm_discovery.json"
    _op.write_text(json.dumps(out, indent=1, default=float)); print("wrote", _op)
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
