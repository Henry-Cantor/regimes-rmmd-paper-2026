"""EXP-5 quartile skill -- post-hoc patch for universality_predictive.json (no model rollout).

The main LOMO analysis stores the per-quartile model NRMSE but not per-quartile persistence, so it cannot
report skill vs persistence on the q4 (active) shots. Persistence needs no model (||ni_t0 - ni_traj[h]|| /
rms(target) per shot), so this adds the per-quartile persistence and skill. Run with --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
HORIZONS = [1, 20, 50, 100]
QUARTS = ("q1", "q2", "q3", "q4")


def _imports():
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)

    def imp(rel, name):
        spec = importlib.util.spec_from_file_location(name, strong / rel)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    rc = imp("training/rmmd_train_eval_impl.py", "rmmd_train_eval_impl")
    ex = imp("theory_validation/extrap_strong.py", "extrap_strong")
    return rc, ex


def persistence_quartiles(rc, ex, evp):
    """NO model. Per-shot persistence NRMSE at each horizon -> activity-stratified persistence per quartile.
    Uses ex.activity_stratified (the SAME rank-based binning the analysis used), so these quartiles line up with
    the model_nrmse quartiles already in the JSON."""
    n = rc._ensure_normalization_stats(Path(evp), checkpoint_dir=None, require=False)
    ds = rc.CompactRolloutDataset(rc._load_phase0_dataset(Path(evp)), max_time=max(HORIZONS), normalization_stats=n)
    acc = {h: {"nrmse": [], "pers": [], "shots": [], "machine": [], "geom_nrmse": [], "geom_pers": [], "gate": []}
           for h in HORIZONS}
    for i in range(len(ds)):
        s = ds[i]
        traj = s["ni_traj"]; T = int(traj.shape[0])
        if T < 1:
            continue
        t0 = s["ni_t0"].numpy()
        for h in HORIZONS:
            if h > T:
                continue
            pnr, _ = rc._normalized_rmse_mae(t0, traj[h - 1].numpy())    # persistence = no-change prediction (ni_t0)
            acc[h]["pers"].append(pnr); acc[h]["nrmse"].append(pnr)     # dummy nrmse=pers; only persistence is read
            acc[h]["shots"].append(i); acc[h]["machine"].append(s.get("machine"))
    strat = ex.activity_stratified(acc, HORIZONS)
    return {str(h): {q: (strat.get(str(h), {}).get(q, {}) or {}).get("persistence_nrmse") for q in QUARTS}
            for h in HORIZONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--machines", nargs="*", required=True)
    ap.add_argument("--holdout", nargs="*", default=[], metavar="name:path", help="EAST:<east.pt> AUGD:<augd.pt>")
    ap.add_argument("--json", default=str(RESULTS / "universality_predictive.json"),
                    help="the universality_predictive.json to PATCH in place")
    args = ap.parse_args()
    rc, ex = _imports()

    evsets = [(m, Path(args.data_root) / f"holdout_{m}" / f"eval_{m}.pt") for m in args.machines]
    evsets += [(spec.split(":", 1)[0], Path(spec.split(":", 1)[1])) for spec in args.holdout]

    pers = {}
    for m, evp in evsets:
        if not evp.exists():
            print(f"[{m}] SKIP ({evp} missing)"); continue
        pers[m] = persistence_quartiles(rc, ex, evp)      # seconds -- no model, no rollout

    # ---- patch the JSON: add per-quartile persistence + skill (= 1 - model/pers) using the model_nrmse in place ----
    jp = Path(args.json)
    report = json.loads(jp.read_text()) if jp.exists() else {"per_machine": {}}
    skill_by_m = {}
    for m, pq in pers.items():
        r = report.get("per_machine", {}).get(m)
        model_q = (r or {}).get("correct_quartiles", {}) if isinstance(r, dict) else {}
        skill = {}
        for h in HORIZONS:
            skill[str(h)] = {}
            for q in QUARTS:
                pv = (pq.get(str(h)) or {}).get(q)
                mv = (model_q.get(str(h)) or {}).get(q)
                skill[str(h)][q] = (1.0 - mv / pv) if (pv and mv is not None) else None
        skill_by_m[m] = skill
        if isinstance(r, dict):
            r["correct_quartile_persistence"] = pq
            r["correct_quartile_skill"] = skill
    if jp.exists():
        jp.write_text(json.dumps(report, indent=1, default=float))
        print(f"patched {jp} with correct_quartile_persistence + correct_quartile_skill for {sorted(pers)}\n")

    # ---- print the honest active-shot metric: q3/q4 skill (>0 = beats persistence on the dynamic shots) ----
    print("skill_vs_persistence @T50 by activity quartile (q1=quiescent .. q4=most dynamic; >0 beats persistence):")
    for m, skill in skill_by_m.items():
        q = skill.get("50", {})
        row = "  ".join(f"{k}={q[k]:+.3f}" if q.get(k) is not None else f"{k}=--" for k in QUARTS)
        print(f"  [{m}] {row}")


if __name__ == "__main__":
    main()
