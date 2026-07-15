#!/usr/bin/env python3
"""Zero-shot extrapolation suite plus STRONG constant fitting. For every supplied model (RMMD, the structural
ablations, and the baselines) this evaluates, with the same semantics as comparison/run_comparison.py:

  1. In-distribution test NRMSE(T) with persistence, bootstrap CIs, and paired Wilcoxon tests.
  2. Holdout zero-shot NRMSE(T) (the holdout is an unseen machine id, so the model adapts only via geometry,
     resonance frequencies, and drivers).
  3. The extrapolation gap (holdout/in-dist ratio per horizon) and the ablation-on-zero-shot table.
  4. Per-machine in-distribution breakdown for the reference model.
  5. STRONG fitting: a separate post-hoc overlay (not used by 1-3) fitting per-machine mean error to
     geometry and profile differences.

Run with --help.
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
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    rc = _import_module(strong / "training" / "rmmd_train_eval_impl.py", "rmmd_train_eval_impl")
    cmp_mod = _import_module(strong / "comparison" / "run_comparison.py", "comparison_run_comparison")
    return rc, cmp_mod


@torch.no_grad()
def eval_dataset(rc, model, dataset, horizons, device, norm_stats, max_shots):
    """Like run_comparison._eval_model but ALSO records machine + persistence per shot
    (needed for the per-machine/STRONG tables) AND geometry-tracking NRMSE: the rollout
    feeds its own predicted geometry back as conditioning, so geometry error is both a
    supplementary forecast metric and a rollout-stability check (geometry is 20% of the
    training loss; abl_geometry shows up here directly). Same rollout, same metric."""
    acc = {h: {"nrmse": [], "shots": [], "machine": [], "pers": [],
               "geom_nrmse": [], "geom_pers": [], "gate": []} for h in horizons}
    n = min(len(dataset), max_shots) if max_shots else len(dataset)
    max_h = max(horizons)
    for i in range(n):
        s = dataset[i]
        T = int(s["ni_traj"].shape[0])
        if T < 1:
            continue
        ni_preds, geom_preds = rc._rollout_compact_shot_to_checkpoints(
            model, s["ni_t0"], s["geom_t0"], s["pre_shot_context"], s["limiter_geometry_tensor"],
            s["ni_traj"], s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}),
            device, norm_stats, max_time_step=min(max_h, T),
            drivers_traj=s.get("drivers_traj"), report_horizons=horizons)
        # DgknetHybrid gate for THIS shot (last forward); nan for models without a gate. Instruments
        # whether the gate opens on dynamic shots and stays ~0 on quiet (the bleed check).
        gate = float(getattr(model, "last_gate_mean", float("nan")))
        for h in horizons:
            if h > T or h not in ni_preds:
                continue
            tgt = s["ni_traj"][h - 1].numpy()
            nr, _ = rc._normalized_rmse_mae(ni_preds[h].numpy(), tgt)
            pnr, _ = rc._normalized_rmse_mae(s["ni_t0"].numpy(), tgt)
            acc[h]["nrmse"].append(nr); acc[h]["shots"].append(i); acc[h]["gate"].append(gate)
            acc[h]["machine"].append(s["machine"]); acc[h]["pers"].append(pnr)
            if h in geom_preds and s["geom_traj"].shape[0] >= h:
                g_tgt = s["geom_traj"][h - 1].numpy()
                gnr, _ = rc._normalized_rmse_mae(geom_preds[h].numpy(), g_tgt)
                gpn, _ = rc._normalized_rmse_mae(s["geom_t0"].numpy(), g_tgt)
                acc[h]["geom_nrmse"].append(gnr); acc[h]["geom_pers"].append(gpn)
    return acc


def summarize(acc, horizons, cmp_mod, ref_acc=None):
    out = {}
    for h in horizons:
        vals = acc[h]["nrmse"]
        if not vals:
            out[h] = None; continue
        mean = float(np.mean(vals)); lo, hi = cmp_mod._boot_ci(vals)
        pers = float(np.mean(acc[h]["pers"])) if acc[h]["pers"] else None
        e = {"nrmse": mean, "ci95": [lo, hi], "n": len(vals),
             "persistence": pers,
             "skill_vs_persistence": (1.0 - mean / pers) if pers else None}
        if acc[h].get("geom_nrmse"):
            gm = float(np.mean(acc[h]["geom_nrmse"]))
            gp = float(np.mean(acc[h]["geom_pers"])) if acc[h]["geom_pers"] else None
            e["geom_nrmse"] = gm
            e["geom_persistence"] = gp
            e["geom_skill_vs_persistence"] = (1.0 - gm / gp) if gp else None
        if ref_acc is not None and ref_acc[h]["nrmse"]:
            ref_map = dict(zip(ref_acc[h]["shots"], ref_acc[h]["nrmse"]))
            mod_map = dict(zip(acc[h]["shots"], acc[h]["nrmse"]))
            p, mdiff, ncom = cmp_mod._paired_pvalue(ref_map, mod_map)
            e["vs_reference"] = {"p_value": p, "mean_diff_vs_ref": mdiff, "n_paired": ncom,
                                 "significant_0.05": (p is not None and p < 0.05)}
        out[h] = e
    return out


def activity_stratified(acc, horizons, n_min: int = 3):
    """Skill vs persistence stratified by SHOT ACTIVITY (per-shot persistence NRMSE = how much
    the true profile actually moves). THE honest zero-shot metric on quasi-stationary machines
    (EAST): pooled persistence skill is dominated by flat shots where there is nothing to
    predict; the model's value shows on the ACTIVE quartiles. Standard forecasting practice
    (activity/event-stratified skill scores)."""
    out = {}
    for h in horizons:
        pers = np.asarray(acc[h]["pers"], dtype=np.float64)
        mod = np.asarray(acc[h]["nrmse"], dtype=np.float64)
        if pers.size < 4 * n_min:
            continue
        # Rank-based equal-count quartiles (not value-based): on quasi-stationary machines many flat shots tie
        # at near-zero persistence NRMSE, so value thresholds leave q1 empty. Ranking splits into 4 equal groups.
        # 0 = most stationary .. 3 = most dynamic.
        ranks = np.empty(pers.size, dtype=np.int64)
        ranks[np.argsort(pers, kind="mergesort")] = np.arange(pers.size)
        bins = np.minimum((ranks * 4) // pers.size, 3)
        gate = np.asarray(acc[h].get("gate", []), dtype=np.float64)
        ent = {}
        for q in range(4):
            sel = bins == q
            if sel.sum() < n_min:
                continue
            mp, pp = float(mod[sel].mean()), float(pers[sel].mean())
            ent[f"q{q + 1}"] = {"n": int(sel.sum()), "model_nrmse": mp, "persistence_nrmse": pp,
                                 "skill_vs_persistence": (1.0 - mp / pp) if pp > 1e-9 else None}
            # gate_mean per quartile: the dgknet-hybrid instrumentation. WANT: ~0 on q1/q2 (no bleed),
            # rising on q4 (the gate discriminates). nan for models without a gate.
            if gate.size == pers.size and np.isfinite(gate[sel]).any():
                ent[f"q{q + 1}"]["gate_mean"] = float(np.nanmean(gate[sel]))
        out[str(h)] = ent
    return out


# ---------------------------------------------------------------------------
# STRONG feature extraction + fitting
# ---------------------------------------------------------------------------
def machine_features(rc, dataset, norm_stats, max_shots):
    """Per-machine means of: geometry tensors, dimensionless beta-vector, edge density."""
    feats = defaultdict(lambda: {"geom": [], "beta": [], "n_edge": [], "n": 0})
    n = min(len(dataset), max_shots) if max_shots else len(dataset)
    for i in range(n):
        s = dataset[i]
        m = s["machine"]
        ni_phys = rc._denormalize_ni_batch(s["ni_t0"].unsqueeze(0), norm_stats)[0].numpy()
        diag = rc.compute_resonance_frequencies({"NI": ni_phys}, m, s.get("pre_shot_scalars", {}) or {})
        # Features must be in physical units: each dataset normalizes with its own stats, so normalized-space
        # distances across datasets are incommensurable.
        geom_phys = rc._denormalize_geometry_batch(s["geom_t0"].unsqueeze(0), norm_stats)[0].numpy()
        feats[m]["geom"].append(np.concatenate([geom_phys.ravel(),
                                                s["limiter_geometry_tensor"].numpy().ravel()]))
        feats[m]["beta"].append([diag["rho_star"], diag["q"], diag["aspect_ratio"],
                                 diag["omega_t"], diag["omega_d"]])
        feats[m]["n_edge"].append(float(ni_phys[-1]))
        feats[m]["n"] += 1
    return {m: {"geom_mean": np.mean(np.stack(v["geom"]), axis=0),
                "beta_mean": np.mean(np.asarray(v["beta"]), axis=0),
                "n_edge_mean": float(np.mean(v["n_edge"])), "n": v["n"]}
            for m, v in feats.items()}


def strong_fit(per_machine_err: dict, feats: dict, horizons, holdout_label: str | None):
    """Least-squares fit of the STRONG (Theorem 4) form over (machine, horizon).
    Fit on TRAINING machines only; predict the held-out machine if present.

    NOTE — this is a POST-HOC INTERPRETIVE OVERLAY, not part of the extrapolation eval:
    the zero-shot NRMSE in report['models'][m]['holdout'] is the RAW model rollout and uses
    NO fitting. strong_fit asks the separate question "is that zero-shot error PREDICTABLE
    from the holdout's dimensionless distance to the training pool?" (Theorem 4 / the SUT
    C2 term). The measured-vs-predicted comparison is reported side by side; the measured
    number stands on its own with or without this fit."""
    train_m = [m for m in per_machine_err if m != holdout_label]
    if len(train_m) < 3:
        return {"error": f"need >=3 training machines, got {train_m}"}
    pool_geom = np.mean(np.stack([feats[m]["geom_mean"] for m in train_m]), axis=0)
    betas = np.stack([feats[m]["beta_mean"] for m in train_m])
    b_mu, b_sd = betas.mean(axis=0), betas.std(axis=0) + 1e-12
    pool_edge = float(np.mean([feats[m]["n_edge_mean"] for m in train_m]))

    # Scale features to O(1) across the training pool so the fitted C's are comparable
    # and the lstsq is well-conditioned (features are physical units of wildly different scale).
    d2f_raw = {m: float(np.sum((feats[m]["geom_mean"] - pool_geom) ** 2)) / pool_geom.size
               for m in feats}
    d2f_scale = max(np.mean([d2f_raw[m] for m in train_m]), 1e-30)
    edge_scale = max(np.mean([abs(feats[m]["n_edge_mean"] - pool_edge) for m in train_m]), 1e-30)

    def x_of(m):
        d2f = d2f_raw[m] / d2f_scale
        dbeta = float(np.linalg.norm((feats[m]["beta_mean"] - b_mu) / b_sd))
        dn = abs(feats[m]["n_edge_mean"] - pool_edge) / edge_scale
        return d2f, dbeta, dn

    Hs = sorted(horizons)
    rows, y, tags = [], [], []
    for m in train_m:
        d2f, dbeta, dn = x_of(m)
        for hi, h in enumerate(Hs):
            err = per_machine_err[m].get(h)
            if err is None:
                continue
            onehot = [1.0 if j == hi else 0.0 for j in range(len(Hs))]   # E_int(T) intercepts
            t_sc = h / float(max(Hs))
            rows.append(onehot + [d2f * t_sc, dbeta * t_sc, dn])
            y.append(err); tags.append((m, h))
    A = np.asarray(rows); yv = np.asarray(y)
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((yv - pred) ** 2)); ss_tot = float(np.sum((yv - yv.mean()) ** 2)) + 1e-12
    nH = len(Hs)
    out = {
        "horizons": Hs, "training_machines": train_m,
        "E_intrinsic_per_horizon": dict(zip(map(str, Hs), coef[:nH].tolist())),
        "C1_geometry": float(coef[nH]), "C2_dimensionless": float(coef[nH + 1]),
        "C3_edge_density": float(coef[nH + 2]),
        "C4_equilibrium": None,
        "C4_note": "N/A: current model has no MHD projection layer (no eps_eq measurable).",
        "fit_R2": 1.0 - ss_res / ss_tot,
        "features_per_machine": {m: dict(zip(["d2F", "dbeta", "dn_edge"], x_of(m))) for m in per_machine_err},
        "residuals": {f"{m}@T{h}": float(r) for (m, h), r in zip(tags, (yv - pred).tolist())},
    }
    if holdout_label and holdout_label in per_machine_err and holdout_label in feats:
        d2f, dbeta, dn = x_of(holdout_label)
        # A geometrically extreme holdout (e.g. spherical NSTX) sits FAR outside the conventional-
        # tokamak training pool; z-scoring by the small training spread makes its distances ~1e10
        # and the LINEAR extrapolation explode. Cap the distances used for PREDICTION (report the
        # UNCAPPED values as the out-of-range indicator) and clip the prediction to a physical range.
        CAPZ = 6.0
        d2f_p, dbeta_p, dn_p = min(d2f, CAPZ), min(dbeta, CAPZ), min(dn, CAPZ)
        out_of_range = bool(max(d2f, dbeta, dn) > CAPZ)
        _meas = [per_machine_err[holdout_label].get(h) for h in Hs if per_machine_err[holdout_label].get(h) is not None]
        _clip = max(3.0, 2.0 * (max(_meas) if _meas else 1.0))
        pred_e, meas_e = {}, {}
        for hi, h in enumerate(Hs):
            p = coef[hi] + coef[nH] * d2f_p * (h / max(Hs)) + coef[nH + 1] * dbeta_p * (h / max(Hs)) + coef[nH + 2] * dn_p
            pred_e[str(h)] = float(min(max(p, 0.0), _clip))
            if per_machine_err[holdout_label].get(h) is not None:
                meas_e[str(h)] = float(per_machine_err[holdout_label][h])
        out["holdout_prediction"] = {"predicted_nrmse": pred_e, "measured_nrmse": meas_e,
                                     "out_of_training_range": out_of_range,
                                     "holdout_feature_distances": {"d2F": float(d2f), "dbeta": float(dbeta), "dn_edge": float(dn)},
                                     "note": "constants fit on training machines ONLY (STRONG out-of-fit test). "
                                             "Distances z-scored by training spread; if out_of_training_range, "
                                             "the holdout is geometrically far beyond the pool (e.g. spherical "
                                             "NSTX) so the prediction is capped (z<=6) + clipped to a physical "
                                             "NRMSE range — read it as 'beyond the linear fit's support', not a tight estimate."}
    out["feature_note"] = ("features in PHYSICAL units (denormalized geometry/NI), scaled by "
                           "training-pool means -> C's comparable; fixes the cross-dataset "
                           "normalization incommensurability of the first run")
    out["protocol_caveat"] = (
        "Training-machine errors are IN-DISTRIBUTION (those machines were trained on), so the "
        "C-constants are fit on a proxy for transfer; the holdout out-of-fit prediction is the "
        "real test. A full leave-one-machine-out (5 retrains) is the "
        "stricter protocol and is deferred (compute); state this in the paper.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indist-data", default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt")
    ap.add_argument("--holdout-data", "--east-data", dest="holdout_data", default=None,
                    help="held-out machine's compact dataset (zero-shot; D3D in the final design). "
                         "--east-data is a backward-compatible alias for this flag.")
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--models", nargs="*", default=[], help="extra label=dir entries (e.g. headline=...)")
    ap.add_argument("--models-json", default=None, help="JSON {label: dir} (LR winners) — merged in")
    ap.add_argument("--horizons", type=int, nargs="*",
                    default=[1, 2, 3, 5, 8, 12, 16, 20, 32, 50, 75, 100])
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--skip-indist", action="store_true",
                    help="skip the in-distribution eval; do ONLY the holdout zero-shot (fast "
                         "preliminary). Disables extrap-gap, STRONG fit, per-machine in-dist.")
    ap.add_argument("--reference", default="full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "results" / "extrap_strong_report.json"))
    args = ap.parse_args()

    rc, cmp_mod = _imports()
    t0 = time.time()
    horizons = sorted(set(args.horizons))

    entries: dict[str, Path] = {}
    if args.ckpt_root:
        for d in sorted(Path(args.ckpt_root).iterdir()):
            if d.is_dir() and cmp_mod._find_checkpoint(d):
                entries[d.name] = d
    if args.models_json:
        mj = Path(args.models_json)
        if mj.exists():
            for label, path in json.loads(mj.read_text()).items():
                entries[label] = Path(path)
        else:
            print(f"[extrap] --models-json {mj} not found; ignoring it and using --ckpt-root "
                  "discovery (fine for a preliminary run before the LR grid finishes).", flush=True)
    for kv in args.models:
        label, _, path = kv.partition("=")
        entries[label] = Path(path)
    if not entries:
        raise SystemExit("No models. Pass --ckpt-root and/or --models/--models-json.")

    def load_ds(path):
        payload = rc._load_phase0_dataset(Path(path))
        norm = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        return rc.CompactRolloutDataset(payload, max_time=max(horizons), normalization_stats=norm), norm

    ds_in, norm_in = (load_ds(args.indist_data) if not args.skip_indist else (None, None))
    ds_hold, norm_hold = (load_ds(args.holdout_data) if args.holdout_data else (None, None))
    if ds_in is None and ds_hold is None:
        raise SystemExit("Nothing to evaluate: --skip-indist is set AND no --holdout-data given.")
    print(f"[extrap] models={list(entries)}"
          + (f"  indist={len(ds_in)} shots" if ds_in is not None else "  (in-dist SKIPPED)")
          + (f"  holdout={len(ds_hold)} shots" if ds_hold is not None else "  (no holdout data)"), flush=True)

    raw = {}
    for label, d in entries.items():
        cp = cmp_mod._find_checkpoint(d)
        if cp is None:
            print(f"[extrap] {label}: no checkpoint, skipping", flush=True); continue
        model, norm_ck, mtype = cmp_mod._build_model(rc, cp, args.device)
        r = {"model_type": mtype, "checkpoint": str(cp),
             "n_params": sum(p.numel() for p in model.parameters())}
        # RIGOR: rollout normalization stats are used ONLY to denormalize NI for the omega
        # computation, so they must be the stats the DATASET was normalized with (the holdout
        # carries its OWN stats, NOT the checkpoint's training stats — using checkpoint stats
        # there would yield wrong physical profiles -> wrong omega_t/omega_d).
        if ds_in is not None:
            r["indist"] = eval_dataset(rc, model, ds_in, horizons, args.device, norm_in, args.max_shots)
        if ds_hold is not None:
            r["holdout"] = eval_dataset(rc, model, ds_hold, horizons, args.device, norm_hold, args.max_shots)
        raw[label] = r
        _dom = "indist" if "indist" in r else "holdout"
        msg = "  ".join(f"T{h}={np.mean(r[_dom][h]['nrmse']):.3f}" for h in horizons if r[_dom][h]["nrmse"])
        print(f"[extrap] {label} ({mtype}) {_dom}: {msg}", flush=True)

    ref = args.reference if args.reference in raw else (next(iter(raw)) if raw else None)
    report = {"meta": {"horizons": horizons, "reference": ref, "indist_data": args.indist_data,
                       "holdout_data": args.holdout_data,
                       "n_indist": (len(ds_in) if ds_in is not None else 0),
                       "n_holdout": len(ds_hold) if ds_hold else 0,
                       "indist_skipped": bool(args.skip_indist), "elapsed_s": None},
              "models": {}}
    for label, r in raw.items():
        e = {"model_type": r["model_type"], "checkpoint": r["checkpoint"], "n_params": r.get("n_params")}
        if "indist" in r:
            e["indist"] = summarize(r["indist"], horizons, cmp_mod,
                                    raw[ref]["indist"] if (ref and label != ref and "indist" in raw[ref]) else None)
            e["indist_activity_stratified"] = activity_stratified(r["indist"], horizons)
        if "holdout" in r:
            e["holdout"] = summarize(r["holdout"], horizons, cmp_mod,
                                     raw[ref].get("holdout") if (ref and label != ref) else None)
            e["holdout_activity_stratified"] = activity_stratified(r["holdout"], horizons)
            if "indist" in e:
                e["extrap_gap"] = {str(h): (e["holdout"][h]["nrmse"] / e["indist"][h]["nrmse"])
                                   for h in horizons
                                   if e["holdout"].get(h) and e["indist"].get(h) and e["indist"][h]["nrmse"]}
        report["models"][label] = e

    # ablation-on-zero-shot deltas vs reference (the generalization table)
    if ref and ds_hold is not None:
        tbl = {}
        for label in raw:
            if label == ref or "holdout" not in raw[label]:
                continue
            row = {}
            for h in horizons:
                a = report["models"][label]["holdout"].get(h)
                b = report["models"][ref]["holdout"].get(h)
                if a and b and a["nrmse"] is not None and b["nrmse"] is not None:
                    row[str(h)] = {"delta_nrmse_vs_ref": a["nrmse"] - b["nrmse"],
                                   "p_value": a.get("vs_reference", {}).get("p_value")}
            tbl[label] = row
        report["zero_shot_ablation_table"] = tbl

    # per-machine breakdown + STRONG fit on the reference model
    if ref:
        per_mach = defaultdict(dict)
        accs = ([("indist", raw[ref]["indist"])] if "indist" in raw[ref] else []) \
             + ([("holdout", raw[ref]["holdout"])] if "holdout" in raw[ref] else [])
        holdout_label = None
        for tag, acc in accs:
            for h in horizons:
                by_m = defaultdict(list)
                for m, v in zip(acc[h]["machine"], acc[h]["nrmse"]):
                    key = m if tag == "indist" else f"{m}(zero-shot)"
                    by_m[key].append(v)
                for m, vals in by_m.items():
                    per_mach[m][h] = float(np.mean(vals))
                    if tag == "holdout":
                        holdout_label = m
        report["per_machine_reference"] = {m: {str(h): v for h, v in d.items()} for m, d in per_mach.items()}

        # STRONG fit needs the TRAINING-machine in-dist errors to fit the C-constants; skip it
        # entirely when --skip-indist (no training-machine errors to fit on).
        if "indist" in raw[ref] and ds_in is not None:
            feats = machine_features(rc, ds_in, norm_in, args.max_shots)
            if ds_hold is not None and holdout_label:
                feats_h = machine_features(rc, ds_hold, norm_hold, args.max_shots)
                for m, v in feats_h.items():
                    feats[holdout_label] = v
            report["strong_fit"] = strong_fit(dict(per_mach), feats, horizons, holdout_label)
        else:
            report["strong_fit"] = {"skipped": "no in-dist eval (--skip-indist): STRONG fit needs training-machine errors"}

    report["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))

    # Per-shot arrays sidecar (notebook stratification/replots without re-eval).
    per_shot = {label: {dom: {str(h): dict(r[dom][h]) for h in horizons}
                        for dom in ("indist", "holdout") if dom in r}
                for label, r in raw.items()}
    ps_path = outp.with_name(outp.stem + ".per_shot.json")
    ps_path.write_text(json.dumps(per_shot))
    print(f"per-shot arrays -> {ps_path}")

    # printed tables
    print(f"\n=== NRMSE: in-dist | holdout zero-shot (ref={ref}) ===")
    hdr = "model".ljust(16) + "".join(f"T{h}".rjust(17) for h in horizons)
    print(hdr); print("-" * len(hdr))
    for label, e in report["models"].items():
        cells = []
        for h in horizons:
            a = e.get("indist", {}).get(h); b = e.get("holdout", {}).get(h)
            sa = f"{a['nrmse']:.3f}" if a and a["nrmse"] is not None else "-"
            sb = f"{b['nrmse']:.3f}" if b and b["nrmse"] is not None else "-"
            cells.append(f"{sa}|{sb}".rjust(17))
        print(label.ljust(16) + "".join(cells))
    if ref:
        print(f"\n=== geometry tracking ({ref}; supplementary — geometry is a CONDITIONING input; "
              "near-persistence = stable feedback) ===")
        for dom in ("indist", "holdout"):
            d = report["models"][ref].get(dom) or {}
            cells = []
            for h in (h for h in horizons if h in (1, 20, 50, 100)):
                v = d.get(h) or {}
                if v.get("geom_nrmse") is not None:
                    cells.append(f"T{h}={v['geom_nrmse']:.3f}(pers {v['geom_persistence']:.3f})")
            if cells:
                print(f"  {dom:7s} " + "  ".join(cells))
    if ref and "holdout_activity_stratified" in report["models"].get(ref, {}):
        print(f"\n=== holdout activity-stratified skill vs persistence ({ref}; q4 = most dynamic shots) ===")
        act = report["models"][ref]["holdout_activity_stratified"]
        for h in ("20", "50", "100"):
            if h in act:
                row = "  ".join(f"{q}: {v['skill_vs_persistence']:+.0%}(n={v['n']})"
                                 for q, v in sorted(act[h].items()) if v["skill_vs_persistence"] is not None)
                print(f"  T{h}: {row}")
        # DgknetHybrid gate-by-quartile (the bleed check): WANT ~0 on q1/q2, rising on q4.
        grow = {h: "  ".join(f"{q}:{v['gate_mean']:.2f}" for q, v in sorted(act[h].items()) if "gate_mean" in v)
                for h in ("20", "50", "100") if h in act}
        grow = {h: r for h, r in grow.items() if r}
        if grow:
            print(f"  -- gate by quartile (dgknet-hybrid; want ~0 on q1/q2, up on q4) --")
            for h, r in grow.items():
                print(f"  T{h}: {r}")
    if "strong_fit" in report and "fit_R2" in report.get("strong_fit", {}):
        sf = report["strong_fit"]
        print(f"\nSTRONG fit: R2={sf['fit_R2']:.3f}  C1(geom)={sf['C1_geometry']:.4f} "
              f"C2(beta)={sf['C2_dimensionless']:.4f} C3(n_edge)={sf['C3_edge_density']:.4f}")
        if "holdout_prediction" in sf:
            print(f"  holdout predicted vs measured (STRONG out-of-fit): "
                  f"{sf['holdout_prediction']['predicted_nrmse']} vs {sf['holdout_prediction']['measured_nrmse']}")
        print("  (NOTE: zero-shot NRMSE above is the RAW rollout — no fitting; strong_fit is the "
              "separate 'is it predictable from dimensionless distance?' overlay.)")
    print(f"Wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
