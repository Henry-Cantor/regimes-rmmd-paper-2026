"""EXP-2 -- does D_res help in the regime the theory predicts?

GIT says D_res should matter for driven plasma and be spurious for quiescent plasma. Step 2.0 dumps the AUGD
abl_dres delta per horizon and activity quartile from the committed extrap report (delta > 0 means removing
D_res worsens, i.e. D_res helps). Step 2.1 (needs a checkpoint) correlates per-shot ||D_res|| against a
driven-ness proxy (PINJ and its time-variability) that does not reuse the NI profile. Run with --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
REPORTS = REPO / "STRONG_RMMD" / "theory_validation" / "results"


def step_2_0():
    """Dump abl_dres regime split from committed extrap reports (EAST quiescent vs AUGD dynamic)."""
    out = {}
    for tag in ("east", "augd"):
        p = REPORTS / f"extrap_strong_report_{tag}.json"
        if not p.exists():
            out[tag] = "NOT FOUND"; continue
        M = json.loads(p.read_text())["models"]
        if "abl_dres" not in M or "full" not in M:
            out[tag] = "NOT FOUND (abl_dres/full missing)"; continue
        def nr(m, h):
            v = (M[m].get("holdout") or {}).get(str(h)); return v.get("nrmse") if isinstance(v, dict) else v
        def q(m, h, qq):
            e = (M[m].get("holdout_activity_stratified") or {}).get(str(h), {})
            return (e.get(qq) or {}).get("model_nrmse")
        pooled = {}
        for h in (20, 50, 100):
            fu, ab = nr("full", h), nr("abl_dres", h)
            if fu is not None and ab is not None:
                pooled[str(h)] = {"full": fu, "abl_dres": ab, "delta_dres_helps": ab - fu}
        quart = {}
        for h in (50, 100):
            quart[str(h)] = {}
            for qq in ("q1", "q2", "q3", "q4"):
                fu, ab = q("full", h, qq), q("abl_dres", h, qq)
                if fu is not None and ab is not None:
                    quart[str(h)][qq] = {"full": fu, "abl_dres": ab, "delta_dres_helps": ab - fu}
        zt = (json.loads(p.read_text()).get("zero_shot_ablation_table") or {}).get("abl_dres", {})
        pvals = {h: {"delta": (zt.get(h) or {}).get("delta_nrmse_vs_ref"), "p": (zt.get(h) or {}).get("p_value")}
                 for h in ("50", "100")}
        out[tag] = {"pooled_by_horizon": pooled, "per_quartile": quart, "paired_p_pooled": pvals,
                    "note": "per-quartile paired p NOT in committed report (needs an extrap rerun that dumps "
                            "per-shot errors, or the ablation harness with --stratified-pvalues) -> NOT FOUND here"}
    return out


def step_2_1(ckpt, dataset_spec, device):
    """Per-shot ||D_res|| vs an INDEPENDENT driven-ness proxy (PINJ), Spearman + perm p. Needs cluster."""
    strong = REPO / "STRONG_RMMD"
    for pth in (str(strong), str(strong / "data_io"), str(REPO)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    def imp(rel, name):
        spec = importlib.util.spec_from_file_location(name, strong / rel)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    rc = imp("training/rmmd_train_eval_impl.py", "rmmd_train_eval_impl")
    cmp_mod = imp("comparison/run_comparison.py", "comparison_run_comparison")
    import torch

    model, norm, _ = cmp_mod._build_model(rc, Path(ckpt), device)
    model.eval()
    name, path = dataset_spec.split(":", 1)
    payload = rc._load_phase0_dataset(Path(path))
    n = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
    ds = rc.CompactRolloutDataset(payload, max_time=2, normalization_stats=n)

    # capture ||D_res|| by patching the resonance kernel forward to record the off-diagonal norm
    rmmd = model.rmmd
    orig = rmmd.kernel.forward
    grabbed = {}
    def patched(z, omega_t, omega_d, context=None):
        kout = orig(z=z, omega_t=omega_t, omega_d=omega_d, context=context)
        grabbed["dres_norm"] = float(torch.linalg.norm(kout.d_res.reshape(kout.d_res.shape[0], -1), dim=1).mean().item())
        grabbed["amp_sum"] = float(kout.amplitudes.sum(dim=1).mean().item())
        return kout
    rmmd.kernel.forward = patched

    dres_norms, pinj_vals, pinj_var = [], [], []
    N = len(ds)
    for i in range(N):
        s = ds[i]
        try:
            rc._rollout_compact_shot_to_checkpoints(
                model, s["ni_t0"], s["geom_t0"], s["pre_shot_context"], s["limiter_geometry_tensor"],
                s["ni_traj"], s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}),
                device, norm, max_time_step=1, drivers_traj=s.get("drivers_traj"), report_horizons=[1])
        except Exception:
            continue
        if "dres_norm" not in grabbed:
            continue
        # INDEPENDENT driven-ness proxy: PINJ (heating power) from pre-shot scalars / drivers, NOT NI-derived
        sc = s.get("pre_shot_scalars", {}) or {}
        pinj = None
        for k in ("PINJ", "pinj", "PNBI", "P_NBI", "PINJ_MEAN"):
            if k in sc and sc[k] is not None:
                pinj = float(sc[k]); break
        drv = s.get("drivers_traj")
        if isinstance(drv, torch.Tensor) and drv.numel():
            d0 = drv.detach().cpu().numpy()
            if pinj is None:
                pinj = float(np.mean(d0[:, 0]))          # driver channel 0 = PINJ (theorems_report meta pinj_channel=0)
            pinj_var.append(float(np.std(d0[:, 0])))     # time-variability of the drive
        else:
            pinj_var.append(np.nan)
        if pinj is None:
            continue
        dres_norms.append(grabbed["dres_norm"]); pinj_vals.append(pinj)
    rmmd.kernel.forward = orig

    def spearman(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        if x.size < 8:
            return float("nan"), float("nan"), int(x.size)
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        rho = float(np.corrcoef(rx, ry)[0, 1]); rng = np.random.default_rng(0)
        null = np.array([np.corrcoef(rx, np.argsort(np.argsort(rng.permutation(ry))))[0, 1] for _ in range(2000)])
        return rho, float((np.abs(null) >= abs(rho)).mean()), int(x.size)

    rho_p, p_p, nP = spearman(dres_norms, pinj_vals)
    rho_v, p_v, nV = spearman(dres_norms, pinj_var)
    return {"dataset": name, "n": len(dres_norms),
            "spearman_dres_vs_PINJ": rho_p, "perm_p_PINJ": p_p,
            "spearman_dres_vs_PINJ_variability": rho_v, "perm_p_PINJ_variability": p_v,
            "dres_tracks_independent_drive": bool((abs(rho_p) > 0.2 and p_p < 0.05) or (abs(rho_v) > 0.2 and p_v < 0.05)),
            "context": "theorems_report EDT NBI split had d_res_high_PINJ=0.9702 vs low=0.9709 (nearly flat) -> "
                       "a flat correlation here would mean the learned D_res does NOT track the drive it should."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dataset", default=None, metavar="name:path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"EXP2_0_regime_split_from_committed": step_2_0()}
    if args.ckpt and args.dataset:
        report["EXP2_1_dres_vs_independent_drive"] = step_2_1(args.ckpt, args.dataset, args.device)
    else:
        report["EXP2_1_dres_vs_independent_drive"] = "NOT RUN (pass --ckpt and --dataset augd:<AUGD.pt> on the cluster)"

    # ---- VERDICT ----
    augd = report["EXP2_0_regime_split_from_committed"].get("augd", {})
    helps_pooled = False; helps_q4 = None
    if isinstance(augd, dict):
        pp = augd.get("paired_p_pooled", {})
        helps_pooled = bool(any((pp.get(h, {}).get("delta") or 0) > 0 and (pp.get(h, {}).get("p") or 1) < 0.05 for h in ("50", "100")))
        q100 = (augd.get("per_quartile", {}) or {}).get("100", {})
        helps_q4 = (q100.get("q4", {}) or {}).get("delta_dres_helps")
    corr = report["EXP2_1_dres_vs_independent_drive"]
    corr_ok = corr.get("dres_tracks_independent_drive") if isinstance(corr, dict) else None
    report["VERDICT_2"] = {
        "prediction_if_true": "abl_dres delta > 0 (p<0.05) on AUGD (dynamic), incl q4; D_res validated 'in regime' iff it ALSO correlates with an independent drive proxy (2.1).",
        "refuted_if": "abl_dres <=0 / n.s. across AUGD quartiles+horizons -> demote D_res to synthetic-only (EXP-3).",
        "augd_dres_helps_pooled_p<0.05": helps_pooled,
        "augd_q4_delta_T100": helps_q4,
        "dres_tracks_independent_drive_2.1": corr_ok,
        "VERDICT_regime_split": "SUPPORTED" if helps_pooled else "REFUTED",
        "VERDICT_mechanism_2.1": ("SUPPORTED" if corr_ok else ("REFUTED/NOT RUN" if corr_ok is not True else "SUPPORTED")),
        "reading": "regime split (2.0) is the paper claim; 2.1 is the honest mechanism check (rule 4). If 2.0 SUPPORTED "
                   "but 2.1 REFUTED -> 'D_res helps the dynamic regime empirically, but its learned magnitude does not "
                   "track the drive; present as an emergent regularizer, not a driven-turbulence readout.'",
    }
    (RESULTS / "dres_regime.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["EXP2_0_regime_split_from_committed"].get("augd", {}), indent=1, default=float))
    print("VERDICT_2:", json.dumps(report["VERDICT_2"], indent=1, default=float))
    print("wrote", RESULTS / "dres_regime.json")


if __name__ == "__main__":
    main()
