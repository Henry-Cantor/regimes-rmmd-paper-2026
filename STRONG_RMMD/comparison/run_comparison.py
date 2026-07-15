#!/usr/bin/env python
"""Comparison-table runner.

Evaluates a set of trained checkpoints (RMMD, ablations, baselines) on the same test set, per horizon, in
normalized space, and emits one comparison-table JSON. For each model and horizon it reports NRMSE, NMAE, and
skill over persistence. Run with --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Diagnostic horizon ladder -> smooth NRMSE-vs-horizon curve. Caps at T=100 (model trains to
# frontier ~50; beyond ~100 the rollout has decayed too far to be informative).
HORIZONS_DEFAULT = (1, 2, 3, 5, 8, 12, 16, 20, 32, 50, 75, 100)


def _import_corr():
    """Load rmmd_train_eval_impl as a module (it lives in training, not a package)."""
    repo = Path(__file__).resolve().parents[2]          # repo root
    strong = repo / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    cpath = strong / "training" / "rmmd_train_eval_impl.py"
    spec = importlib.util.spec_from_file_location("rmmd_train_eval_impl", cpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rmmd_train_eval_impl"] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_checkpoint(d: Path) -> Path | None:
    # Checkpoints are saved gzipped (.pt.gz); return the BASE .pt path so the gz-aware loader
    # (_torch_load_checkpoint_any -> _checkpoint_paths) resolves either .pt or .pt.gz.
    def _base(p: Path) -> Path:
        return p.with_suffix("") if p.suffix == ".gz" else p
    if d.is_file():
        return _base(d)
    for name in ("checkpoint_best.pt", "checkpoint_best.pt.gz"):
        if (d / name).exists():
            return d / "checkpoint_best.pt"
    eps = sorted(d.glob("checkpoint_epoch_*.pt")) + sorted(d.glob("checkpoint_epoch_*.pt.gz"))
    return _base(sorted(eps)[-1]) if eps else None


def _build_model(rc, ckpt_path: Path, device: str):
    ck = rc._torch_load_checkpoint_any(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "model_state" in ck:
        sd = ck["model_state"]
        cfg = ck.get("config", {}) or {}
        norm = ck.get("normalization_stats", {}) or {}
    else:
        sd, cfg, norm = ck, {}, {}
    inferred = rc._infer_model_dimensions(sd)
    g = lambda k, d: cfg.get(k, inferred.get(k, d))
    model = rc._make_model(
        cfg.get("machine_names") or ["default"],
        state_dim=int(g("state_dim", 40)), latent_dim=int(g("latent_dim", 384)),
        latent_profile=int(g("latent_profile", 160)), latent_geom=int(g("latent_geom", 160)),
        machine_embedding_dim=int(g("machine_embedding_dim", 48)), n_harmonics=int(g("n_harmonics", 4)),
        use_transport_step=bool(cfg.get("use_transport_step", True)),
        ablate_drivers=bool(cfg.get("ablate_drivers", False)),
        ablate_geometry=bool(cfg.get("ablate_geometry", False)),
        ablate_dres=bool(cfg.get("ablate_dres", False)),
        model_type=str(cfg.get("model_type", "rmmd")),
        baseline_latent_dim=int(cfg.get("baseline_latent_dim", 128)),
    ).to(device)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        rc._load_compatible_state_dict(model, sd)
    model.eval()
    return model, norm, str(cfg.get("model_type", "rmmd"))


@torch.no_grad()
def _eval_model(rc, model, dataset, horizons, device, norm_stats, max_shots, persist_acc=None):
    """Return per-shot arrays {h: {'nrmse':[...],'nmae':[...],'shots':[...]}} (shot index kept so the
    caller can PAIR models for significance tests); if persist_acc dict given, fill persistence too."""
    acc = {h: {"nrmse": [], "nmae": [], "shots": []} for h in horizons}
    n = min(len(dataset), max_shots) if max_shots else len(dataset)
    max_h = max(horizons)
    _t0 = time.time()
    for i in range(n):
        if i % 20 == 0:
            print(f"[comparison]   ...rolling shot {i}/{n} (max_h={max_h}, {time.time() - _t0:.0f}s)", flush=True)
        s = dataset[i]
        ni_traj = s["ni_traj"]
        T = int(ni_traj.shape[0])
        if T < 1:
            continue
        ni_t0, geom_t0 = s["ni_t0"], s["geom_t0"]
        ni_preds, _ = rc._rollout_compact_shot_to_checkpoints(
            model, ni_t0, geom_t0, s["pre_shot_context"], s["limiter_geometry_tensor"],
            ni_traj, s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}), device, norm_stats,
            max_time_step=min(max_h, T), drivers_traj=s.get("drivers_traj"), report_horizons=horizons,
        )
        for h in horizons:
            if h > T or h not in ni_preds:
                continue
            tgt = ni_traj[h - 1].numpy()
            nr, nm = rc._normalized_rmse_mae(ni_preds[h].numpy(), tgt)
            acc[h]["nrmse"].append(nr); acc[h]["nmae"].append(nm); acc[h]["shots"].append(i)
            if persist_acc is not None:
                pnr, pnm = rc._normalized_rmse_mae(ni_t0.numpy(), tgt)
                persist_acc.setdefault(h, {"nrmse": [], "nmae": []})
                persist_acc[h]["nrmse"].append(pnr); persist_acc[h]["nmae"].append(pnm)
    # per-shot arrays kept (aligned by shot index) so the caller can pair models for stats.
    return {h: {"nrmse": v["nrmse"], "nmae": v["nmae"], "shots": v["shots"], "n": len(v["nrmse"])}
            for h, v in acc.items()}


def _boot_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    """95% bootstrap CI of the mean."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _paired_pvalue(ref_map, model_map):
    """Paired test that `model` differs from `ref` on the SAME shots. Wilcoxon signed-rank (scipy)
    if available; else a paired sign/bootstrap fallback. Returns (p_value, mean_diff(model-ref), n)."""
    common = sorted(set(ref_map) & set(model_map))
    if len(common) < 8:
        return (None, None, len(common))
    r = np.array([ref_map[i] for i in common], dtype=np.float64)
    m = np.array([model_map[i] for i in common], dtype=np.float64)
    d = m - r  # >0 means model is WORSE (higher NRMSE) than ref
    try:
        from scipy.stats import wilcoxon
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(wilcoxon(m, r, zero_method="wilcox").pvalue)
    except Exception:
        # paired bootstrap two-sided p-value on the mean difference
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, d.size, size=(5000, d.size))].mean(axis=1)
        frac = float((bs <= 0).mean())
        p = 2.0 * min(frac, 1 - frac)
    return (p, float(d.mean()), len(common))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--ckpt-root", default=None, help="auto-discover label=subdir checkpoints under here")
    ap.add_argument("--models", nargs="*", default=[], help="explicit label=path entries")
    ap.add_argument("--models-json", default=None, help="JSON {label: dir} (e.g. LR-winner map); takes precedence")
    ap.add_argument("--horizons", type=int, nargs="*", default=list(HORIZONS_DEFAULT))
    ap.add_argument("--max-shots", type=int, default=0, help="0 = all test shots")
    ap.add_argument("--reference", default="full", help="model label others are paired-tested against")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="STRONG_RMMD/comparison/results/comparison_table.json")
    args = ap.parse_args()
    # On CPU the per-step matrix ops are tiny; on a many-core node BLAS oversubscribes threads
    # (each op spawns dozens -> pure overhead -> ~100x slowdown). Pin to 1 thread on CPU.
    if args.device == "cpu":
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        print("[comparison] device=cpu -> torch threads pinned to 1 (avoids BLAS oversubscription). "
              "Pass --device cuda on a GPU node for ~50x speedup.", flush=True)

    rc = _import_corr()
    horizons = sorted(set(int(h) for h in args.horizons))

    # discover models
    entries = {}
    if args.models_json:
        for label, path in json.loads(Path(args.models_json).read_text()).items():
            entries[label] = Path(path)
    for kv in args.models:
        label, _, path = kv.partition("=")
        entries[label] = Path(path)
    if args.ckpt_root and not entries:
        for d in sorted(Path(args.ckpt_root).iterdir()):
            if d.is_dir() and d.name not in entries and _find_checkpoint(d):
                entries[d.name] = d
    if not entries:
        raise SystemExit("No models found. Pass --models label=dir ... or --ckpt-root <dir>.")

    payload = rc._load_phase0_dataset(Path(args.test_data))
    norm_for_ds = rc._ensure_normalization_stats(Path(args.test_data), checkpoint_dir=None, require=False)
    dataset = rc.CompactRolloutDataset(payload, max_time=max(horizons), normalization_stats=norm_for_ds)
    print(f"[comparison] test shots={len(dataset)}  horizons={horizons}  models={list(entries)}", flush=True)

    persist_acc: dict = {}
    results = {}
    for label, d in entries.items():
        cp = _find_checkpoint(d)
        if cp is None:
            print(f"[comparison] {label}: no checkpoint in {d}, skipping", flush=True); continue
        model, norm, mtype = _build_model(rc, cp, args.device)
        n_params = sum(p.numel() for p in model.parameters())
        # persistence is model-independent; accumulate it only on the first model pass
        res = _eval_model(rc, model, dataset, horizons, args.device, norm or norm_for_ds,
                          args.max_shots, persist_acc if not persist_acc else None)
        results[label] = {"model_type": mtype, "checkpoint": str(cp), "per_shot": res,
                          "n_params": n_params}
        means = {h: (float(np.mean(res[h]["nrmse"])) if res[h]["nrmse"] else None) for h in horizons}
        print(f"[comparison] {label} ({mtype}): " +
              "  ".join(f"T{h}={means[h]:.3f}" for h in horizons if means[h] is not None), flush=True)

    persistence = {h: {"nrmse": float(np.mean(v["nrmse"])) if v["nrmse"] else None,
                       "nmae": float(np.mean(v["nmae"])) if v["nmae"] else None} for h, v in persist_acc.items()}

    # reference model for the PAIRED significance tests (default 'full' = RMMD).
    ref_label = args.reference if args.reference in results else (next(iter(results)) if results else None)

    def _shotmap(label, h):
        ps = results[label]["per_shot"][h]
        return {i: v for i, v in zip(ps["shots"], ps["nrmse"])}

    summary = {}
    for label, r in results.items():
        res = r["per_shot"]; by_h = {}
        for h in horizons:
            vals = res[h]["nrmse"]
            mean = float(np.mean(vals)) if vals else None
            lo, hi = _boot_ci(vals) if vals else (None, None)
            p = persistence.get(h, {}).get("nrmse")
            e = {"nrmse": mean, "nmae": float(np.mean(res[h]["nmae"])) if res[h]["nmae"] else None,
                 "n": res[h]["n"], "ci95": [lo, hi],
                 "skill_vs_persistence": (1.0 - mean / p) if (mean is not None and p) else None}
            if ref_label and label != ref_label:
                pval, mdiff, ncom = _paired_pvalue(_shotmap(ref_label, h), _shotmap(label, h))
                e["vs_reference"] = {"reference": ref_label, "p_value": pval, "mean_diff_vs_ref": mdiff,
                                     "n_paired": ncom, "significant_0.05": (pval is not None and pval < 0.05)}
            by_h[h] = e
        summary[label] = {"model_type": r["model_type"], "checkpoint": r["checkpoint"],
                          "n_params": r.get("n_params"), "by_horizon": by_h}

    out = {"horizons": horizons, "reference": ref_label, "persistence": persistence, "models": summary,
           "test_data": str(args.test_data), "n_shots": len(dataset),
           "stats": {"ci": "95% bootstrap (2000 resamples)", "test": "paired Wilcoxon signed-rank vs reference (scipy; bootstrap fallback)"}}
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    # printed table: NRMSE ±halfCI ; '*' = significantly different from the reference (paired p<0.05)
    print(f"\n=== NRMSE by horizon  (mean ±½·95%CI;  * = paired p<0.05 vs '{ref_label}') ===")
    hdr = "model".ljust(15) + "".join(f"T{h}".rjust(15) for h in horizons)
    print(hdr); print("-" * len(hdr))
    def cell(e):
        if e is None or e["nrmse"] is None: return "-".rjust(15)
        half = (e["ci95"][1] - e["ci95"][0]) / 2 if e["ci95"][0] is not None else 0.0
        star = "*" if e.get("vs_reference", {}).get("significant_0.05") else " "
        return f"{e['nrmse']:.3f}±{half:.3f}{star}".rjust(15)
    print("persistence".ljust(15) + "".join((f"{persistence[h]['nrmse']:.3f}" if persistence.get(h, {}).get('nrmse') is not None else "-").rjust(15) for h in horizons))
    for label in summary:
        np_str = f"{summary[label]['n_params']/1e6:.1f}M" if summary[label].get("n_params") else "?"
        print(f"{label:<11}{np_str:>4}" + "".join(cell(summary[label]["by_horizon"].get(h)) for h in horizons))
    print(f"\nWrote {outp}  (reference for tests: {ref_label})")


if __name__ == "__main__":
    main()
