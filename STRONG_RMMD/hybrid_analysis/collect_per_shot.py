"""Per-shot RMMD-vs-DGKNet error collector. (Superseded by the driver-keyed router in
../decisive_experiments/exp4b_router_sota.py; kept for reference.) Writes per_shot.json for
find_threshold.py. Run with --help.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]   # hybrid_analysis -> STRONG_RMMD -> repo root


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


def _shot_features(s) -> dict:
    """INFERENCE-AVAILABLE features only (known before the rollout): the prescribed driver trajectory
    and the starting NI profile. NO use of ni_traj (that would leak the answer)."""
    feats = {}
    drv = s.get("drivers_traj")
    if isinstance(drv, torch.Tensor) and drv.ndim == 2 and drv.shape[0] >= 2:
        d = drv.detach().cpu().numpy().astype(np.float64)            # (T, n_drivers)
        rng = np.ptp(d, axis=0)                                      # peak-to-peak per channel
        scale = np.maximum(np.abs(d).mean(axis=0), 1e-6)
        feats["driver_ptp_norm"] = float(np.max(rng / scale))       # biggest relative driver swing
        feats["driver_std_mean"] = float(np.std(d, axis=0).mean())  # mean temporal std
        feats["driver_max_slope"] = float(np.max(np.abs(np.diff(d, axis=0))))  # fastest step
    else:
        feats["driver_ptp_norm"] = feats["driver_std_mean"] = feats["driver_max_slope"] = 0.0
    ni0 = s["ni_t0"].detach().cpu().numpy().astype(np.float64)
    feats["ni_mean0"] = float(ni0.mean())
    feats["ni_peaking"] = float(ni0[: max(1, len(ni0) // 4)].mean() / (np.abs(ni0).mean() + 1e-9))  # core/avg
    # STARTING-STATE GEOMETRY (also inference-available): the switch may key on machine shape / starting
    # geometry, not just driver dynamism (per the hypothesis that geometry sets the transport regime).
    geom = s.get("geom_t0")
    if isinstance(geom, torch.Tensor) and geom.numel():
        g = geom.detach().cpu().numpy().astype(np.float64).ravel()
        feats["geom_mean0"] = float(g.mean())
        feats["geom_std0"] = float(g.std())
        feats["geom_ptp0"] = float(np.ptp(g))
    else:
        feats["geom_mean0"] = feats["geom_std0"] = feats["geom_ptp0"] = 0.0
    return feats


@torch.no_grad()
def collect(rc, model_r, model_d, dataset, H, device, norm_r, norm_d, name, max_shots):
    n = min(len(dataset), max_shots) if max_shots else len(dataset)
    rows = []
    for i in range(n):
        s = dataset[i]
        T = int(s["ni_traj"].shape[0])
        if T < H:                      # only shots that reach the horizon (fair pairing)
            continue
        tgt = s["ni_traj"][H - 1].numpy()
        pers, _ = rc._normalized_rmse_mae(s["ni_t0"].numpy(), tgt)   # ORACLE activity label

        def run(model, norm):
            ni_preds, _ = rc._rollout_compact_shot_to_checkpoints(
                model, s["ni_t0"], s["geom_t0"], s["pre_shot_context"], s["limiter_geometry_tensor"],
                s["ni_traj"], s["geom_traj"], s["machine"], s.get("pre_shot_scalars", {}),
                device, norm, max_time_step=min(H, T), drivers_traj=s.get("drivers_traj"),
                report_horizons=[H])
            if H not in ni_preds:
                return None
            nr, _ = rc._normalized_rmse_mae(ni_preds[H].numpy(), tgt)
            return float(nr)

        nr_r, nr_d = run(model_r, norm_r), run(model_d, norm_d)
        if nr_r is None or nr_d is None:
            continue
        rec = {"dataset": name, "shot": i, "machine": str(s["machine"]),
               "nrmse_rmmd": nr_r, "nrmse_dgknet": nr_d, "activity": float(pers),
               "dgknet_wins": bool(nr_d < nr_r)}
        rec.update(_shot_features(s))
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True, help="dir holding full/ and base_dgknet/")
    ap.add_argument("--dataset", action="append", required=True, metavar="name:path",
                    help="repeatable; e.g. --dataset val:val.pt --dataset east:EAST.pt")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "STRONG_RMMD/hybrid_analysis/results/per_shot.json"))
    args = ap.parse_args()

    rc, cmp_mod = _imports()
    ck = Path(args.ckpt_root)
    cp_r, cp_d = cmp_mod._find_checkpoint(ck / "full"), cmp_mod._find_checkpoint(ck / "base_dgknet")
    if not cp_r or not cp_d:
        print(f"need both full/ and base_dgknet/ checkpoints under {ck}", file=sys.stderr)
        return 2
    model_r, norm_r, _ = cmp_mod._build_model(rc, cp_r, args.device)
    model_d, norm_d, _ = cmp_mod._build_model(rc, cp_d, args.device)

    all_rows = []
    for spec in args.dataset:
        name, path = spec.split(":", 1)
        payload = rc._load_phase0_dataset(Path(path))
        norm = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        ds = rc.CompactRolloutDataset(payload, max_time=args.horizon, normalization_stats=norm)
        # each model carries its OWN training-norm stats (norm_r/norm_d), not the dataset's
        rows = collect(rc, model_r, model_d, ds, args.horizon, args.device, norm_r, norm_d,
                       name, args.max_shots or None)
        nd = sum(r["dgknet_wins"] for r in rows)
        print(f"[{name}] {len(rows)} shots paired | dgknet wins {nd} ({nd/max(1,len(rows)):.0%})", flush=True)
        all_rows += rows

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"horizon": args.horizon, "n": len(all_rows), "shots": all_rows}, indent=1))
    print(f"wrote {len(all_rows)} shot records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
