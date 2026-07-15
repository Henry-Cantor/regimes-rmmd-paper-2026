#!/usr/bin/env python
"""Model-free check: does rel_ptp (drive variability = ptp(PINJ)/mean|PINJ|, the router's route feature)
predict a shot's activity better than mean PINJ (drive level)? Activity = per-shot persistence NRMSE
||ni[T]-ni[0]|| / ||ni[T]||. rel_ptp is machine-scale-invariant while mean(PINJ) mixes machine scales, so
the per-holdout rows are the clean within-machine test. PINJ is driver channel 0.
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_USER = os.environ.get("USER", "USER")


def _imports():
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("rmmd_train_eval_impl", strong / "training" / "rmmd_train_eval_impl.py")
    rc = importlib.util.module_from_spec(spec); sys.modules["rmmd_train_eval_impl"] = rc; spec.loader.exec_module(rc)
    return rc


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if x.size < 4:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def _auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    m = np.isfinite(score); score, label = score[m], label[m]
    pos, neg = (label == 1).sum(), (label == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return float((r[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=f"/scratch/gpfs/{_USER}/strong_rmmd")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--out-json", default=str(HERE / "results" / "activity_drive_check.json"))
    a = ap.parse_args()
    rc = _imports()
    dr = Path(a.data_root)
    datasets = {"pool": dr / "phase0NEW" / "dataset_test_compact.pt",
                "east": dr / "phase0NEW_east" / "dataset_test_compact.pt",
                "augd": dr / "phase0NEW_augd" / "dataset_test_compact.pt"}
    T = a.horizon
    RP, MP, ACT, DS = [], [], [], []
    for name, path in datasets.items():
        if not path.exists():
            print(f"[skip] {name}: missing {path}"); continue
        norm = rc._ensure_normalization_stats(Path(path), checkpoint_dir=None, require=False)
        ds = rc.CompactRolloutDataset(rc._load_phase0_dataset(Path(path)), max_time=100, normalization_stats=norm)
        n = 0
        for i in range(len(ds)):
            s = ds[i]
            ni, drv = s.get("ni_traj"), s.get("drivers_traj")
            if ni is None or drv is None:
                continue
            ni = ni.detach().cpu().numpy().astype(float) if hasattr(ni, "detach") else np.asarray(ni, float)
            p = drv.detach().cpu().numpy().astype(float) if hasattr(drv, "detach") else np.asarray(drv, float)
            if ni.ndim != 2 or ni.shape[0] < 2 or p.ndim != 2 or p.shape[0] < 2:
                continue
            pinj = p[:, 0]
            RP.append(float(np.ptp(pinj) / (np.abs(pinj).mean() + 1e-9)))    # rel_ptp = drive variability
            MP.append(float(pinj.mean()))                                    # mean PINJ = drive level
            t = min(T, ni.shape[0] - 1)
            ACT.append(float(np.linalg.norm(ni[t].ravel() - ni[0].ravel()) / (np.linalg.norm(ni[t].ravel()) + 1e-9)))
            DS.append(name); n += 1
        print(f"[{name}] {n} shots")
    RP, MP, ACT, DS = map(np.array, (RP, MP, ACT, DS))

    def block(mask):
        act, rp, mp = ACT[mask], RP[mask], MP[mask]
        if act.size < 8:
            return None
        hi = (act >= np.nanquantile(act, 0.75)).astype(int)                  # top-quartile (most dynamic) shots
        s_rp, s_mp = _spearman(rp, act), _spearman(mp, act)
        au_rp, au_mp = _auc(rp, hi), _auc(mp, hi)
        return {"n": int(act.size),
                "spearman_rel_ptp_vs_activity": s_rp, "spearman_meanPINJ_vs_activity": s_mp,
                "auc_rel_ptp_predicts_q4": au_rp, "auc_meanPINJ_predicts_q4": au_mp,
                "rel_ptp_better": bool(abs(s_rp) > abs(s_mp)
                                       and (au_rp if np.isfinite(au_rp) else 0) > (au_mp if np.isfinite(au_mp) else 0))}
    out = {"horizon": T, "activity_def": "persistence NRMSE = ||ni[T]-ni[0]|| / ||ni[T]|| (model-free)",
           "route_feature": "rel_ptp = ptp(PINJ)/mean|PINJ| (drive variability) vs mean(PINJ) (drive level)",
           "by": {}}
    b = block(np.ones(ACT.shape, bool))
    if b:
        out["by"]["pooled"] = b
    for name in ("pool", "east", "augd"):
        bb = block(DS == name)
        if bb:
            out["by"][name] = bb
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1, default=float))
    print(f"\n=== rel_ptp vs mean PINJ for predicting ACTIVITY (persistence NRMSE @T{T}) ===")
    print(f"{'set':7} {'n':>5} {'Sp(rel_ptp)':>12} {'Sp(meanPINJ)':>13} {'AUC(rel_ptp)':>13} {'AUC(meanPINJ)':>14}  winner")
    for k, bl in out["by"].items():
        w = "rel_ptp" if bl["rel_ptp_better"] else "mean_PINJ"
        print(f"{k:7} {bl['n']:>5} {bl['spearman_rel_ptp_vs_activity']:>12.3f} {bl['spearman_meanPINJ_vs_activity']:>13.3f} "
              f"{bl['auc_rel_ptp_predicts_q4']:>13.3f} {bl['auc_meanPINJ_predicts_q4']:>14.3f}  {w}")
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
