"""EXP-5 (LOMO) data prep -- build the leave-one-machine-out folds from the per-machine-normalized pool.

For each held-out training machine, train/val is built from the remaining machines and eval is all of the
held-out machine's shots. All datasets are already per-machine normalized, so the merge is scale-consistent.
Run with --help.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _machine(s):
    m = s.get("machine", "UNKNOWN") if isinstance(s, dict) else "UNKNOWN"
    return m.decode() if isinstance(m, bytes) else str(m)


def _key(p):
    for k in ("data", "samples"):
        if isinstance(p.get(k), list):
            return k
    raise ValueError("payload has no 'data'/'samples' list")


def _subset(payload, keep_fn):
    """Return a shallow copy of `payload` keeping only the shots whose machine satisfies keep_fn (+ its count).
    All other keys (metadata, normalization_stats_by_machine, ...) are carried through unchanged."""
    k = _key(payload)
    idx = [i for i, s in enumerate(payload[k]) if keep_fn(_machine(s))]
    new = {kk: vv for kk, vv in payload.items() if kk not in (k, "sample_index")}
    new[k] = [payload[k][i] for i in idx]
    si = payload.get("sample_index")
    if isinstance(si, list):
        new["sample_index"] = [si[i] for i in idx if i < len(si)]
    return new, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--machines", nargs="*", default=None, help="default = all machines in train")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra datasets (e.g. EAST, AUGD) to ADD to every fold's training pool -> train-on-6")
    ap.add_argument("--no-own-stats", action="store_true",
                    help="skip attaching the held-out machine's OWN per-machine stats to eval_m (default = attach, "
                         "which is REQUIRED for correct zero-shot transfer, matching the AUGD/EAST holdout datasets).")
    ap.add_argument("--eval-only", action="store_true",
                    help="ONLY rebuild the (small) eval_m sets; leave the existing train/val folds untouched. "
                         "Use when only the eval feed changed (no retraining) -> much faster.")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr = torch.load(args.train, map_location="cpu", weights_only=False)
    va = torch.load(args.val, map_location="cpu", weights_only=False)

    # extra-machine shots (EAST/AUGD) merged into every fold's train/val pool (train-on-6). They are already
    # per-machine normalized (own-built by phase0), consistent with the pool's per-machine normalization.
    extra_shots = []
    for ef in args.extra:
        ep = torch.load(ef, map_location="cpu", weights_only=False)
        extra_shots += ep[_key(ep)]
    if extra_shots:                                       # split 80/20 -> present in BOTH train and val pools
        idx = np.random.default_rng(0).permutation(len(extra_shots)); cut = int(0.8 * len(extra_shots))
        extra_tr = [extra_shots[i] for i in idx[:cut]]; extra_va = [extra_shots[i] for i in idx[cut:]]
        print(f"extra shots merged into EVERY fold: {len(extra_tr)} train + {len(extra_va)} val "
              f"from {sorted(set(_machine(s) for s in extra_shots))}")
    else:
        extra_tr, extra_va = [], []

    machines = args.machines or sorted({_machine(s) for s in tr[_key(tr)]})
    print("machines:", machines)
    summary = {}
    for m in machines:
        fold = out / f"holdout_{m}"; fold.mkdir(exist_ok=True)
        ntr = nvo = -1
        if not args.eval_only:                                # (skip the big train/val folds if only eval changed)
            tr_others, ntr = _subset(tr, lambda x: x != m)    # train on the OTHER training machines
            va_others, nvo = _subset(va, lambda x: x != m)
            if extra_shots:                                   # + EAST/AUGD in BOTH train and val -> train-on-6
                tr_others[_key(tr_others)] = tr_others[_key(tr_others)] + extra_tr; ntr += len(extra_tr)
                va_others[_key(va_others)] = va_others[_key(va_others)] + extra_va; nvo += len(extra_va)
            torch.save(tr_others, fold / "dataset_train_compact.pt")
            torch.save(va_others, fold / "dataset_val_compact.pt")

        # eval set = ALL of m's shots (m is held out ENTIRELY -> use train+val, not just the ~30 val shots)
        tr_m, _ = _subset(tr, lambda x: x == m); va_m, _ = _subset(va, lambda x: x == m)
        eval_m = tr_m; eval_m[_key(eval_m)] = tr_m[_key(tr_m)] + va_m[_key(va_m)]; neval = len(eval_m[_key(eval_m)])
        if not args.no_own_stats:
            # The pool is normalized PER-MACHINE (build_phase0new_from_cdfs.py:797), so m's eval profiles are ALREADY
            # own-normalized -- do NOT re-normalize. Just ATTACH m's own per-machine stats so the model denormalizes
            # correctly for the omega computation, exactly like the committed AUGD/EAST holdouts.
            own = (tr.get("normalization_stats_by_machine") or {}).get(m)
            if not isinstance(own, dict):
                raise ValueError(f"pool has no normalization_stats_by_machine[{m}] -- rebuild the pool with "
                                 f"build_phase0new (it stores per-machine stats), or pass --no-own-stats.")
            eval_m["normalization_stats"] = own                    # embedded stats (priority #4)
            meta = eval_m.get("metadata")                          # AND metadata y_stats/geom_stats (priority #3, read first)
            if isinstance(meta, dict):
                ni = own.get("kinetic_profiles.NI") or {}; gm = own.get("geometry_tensor") or {}
                if ni.get("mean_per_element") is not None:
                    meta["y_stats"] = {"mean": ni["mean_per_element"], "std": ni["std_per_element"]}
                if gm.get("mean_per_element") is not None:
                    meta["geom_stats"] = {"mean": gm["mean_per_element"], "std": gm["std_per_element"]}
            eval_m.pop("normalization_stats_by_machine", None)     # drop the pool-wide map; m's own is now attached
            t0 = np.stack([np.asarray(s["ni_t0"], float).reshape(-1)[:40]     # sanity: ni_t0 should already be ~unit
                           for s in eval_m[_key(eval_m)] if s.get("ni_t0") is not None])
            print(f"  [{m}] eval already own-normalized (phase0 per-machine): ni_t0 per-elem |mean|~"
                  f"{float(np.abs(t0.mean(0)).mean()):.3g} std~{float(t0.std(0).mean()):.3g} (expect ~0 / ~1); "
                  f"attached phase0 per-machine stats, NO re-normalize")
        torch.save(eval_m, fold / f"eval_{m}.pt")
        summary[m] = {"train": ntr, "val": nvo, "eval_m_ALL_shots": neval, "own_stats": not args.no_own_stats,
                      "eval_only": args.eval_only}
        print(f"  holdout {m}: train={ntr} val={nvo} eval_m(ALL)={neval} -> {fold}")
    print("\nSHOT COUNTS:")
    for m, c in summary.items():
        print(f"  {m}: {c}")


if __name__ == "__main__":
    main()
