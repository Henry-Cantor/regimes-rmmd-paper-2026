"""EXP-1 helper -- split the combined compact dataset into per-machine datasets so each machine can be trained
with a separate model. Preserves the payload structure (normalization stats, sample index) and filters the
sample list to one machine; writes one train and one val .pt per machine. Run with --help.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def _machine_of(sample):
    m = sample.get("machine", "UNKNOWN") if isinstance(sample, dict) else "UNKNOWN"
    return m.decode() if isinstance(m, bytes) else str(m)


def _sample_list_key(payload):
    for k in ("data", "samples"):
        if isinstance(payload.get(k), list):
            return k
    return None


def filter_payload(payload, machine):
    """Return a NEW payload keeping only `machine`'s samples; preserves all other keys."""
    key = _sample_list_key(payload)
    if key is None:
        raise ValueError("payload has neither 'data' nor 'samples' list (sharded payloads not supported here)")
    idx = [i for i, s in enumerate(payload[key]) if _machine_of(s) == machine]
    new = {k: v for k, v in payload.items() if k not in (key, "sample_index")}
    new[key] = [payload[key][i] for i in idx]
    si = payload.get("sample_index")
    if isinstance(si, list) and len(si) == len(payload[key]) + 0:  # align if present
        new["sample_index"] = [si[i] for i in idx if i < len(si)]
    return new, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--machines", nargs="*", default=None, help="default = all machines found in train")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train = torch.load(args.train, map_location="cpu", weights_only=False)
    val = torch.load(args.val, map_location="cpu", weights_only=False)
    key = _sample_list_key(train)
    machines = args.machines or sorted({_machine_of(s) for s in train[key]})
    print("machines:", machines)
    summary = {}
    for m in machines:
        tr, ntr = filter_payload(train, m)
        va, nva = filter_payload(val, m)
        md = out / m; md.mkdir(exist_ok=True)
        torch.save(tr, md / "dataset_train_compact.pt")
        torch.save(va, md / "dataset_val_compact.pt")
        summary[m] = {"n_train": ntr, "n_val": nva}
        print(f"  {m}: train={ntr} val={nva} -> {md}")
    print("\nSHOT COUNTS (log these — small-data machines can overfit, per spec step 2):")
    for m, c in summary.items():
        print(f"  {m}: {c}")


if __name__ == "__main__":
    main()
