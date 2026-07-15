"""FusedRMMD -- RMMD blended with a DGKNet-style operator by a per-radius learned gate. Explored as a fused
variant; the deployed hybrid is the per-shot router in ../decisive_experiments. Kept for the ablation record.
"""
from __future__ import annotations

import dataclasses
import gzip
import io
import os
from pathlib import Path

import torch
import torch.nn as nn

from strong_rmmd.dgknet_hybrid_rmmd import DgknetHybridRMMD
from strong_rmmd.multi_machine_rmmd import MultiMachineRMMD, MultiMachineOutput


def _resolve_ckpt(p):
    """Resolve a checkpoint that may be given as a dir, a .pt, or a .pt.gz (return the real path or None)."""
    p = Path(p)
    if p.is_dir():
        for c in ("checkpoint_best.pt", "checkpoint_best.pt.gz", "checkpoint_best.pt.gz".replace(".gz", "")):
            if (p / c).exists():
                return p / c
        return None
    if p.exists():
        return p
    gz = Path(str(p) + ".gz")
    if gz.exists():
        return gz
    if str(p).endswith(".gz") and Path(str(p)[:-3]).exists():
        return Path(str(p)[:-3])
    return None


def _load_ckpt_any(path):
    """torch.load a checkpoint that may be gzip-compressed (.pt.gz), as the rest of the pipeline saves them."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


class FusedRMMD(DgknetHybridRMMD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nr, nd = int(self.n_radial), int(self.n_drivers)
        # PER-RADIUS gate (parent's was a per-shot scalar). Init bias 0 -> sigmoid 0.5 = the fixed
        # ensemble that already beats both -> training can only improve from a winning start.
        self.gate_head = nn.Sequential(nn.Linear(nr + nd + nr, 64), nn.SiLU(), nn.Linear(64, nr))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 0.0)

        # cheap mode: load the trained RMMD into the parent + freeze it; train only gate + DGKNet skip.
        ckpt = os.environ.get("FUSED_RMMD_CKPT")
        if ckpt:
            ckpt = _resolve_ckpt(ckpt)
            if ckpt is not None:
                sd = _load_ckpt_any(ckpt)             # handles .pt AND .pt.gz
                if isinstance(sd, dict):
                    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
                missing, unexpected = self.load_state_dict(sd, strict=False)
                print(f"[FusedRMMD] loaded RMMD parent from {ckpt} "
                      f"(missing {len(missing)} = gate/skip, unexpected {len(unexpected)})", flush=True)
            else:
                print(f"[FusedRMMD] WARNING: FUSED_RMMD_CKPT={os.environ.get('FUSED_RMMD_CKPT')} not found "
                      "(.pt/.pt.gz); RMMD parent stays at init.", flush=True)
        if os.environ.get("FUSED_FREEZE_RMMD", "1") == "1":
            trainable = ("gate_head", "dgk_")   # gate + the DGKNet skip (dgk_enc/dgk_ctx/dgk_koopman/dgk_dec)
            n_train = 0
            for name, p in self.named_parameters():
                p.requires_grad = any(name.startswith(t) for t in trainable)
                n_train += p.requires_grad
            print(f"[FusedRMMD] frozen RMMD parent; {n_train} trainable tensors (gate + DGKNet skip)", flush=True)

    def forward(self, x_t, *args, **kwargs) -> MultiMachineOutput:
        # RMMD forward (bypass the parent's per-shot-switch forward)
        out = MultiMachineRMMD.forward(self, x_t, *args, **kwargs)
        if self.ablate_skip:
            return out
        batch = x_t if isinstance(x_t, dict) else kwargs.get("batch_data")
        if not isinstance(batch, dict) and len(args) >= 4 and isinstance(args[3], dict):
            batch = args[3]
        if not isinstance(batch, dict):
            return out
        feats = self._features(batch, out.x_next, out.x_next.device, out.x_next.dtype)
        if feats is None:
            return out
        ni_cur, drv_v, tendency = feats
        # DGKNet-style skip (same operator as the parent hybrid)
        skip_in = torch.cat([ni_cur, drv_v], dim=-1)
        z_next = self.dgk_koopman(self.dgk_enc(skip_in), self.dgk_ctx(skip_in))
        ni_dgk = ni_cur + self.dgk_dec(z_next)
        self.last_ni_dgk = ni_dgk
        # PER-RADIUS continuous blend (no dead-zone): g in [0,1]^n_radial
        logit = self.gate_head(torch.cat([ni_cur, drv_v, tendency], dim=-1))   # (B, n_radial)
        g = torch.sigmoid(logit)
        self.last_gate_logit = logit
        self.last_gate_mean = float(g.mean().item())
        x_next = (1.0 - g) * out.x_next + g * ni_dgk
        x_next = torch.nan_to_num(torch.clamp(x_next, min=-8.0, max=8.0), nan=0.0, posinf=0.0, neginf=0.0)
        return dataclasses.replace(out, x_next=x_next)


__all__ = ["FusedRMMD"]
