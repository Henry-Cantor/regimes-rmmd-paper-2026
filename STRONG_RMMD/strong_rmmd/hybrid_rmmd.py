"""HybridRMMD -- the RMMD operator plus a gated MLP skip that bypasses the Koopman bottleneck. Explored as a
transfer variant; the deployed hybrid is the per-shot router in ../decisive_experiments. Kept for the
ablation record. Did NOT work.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from strong_rmmd.multi_machine_rmmd import MultiMachineRMMD, MultiMachineOutput


class HybridRMMD(MultiMachineRMMD):
    def __init__(self, *args, hybrid_skip_hidden: int = 512, ablate_skip: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablate_skip = bool(ablate_skip)
        n_radial = int(self.n_radial)
        in_dim = n_radial + int(self.n_drivers)         # OPERATOR-FREE inputs: current NI + drivers
        h = int(hybrid_skip_hidden)
        self.mlp_skip = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, n_radial),
        )
        for m in self.mlp_skip:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # zero the LAST layer so the skip output is 0 at init -> x_next == RMMD's at init.
        nn.init.zeros_(self.mlp_skip[-1].weight)
        nn.init.zeros_(self.mlp_skip[-1].bias)
        # learned mixing gate; sigmoid(0)=0.5 so gradient can grow the (initially-zero) skip.
        self.skip_gate = nn.Parameter(torch.tensor(0.0))

    def _skip_inputs(self, batch_data, device, dtype):
        """OPERATOR-FREE features for the skip: current NI (n_radial) + per-step drivers
        (n_drivers), extracted EXACTLY as the parent forward does (so they match)."""
        ni_cur = batch_data.get("ni_t0")
        if not isinstance(ni_cur, torch.Tensor):
            return None
        ni_cur = ni_cur.to(device).reshape(ni_cur.shape[0], -1)[:, : self.n_radial].to(dtype)
        B = ni_cur.shape[0]
        drv = batch_data.get("drivers")
        if self.ablate_drivers or not isinstance(drv, torch.Tensor):
            drivers_vec = torch.zeros(B, self.n_drivers, device=device, dtype=dtype)
        else:
            drivers_vec = drv.to(device).to(dtype).view(B, -1)
            k = drivers_vec.shape[-1]
            if k < self.n_drivers:
                drivers_vec = torch.cat(
                    [drivers_vec, torch.zeros(B, self.n_drivers - k, device=device, dtype=dtype)], dim=-1)
            elif k > self.n_drivers:
                drivers_vec = drivers_vec[:, : self.n_drivers]
        return torch.cat([ni_cur, drivers_vec], dim=-1)

    def forward(self, x_t, *args, **kwargs) -> MultiMachineOutput:
        out = super().forward(x_t, *args, **kwargs)
        if self.ablate_skip:
            return out
        # Resolve the EFFECTIVE batch dict exactly as the parent does: x_t if it's the dict, else
        # the `batch_data` argument (kwarg, or 5th positional after x_t/machine_names/omega_t/omega_d).
        batch = x_t if isinstance(x_t, dict) else kwargs.get("batch_data")
        if not isinstance(batch, dict) and len(args) >= 4 and isinstance(args[3], dict):
            batch = args[3]
        if not isinstance(batch, dict):
            return out
        feats = self._skip_inputs(batch, out.x_next.device, out.x_next.dtype)
        if feats is None:
            return out
        skip = self.mlp_skip(feats)
        gate = torch.sigmoid(self.skip_gate)
        x_next = torch.clamp(out.x_next + gate * skip, min=-8.0, max=8.0)
        x_next = torch.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0)
        return dataclasses.replace(out, x_next=x_next)

    def skip_gate_value(self) -> float:
        """Diagnostic: how much flexible correction the model learned to use (0=pure RMMD)."""
        return float(torch.sigmoid(self.skip_gate).item())
