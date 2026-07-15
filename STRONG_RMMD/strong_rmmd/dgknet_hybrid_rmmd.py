"""DgknetHybridRMMD -- RMMD on quiet shots and the DGKNet (metriplectic-Koopman) operator on dynamic shots,
blended by a per-shot detector gate. Explored as a hybrid variant; the deployed hybrid is the per-shot router
in ../decisive_experiments. Kept for the ablation record.
"""
from __future__ import annotations

import dataclasses
import os

import torch
import torch.nn as nn

from strong_rmmd.multi_machine_rmmd import MultiMachineRMMD, MultiMachineOutput

# Dead-zone gate threshold: the gate is hard-zero below this threshold (quiet shots use pure RMMD) and
# continuous above it (the blend applies on dynamic shots).
_GATE_DEADZONE = float(os.environ.get("DGK_GATE_DEADZONE", "0.15"))


def _even(n: int) -> int:
    return int(n) if int(n) % 2 == 0 else int(n) + 1


class DgknetHybridRMMD(MultiMachineRMMD):
    def __init__(self, *args, dgk_koopman_dim: int = 64, dgk_ctx_dim: int = 32,
                 ablate_skip: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablate_skip = bool(ablate_skip)
        nr = int(self.n_radial)
        nd = int(self.n_drivers)
        kd = _even(dgk_koopman_dim)
        self.dgk_koopman_dim = kd
        self.dgk_ctx_dim = int(dgk_ctx_dim)

        # The dgknet skip operator (symplectic + diagonal dissipation). Ensure the repo root is on the
        # path so `dgknet_baseline` resolves regardless of entry point.
        import sys as _sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[2]   # strong_rmmd -> STRONG_RMMD -> repo root
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from dgknet_baseline.phases.phase2_dgknet_architecture import MetriplecticKoopman
        self.dgk_enc = nn.Sequential(nn.Linear(nr + nd, 128), nn.GELU(), nn.Linear(128, kd))
        self.dgk_ctx = nn.Sequential(nn.Linear(nr + nd, 64), nn.GELU(), nn.Linear(64, self.dgk_ctx_dim))
        self.dgk_koopman = MetriplecticKoopman(koopman_dim=kd, geometry_dim=self.dgk_ctx_dim)
        self.dgk_dec = nn.Linear(kd, nr)
        nn.init.normal_(self.dgk_dec.weight, std=1e-3); nn.init.zeros_(self.dgk_dec.bias)

        # --- PER-SHOT detector gate g(ni, drivers, |RMMD tendency|) in [0,1]. The tendency term lets
        # the gate see how much change the RMMD itself predicts (a dynamism signal). Init ~closed.
        self.gate_head = nn.Sequential(
            nn.Linear(nr + nd + nr, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -3.0)   # sigmoid(-3) ~= 0.047 -> ~pure RMMD at init

        self.last_gate_mean = 0.0          # diagnostic: mean APPLIED gate on the last forward
        self.last_gate_soft = 0.0          # mean pre-dead-zone sigmoid
        self.last_gate_logit = None        # (B,1) RAW logit -> per-step activity-supervision target
        self.last_ni_dgk = None            # (B,n_radial) dgknet-skip prediction -> skip-competence loss
        self.gate_by_machine: dict[str, float] = {}

    def _features(self, batch, rmmd_x_next, device, dtype):
        """Operator-free features, extracted EXACTLY as the parent does so they match the rollout."""
        ni_cur = batch.get("ni_t0")
        if not isinstance(ni_cur, torch.Tensor):
            return None
        ni_cur = ni_cur.to(device).reshape(ni_cur.shape[0], -1)[:, : self.n_radial].to(dtype)
        B = ni_cur.shape[0]
        drv = batch.get("drivers")
        if self.ablate_drivers or not isinstance(drv, torch.Tensor):
            drv_v = torch.zeros(B, self.n_drivers, device=device, dtype=dtype)
        else:
            drv_v = drv.to(device).to(dtype).view(B, -1)
            k = drv_v.shape[-1]
            if k < self.n_drivers:
                drv_v = torch.cat([drv_v, torch.zeros(B, self.n_drivers - k, device=device, dtype=dtype)], dim=-1)
            elif k > self.n_drivers:
                drv_v = drv_v[:, : self.n_drivers]
        tendency = (rmmd_x_next.detach() - ni_cur).abs()       # |predicted NI change| (dynamism signal)
        return ni_cur, drv_v, tendency

    def forward(self, x_t, *args, **kwargs) -> MultiMachineOutput:
        out = super().forward(x_t, *args, **kwargs)
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

        # dgknet skip prediction (residual decode, like the DGKNet baseline)
        skip_in = torch.cat([ni_cur, drv_v], dim=-1)
        z_next = self.dgk_koopman(self.dgk_enc(skip_in), self.dgk_ctx(skip_in))
        ni_dgk = ni_cur + self.dgk_dec(z_next)
        self.last_ni_dgk = ni_dgk          # exposed for the skip-competence loss (train it everywhere)

        # PER-SHOT gate
        logit = self.gate_head(torch.cat([ni_cur, drv_v, tendency], dim=-1))   # (B,1)
        g_soft = torch.sigmoid(logit)
        # DEAD-ZONE: hard-0 below _GATE_DEADZONE (quiet -> EXACTLY pure RMMD, no q1 bleed), continuous
        # above (the BLEND on dynamic: (1-g) keeps SOME RMMD off-diagonal, g the dgknet operator).
        g = torch.clamp((g_soft - _GATE_DEADZONE) / (1.0 - _GATE_DEADZONE), min=0.0, max=1.0)
        self.last_gate_logit = logit                 # RAW logit -> per-step activity supervision target
        self.last_gate_soft = float(g_soft.mean().item())
        self.last_gate_mean = float(g.mean().item())   # APPLIED (post-dead-zone) gate
        names = args[0] if args else kwargs.get("machine_names")
        if isinstance(names, (list, tuple)) and len(names) == g.shape[0]:
            gv = g.detach().view(-1).cpu()
            for nm in set(names):
                self.gate_by_machine[str(nm)] = float(gv[[i for i, x in enumerate(names) if x == nm]].mean())

        x_next = (1.0 - g) * out.x_next + g * ni_dgk            # RMMD on quiet (g~0), dgknet on q4 (g~1)
        x_next = torch.nan_to_num(torch.clamp(x_next, min=-8.0, max=8.0), nan=0.0, posinf=0.0, neginf=0.0)
        return dataclasses.replace(out, x_next=x_next)

    def gate_value(self) -> float:
        return float(self.last_gate_mean)
