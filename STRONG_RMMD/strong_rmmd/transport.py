"""Conservative transport step for the NI profile.

Predicts NI(t+1) with a semi-implicit (backward-Euler) flux-form continuity step:

    dn/dt = (1/V') d/drho[ V'(D dn/drho - v n) ] + S
    (I - dt L) n_{t+1} = n_t + dt S        # L tridiagonal

The network outputs dimensionless coefficients (diffusion number nu = dt*D/drho^2,
convection number, source), bounded and initialized near zero so the step starts as
the identity and then learns transport. Properties (verified in tests): conservation
(sum_i V'_i dN_i = dt sum_i V'_i S_i under zero-flux boundaries), contraction
(D >= 0 => ||(I - dt L)^-1|| <= 1, so error is bounded at every horizon), and
differentiability through the tridiagonal solve. D is resonance-conditioned.
"""
from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F

# Driver-gate for the transport step: a per-batch gate blends time-stepping from backward-Euler
# (theta=1, the default) toward Crank-Nicolson (theta=0.5); theta = 1 - A*g, clamped >= 0.5 so it stays
# unconditionally stable. g~0 leaves it at backward-Euler.
_DG_TRANSPORT_RELAX = float(os.environ.get("DG_TRANSPORT_RELAX", "0.5"))


class ConservativeTransportStep(nn.Module):
    def __init__(self, n_radial: int, feat_dim: int, dt: float = 1.0,
                 nu_max: float = 2.0, v_max: float = 0.3, s_max: float = 0.1,
                 use_pinch: bool = True):
        super().__init__()
        self.n = int(n_radial)
        self.dt = float(dt)
        self.use_pinch = bool(use_pinch)
        self.nu_max = float(nu_max)     # cap on per-step diffusion number dt*D/drho^2
        self.v_max = float(v_max)       # cap on per-step convection number
        self.s_max = float(s_max)       # cap on per-step source
        h = max(128, feat_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
        )
        self.d_head = nn.Linear(h, n_radial)        # -> sigmoid -> diffusion number (0, nu_max)
        self.v_head = nn.Linear(h, n_radial)        # -> tanh -> convection number (pinch)
        self.s_head = nn.Linear(h, n_radial)        # -> tanh -> source
        self.logvp_head = nn.Linear(h, n_radial)    # -> exp -> V' > 0 (volume element)
        for head in (self.v_head, self.s_head, self.logvp_head):
            nn.init.zeros_(head.bias); nn.init.normal_(head.weight, std=1e-3)
        # D ~ 0 at init (nu = nu_max*sigmoid(-6) ~ 0.005) => step ~ IDENTITY => sharp T1.
        nn.init.normal_(self.d_head.weight, std=1e-3); nn.init.constant_(self.d_head.bias, -6.0)

    def coefficients(self, feat: torch.Tensor, dt: float):
        h = self.trunk(feat)
        drho = 1.0 / (self.n - 1)
        nu = self.nu_max * torch.sigmoid(self.d_head(h))             # (B,N) diffusion number
        D = nu * (drho * drho) / dt                                  # physical D for the operator
        if self.use_pinch:
            v = torch.tanh(self.v_head(h)) * self.v_max * (drho / dt)  # convection number bounded
        else:
            v = torch.zeros_like(D)
        S = torch.tanh(self.s_head(h)) * self.s_max                  # bounded per-step source
        Vp = torch.exp(torch.clamp(self.logvp_head(h), -3.0, 3.0))   # (B,N) > 0, ~1 default
        return D, v, S, Vp

    def build_operator(self, D, v, Vp):
        """Tridiagonal continuity operator L (B,N,N) in flux/divergence form (conservative);
        central diffusion + convection; zero-flux at axis & edge faces."""
        B, N = D.shape
        drho = 1.0 / (N - 1)
        Df = 0.5 * (D[:, :-1] + D[:, 1:])
        Vpf = 0.5 * (Vp[:, :-1] + Vp[:, 1:])
        vf = 0.5 * (v[:, :-1] + v[:, 1:])
        gd = Vpf * Df / drho                 # diffusion face conductance (B,N-1)
        hc = Vpf * vf                        # convection face coeff       (B,N-1)
        inv = 1.0 / (Vp * drho)              # 1/(V'_i drho)               (B,N)
        z = torch.zeros(B, 1, device=D.device, dtype=D.dtype)
        gd_up = torch.cat([gd, z], dim=1);  hc_up = torch.cat([hc, z], dim=1)
        gd_lo = torch.cat([z, gd], dim=1);  hc_lo = torch.cat([z, hc], dim=1)
        upper = inv * (gd_up - 0.5 * hc_up)
        lower = inv * (gd_lo + 0.5 * hc_lo)
        main = inv * (-(gd_up + 0.5 * hc_up) - (gd_lo - 0.5 * hc_lo))
        L = (torch.diag_embed(main)
             + torch.diag_embed(upper[:, :N - 1], offset=1)
             + torch.diag_embed(lower[:, 1:], offset=-1))
        return L

    def forward(self, ni: torch.Tensor, feat: torch.Tensor, dt: float | None = None,
                relax_gate: torch.Tensor | None = None):
        B, N = ni.shape
        dt = self.dt if dt is None else float(dt)
        D, v, S, Vp = self.coefficients(feat, dt)
        L = self.build_operator(D, v, Vp)
        eye = torch.eye(N, device=ni.device, dtype=ni.dtype).unsqueeze(0)
        if relax_gate is None:
            A = eye - dt * L                                   # backward Euler (theta=1)
            rhs = (ni + dt * S).unsqueeze(-1)
        else:
            # DRIVER-GATE theta-method: theta=1 (backward Euler) on quiet shots -> theta>=0.5
            # (Crank-Nicolson) on transients. (I - theta*dt*L) n_{t+1} = (I + (1-theta)*dt*L) n_t + dt*S.
            theta = (1.0 - _DG_TRANSPORT_RELAX * relax_gate.view(B, 1)).clamp(min=0.5, max=1.0)  # [B,1]
            Ln = torch.einsum("bij,bj->bi", L, ni)             # L @ n_t, [B,N]
            A = eye - theta.view(B, 1, 1) * dt * L
            rhs = (ni + (1.0 - theta) * dt * Ln + dt * S).unsqueeze(-1)
        ni_next = torch.linalg.solve(A, rhs).squeeze(-1)
        return ni_next, {"D": D, "v": v, "S": S, "Vp": Vp, "L": L}


__all__ = ["ConservativeTransportStep"]
