"""Baseline one-step propagators for the RMMD comparison table.

MLP / LSTM / NeuralODE / FNO — each a STATELESS per-step *residual* map x_{t+1} = x_t + f(x_t, ctx),
which is the SAME setup as MultiMachineRMMD (it also re-encodes the current NI every rollout step).
So the autoregressive unit-step rollout is the shared time-recurrence and the comparison is fair:
identical data, identical rollout/curriculum/loss harness, identical drivers — only the one-step
architecture differs.

Each consumes the compact inputs (ni_t0, geometry_tensor, pre_shot_context, drivers, step_dt,
machine) and returns a MultiMachineOutput with:
  - x_next         : NI prediction (residual from current NI)
  - geometry_pred  : identity (current geometry) — baselines do not model equilibrium evolution
  - latent_next    : a latent (matches state_legacy_encoder dim) for the harness's z_pred
  - rmmd           : a ZEROED stub (baselines have no RMMD physics; train_command zeros the physics
                     loss weights for baselines, so they are judged on the DATA loss only).
shared_private_penalty() returns 0.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .multi_machine_rmmd import MultiMachineOutput, PRE_SHOT_CONTEXT_DIM
from .rmmd_block import RMMDOutput

N_RADIAL = 40
N_PSI = 40
N_FOURIER = 66
_GEOM_FLAT = N_PSI * N_FOURIER


def _stub_rmmd(latent: torch.Tensor, n_modes: int = 16) -> RMMDOutput:
    """Zeroed RMMD output so RMMD-specific loss terms vanish (weights are also zeroed for
    baselines). d_psd is a tiny identity to avoid any divide-by-norm in the loss."""
    b = latent.shape[0]
    dev, dt = latent.device, latent.dtype
    zmat = torch.zeros(b, n_modes, n_modes, device=dev, dtype=dt)
    eye = torch.eye(n_modes, device=dev, dtype=dt).unsqueeze(0).expand(b, n_modes, n_modes).contiguous()
    return RMMDOutput(
        x_next=latent, z_next=latent, d_res=zmat, d_total=zmat,
        symmetry_error=torch.zeros((), device=dev, dtype=dt), d_psd=1e-4 * eye, k_sym=zmat,
    )


class _BaselineBase(nn.Module):
    """Shared input encoding + interface plumbing for all baselines."""

    def __init__(
        self,
        machine_names: Sequence[str],
        n_radial: int = N_RADIAL,
        latent_dim: int = 128,
        ctx_dim: int = 256,
        n_drivers: int = 8,
        machine_embedding_dim: int = 24,
    ) -> None:
        super().__init__()
        names = list(dict.fromkeys(str(m) for m in machine_names)) or ["default"]
        self.machine_names = names
        self.machine_to_idx = {m: i for i, m in enumerate(names)}
        self.n_radial = int(n_radial)
        self.latent_dim = int(latent_dim)
        self.ctx_dim = int(ctx_dim)
        self.n_drivers = int(n_drivers)

        self.machine_embedding = nn.Embedding(len(names), machine_embedding_dim)
        # state_legacy_encoder: required by the harness to compute z_true from current NI.
        self.state_legacy_encoder = nn.Sequential(
            nn.Linear(self.n_radial, 256), nn.GELU(), nn.Linear(256, self.latent_dim)
        )
        self.pre_shot_enc = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, 256), nn.GELU(), nn.Linear(256, ctx_dim)
        )
        self.driver_enc = nn.Sequential(nn.Linear(self.n_drivers, 64), nn.GELU(), nn.Linear(64, ctx_dim))
        self.geom_enc = nn.Sequential(nn.Linear(_GEOM_FLAT, 256), nn.GELU(), nn.Linear(256, ctx_dim))
        self.dt_enc = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, ctx_dim))
        self.mach_proj = nn.Linear(machine_embedding_dim, ctx_dim)
        self.ctx_norm = nn.LayerNorm(ctx_dim)

    # -- interface helpers -------------------------------------------------
    def shared_private_penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.machine_embedding.weight.device)

    def _machine_idx(self, names: Sequence[str], device) -> torch.Tensor:
        return torch.tensor([self.machine_to_idx.get(str(m), 0) for m in names], device=device, dtype=torch.long)

    @staticmethod
    def _clean_pre_shot(pre_shot: torch.Tensor) -> torch.Tensor:
        pre_shot = torch.nan_to_num(pre_shot, nan=0.0, posinf=0.0, neginf=0.0)
        big = pre_shot.abs().amax(dim=1, keepdim=True) > 100.0
        if torch.any(big):
            pre_shot = torch.sign(pre_shot) * torch.log1p(pre_shot.abs())
        return torch.clamp(pre_shot, min=-12.0, max=12.0)

    def _context(self, geom, pre_shot, drivers, step_dt, names, device, B) -> torch.Tensor:
        ctx = self.pre_shot_enc(self._clean_pre_shot(pre_shot.to(device)))
        if isinstance(drivers, torch.Tensor):
            d = drivers.to(device).view(B, -1)
            if d.shape[-1] != self.n_drivers:
                d = (torch.cat([d, torch.zeros(B, self.n_drivers, device=device)], -1)[:, : self.n_drivers]
                     if d.shape[-1] < self.n_drivers else d[:, : self.n_drivers])
            ctx = ctx + self.driver_enc(d.to(ctx.dtype))
        if isinstance(geom, torch.Tensor):
            ctx = ctx + self.geom_enc(geom.to(device).reshape(B, -1).to(ctx.dtype))
        if isinstance(step_dt, torch.Tensor):
            ctx = ctx + self.dt_enc(step_dt.to(device).view(B, 1).to(ctx.dtype))
        ctx = ctx + self.mach_proj(self.machine_embedding(self._machine_idx(names, device)))
        return self.ctx_norm(ctx)

    def _core(self, ni_curr: torch.Tensor, ctx: torch.Tensor):  # -> (ni_delta, latent)
        raise NotImplementedError

    def forward(self, x_t=None, machine_names=None, omega_t=None, omega_d=None, batch_data=None):
        if batch_data is None and isinstance(x_t, dict):
            batch_data = x_t
            x_t = None
        bd: Dict = batch_data or {}
        ni_curr = bd.get("ni_t0")
        if not isinstance(ni_curr, torch.Tensor):
            ni_curr = x_t
        device = ni_curr.device
        B = ni_curr.shape[0]
        ni_flat = ni_curr.reshape(B, -1)[:, : self.n_radial]
        names = [str(m) for m in (machine_names or ["default"] * B)]
        geom = bd.get("geometry_tensor")
        ctx = self._context(geom, bd.get("pre_shot_context"), bd.get("drivers"), bd.get("step_dt"), names, device, B)
        ni_delta, latent = self._core(ni_flat, ctx)
        x_next = torch.clamp(ni_flat + ni_delta, min=-8.0, max=8.0)
        geom_pred = geom.to(device) if isinstance(geom, torch.Tensor) else None  # identity geometry
        emb = self.machine_embedding(self._machine_idx(names, device))
        return MultiMachineOutput(
            x_next=x_next, machine_embedding=emb, rmmd=_stub_rmmd(latent),
            geometry_pred=geom_pred, latent_next=latent,
        )


class MLPBaseline(_BaselineBase):
    """Feedforward residual map."""

    def __init__(self, *a, hidden: int = 512, **k):
        super().__init__(*a, **k)
        self.net = nn.Sequential(
            nn.Linear(self.n_radial + self.ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.latent_dim), nn.GELU(),
        )
        self.delta_head = nn.Linear(self.latent_dim, self.n_radial)
        nn.init.normal_(self.delta_head.weight, std=1e-3); nn.init.zeros_(self.delta_head.bias)  # ~identity init

    def _core(self, ni_curr, ctx):
        h = self.net(torch.cat([ni_curr, ctx], dim=-1))
        return self.delta_head(h), h


class LSTMBaseline(_BaselineBase):
    """LSTM that scans the radial profile (each rho-point conditioned on the context vector),
    then predicts the next-step profile residual from the per-point outputs."""

    def __init__(self, *a, hidden: int = 256, layers: int = 2, **k):
        super().__init__(*a, **k)
        self.hidden = hidden
        self.in_proj = nn.Linear(1 + self.ctx_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.delta_head = nn.Linear(2 * hidden, 1)
        self.latent_proj = nn.Linear(2 * hidden, self.latent_dim)
        nn.init.normal_(self.delta_head.weight, std=1e-3); nn.init.zeros_(self.delta_head.bias)

    def _core(self, ni_curr, ctx):
        B = ni_curr.shape[0]
        seq = ni_curr.unsqueeze(-1)                                  # [B, R, 1]
        ctx_b = ctx.unsqueeze(1).expand(B, self.n_radial, ctx.shape[-1])
        x = self.in_proj(torch.cat([seq, ctx_b], dim=-1))            # [B, R, H]
        out, _ = self.lstm(x)                                        # [B, R, 2H]
        ni_delta = self.delta_head(out).squeeze(-1)                  # [B, R]
        latent = self.latent_proj(out.mean(dim=1))                   # [B, L]
        return ni_delta, latent


class NeuralODEBaseline(_BaselineBase):
    """Latent Neural ODE: encode (ni, ctx) -> z0, integrate dz/dt = f(z, ctx) over the unit step
    with fixed-step RK4 (no torchdiffeq dependency), decode the residual."""

    def __init__(self, *a, hidden: int = 256, n_steps: int = 4, **k):
        super().__init__(*a, **k)
        self.n_steps = int(n_steps)
        self.enc = nn.Sequential(nn.Linear(self.n_radial + self.ctx_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.latent_dim))
        self.f = nn.Sequential(nn.Linear(self.latent_dim + self.ctx_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, self.latent_dim))
        self.dec = nn.Linear(self.latent_dim, self.n_radial)
        nn.init.normal_(self.dec.weight, std=1e-3); nn.init.zeros_(self.dec.bias)

    def _deriv(self, z, ctx):
        return self.f(torch.cat([z, ctx], dim=-1))

    def _core(self, ni_curr, ctx):
        z = self.enc(torch.cat([ni_curr, ctx], dim=-1))
        h = 1.0 / self.n_steps
        for _ in range(self.n_steps):                                # RK4 over t in [0,1]
            k1 = self._deriv(z, ctx)
            k2 = self._deriv(z + 0.5 * h * k1, ctx)
            k3 = self._deriv(z + 0.5 * h * k2, ctx)
            k4 = self._deriv(z + h * k3, ctx)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.dec(z), z


class DGKNetBaseline(_BaselineBase):
    """DGKNet baseline — OUR OWN predecessor model, built from existing SOTA methods
    (Metriplectic-Koopman operator: symplectic + negative-semidefinite *diagonal-structured*
    dissipation; NO resonant off-diagonal D_res, no conservative transport step, no SUT
    alignment), wrapped in the shared harness. This is a same-authors SOTA-methods composite
    control — it isolates the contribution of the NOVEL components from the general
    metriplectic-Koopman framework, without the implementation-quality confounds of
    third-party reimplementations. Encode (ni, ctx) -> z; z_next = MetriplecticKoopman(z, ctx);
    decode the residual. Reuses the actual DGKNet operator class (dgknet_baseline)."""

    def __init__(self, *a, hidden: int = 256, **k):
        super().__init__(*a, **k)
        # Lazy import DGKNet's deps only when this baseline is built; ensure the repo root is on the
        # path so `dgknet_baseline` resolves regardless of entry point.
        import sys as _sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[2]   # strong_rmmd -> STRONG_RMMD -> repo root
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from dgknet_baseline.phases.phase2_dgknet_architecture import MetriplecticKoopman
        kd = self.latent_dim if self.latent_dim % 2 == 0 else self.latent_dim + 1  # symplectic needs even
        self.koopman_dim = kd
        self.enc = nn.Sequential(nn.Linear(self.n_radial + self.ctx_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, kd))
        self.koopman = MetriplecticKoopman(koopman_dim=kd, geometry_dim=self.ctx_dim)
        self.dec = nn.Linear(kd, self.n_radial)
        nn.init.normal_(self.dec.weight, std=1e-3); nn.init.zeros_(self.dec.bias)

    def _core(self, ni_curr, ctx):
        z = self.enc(torch.cat([ni_curr, ctx], dim=-1))
        z_next = self.koopman(z, ctx)
        return self.dec(z_next), z_next


class _SpectralConv1d(nn.Module):
    """1-D Fourier layer: rFFT along the radial axis, keep the lowest `modes`, complex-linear mix, iFFT."""

    def __init__(self, in_c: int, out_c: int, modes: int):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / (in_c * out_c)
        self.weight = nn.Parameter(scale * torch.rand(in_c, out_c, self.modes, 2))  # (real, imag)

    def forward(self, x):                                        # x: [B, C, R]
        R = x.shape[-1]
        xf = torch.fft.rfft(x, dim=-1)                           # [B, C, R//2+1]
        m = min(self.modes, xf.shape[-1])
        w = torch.view_as_complex(self.weight.contiguous())      # [in, out, modes] complex
        out = torch.zeros(x.shape[0], w.shape[1], xf.shape[-1], dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bim,iom->bom", xf[..., :m], w[..., :m])
        return torch.fft.irfft(out, n=R, dim=-1)                  # [B, out, R]


class FNOBaseline(_BaselineBase):
    """1-D Fourier Neural Operator over the radial profile. Each rho-point is lifted from its NI value + the
    shared context vector; L Fourier layers (spectral conv along radius + pointwise skip) mix the radial modes
    globally; a pointwise head projects to the next-step residual. A genuine spectral-operator baseline, same
    residual/rollout harness as the others (x_{t+1} = x_t + FNO(x_t, ctx))."""

    def __init__(self, *a, width: int = 64, modes: int = 16, layers: int = 4, hidden: Optional[int] = None, **k):
        super().__init__(*a, **k)
        if hidden:                                               # capacity knob from make_baseline (latent_dim>128)
            width = max(width, int(hidden) // 2)
        self.width, self.modes = int(width), int(modes)
        self.lift = nn.Linear(1 + self.ctx_dim, self.width)
        self.specs = nn.ModuleList([_SpectralConv1d(self.width, self.width, self.modes) for _ in range(layers)])
        self.pws = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(self.width, 128), nn.GELU(), nn.Linear(128, 1))
        self.latent_proj = nn.Linear(self.width, self.latent_dim)
        nn.init.normal_(self.proj[-1].weight, std=1e-3); nn.init.zeros_(self.proj[-1].bias)   # ~identity init

    def _core(self, ni_curr, ctx):
        B, R = ni_curr.shape
        ctx_b = ctx.unsqueeze(1).expand(B, R, ctx.shape[-1])
        h = self.lift(torch.cat([ni_curr.unsqueeze(-1), ctx_b], dim=-1)).transpose(1, 2)   # [B, width, R]
        for spec, pw in zip(self.specs, self.pws):
            h = torch.nn.functional.gelu(spec(h) + pw(h))
        h = h.transpose(1, 2)                                     # [B, R, width]
        ni_delta = self.proj(h).squeeze(-1)                       # [B, R]
        latent = self.latent_proj(h.mean(dim=1))                  # [B, latent_dim]
        return ni_delta, latent


_BASELINES = {"mlp": MLPBaseline, "lstm": LSTMBaseline, "node": NeuralODEBaseline,
              "dgknet": DGKNetBaseline, "fno": FNOBaseline}


def make_baseline(kind: str, machine_names: Sequence[str], n_drivers: int = 8,
                  latent_dim: int = 128, machine_embedding_dim: int = 24) -> _BaselineBase:
    kind = str(kind).lower()
    if kind not in _BASELINES:
        raise ValueError(f"unknown baseline '{kind}'; choose from {sorted(_BASELINES)}")
    kwargs = dict(machine_names=machine_names, n_drivers=n_drivers,
                  latent_dim=latent_dim, machine_embedding_dim=machine_embedding_dim)
    # Capacity scaling: a baseline's parameters live mostly in its `hidden` width, so scale hidden with
    # latent_dim above the 128 default to make --baseline-latent-dim a real capacity knob.
    if latent_dim > 128:
        kwargs["hidden"] = 2 * int(latent_dim)
    return _BASELINES[kind](**kwargs)
