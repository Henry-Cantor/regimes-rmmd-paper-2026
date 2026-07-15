"""Single-machine RMMD block for STRONG-RMMD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import os
import logging
from torch import nn
from torch.nn import functional as F

from strong_rmmd.resonance_kernel import LorentzianResonanceKernel
from strong_rmmd.memory_kernel import lorentzian_time_kernel

# Driver-gate (optional, via relax_gate). A per-batch gate g in [0,1] relaxes the resonant contraction
# on transients: dissipation gain *= (1 - A*g), step cap += B*g. g~0 leaves the ungated block unchanged.
_DG_DISSIP_RELAX = float(os.environ.get("DG_DISSIP_RELAX", "0.8"))
_DG_STEP_RELAX = float(os.environ.get("DG_STEP_RELAX", "1.0"))


@dataclass
class RMMDOutput:
    x_next: torch.Tensor
    z_next: torch.Tensor
    d_res: torch.Tensor
    d_total: torch.Tensor
    symmetry_error: torch.Tensor
    d_psd: torch.Tensor | None = None
    k_sym: torch.Tensor | None = None
    k_dis: torch.Tensor | None = None
    d_res_time: torch.Tensor | None = None
    step_scale_mean: torch.Tensor | None = None


class RMMDBlock(nn.Module):
    """One-step resonance-mediated metriplectic latent update.

    Dynamics are integrated with an explicit Euler step:
        z_{t+1} = z_t + dt * (z_t C^T - z_t D^T)

    where `C` is a learned conservative operator and
    `D = D_diag + D_res` is a dissipative operator.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        context_dim: int = 0,
        n_harmonics: int = 4,
        dt: float = 1.0,
        use_time_kernel: bool = False,
        n_taus: int = 8,
        max_step_ratio: float = 0.5,
        ablate_offdiag: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.dt = dt
        # ABLATION (no-D_res): use diagonal-only dissipation, removing the novel resonant
        # off-diagonal cross-mode coupling (reduces to standard diagonal metriplectic damping).
        self.ablate_offdiag = bool(ablate_offdiag)
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even")

        self.half_dim = latent_dim // 2
        self.context_dim = context_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 2 * latent_dim),
            nn.SiLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.SiLU(),
            nn.Linear(2 * latent_dim, state_dim),
        )

        self.A_sym_param = nn.Parameter(torch.randn(self.half_dim, self.half_dim) * 0.02)
        self.L_dis_param = nn.Parameter(torch.randn(self.half_dim, self.half_dim) * 0.02)
        self.diag_dissipative = nn.Parameter(torch.zeros(latent_dim))

        gate_out_dim = self.half_dim * self.half_dim
        self.geom_gate_sym = nn.Sequential(
            nn.Linear(context_dim, gate_out_dim),
            nn.Tanh(),
        )
        self.geom_gate_dis = nn.Sequential(
            nn.Linear(context_dim, gate_out_dim),
            nn.Tanh(),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(context_dim, 2 * latent_dim),
            nn.SiLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.residual_action_scale = nn.Parameter(torch.tensor(0.10))
        self.residual_gate_scale = nn.Parameter(torch.tensor(0.05))
        self.dissipation_step = 0.12
        # Trust-region cap on one-step latent displacement (fraction of ||z||+1), which bounds
        # per-step error amplification.
        self.max_step_ratio = float(max_step_ratio)

        # Learnable gain g in (0, max_dissipation_gain) for the bounded contraction step in forward();
        # init sigmoid(-1.4)*0.5 ~= 0.13.
        self.dissipation_gain = nn.Parameter(torch.tensor(-1.4))
        self.max_dissipation_gain = 0.5
        # Diagnostics: fraction of latent energy removed by the dissipative step, and the off-diagonal
        # Frobenius share ||D_res||_F / ||D_psd||_F.
        self.last_dissip_frac = 0.0
        self.last_offdiag_frac = 0.0

        self.kernel = LorentzianResonanceKernel(
            latent_dim=latent_dim,
            n_harmonics=n_harmonics,
            context_dim=context_dim,
        )

        self.use_time_kernel = use_time_kernel
        self.n_taus = n_taus
        if self.use_time_kernel:
            taus = torch.linspace(0.0, float(n_taus - 1) * float(self.dt), n_taus)
            self.register_buffer("taus", taus)

    def _get_K_sym(self, A_skew: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.half_dim, device=A_skew.device, dtype=A_skew.dtype)
        I_plus_A = I + A_skew
        I_minus_A = I - A_skew
        jitter = 1e-3 if A_skew.dtype == torch.float32 else 1e-6
        I_plus_A = I_plus_A + jitter * I
        try:
            return torch.linalg.solve(I_plus_A, I_minus_A)
        except RuntimeError:
            return torch.linalg.lstsq(I_plus_A, I_minus_A).solution

    def _get_K_dis(self, L: torch.Tensor) -> torch.Tensor:
        return -L.T @ L

    def _operator_norm(self, mat: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
        """Per-batch spectral-norm (lambda_max) estimate via power iteration.

        Detached on purpose: it is used only to SCALE the dissipative step so the step
        operator (I - g * D_psd/lambda_max) stays in [1-g, 1] (non-expansive).  We do
        not learn through the normaliser — the standard spectral-normalisation trick.
        mat is assumed symmetric PSD; returns shape (batch,).
        """
        b, d, _ = mat.shape
        with torch.no_grad():
            v = torch.randn(b, d, device=mat.device, dtype=mat.dtype)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
            for _ in range(max(1, n_iter)):
                v = torch.einsum("bij,bj->bi", mat, v)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
            lam = torch.einsum("bij,bj->bi", mat, v).norm(dim=1)
        return lam.clamp_min(1e-6)

    def forward(
        self,
        x_t: torch.Tensor,
        omega_t: torch.Tensor,
        omega_d: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        z_t: Optional[torch.Tensor] = None,
        relax_gate: Optional[torch.Tensor] = None,
    ) -> RMMDOutput:
        if z_t is None:
            z_t = self.encoder(x_t)

        if context is None:
            context = torch.zeros(z_t.shape[0], self.context_dim, device=z_t.device, dtype=z_t.dtype)

        kernel_out = self.kernel(z=z_t, omega_t=omega_t, omega_d=omega_d, context=context)
        d_res = kernel_out.d_res
        d_res_time = None

        # Build skew-symmetric A and keep entries bounded to avoid near-singular Cayley transforms.
        A_raw = self.A_sym_param
        A_skew = A_raw - A_raw.T
        A_skew = torch.tanh(A_skew) * 0.8
        K_sym = self._get_K_sym(A_skew)
        K_dis = self._get_K_dis(self.L_dis_param)

        eye = torch.eye(self.half_dim, device=z_t.device, dtype=z_t.dtype)

        sym_gate = self.geom_gate_sym(context).view(z_t.shape[0], self.half_dim, self.half_dim)
        dis_gate = self.geom_gate_dis(context).view(z_t.shape[0], self.half_dim, self.half_dim)

        # Reduce gate influence to avoid large, dense per-batch operator perturbations.
        K_sym_mod = K_sym.unsqueeze(0) + self.residual_gate_scale * sym_gate
        K_dis_mod = K_dis.unsqueeze(0) + self.residual_gate_scale * dis_gate
        K_sym_mod = torch.nan_to_num(K_sym_mod, nan=0.0, posinf=0.0, neginf=0.0)
        K_dis_mod = torch.nan_to_num(K_dis_mod, nan=0.0, posinf=0.0, neginf=0.0)

        # Keep gate-modulated operators in a numerically safe range without nullifying direction.
        sym_fro = torch.norm(K_sym_mod, dim=(1, 2), keepdim=True)
        dis_fro = torch.norm(K_dis_mod, dim=(1, 2), keepdim=True)
        K_sym_mod = K_sym_mod / torch.clamp(sym_fro / 8.0, min=1.0)
        K_dis_mod = K_dis_mod / torch.clamp(dis_fro / 8.0, min=1.0)

        z_q, z_p = z_t[:, : self.half_dim], z_t[:, self.half_dim :]
        sym_residual = torch.bmm(z_q.unsqueeze(1), (K_sym_mod - eye.unsqueeze(0)).transpose(1, 2)).squeeze(1)
        z_q_next = z_q + self.dissipation_step * sym_residual
        z_p_next = z_p + self.dissipation_step * torch.bmm(z_p.unsqueeze(1), K_dis_mod.transpose(1, 2)).squeeze(1)
        z_next = torch.cat([z_q_next, z_p_next], dim=1)

        residual = self.residual_gate(context)
        z_next = z_next + self.residual_gate_scale * torch.tanh(residual)

        # Resonance-mediated dissipative contraction. The kernel emits a PSD operator D_psd; a bounded
        # step z <- z - g * (D_psd / lambda_max) z has eigenvalues in [1 - g, 1], so it is non-expansive
        # and keeps the autoregressive rollout error bounded. D_psd (not the diagonal-removed d_res) is
        # used because only the PSD form guarantees contraction while retaining the off-diagonal coupling.
        d_psd_sym = 0.5 * (kernel_out.d_psd + kernel_out.d_psd.transpose(-1, -2))
        # Off-diagonal (cross-mode) share of the dissipation operator, tracked as a diagnostic.
        with torch.no_grad():
            diag_only = torch.diag_embed(torch.diagonal(d_psd_sym, dim1=-2, dim2=-1))
            off = d_psd_sym - diag_only
            fro_off = off.reshape(off.shape[0], -1).norm(dim=1)
            fro_tot = d_psd_sym.reshape(d_psd_sym.shape[0], -1).norm(dim=1) + 1e-12
            self.last_offdiag_frac = float((fro_off / fro_tot).mean().item())
        # ABLATION: strip the resonant off-diagonal coupling, keep diagonal-only dissipation.
        if self.ablate_offdiag:
            d_psd_sym = torch.diag_embed(torch.diagonal(d_psd_sym, dim1=-2, dim2=-1))
            self.last_offdiag_frac = 0.0
        lam_max = self._operator_norm(d_psd_sym)
        d_psd_hat = d_psd_sym / (lam_max.view(-1, 1, 1) + 1e-3)
        gain = torch.sigmoid(self.dissipation_gain) * self.max_dissipation_gain
        if relax_gate is not None:                      # DRIVER-GATE: less contraction on transients
            gain = gain * (1.0 - _DG_DISSIP_RELAX * relax_gate.view(-1, 1).to(z_next.dtype))
        dissip_step = gain * torch.einsum("bij,bj->bi", d_psd_hat, z_next)
        z_pre_dissip = z_next
        z_next = z_next - dissip_step
        # Diagnostic: relative latent energy removed by the dissipative step.
        with torch.no_grad():
            e_before = (z_pre_dissip ** 2).sum(dim=1) + 1e-12
            e_after = (z_next ** 2).sum(dim=1)
            self.last_dissip_frac = float(((e_before - e_after) / e_before).mean().item())

        # Trust-region limiter: cap one-step latent displacement relative to current scale.
        # This preserves direction while preventing rare spikes from destabilizing training.
        delta = z_next - z_t
        delta_norm = torch.norm(delta, dim=1, keepdim=True)
        z_norm = torch.norm(z_t, dim=1, keepdim=True)
        step_ratio = self.max_step_ratio
        if relax_gate is not None:                      # DRIVER-GATE: allow bigger steps on transients
            step_ratio = self.max_step_ratio + _DG_STEP_RELAX * relax_gate.view(-1, 1).to(z_t.dtype)
        max_delta = step_ratio * (z_norm + 1.0)
        step_scale = torch.clamp(max_delta / torch.clamp(delta_norm, min=1e-6), max=1.0)
        z_next = z_t + delta * step_scale
        z_next = torch.nan_to_num(z_next, nan=0.0, posinf=0.0, neginf=0.0)

        diag_vals = F.softplus(self.diag_dissipative)
        d_diag = torch.diag_embed(diag_vals.unsqueeze(0).expand(z_t.shape[0], -1))
        d_total = d_diag + d_res
        x_next = self.decoder(z_next)

        symmetry_error = torch.mean(torch.abs(d_res - d_res.transpose(-1, -2)), dim=(-2, -1))
        # Optionally build a time-domain representation of the resonance kernel
        if self.use_time_kernel:
            # spectral weights per mode (bsz, n_modes)
            spectral = kernel_out.amplitudes * kernel_out.lorentz_weights
            # per-mode outer products (n_modes, latent_dim, latent_dim)
            mode_outer = self.kernel.mode_vectors[:, :, None] * self.kernel.mode_vectors[:, None, :]
            # per-mode gammas and mode frequencies: omega0 = omega_d * harmonic_index
            omega0 = omega_d[:, None] * self.kernel.harmonic_indices[None, :].to(omega_d.device)
            gammas = kernel_out.gammas
            # compute time-domain lorentzian per batch, mode and tau -> (bsz, n_modes, n_taus)
            taus = self.taus.to(z_t.device)
            # ensure shapes for broadcasting
            K_time = lorentzian_time_kernel(taus[None, None, :], gammas[:, :, None], omega0[:, :, None])
            # weighted sum over modes into time-resolved d_res: (bsz, n_taus, latent_dim, latent_dim)
            d_res_time = torch.einsum("bm,mij,bmt->btij", spectral, mode_outer, K_time)
            # symmetrize and zero diagonal per-time
            d_res_time = 0.5 * (d_res_time + d_res_time.transpose(-1, -2))
            diag = torch.diagonal(d_res_time, dim1=-2, dim2=-1)
            d_res_time = d_res_time - torch.diag_embed(diag)

        # Optional environment-gated debug logging to inspect per-term magnitudes.
        try:
            if os.environ.get("RMMD_DEBUG", "0") == "1":
                logger = logging.getLogger("rmmd_block")
                A_norm = float(torch.norm(A_skew).item())
                K_sym_fro = float(sym_fro.mean().item()) if 'sym_fro' in locals() else float(torch.norm(K_sym_mod).item())
                K_dis_fro = float(dis_fro.mean().item()) if 'dis_fro' in locals() else float(torch.norm(K_dis_mod).item())
                sym_res_norm = float(torch.norm(sym_residual).item())
                d_res_act_norm = float(torch.norm(d_res_action).item())
                diag_max = float(torch.max(diag_vals).item()) if 'diag_vals' in locals() else 0.0
                step_s = float(step_scale.mean().item()) if 'step_scale' in locals() else 0.0
                logger.info(
                    "RMMDBlock debug: A_norm=%.6g K_sym_fro=%.6g K_dis_fro=%.6g sym_res_norm=%.6g d_res_act_norm=%.6g diag_max=%.6g step_scale=%.6g",
                    A_norm,
                    K_sym_fro,
                    K_dis_fro,
                    sym_res_norm,
                    d_res_act_norm,
                    diag_max,
                    step_s,
                )
        except Exception:
            pass

        return RMMDOutput(
            x_next=x_next,
            z_next=z_next,
            d_res=d_res,
            d_total=d_total,
            d_psd=kernel_out.d_psd,
            k_sym=K_sym_mod,
            k_dis=K_dis_mod,
            symmetry_error=symmetry_error,
            d_res_time=d_res_time,
            step_scale_mean=step_scale.mean(),
        )
