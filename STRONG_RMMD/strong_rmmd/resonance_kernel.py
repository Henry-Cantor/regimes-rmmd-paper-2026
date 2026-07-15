"""Resonance kernel modules for STRONG-RMMD.

This module models off-diagonal dissipative coupling using a compact sum of
Lorentzian-weighted latent mode outer products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ResonanceKernelOutput:
    d_res: torch.Tensor
    d_psd: torch.Tensor
    lorentz_weights: torch.Tensor
    amplitudes: torch.Tensor
    gammas: torch.Tensor


class LorentzianResonanceKernel(nn.Module):
    """Construct off-diagonal resonance-mediated dissipation matrices.

    The kernel computes
        D_psd = sum_k a_k(z) * L_k(omega_t, omega_d, gamma_k(z)) * (v_k v_k^T)
    and then returns
        D_res = D_psd with diagonal removed.

    `D_psd` is symmetric positive semidefinite by construction. `D_res` is the
    off-diagonal coupling matrix used by RMMD updates.
    """

    def __init__(
        self,
        latent_dim: int,
        n_harmonics: int = 4,
        context_dim: int = 0,
        min_gamma: float = 1e-3,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if n_harmonics <= 0:
            raise ValueError("n_harmonics must be > 0")

        self.latent_dim = latent_dim
        self.n_harmonics = n_harmonics
        self.min_gamma = min_gamma

        # Harmonic basis vectors are learned and shared across machines.
        # Keep mode vectors numerically bounded via tanh in forward pass.
        self.mode_vectors = nn.Parameter(torch.randn(n_harmonics, latent_dim) * 0.05)
        # maximum amplitude scale for learned spectral amplitudes
        self.amp_scale = float(2.0)

        in_dim = latent_dim + 2 + context_dim
        self.amp_head = nn.Sequential(
            nn.Linear(in_dim, 2 * n_harmonics),
            nn.SiLU(),
            nn.Linear(2 * n_harmonics, n_harmonics),
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(in_dim, 2 * n_harmonics),
            nn.SiLU(),
            nn.Linear(2 * n_harmonics, n_harmonics),
        )

        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonic_indices", harmonics)

    def _lorentz_weights(
        self,
        omega_t: torch.Tensor,
        omega_d: torch.Tensor,
        gammas: torch.Tensor,
    ) -> torch.Tensor:
        centers = omega_d[:, None] * self.harmonic_indices[None, :]
        detuning = omega_t[:, None] - centers
        return 1.0 / (1.0 + (detuning / gammas) ** 2)

    def forward(
        self,
        z: torch.Tensor,
        omega_t: torch.Tensor,
        omega_d: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> ResonanceKernelOutput:
        if z.ndim != 2:
            raise ValueError("z must have shape (batch, latent_dim)")

        bsz, k = z.shape
        if k != self.latent_dim:
            raise ValueError(f"expected latent_dim={self.latent_dim}, got {k}")

        omega_t = omega_t.reshape(bsz).to(z.dtype)
        omega_d = omega_d.reshape(bsz).to(z.dtype)

        pieces = [z, omega_t[:, None], omega_d[:, None]]
        if context is not None:
            pieces.append(context)
        inp = torch.cat(pieces, dim=-1)

        # Bound amplitudes to a limited positive range to prevent large PSD magnitudes.
        amplitudes = torch.sigmoid(self.amp_head(inp)) * self.amp_scale
        gammas = F.softplus(self.gamma_head(inp)) + self.min_gamma
        lorentz = self._lorentz_weights(omega_t, omega_d, gammas)
        weights = amplitudes * lorentz

        # Bound mode vectors to keep outer products numerically stable.
        mode_vecs_bounded = torch.tanh(self.mode_vectors) * 0.5
        mode_outer = mode_vecs_bounded[:, :, None] * mode_vecs_bounded[:, None, :]
        d_psd = torch.einsum("bm,mij->bij", weights, mode_outer)
        d_psd = 0.5 * (d_psd + d_psd.transpose(-1, -2))

        d_res = d_psd.clone()
        d_res.diagonal(dim1=-2, dim2=-1).zero_()
        return ResonanceKernelOutput(
            d_res=d_res,
            d_psd=d_psd,
            lorentz_weights=lorentz,
            amplitudes=amplitudes,
            gammas=gammas,
        )
