"""Time-domain memory kernel utilities for TIER 3.

Provides Lorentzian resonance kernel K_ij(tau) and frequency-integrated D_res construction.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def lorentzian_time_kernel(tau: torch.Tensor, gamma: float, omega0: float) -> torch.Tensor:
    """Compute Lorentzian-like damped cosine kernel K(tau) = exp(-gamma*tau/2) * cos(omega0*tau).

    Args:
        tau: tensor of time lags (any shape)
        gamma: damping rate (scalar)
        omega0: resonant frequency (scalar)

    Returns:
        Tensor same shape as tau
    """
    return torch.exp(-0.5 * gamma * tau) * torch.cos(omega0 * tau)


def frequency_integrated_resonance(d_weights: torch.Tensor, taus: torch.Tensor, gamma: float, omega0: float) -> torch.Tensor:
    """Given spectral weights per mode and a set of taus, return time-domain D_res(taus).

    d_weights: (..., n_modes) spectral amplitudes
    taus: (n_taus,) time lag samples
    returns: (..., n_taus) kernel values
    """
    # Ensure shapes: broadcast d_weights over taus
    w = d_weights
    taus = taus.reshape(-1)
    # compute kernel for each mode and tau
    # w: (..., n_modes), taus: (n_taus,)
    # result: (..., n_taus, n_modes) then sum over modes
    gamma_t = float(gamma)
    omega0_t = float(omega0)
    tau_mat = taus.unsqueeze(0)  # (1, n_taus)
    k = torch.exp(-0.5 * gamma_t * tau_mat) * torch.cos(omega0_t * tau_mat)  # (1, n_taus)
    # if w has modes, sum modes weighted by mean amplitude as simple proxy
    if w.ndim >= 1:
        # reduce modes to a single amplitude per batch
        amps = w.mean(dim=-1, keepdim=True)  # (..., 1)
        k = k * amps.unsqueeze(-1)  # (..., n_taus)
        k = k.squeeze(-1)
    return k
