"""Resonance frequency utilities for STRONG-RMMD (gyro-Bohm normalized).

These frequencies (omega_t, omega_d) condition the Lorentzian resonance kernel in
the RMMD block, and they are the physical anchor for the Spectral Universality
Theorem (SUT). For "the Koopman/dissipation spectrum is universal across tokamaks"
to be a *physics* statement rather than a fitting artifact, the frequencies fed to
the kernel must be expressed in proper gyro-Bohm (GB) normalized units — i.e.
normalized by the GB rate c_s / a — so a NSTX mode and an ITER mode are compared
on the same dimensionless axis.

Definitions (all dimensionless, GB-normalized):

    c_s     = sqrt(T_e / m_i)            ion sound speed
    Omega_i = e B / m_i                  ion cyclotron frequency
    rho_s   = c_s / Omega_i              ion-sound gyroradius
    rho*    = rho_s / a                   GB small parameter
    omega_GB = c_s / a                    gyro-Bohm rate (normalization)

    omega_d  (drift / diamagnetic, sets resonance harmonic centers m*omega_d):
        omega*_n / omega_GB = (c_s / L_n) / (c_s / a) = a / L_n
        -> the GB-normalized density-gradient drift frequency. Computed from the
           NI profile gradient (available at train time), so it varies per shot.

    omega_t  (transit / operating frequency, tested against the resonances):
        omega_transit / omega_GB = (v_th / (q R)) / (c_s / a) ~ (a / (q R))
        -> a GB-normalized transit frequency from machine geometry/scalars; varies
           across machines (inverse aspect ratio / safety factor). Wave-particle
           resonance occurs when omega_t ~ m * omega_d (transit-drift resonance).

This keeps the kernel inputs O(1) and, crucially, makes the *learned* mode
frequencies read off the operator (see sut_analysis.py) already GB-normalized.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

# Physical constants (SI)
_E_CHARGE = 1.602176634e-19      # C
_M_PROTON = 1.67262192369e-27    # kg
_KEV_TO_J = 1.0e3 * _E_CHARGE    # keV -> Joules


def _scale_length(profile: np.ndarray, grid: np.ndarray) -> float:
    """Median normalized gradient scale length L = |f / (df/dx)| on the rho grid.

    Returned in units of the grid (rho in [0, 1]), i.e. as a fraction of the
    minor radius. a / L is then simply 1 / L (the minor radius cancels), which is
    exactly the GB-normalized inverse gradient scale length.
    """
    arr = np.asarray(profile, dtype=np.float64).reshape(-1)
    x = np.asarray(grid, dtype=np.float64).reshape(-1)
    if arr.size < 2 or x.size < 2 or arr.size != x.size:
        return 1.0
    grad = np.gradient(arr, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.abs(arr / (grad + 1e-12))
    local = local[np.isfinite(local)]
    local = local[(local > 1e-3) & (local < 1e3)]
    return float(np.median(local)) if local.size else 1.0


def _first(meta: Mapping[str, float], keys, default: float) -> float:
    for k in keys:
        if k in meta and meta[k] is not None:
            try:
                v = float(meta[k])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                continue
    return float(default)


def compute_resonance_frequencies(
    profiles: Mapping[str, np.ndarray],
    machine_name: str,
    cdf_metadata: Mapping[str, float],
) -> Dict[str, float]:
    """Compute gyro-Bohm normalized resonance frequencies for one shot.

    Returns a dict with omega_t, omega_d (GB-normalized, fed to the kernel) plus
    physical diagnostics (rho_star, c_s, omega_gb, ...) used by SUT analysis.
    Backward-compatible keys l_n, l_t, l_ti are retained.
    """
    meta = cdf_metadata or {}

    te = np.asarray(profiles.get("TE", []), dtype=np.float64)
    ti = np.asarray(profiles.get("TI", []), dtype=np.float64)
    ne = np.asarray(profiles.get("NE", []), dtype=np.float64)
    ni = np.asarray(profiles.get("NI", []), dtype=np.float64)

    # Density gradient drive comes from whichever density profile is present
    # (NI in compact training; NE otherwise).
    dens = ni if ni.size else ne
    n = max(te.size, ti.size, ne.size, ni.size, 2)
    rho = np.linspace(0.0, 1.0, n, dtype=np.float64)

    l_n = _scale_length(dens if dens.size else rho, rho)      # density scale length (rho units)
    l_t = _scale_length(te if te.size else rho, rho)          # Te scale length
    l_ti = _scale_length(ti if ti.size else rho, rho)         # Ti scale length

    # --- Physical machine scalars (with robust fallbacks) ---
    b_field = abs(_first(meta, ("B_T", "BTDIA", "BT", "B0"), 1.0))                  # T
    r_major = abs(_first(meta, ("R_major", "RAXIS", "RMAJOR", "RGEO"), 1.0))        # m
    a_minor = abs(_first(meta, ("a_minor", "RMINOR", "AMIN", "RMAJB"), 0.5))        # m
    te_kev = abs(_first(meta, ("TE_keV", "TEKEV", "TE0", "TE_AXIS", "TE"), 1.0))    # keV
    q_safety = abs(_first(meta, ("q95", "Q95", "QEDGE", "QAXIS", "q"), 2.0))        # -
    a_mass = abs(_first(meta, ("AMAIN", "A_MAIN", "ZA", "AMASS"), 2.0))             # amu (D=2)

    a_minor = max(a_minor, 1e-3)
    r_major = max(r_major, 1e-3)
    q_safety = max(q_safety, 0.5)
    m_i = max(a_mass, 1.0) * _M_PROTON

    # Gyro-Bohm reference quantities.
    t_e_j = max(te_kev, 1e-3) * _KEV_TO_J
    c_s = float(np.sqrt(t_e_j / m_i))                 # m/s
    omega_ci = _E_CHARGE * b_field / m_i              # rad/s
    rho_s = c_s / max(omega_ci, 1e-9)                 # m
    rho_star = rho_s / a_minor                         # -
    omega_gb = c_s / a_minor                           # rad/s (GB rate)

    # GB-normalized frequencies fed to the resonance kernel.
    # omega_d: density diamagnetic drift = a / L_n  (minor radius cancels in 1/L_n).
    omega_d = 1.0 / max(l_n, 1e-3)
    # omega_t: transit frequency in GB units ~ a / (q R) (v_th ~ c_s reference).
    omega_t = a_minor / (q_safety * r_major)

    # Keep inputs in a numerically comfortable O(1-10) band.
    omega_d = float(np.clip(omega_d, 1e-3, 50.0))
    omega_t = float(np.clip(omega_t, 1e-4, 50.0))

    return {
        "omega_t": omega_t,
        "omega_d": omega_d,
        # physical diagnostics (used by SUT normalization / reporting)
        "omega_gb": float(omega_gb),
        "rho_star": float(rho_star),
        "c_s": float(c_s),
        "rho_s": float(rho_s),
        "q": float(q_safety),
        "aspect_ratio": float(r_major / a_minor),
        # backward-compatible scale lengths
        "l_n": float(l_n),
        "l_t": float(l_t),
        "l_ti": float(l_ti),
    }
