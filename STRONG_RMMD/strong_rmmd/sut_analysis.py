"""Spectral Universality Theorem (SUT) analysis. Legacy (SUT believed to be result, but circular is it is a loss method)

Tests whether the learned Koopman/dissipation spectrum is consistent across tokamaks when expressed in
gyro-Bohm normalized units. For each machine, its t=0 shots are pushed through the model's encoder to obtain a
representative latent operating point and context, and the operator spectra are extracted and compared across
machines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch import nn


# Normalized unit step (must match the rollout's dt encoding).
def _normalized_step_dt(dt: float = 1.0, dt_max: float = 1000.0) -> float:
    return float(math.log1p(dt) / math.log1p(dt_max))


@dataclass
class MachineSpectrum:
    machine: str
    eigenvalues: np.ndarray          # complex (D,)
    frequencies: np.ndarray          # selected top-N0 mode frequencies (sorted asc by |f|)
    damping: np.ndarray              # matching damping for selected modes
    omega_t: float = 0.0
    omega_d: float = 0.0
    omega_gb: float = 0.0
    rho_star: float = 0.0


@dataclass
class SUTResult:
    n_modes: int
    machines: List[str] = field(default_factory=list)
    spectra: Dict[str, MachineSpectrum] = field(default_factory=dict)
    sigma_over_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    null_sigma_over_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frac_below_threshold: float = 0.0
    null_frac_below_threshold: float = 0.0
    threshold: float = 0.20
    passed: bool = False

    def summary(self) -> Dict[str, object]:
        return {
            "n_machines": len(self.machines),
            "machines": self.machines,
            "n_modes": int(self.n_modes),
            "threshold": float(self.threshold),
            "sigma_over_mu": [float(x) for x in self.sigma_over_mu],
            "null_sigma_over_mu": [float(x) for x in self.null_sigma_over_mu],
            "frac_modes_below_threshold": float(self.frac_below_threshold),
            "null_frac_modes_below_threshold": float(self.null_frac_below_threshold),
            "median_sigma_over_mu": float(np.median(self.sigma_over_mu)) if self.sigma_over_mu.size else float("nan"),
            "null_median_sigma_over_mu": float(np.median(self.null_sigma_over_mu)) if self.null_sigma_over_mu.size else float("nan"),
            "passed": bool(self.passed),
            "per_machine": {
                m: {
                    "omega_t": s.omega_t,
                    "omega_d": s.omega_d,
                    "omega_gb": s.omega_gb,
                    "rho_star": s.rho_star,
                    "frequencies": [float(x) for x in s.frequencies],
                    "damping": [float(x) for x in s.damping],
                }
                for m, s in self.spectra.items()
            },
        }


@torch.no_grad()
def compact_latent_and_context(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    machine_names: Sequence[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror MultiMachineRMMD compact-mode encoder to recover (z, context).

    Returns z (B, latent_dim) and context (B, context_width) exactly as the model
    builds them before the RMMD latent step.
    """
    pre_shot = batch_data["pre_shot_context"].to(device)
    limiter = batch_data["limiter_geometry_tensor"].to(device)
    machine_idx = model._machine_indices(machine_names).to(device)
    machine_emb = model.machine_embedding(machine_idx)

    z_preshot = model.pre_shot_initializer(pre_shot)
    ni_current = batch_data["ni_t0"].to(device)
    z_preshot = z_preshot + model.state_legacy_encoder(ni_current.reshape(ni_current.shape[0], -1))
    z = model.koopman_encoder(z_preshot)

    step_dt = batch_data.get("step_dt")
    if step_dt is None:
        step_dt = torch.full((z.shape[0], 1), _normalized_step_dt(1.0), device=device)
    z = z + model.compact_dt_encoder(step_dt.to(device).view(-1, 1).to(z.dtype))

    geom_current = batch_data.get("geometry_tensor")
    if isinstance(geom_current, torch.Tensor):
        geom_context = model.geom_context_norm(
            model.limiter_geometry_encoder(limiter.to(device))
            + model.geometry_encoder(geom_current.to(device))
        )
    else:
        geom_context = model.geom_context_norm(model.limiter_geometry_encoder(limiter.to(device)))

    context = model.rmmd_context_norm(
        model.rmmd_context_proj(torch.cat([machine_emb, geom_context], dim=-1))
    )
    return z, context


def latent_jacobian_eigenvalues(
    model: nn.Module,
    z0: torch.Tensor,
    context0: torch.Tensor,
    omega_t: float,
    omega_d: float,
    device: str,
) -> np.ndarray:
    """Eigenvalues of d z_next / d z_t at operating point z0 (one sample)."""
    z0 = z0.detach().to(device).reshape(-1)
    context0 = context0.detach().to(device).reshape(-1)
    ot = torch.tensor([float(omega_t)], device=device)
    od = torch.tensor([float(omega_d)], device=device)

    def step(z: torch.Tensor) -> torch.Tensor:
        zt = z.reshape(1, -1)
        out = model.rmmd(x_t=zt, omega_t=ot, omega_d=od, context=context0.reshape(1, -1), z_t=zt)
        return out.z_next.reshape(-1)

    jac = torch.autograd.functional.jacobian(step, z0, vectorize=True)
    jac = jac.detach().to(torch.float64)
    eig = torch.linalg.eigvals(jac)
    return eig.cpu().numpy()


def _select_top_modes(eig: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick the n_modes least-damped (|lambda| closest to 1) modes; return (freq, damping)
    sorted ascending by |frequency|. Conjugate pairs collapse to |frequency|."""
    freq = np.angle(eig)                      # rad/step, in (-pi, pi]
    damping = -np.log(np.abs(eig) + 1e-12)    # >0 = decaying
    # Persistence = closeness of |lambda| to 1 (least damped first).
    order = np.argsort(np.abs(damping))
    sel = order[: max(1, n_modes)]
    f_sel = np.abs(freq[sel])
    d_sel = damping[sel]
    asc = np.argsort(f_sel)
    return f_sel[asc], d_sel[asc]


def machine_spectrum(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    machine_names: Sequence[str],
    omega_t: float,
    omega_d: float,
    device: str,
    n_modes: int,
    n_probe: int = 8,
) -> MachineSpectrum:
    """Build the GB-normalized spectrum for one machine at its mean operating point."""
    z, context = compact_latent_and_context(model, batch_data, machine_names, device)
    n = min(n_probe, z.shape[0])
    z0 = z[:n].mean(dim=0)
    context0 = context[:n].mean(dim=0)
    eig = latent_jacobian_eigenvalues(model, z0, context0, omega_t, omega_d, device)
    freq, damp = _select_top_modes(eig, n_modes)
    return MachineSpectrum(
        machine=str(machine_names[0]),
        eigenvalues=eig,
        frequencies=freq,
        damping=damp,
        omega_t=float(omega_t),
        omega_d=float(omega_d),
    )


def _sigma_over_mu(freqs_by_machine: List[np.ndarray]) -> np.ndarray:
    """Per-aligned-mode sigma/mu across machines. Modes aligned by ascending |freq|."""
    n_modes = min(len(f) for f in freqs_by_machine)
    if n_modes == 0:
        return np.zeros(0)
    stack = np.stack([f[:n_modes] for f in freqs_by_machine], axis=0)  # (M, n_modes)
    mu = np.mean(stack, axis=0)
    sigma = np.std(stack, axis=0)
    return sigma / (np.abs(mu) + 1e-9)


def run_sut_test(
    model: nn.Module,
    machine_batches: Dict[str, Dict[str, torch.Tensor]],
    machine_omegas: Dict[str, tuple[float, float]],
    device: str = "cpu",
    n_modes: int | None = None,
    threshold: float = 0.20,
    null_seed: int = 0,
    machine_extras: Dict[str, Dict[str, float]] | None = None,
) -> SUTResult:
    """Run the SUT universality test + permuted-omega null model.

    machine_batches : {machine -> collated batch dict (t=0 fields)}
    machine_omegas  : {machine -> (omega_t, omega_d)} GB-normalized
    n_modes         : top-N0 modes to compare (default latent_dim // 4)
    """
    model.eval()
    machines = sorted(machine_batches.keys())
    latent_dim = int(getattr(model, "latent_dim", 256))
    if n_modes is None:
        n_modes = max(1, latent_dim // 4)

    spectra: Dict[str, MachineSpectrum] = {}
    for m in machines:
        ot, od = machine_omegas.get(m, (1.0, 1.0))
        names = [m] * int(machine_batches[m]["ni_t0"].shape[0])
        spec = machine_spectrum(model, machine_batches[m], names, ot, od, device, n_modes)
        if machine_extras and m in machine_extras:
            spec.omega_gb = float(machine_extras[m].get("omega_gb", 0.0))
            spec.rho_star = float(machine_extras[m].get("rho_star", 0.0))
        spectra[m] = spec

    freqs = [spectra[m].frequencies for m in machines]
    som = _sigma_over_mu(freqs)
    frac = float(np.mean(som < threshold)) if som.size else 0.0

    # NULL: permute GB omegas across machines (wrong normalization) and recompute.
    rng = np.random.default_rng(null_seed)
    perm = list(machines)
    if len(perm) > 1:
        while True:
            shuffled = list(rng.permutation(perm))
            if any(a != b for a, b in zip(shuffled, perm)):
                break
    else:
        shuffled = perm
    null_freqs: List[np.ndarray] = []
    for m, src in zip(machines, shuffled):
        ot, od = machine_omegas.get(src, (1.0, 1.0))  # mismatched omega
        names = [m] * int(machine_batches[m]["ni_t0"].shape[0])
        spec = machine_spectrum(model, machine_batches[m], names, ot, od, device, n_modes)
        null_freqs.append(spec.frequencies)
    null_som = _sigma_over_mu(null_freqs)
    null_frac = float(np.mean(null_som < threshold)) if null_som.size else 0.0

    result = SUTResult(
        n_modes=n_modes,
        machines=machines,
        spectra=spectra,
        sigma_over_mu=som,
        null_sigma_over_mu=null_som,
        frac_below_threshold=frac,
        null_frac_below_threshold=null_frac,
        threshold=threshold,
    )
    # PASS: real spectra agree (most modes below threshold) AND the null is clearly
    # worse (universality is GB-driven, not trivial).
    result.passed = bool(frac >= 0.75 and frac > null_frac + 0.20)
    return result


__all__ = [
    "MachineSpectrum",
    "SUTResult",
    "compact_latent_and_context",
    "latent_jacobian_eigenvalues",
    "machine_spectrum",
    "run_sut_test",
]
