"""Theorem validation implementations for STRONG-RMMD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class GITResult:
    """Generalized Ito Theorem validation."""
    kl_predicted: torch.Tensor
    kl_measured: torch.Tensor
    r_squared: float
    c_scaling: float
    passes: bool


@dataclass
class SUTResult:
    """Spectral Universality Theorem validation."""
    eigenvalues: Dict[str, torch.Tensor]
    mode_indices_per_machine: Dict[str, np.ndarray]
    relative_variance: float
    passes: bool


def validate_git(
    nrmse_horizon_t: np.ndarray,
    d_res_frobenius: np.ndarray,
    time_lags: np.ndarray,
) -> GITResult:
    """
    Validate GIT: KL divergence ∝ T · ||D_res||_F^2

    Expected relationship: KL ∝ T · ||D||²
    where T is time/horizon and D_res is off-diagonal dissipation strength.
    """
    if len(nrmse_horizon_t) < 10:
        return GITResult(
            kl_predicted=torch.tensor([]),
            kl_measured=torch.tensor([]),
            r_squared=0.0,
            c_scaling=0.0,
            passes=False,
        )

    nrmse_arr = np.asarray(nrmse_horizon_t, dtype=np.float32)
    d_arr = np.asarray(d_res_frobenius, dtype=np.float32)
    t_arr = np.asarray(time_lags, dtype=np.float32)

    predicted_variable = t_arr * (d_arr ** 2)
    measured_variable = nrmse_arr ** 2

    valid = np.isfinite(predicted_variable) & np.isfinite(measured_variable)
    predicted_variable = predicted_variable[valid]
    measured_variable = measured_variable[valid]

    if len(predicted_variable) < 3:
        return GITResult(
            kl_predicted=torch.tensor(predicted_variable),
            kl_measured=torch.tensor(measured_variable),
            r_squared=0.0,
            c_scaling=0.0,
            passes=False,
        )

    corr = np.corrcoef(predicted_variable, measured_variable)[0, 1]
    r_sq = corr ** 2 if np.isfinite(corr) else 0.0

    coeffs = np.polyfit(predicted_variable, measured_variable, 1)
    c_scale = coeffs[0]

    passes = r_sq > 0.95
    return GITResult(
        kl_predicted=torch.tensor(predicted_variable),
        kl_measured=torch.tensor(measured_variable),
        r_squared=r_sq,
        c_scaling=c_scale,
        passes=passes,
    )


def validate_sut(
    eigenvalue_dict: Dict[str, np.ndarray],
    n_top_modes: int | None = None,
) -> SUTResult:
    """
    Validate SUT: Top K modes are universal across machines.

    Expected: relative variance of top N₀=K/4 eigenvalues across machines < 0.2
    Indicates these modes are machine-independent (universal).
    """
    if not eigenvalue_dict:
        return SUTResult(
            eigenvalues={},
            mode_indices_per_machine={},
            relative_variance=1.0,
            passes=False,
        )

    machine_names = list(eigenvalue_dict.keys())
    if len(machine_names) < 2:
        return SUTResult(
            eigenvalues=eigenvalue_dict,
            mode_indices_per_machine={},
            relative_variance=1.0,
            passes=False,
        )

    eigs_list = []
    for machine_name in machine_names:
        eigs = np.sort(np.abs(np.real(eigenvalue_dict[machine_name])))[::-1]
        eigs_list.append(eigs)

    max_modes = min(len(eigs) for eigs in eigs_list)
    n_top = n_top_modes or max(1, max_modes // 4)
    n_top = min(n_top, max_modes)

    if n_top < 1:
        return SUTResult(
            eigenvalues=eigenvalue_dict,
            mode_indices_per_machine={},
            relative_variance=1.0,
            passes=False,
        )

    top_eigs = np.array([eigs[:n_top] for eigs in eigs_list])
    mean_eigs = np.mean(top_eigs, axis=0)
    std_eigs = np.std(top_eigs, axis=0)
    relative_var = float(np.mean(std_eigs / (mean_eigs + 1e-8)))

    passes = relative_var < 0.2

    mode_indices = {name: np.arange(n_top) for name in machine_names}
    return SUTResult(
        eigenvalues=eigenvalue_dict,
        mode_indices_per_machine=mode_indices,
        relative_variance=relative_var,
        passes=passes,
    )


def validate_strong_bound(
    residuals_true_pred: Dict[str, np.ndarray],
    d_res_matrices: Dict[str, np.ndarray],
    threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Validate STRONG bound: E||error|| ≤ C₁ · (linear) + C₂ · (D_res)

    Returns per-machine correlation coefficients and fit quality.
    """
    results = {}

    for machine_name, residual_array in residuals_true_pred.items():
        if machine_name not in d_res_matrices:
            results[machine_name] = {"r_squared": 0.0, "passes": False}
            continue

        residual_norm_arr = np.linalg.norm(residual_array.reshape(residual_array.shape[0], -1), axis=1)
        d_res_norm = np.linalg.norm(d_res_matrices[machine_name], ord="fro")

        if np.max(residual_norm_arr) < 1e-8 or d_res_norm < 1e-8:
            results[machine_name] = {"r_squared": 0.0, "passes": False}
            continue

        predicted_bound = d_res_norm * np.ones_like(residual_norm_arr)
        valid = np.isfinite(residual_norm_arr) & np.isfinite(predicted_bound)
        corr = np.corrcoef(residual_norm_arr[valid], predicted_bound[valid])[0, 1]
        r_sq = corr ** 2 if np.isfinite(corr) else 0.0

        results[machine_name] = {"r_squared": r_sq, "passes": r_sq > 0.1}

    return results
